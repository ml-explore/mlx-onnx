import importlib
import os
from typing import Any, Tuple

import mlx.core as mx
import numpy as np
import onnx
from onnx.helper import tensor_dtype_to_np_dtype

onnx_ops = importlib.import_module("mlx.onnx.ops")
DEBUG = os.getenv("DEBUG", "0") == "1"


class MlxBackendWrapper:
    @classmethod
    def prepare(cls, model: onnx.ModelProto, device: str):
        return MlxBackend(model)

    @classmethod
    def supports_device(cls, device: str) -> bool:
        return device.lower() in ["cpu", "gpu"]


class MlxBackend:
    def __init__(self, model: onnx.ModelProto):
        self._model = model
        self._cache = {}
        self.initializer_arrays()

    def initializer_arrays(self):
        for i in self._model.graph.initializer:
            if i.name in self._cache:
                continue
            self._cache[i.name] = self.parse_array(i)

    def parse_array(self, inp: onnx.TensorProto) -> mx.array:
        if inp.data_type == onnx.TensorProto.FLOAT and len(inp.float_data) > 0:
            return mx.array(
                np.array(inp.float_data, dtype=np.float32).reshape(inp.dims),
                dtype=mx.float32,
            )
        elif inp.data_type == onnx.TensorProto.INT32 and len(inp.int32_data) > 0:
            return mx.array(
                np.array(inp.int32_data, dtype=np.int32).reshape(inp.dims),
                dtype=mx.int32,
            )
        elif inp.data_type == onnx.TensorProto.INT64 and len(inp.int64_data) > 0:
            return mx.array(
                np.array(inp.int64_data, dtype=np.int64).reshape(inp.dims),
                dtype=mx.int64,
            )
        elif len(inp.raw_data) > 0:
            return mx.array(
                np.frombuffer(
                    inp.raw_data, dtype=tensor_dtype_to_np_dtype(inp.data_type)
                ).reshape(inp.dims)
            )
        else:
            raise NotImplementedError(
                f"Not implemented for {inp.data_type} {inp.name} {inp.dims}"
            )

    def get_input_dict(self, inputs):
        input_names = [x.name for x in self._model.graph.input]
        init_names = set([x.name for x in self._model.graph.initializer])
        real_inputs = [x for x in input_names if x not in init_names]
        return dict(zip(real_inputs, inputs))

    def parse_attributes(self, attrs):
        res = {}
        for x in attrs:
            if x.type == onnx.AttributeProto.FLOAT:
                res[x.name] = float(x.f)
            elif x.type == onnx.AttributeProto.INT:
                res[x.name] = int(x.i)
            elif x.type == onnx.AttributeProto.STRING:
                res[x.name] = x.s.decode("utf-8")
            elif x.type == onnx.AttributeProto.TENSOR:
                res[x.name] = self.parse_array(x.t)
            # Sometimes this gets passed as args to functions that expect mx.array, so just converting
            # them here to simplify the op code
            elif x.type == onnx.AttributeProto.FLOATS:
                res[x.name] = mx.array([float(f) for f in x.floats], dtype=mx.float32)
            elif x.type == onnx.AttributeProto.INTS:
                res[x.name] = mx.array([int(i) for i in x.ints], dtype=mx.int64)
            elif x.type == onnx.AttributeProto.STRINGS:
                res[x.name] = tuple(s.decode("utf-8") for s in x.strings)
            elif x.type == onnx.AttributeProto.GRAPH:
                raise NotImplementedError(f"Attribute type graph not implemented")
            else:
                raise NotImplementedError(f"Attribute type {x.type} not implemented")
        return res

    def run(self, *inputs, **kwargs: Any) -> Tuple[mx.array, ...]:
        self.initializer_arrays()
        inmap = self.get_input_dict(inputs)

        for i in self._model.graph.input:
            if i.name in self._cache:
                continue
            if i.name in inmap:
                if isinstance(inmap[i.name], mx.array):
                    self._cache[i.name] = inmap[i.name]
                elif isinstance(inmap[i.name], list):
                    self._cache[i.name] = [mx.array(x) for x in inmap[i.name]]
                elif isinstance(inmap[i.name], np.ndarray):
                    self._cache[i.name] = mx.array(inmap[i.name])
                elif inmap[i.name] is None:
                    self._cache[i.name] = None
                else:
                    raise NotImplementedError(
                        f"Input type {inmap[i.name]} not implemented"
                    )
        for i, node in enumerate(self._model.graph.node):
            args = [self._cache[x] if x in self._cache else None for x in node.input]
            opt = self.parse_attributes(node.attribute)
            if DEBUG:
                print(
                    f"Running op {node.input} {node.op_type} with args {len(args)} and opt {opt}"
                )
            # Special case for split as outputs might need to be inferred from node
            if node.op_type == "Split":
                if "num_outputs" not in opt and len(args) != 2:
                    opt["num_outputs"] = len(node.output)
                res = getattr(onnx_ops, node.op_type)(*args, **opt)
            elif hasattr(onnx_ops, node.op_type):
                res = getattr(onnx_ops, node.op_type)(*args, **opt)
            else:
                raise NotImplementedError(f"Operation {node.op_type} not implemented")

            if not isinstance(res, tuple):
                res = (res,)
            if len(node.output) > len(res):
                raise ValueError(
                    f"Expected {len(node.output)} outputs but got {len(res)}"
                )
            for name, out in zip(node.output, res):
                self._cache[name] = out
        return tuple(self._cache[out.name] for out in self._model.graph.output)
