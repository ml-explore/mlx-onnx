import mlx.core as mx
from onnx import TensorProto

DTYPE_MAP = {
    TensorProto.FLOAT: mx.float32,
    TensorProto.UINT8: mx.uint8,
    TensorProto.INT8: mx.int8,
    TensorProto.UINT16: mx.uint16,
    TensorProto.INT16: mx.int16,
    TensorProto.INT32: mx.int32,
    TensorProto.INT64: mx.int64,
    TensorProto.BOOL: mx.bool_,
    TensorProto.FLOAT16: mx.float16,
    TensorProto.UINT32: mx.uint32,
    TensorProto.UINT64: mx.uint64,
    TensorProto.BFLOAT16: mx.bfloat16,
    TensorProto.COMPLEX64: mx.complex64,
}


def dtype_helper(dtype: TensorProto.DataType) -> mx.Dtype:
    assert dtype in DTYPE_MAP, f"Unsupported dtype {dtype}"
    return DTYPE_MAP[dtype]
