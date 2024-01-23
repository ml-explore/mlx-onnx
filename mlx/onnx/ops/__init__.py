import functools
import math
from typing import List, Optional, Union

import mlx.core as mx
import mlx.nn.layers as layers
import mlx.nn.losses as losses
import onnx

from .helper import DTYPE_MAP
from .op_conv import Conv
from .op_depth import DepthToSpace, SpaceToDepth
from .op_dropout import Dropout
from .op_image import ImageDecoder
from .op_lrn import LRN
from .op_norm import (
    BatchNormalization,
    GroupNormalization,
    InstanceNormalization,
    LayerNormalization,
)
from .op_onehot import OneHot
from .op_pad import Pad
from .op_pool import AveragePool, MaxPool
from .op_sequence import (
    ConcatFromSequence,
    SequenceAt,
    SequenceConstruct,
    SequenceEmpty,
    SequenceErase,
    SequenceInsert,
    SequenceLength,
    SplitToSequence,
)
from .op_slice import Slice
from .op_split import Split
from .op_topk import TopK
from .op_window import BlackmanWindow, HammingWindow, HannWindow

# Reference Docs: https://onnx.ai/onnx/operators/


def Add(x: mx.array, y: mx.array, broadcast=None, axis=None):
    return x + y


def Sub(x: mx.array, y: mx.array):
    return x - y


def Mul(x: mx.array, y: mx.array):
    return x * y


def Div(x: mx.array, y: mx.array):
    return x / y


def Neg(x: mx.array):
    return -x


def Pow(x: mx.array, y: mx.array):
    return (x**y).astype(x.dtype)


def Sqrt(x: mx.array):
    return x.sqrt()


def Abs(x: mx.array):
    return x.abs()


def Exp(x: mx.array):
    return x.exp()


def Log(x: mx.array):
    return x.log()


def Sin(x: mx.array):
    return x.sin()


def Sinh(x: mx.array):
    return mx.sinh(x)


def Asin(x: mx.array):
    return mx.arcsin(x)


def Asinh(x: mx.array):
    return mx.arcsinh(x)


def Cos(x: mx.array):
    return x.cos()


def Cosh(x: mx.array):
    return mx.cosh(x)


def Acos(x: mx.array):
    return mx.arccos(x)


def Acosh(x: mx.array):
    return mx.arccosh(x)


def Tan(x: mx.array):
    return x.sin() / x.cos()


def Tanh(x: mx.array):
    return mx.sinh(x) / mx.cosh(x)


def Atan(x: mx.array):
    return mx.arctan(x)


def Atanh(x: mx.array):
    return mx.arctanh(x)


def Relu(x: mx.array):
    return layers.relu(x)


def Floor(x: mx.array):
    return mx.floor(x)


def Ceil(x: mx.array):
    return mx.ceil(x)


def Sigmoid(x: mx.array):
    return mx.sigmoid(x)


def Sign(x: mx.array):
    return mx.sign(x)


def Softplus(x: mx.array):
    return layers.softplus(x)


def HardSwish(x: mx.array):
    return layers.hardswish(x)


def HardSigmoid(x: mx.array, alpha=0.2, beta=0.5):
    return mx.clip(x * alpha + beta, 0, 1)


def Softsign(x: mx.array):
    return layers.softsign(x)


def MatMul(x: mx.array, y: mx.array):
    return x @ y


def MatMulInteger(
    x: mx.array,
    y: mx.array,
    a_zero_point: Optional[mx.array] = None,
    b_zero_point: Optional[mx.array] = None,
):
    x = x.astype(mx.float32)
    y = y.astype(mx.float32)
    if a_zero_point is not None:
        x = x - a_zero_point
    if b_zero_point is not None:
        y = y - b_zero_point
    return (x @ y).astype(mx.int32)


def Cast(x: mx.array, to: int, saturate=1):
    if to == onnx.TensorProto.DOUBLE:
        raise NotImplementedError("mlx does not support double data type")
    return x.astype(DTYPE_MAP[to])


def CastLike(x: mx.array, target_type: mx.array, saturate=1):
    return x.astype(target_type.dtype)


def ConstantOfShape(x: mx.array, value: mx.array = None):
    if value is None:
        value = mx.array([0])
    shape = x.tolist()
    return mx.ones(shape, dtype=value.dtype) * (value if shape[0] != 0 else 1)


def Tile(x: mx.array, repeats: mx.array):
    return mx.tile(x, repeats.tolist())


def Shape(x: mx.array, end=None, start=0):
    return mx.array(x.shape[start:end], dtype=mx.int64)


def Constant(
    value: mx.array = None,
    value_float=None,
    value_floats=None,
    value_int=None,
    value_ints=None,
    value_string=None,
    value_strings=None,
):
    if value is not None:
        return value
    if value_float is not None:
        return mx.array(value_float, dtype=mx.float32)
    if value_floats is not None:
        return mx.array(list(value_floats), dtype=mx.float32)
    if value_int is not None:
        return mx.array(value_int, dtype=mx.int32)
    if value_ints is not None:
        return mx.array(list(value_ints), dtype=mx.int32)
    if value_string is not None or value_strings is not None:
        raise NotImplementedError()


def Less(x: mx.array, y: mx.array):
    return x < y


def LessOrEqual(x: mx.array, y: mx.array):
    return x <= y


def Equal(x: mx.array, y: mx.array):
    return x == y


def Greater(x: mx.array, y: mx.array):
    return x > y


def GreaterOrEqual(x: mx.array, y: mx.array):
    return x >= y


def Where(condition: mx.array, x: mx.array, y: mx.array):
    return mx.where(condition, x, y)


def LeakyRelu(x: mx.array, alpha=0.01):
    return layers.leaky_relu(x, alpha)


def And(x: mx.array, y: mx.array):
    return x & y


def Or(x: mx.array, y: mx.array):
    return x | y


def Trilu(x: mx.array, k=0, upper=1):
    if isinstance(k, mx.array):
        k = k.item()
    return mx.triu(x, k) if upper else mx.tril(x, k)


def Transpose(x: mx.array, perm: mx.array = None):
    return x.transpose() if perm is None else x.transpose(perm.tolist())


def Identity(x: mx.array):
    return x


def Sum(*args: List[mx.array]):
    return functools.reduce(mx.array.__add__, args)


def Mean(*args: List[mx.array]):
    return Sum(*args) / len(args)


def Max(*args: List[mx.array]):
    return functools.reduce(mx.maximum, args)


def Min(*args: List[mx.array]):
    return functools.reduce(mx.minimum, args)


def Elu(x: mx.array, alpha=1.0):
    return layers.elu(x, alpha)


def Celu(x: mx.array, alpha=1.0):
    return layers.celu(x, alpha)


def Reciprocal(x: mx.array):
    return x.reciprocal()


def Mish(x: mx.array):
    return layers.mish(x)


def PRelu(x: mx.array, slope: mx.array):
    slope = slope[0] if slope.shape[-1] != x.shape[-1] else slope
    return layers.prelu(x, slope)


def Selu(x: mx.array, alpha=1.67326319217681884765625, gamma=1.05070102214813232421875):
    return gamma * (layers.relu(x) - layers.relu(-alpha * x.exp() + alpha))


def Clip(x: mx.array, min=float("-inf"), max=float("inf")):
    if max is None:
        return x
    if min is None:
        return x
    return mx.clip(x, min, max).astype(x.dtype)


def Range(start: mx.array, limit: mx.array, delta: mx.array):
    return mx.arange(start.item(), limit.item(), delta.item())


def Size(x: Union[mx.array, list[int]]):
    return mx.array(math.prod(x if isinstance(x, list) else x.shape), dtype=mx.int64)


def Shrink(x: mx.array, bias=0.0, lambd=0.5):
    return (x < -lambd) * (x + bias) + (x > lambd) * (x - bias)


def Reshape(x: mx.array, shape: mx.array, allowzero=0):
    new_shape = [
        int(d) if d != 0 else (0 if allowzero else x.shape[i])
        for i, d in enumerate(shape.tolist())
    ]
    return x.reshape(new_shape)


def Squeeze(x: mx.array, axes: mx.array = None):
    return mx.squeeze(x, axes.tolist() if axes is not None else None)


def Unsqueeze(x: mx.array, axes: mx.array):
    return mx.expand_dims(x, axes.tolist())


def Flatten(x: mx.array, axis=1):
    new_shape = math.prod([1] + x.shape[:axis])
    return mx.reshape(
        x,
        (
            new_shape,
            -1,
        ),
    )


def axes_helper(axes: Optional[mx.array] = None, noop_with_empty_axes=0):
    if isinstance(axes, tuple):
        return axes
    if axes is not None and isinstance(axes, mx.array) and axes.size > 0:
        return axes.tolist()
    return [] if noop_with_empty_axes else None


def ReduceMax(x: mx.array, axes: mx.array = None, keepdims=1, noop_with_empty_axes=0):
    return x.max(axes_helper(axes, noop_with_empty_axes), keepdims=keepdims)


def ReduceMin(x: mx.array, axes: mx.array = None, keepdims=1, noop_with_empty_axes=0):
    if math.prod(x.shape) == 0:
        return mx.array(float("inf")).astype(x.dtype)
    return x.min(axes_helper(axes, noop_with_empty_axes), keepdims=keepdims)


def ReduceMean(x: mx.array, axes: mx.array = None, keepdims=1, noop_with_empty_axes=0):
    return x.mean(axes_helper(axes, noop_with_empty_axes), keepdims=keepdims)


def ReduceProd(x: mx.array, axes: mx.array = None, keepdims=1, noop_with_empty_axes=0):
    return x.prod(axes_helper(axes, noop_with_empty_axes), keepdims=keepdims)


def ReduceL1(x: mx.array, axes: mx.array = None, keepdims=1, noop_with_empty_axes=0):
    return x.abs().sum(axes_helper(axes, noop_with_empty_axes), keepdims=keepdims)


def ReduceL2(x: mx.array, axes: mx.array = None, keepdims=1, noop_with_empty_axes=0):
    return (
        x.square()
        .sum(axes_helper(axes, noop_with_empty_axes), keepdims=keepdims)
        .sqrt()
    )


def ReduceSum(x: mx.array, axes: mx.array = None, keepdims=1, noop_with_empty_axes=0):
    return x.sum(axes_helper(axes, noop_with_empty_axes), keepdims=keepdims)


def ReduceLogSum(
    x: mx.array, axes: mx.array = None, keepdims=1, noop_with_empty_axes=0
):
    return x.sum(axes_helper(axes, noop_with_empty_axes), keepdims=keepdims).log()


def ReduceLogSumExp(
    x: mx.array, axes: mx.array = None, keepdims=1, noop_with_empty_axes=0
):
    return x.exp().sum(axes_helper(axes, noop_with_empty_axes), keepdims=keepdims).log()


def ReduceSumSquare(
    x: mx.array, axes: mx.array = None, keepdims=1, noop_with_empty_axes=0
):
    return x.square().sum(axes_helper(axes, noop_with_empty_axes), keepdims=keepdims)


def Concat(*args: List[mx.array], axis):
    return mx.concatenate(args, axis=axis)


def Gemm(
    A: mx.array,
    B: mx.array,
    C: Optional[mx.array] = None,
    alpha=1.0,
    beta=1.0,
    transA=0,
    transB=0,
    broadcast=0,
):
    if transA:
        A = A.transpose()
    if transB:
        B = B.transpose()
    ret = alpha * (A @ B)
    if C is not None:
        ret += beta * C
    return ret


def Softmax(x: mx.array, axis=-1):
    return layers.softmax(x, axis=axis)


def LogSoftmax(x: mx.array, axis=-1):
    return layers.log_softmax(x, axis=axis)


def Gelu(x: mx.array, approximate="none"):
    return layers.gelu(x) if approximate == "none" else layers.gelu_fast_approx(x)


def Erf(x: mx.array):
    return mx.erf(x)


def Round(x: mx.array):
    return x.round()


def ArgMax(x: mx.array, axis=0, keepdims=1, select_last_index=0):
    return mx.argmax(x, axis=axis, keepdims=keepdims).astype(mx.int64)


def ArgMin(x: mx.array, axis=0, keepdims=1, select_last_index=0):
    return mx.argmin(x, axis=axis, keepdims=keepdims).astype(mx.int64)


def Expand(x: mx.array, shape: mx.array):
    return x * mx.ones(shape.tolist())


def CumSum(x: mx.array, axis: mx.array, exclusive=0, reverse=0):
    return mx.cumsum(x, axis.item(), reverse=reverse, inclusive=not exclusive)


def EyeLike(x: mx.array, dtype=None, k=0):
    if dtype is None:
        dtype = x.dtype
    else:
        dtype = DTYPE_MAP[dtype]
    return mx.eye(x.shape[0], x.shape[1], k=k, dtype=dtype)


def Gather(x: mx.array, indices: mx.array, axis=0):
    return mx.take(x, indices, axis=axis)


def GatherElements(x: mx.array, indices: mx.array, axis=0):
    return mx.take_along_axis(x, indices, axis=axis)


def Not(x: mx.array):
    return ~x


def Mod(x: mx.array, y: mx.array, fmod=0):
    return x % y


def OptionalHasElement(x: Optional[mx.array] = None):
    return mx.array(x is not None and len(x) > 0, dtype=mx.bool_)


def OptionalGetElement(x: Optional[mx.array] = None):
    return x if x is not None else mx.array([])


def IsInf(x: mx.array, detect_negative=1, detect_positive=1):
    return (x == float("inf")) * bool(detect_positive) + (x == float("-inf")) * bool(
        detect_negative
    )


def IsNaN(x: mx.array):
    return x != x


def ThresholdedRelu(x: mx.array, alpha=1.0):
    return mx.where(x > alpha, x, 0)


def Binarizer(x: mx.array, threshold=0.0):
    return mx.where(x > threshold, 1.0, 0.0)


def GlobalAveragePool(x: mx.array):
    return x.mean(axis=tuple(range(2, x.ndim)), keepdims=True)


def GlobalMaxPool(x: mx.array):
    return x.max(axis=tuple(range(2, x.ndim)), keepdims=True)


def Xor(x: mx.array, y: mx.array):
    return mx.where(x == y, False, True)


def Compress(x: mx.array, condition: mx.array, axis=None):
    if axis is None:
        x = x.flatten()
        axis = 0
    axis = axis if axis >= 0 else axis + x.ndim

    # TODO: Replace with bool indexing when added
    temp = []
    for i, v in enumerate(condition.tolist()):
        if v:
            temp.append(i)
    temp = mx.array(temp, dtype=mx.int64)
    return x[tuple([slice(None) if i != axis else temp for i in range(x.ndim)])]
