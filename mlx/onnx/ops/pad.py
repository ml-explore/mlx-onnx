import mlx.core as mx
from typing import List, Union, Optional
import math

def convert_pad(onnx_pads: List[int], ndims:Optional[int]=None, axes:Optional[int]=None):
    """
    Convert onnx padding to mlx padding
    Onnx padding is [x1_begin, x2_begin...x1_end, x2_end,...]
    Mlx padding is [(x1_begin, x1_end), (x2_begin, x2_end)...]
    """
    if ndims and len(onnx_pads) // 2 != ndims:
        onnx_pads = onnx_pads * ndims
    if ndims is None:
        ndims = len(onnx_pads) // 2
    if axes is None:
        axes = list(range(ndims))
    res = [(0,0)] * ndims
    naxes = len(axes)
    for i in range(naxes):
        res[axes[i]] = (onnx_pads[i], onnx_pads[i+naxes])
    return res

def auto_pad(shape: List[int], auto_pad:str, strides: Optional[Union[int, List[int]]], kernel_shape: List[int]):
    """
    Convert auto_pad to valid padding, valid options for auto_pad are: NOTSET, SAME_UPPER, SAME_LOWER, VALID
    Default value is NOTSET which means explicit padding is used
    SAME_UPPER or SAME_LOWER means pad the input so that `out_shape[i] = ceil(in_shape[i] / strides[i])` for each axis `i`.
    """
    res = []
    if auto_pad == "NOTSET":
        return res
    if strides is None:
        strides = [1] * len(kernel_shape)
    if isinstance(strides, int):
        strides = [strides] * len(kernel_shape)
    if auto_pad in ("SAME_UPPER", "SAME_LOWER"):
        for (dim, stride, kdim) in zip(shape[-len(kernel_shape):], strides, kernel_shape):
            res.append((math.ceil(dim / stride)-1)*stride+((kdim-1)+1)-dim)
        temp = []
        for s in res:
            temp.append(s // 2)
            temp.append(s-s // 2)
        res = temp
        return res[::2] + res[1::2] if auto_pad == "SAME_UPPER" else res[1::2] + res[::2]

    raise NotImplementedError(f"auto_pad {auto_pad} not implemented")