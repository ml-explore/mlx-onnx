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
