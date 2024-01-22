import mlx.core as mx
from typing import Optional, List
from .pad import convert_pad
import math

def compute_strides(shape: List[int]):
    return list(reversed(mx.cumprod(mx.array([1] + list(reversed(shape))))[:-1].tolist()))

def MaxPool(x: mx.array, kernel_shape=None, auto_pad="NOTSET", ceil_mode=0, dilations:Optional[mx.array]=None, pads=None, storage_order=0, strides=None):
    """
    x: [Batch, Channel, Height, Width]
    storage_order: how the data is layed out in the array 0 = row, 1 = col 
    ceil_mode: whether to use ceil mode when output calculating the shape 1 = floor 0 = ceil
    pads: [x1_begin, x2_begin...x1_end, x2_end,...]
    """
    assert x.ndim >= 3, "MaxPool only supports >= 3D input"
    assert auto_pad == "NOTSET", "MaxPool only supports auto_pad=NOTSET for now"
    assert storage_order == 0, "MaxPool only supports storage_order=0 for now"

    if isinstance(kernel_shape, mx.array):
        kernel_shape = kernel_shape.tolist()
    if isinstance(strides, mx.array):
        strides = strides.tolist()
    if strides is None:
        strides = [1] * len(kernel_shape)
    if isinstance(pads, mx.array):
        pads = pads.tolist()
    if pads is None:
        pads = [0] * len(kernel_shape) * 2
    if any([p > 0 for p in pads]):
        pads = convert_pad(pads)
        # if ceil_mode == 1:
            # pads = [(p[0], p[1]+1) for p in pads]
        x = mx.pad(x, pad_width=[(0,0), (0,0)] + pads, constant_values=float("-inf"))

    if dilations is None:
        dilations = [1] * len(kernel_shape)
    if isinstance(dilations, mx.array):
        dilations = dilations.tolist()
    if any([d > 1 for d in dilations]):
        raise NotImplementedError("MaxPool does not support dilation > 1")

    if ceil_mode == 1:
        x = mx.pad(x, pad_width=[(0,0), (0,0)] + [(0,1)]*(x.ndim-2), constant_values=float("-inf"))

    if x.ndim == 3:
        res = _max_pool1d(x, kernel_shape, strides, ceil_mode)
    elif x.ndim == 4:
        res = _max_pool2d(x, kernel_shape, strides, ceil_mode)
    elif x.ndim == 5:
        res = _max_pool3d(x, kernel_shape, strides, ceil_mode)

    r_len, og_len = math.prod(res.shape), math.prod(x.shape)
    # get the indicies
    xf = x.flatten()
    rf = x.flatten()
    return (res)

def _max_pool1d(x: mx.array, kernel_shape: List[int], strides: List[int], ceil_mode: int):
    [bs, ch, h] = x.shape
    [b_stride, c_stride, h_stride] = compute_strides(x.shape)
    _rop = lambda x: math.floor(x) if ceil_mode == 0 else math.ceil(x)
    windows = mx.as_strided(
        x, 
        shape=(
            bs,
            ch,
            _rop((h - kernel_shape[0]) / strides[0]) + 1,
            kernel_shape[0],
        ),
        strides=(
            b_stride,
            c_stride,
            h_stride * strides[0],
            h_stride,
        )
    )
    return mx.max(windows, axis=(3))

def _max_pool2d(x: mx.array, kernel_shape: List[int], strides: List[int], ceil_mode: int):
    [bs, ch, h, w] = x.shape
    [b_stride, c_stride, h_stride, w_stride] = compute_strides(x.shape)
    _rop = lambda x: math.floor(x) if ceil_mode == 0 else math.ceil(x)
    windows = mx.as_strided(
        x, 
        shape=(
            bs,
            ch,
            _rop((h - kernel_shape[0]) / strides[0]) + 1,
            _rop((w - kernel_shape[1]) / strides[1]) + 1,
            kernel_shape[0],
            kernel_shape[1],
        ),
        strides=(
            b_stride,
            c_stride,
            h_stride * strides[0],
            w_stride * strides[1],
            h_stride,
            w_stride,
        )
    )
    return mx.max(windows, axis=(4, 5))

def _max_pool3d(x: mx.array, kernel_shape: List[int], strides: List[int], ceil_mode: int):
    [bs, ch, h, w, d] = x.shape
    [b_stride, c_stride, h_stride, w_stride, d_stride] = compute_strides(x.shape)
    _rop = lambda x: math.floor(x) if ceil_mode == 0 else math.ceil(x)
    windows = mx.as_strided(
        x, 
        shape=(
            bs,
            ch,
            _rop((h - kernel_shape[0]) / strides[0]) + 1,
            _rop((w - kernel_shape[1]) / strides[1]) + 1,
            _rop((d - kernel_shape[2]) / strides[2]) + 1,
            kernel_shape[0],
            kernel_shape[1],
            kernel_shape[2],
        ),
        strides=(
            b_stride,
            c_stride,
            h_stride * strides[0],
            w_stride * strides[1],
            d_stride * strides[2],
            h_stride,
            w_stride,
            d_stride,
        )
    )
    return mx.max(windows, axis=(5, 6, 7))