import mlx.core as mx
from typing import Optional, List, Callable
from .pad import convert_pad, auto_pad as ap
import math

def compute_strides(shape: List[int]):
    return list(reversed(mx.cumprod(mx.array([1] + list(reversed(shape))))[:-1].tolist()))

def MaxPool(x: mx.array, kernel_shape=None, auto_pad="NOTSET", ceil_mode=0, dilations:Optional[mx.array]=None, pads=None, storage_order=0, strides=None):
    return Pool(x, mx.max, float("-inf"), kernel_shape, auto_pad, ceil_mode, dilations, pads, storage_order, strides)

def AveragePool(x: mx.array, kernel_shape=None, auto_pad="NOTSET", ceil_mode=0, dilations:Optional[mx.array]=None, pads=None, storage_order=0, strides=None, count_include_pad=0):
    res = Pool(x, mx.mean, 0, kernel_shape, auto_pad, ceil_mode, dilations, pads, storage_order, strides)
    if count_include_pad:
        return res
    div = Pool(mx.ones_like(x), mx.mean, 0, kernel_shape, auto_pad, ceil_mode, dilations, pads, storage_order, strides)
    return res / div

def Pool(x: mx.array, op: Callable[..., mx.array], pad_fill: float, kernel_shape=None, auto_pad="NOTSET", ceil_mode=0, dilations:Optional[mx.array]=None, pads=None, storage_order=0, strides=None):
    """
    x: [Batch, Channel, Height, Width]
    storage_order: how the data is layed out in the array 0 = row, 1 = col 
    ceil_mode: whether to use ceil mode when output calculating the shape 1 = floor 0 = ceil
    pads: [x1_begin, x2_begin...x1_end, x2_end,...]
    """
    assert x.ndim >= 3, "Pool only supports >= 3D input"
    assert storage_order == 0, "Pool only supports storage_order=0 for now"
    
    if dilations is None:
        dilations = [1] * len(kernel_shape)
    if isinstance(dilations, mx.array):
        dilations = dilations.tolist()
    if any([d > 1 for d in dilations]):
        raise NotImplementedError("Pool does not support dilation > 1")
    
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
    if auto_pad != "NOTSET":
        pads = ap(x.shape, auto_pad, strides, kernel_shape)
    if any([p > 0 for p in pads]):
        pads = convert_pad(pads)
        x = mx.pad(x, pad_width=[(0,0), (0,0)] + pads, constant_values=pad_fill)

    if ceil_mode == 1:
        x = mx.pad(x, pad_width=[(0,0), (0,0)] + [(0,1)]*(x.ndim-2), constant_values=pad_fill)

    if x.ndim == 3:
        res = _pool1d(x, op, kernel_shape, strides, ceil_mode)
    elif x.ndim == 4:
        res = _pool2d(x, op, kernel_shape, strides, ceil_mode)
    elif x.ndim == 5:
        res = _pool3d(x, op, kernel_shape, strides, ceil_mode)
    return (res)

def _pool1d(x: mx.array, op: Callable[..., mx.array], kernel_shape: List[int], strides: List[int], ceil_mode: int):
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
    return op(windows, axis=(3))

def _pool2d(x: mx.array, op: Callable[..., mx.array], kernel_shape: List[int], strides: List[int], ceil_mode: int):
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
    return op(windows, axis=(4, 5))

def _pool3d(x: mx.array, op: Callable[..., mx.array], kernel_shape: List[int], strides: List[int], ceil_mode: int):
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
    return op(windows, axis=(5, 6, 7))