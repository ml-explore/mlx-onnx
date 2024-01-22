import mlx.core as mx
from typing import Optional
from .pad import convert_pad, auto_pad as ap

def Conv(x: mx.array, weight: mx.array, bias: Optional[mx.array]=None, dilations:Optional[mx.array]=None, group=1, auto_pad="NOTSET", kernel_shape:Optional[mx.array]=None, pads:Optional[mx.array]=None, strides:Optional[mx.array]=None):
    assert group == 1, f"mlx only supports 1 group, got {group}"
    if dilations is not None:
        assert all(x == 1 for x in dilations.tolist()), "mlx only supports dilation 1"
    
    if isinstance(kernel_shape, mx.array):
        kernel_shape = kernel_shape.tolist()
    if isinstance(strides, mx.array):
        strides = strides.tolist()
    if strides is None:
        strides = [1] * len(kernel_shape)
    if isinstance(pads, mx.array):
        pads = pads.tolist()
    if pads is None:
        pads = [0] * len(kernel_shape)
    
    if x.ndim < weight.ndim:
        x = mx.expand_dims(x, 0)
    
    if auto_pad != "NOTSET":
        padding = convert_pad(ap(x.shape, auto_pad, strides, kernel_shape))
        x = mx.pad(x, pad_width=[(0,0), (0,0)] + padding, constant_values=0)
    
    if x.ndim == 3:
        c = mx.conv1d(x.transpose(0, 2, 1), weight.transpose(0, 2, 1), padding=pads[0] if pads is not None else 0, stride=strides[0] if strides is not None else 1)
        c = c + bias if bias is not None else c 
        return c.transpose(0, 2, 1)
    elif x.ndim == 4:
        c = mx.conv2d(x.transpose(0, 2, 3, 1), weight.transpose(0, 2, 3, 1), padding=pads[:2] if pads is not None else 0, stride=strides if strides is not None else 1)
        c = c + bias if bias is not None else c
        return c.transpose(0, 3, 1, 2)
    else:
        raise NotImplementedError("mlx does not support conv other than 1d and 2d")
    