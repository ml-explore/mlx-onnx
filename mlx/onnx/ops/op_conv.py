import mlx.core as mx
from typing import Optional

def Conv(x: mx.array, weight: mx.array, bias: Optional[mx.array]=None, dilations:Optional[mx.array]=None, group=1, auto_pad="NOTSET", kernel_shape:Optional[mx.array]=None, pads:Optional[mx.array]=None, strides:Optional[mx.array]=None):
    assert group == 1, f"mlx only supports 1 group, got {group}"
    if dilations is not None:
        assert all(x == 1 for x in dilations.tolist()), "mlx only supports dilation 1"
    if x.ndim == 3:
        c = mx.conv1d(x.transpose(0, 2, 1), weight.transpose(0, 2, 1), padding=pads.tolist()[0] if pads is not None else 0, stride=strides.tolist()[0] if strides is not None else 1)
        c = c + bias if bias is not None else c 
        return c.transpose(0, 2, 1)
    elif x.ndim == 4:
        c = mx.conv2d(x.transpose(0, 2, 3, 1), weight.transpose(0, 2, 3, 1), padding=pads.tolist()[:2] if pads is not None else 0, stride=strides.tolist() if strides is not None else 1)
        c = c + bias if bias is not None else c
        return c.transpose(0, 3, 1, 2)
    else:
        raise NotImplementedError("mlx does not support conv other than 1d and 2d")
    