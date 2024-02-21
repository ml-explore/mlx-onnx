from typing import Optional

import mlx.core as mx

from .pad import convert_pad


def Pad(
    x: mx.array,
    pads: mx.array,
    constant_value=0.0,
    axes: Optional[mx.array] = None,
    mode="constant",
    value: Optional[float] = None,
):
    assert mode == "constant", f"Only constant padding is supported, got {mode}"
    if value is not None:
        constant_value = value
    if isinstance(pads, mx.array):
        pads = pads.tolist()
    if isinstance(axes, mx.array):
        axes = axes.tolist()
    pads = convert_pad(pads, x.ndim, axes)
    return mx.pad(x, pads, constant_value)
