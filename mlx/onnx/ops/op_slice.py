import mlx.core as mx
from typing import Optional

def Slice(
    x: mx.array,
    starts: mx.array,
    ends: mx.array,
    axes: Optional[mx.array] = None,
    steps: Optional[mx.array] = None,
):
    if axes is None:
        axes = mx.arange(x.ndim)
    if steps is None:
        steps = mx.ones(starts.shape, dtype=mx.int64)
    slices = [slice(0, d) for d in x.shape]
    for start, end, axe, step in zip(starts, ends, axes, steps):
        slices[axe.item()] = slice(start.item(), end.item(), step.item())
    return x[tuple(slices)]