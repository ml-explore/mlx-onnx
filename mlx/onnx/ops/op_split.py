import mlx.core as mx
import math
from typing import Optional

def Split(x: mx.array, split: Optional[mx.array] = None, num_outputs=None, axis=0):
    if split is None:
        if x.shape[axis] % num_outputs == 0:
            split = [x.shape[axis] // num_outputs] * num_outputs
        else:
            cnt = math.ceil(x.shape[axis] / num_outputs)
            split = [cnt] * (num_outputs - 1) + [
                x.shape[axis] - cnt * (num_outputs - 1)
            ]
        split = mx.array(split, dtype=mx.int64)
    sli = [slice(0, s) for s in x.shape]
    res = []
    pos = 0
    for spl in split.tolist():
        sli[axis] = slice(pos, pos + spl)
        pos += spl
        res.append(x[tuple(sli)])
    return tuple(res)