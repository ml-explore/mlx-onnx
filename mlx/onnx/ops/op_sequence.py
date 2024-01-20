import mlx.core as mx
from typing import List, Optional

def SplitToSequence(x: mx.array, split: Optional[mx.array]=None, axis:int=0, keepdims=0):
    if split is None:
        split_len = [1] * x.shape[axis]
    elif split.ndim == 0:
        dim = x.shape[axis]
        _len = split.item()
        n = dim // int(_len)
        split_len = [_len] * n
        left = dim - _len * n
        if left > 0:
            split_len.append(left)
    else:
        split_len = split.tolist()
    sli = [slice(0, s) for s in x.shape]
    res = []
    pos = 0
    for spl in split_len:
        sli[axis] = slice(pos, pos + spl)
        pos += spl
        res.append(x[tuple(sli)])
    return res

def SequenceConstruct(*args: List[mx.array]):
    return [*args]

def SequenceLength(x):
    return mx.array(len(x), dtype=mx.int64)

def SequenceEmpty():
    return []

def SequenceAt(seq: List[mx.array], index: mx.array):
    if isinstance(index, mx.array):
        index = index.item()
    return seq[index]

def SequenceErase(seq: List[mx.array], index: Optional[mx.array]=None):
    if index is None:
        index = -1
    else:
        index = index.item()
    return seq[:index] + seq[index + 1:]

def ConcatFromSequence(seq: List[mx.array], axis: int=0, new_axis=0):
    if new_axis == 1:
        sc = [s[..., None] for s in seq]
        return mx.concatenate(sc, axis=axis)
    return mx.concatenate(seq, axis=axis)

def SequenceInsert(seq: List[mx.array], value: mx.array, ind=None):
    if ind is not None:
        ind = ind.item()
    if ind is None:
        seq.append(value)
    else:
        seq.insert(ind, value)
    return seq