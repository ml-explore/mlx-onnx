import mlx.core as mx
from typing import Optional

def NegativeLogLikelihoodLoss(scores: mx.array, target: mx.array, weight: Optional[mx.array]=None, ignore_index=None, reduction="mean"):
    print(weight.shape if weight is not None else None, scores.shape, target.shape)
    if ignore_index is not None: weight = mx.where(target == ignore_index, 0, weight if weight is not None else 1)
    loss = -mx.take_along_axis(scores, target[..., None], 1).squeeze(-1)
    if weight is not None:
        weight = weight[target]
        loss = loss * weight
    if reduction == "mean":
        return loss.mean() if weight is None else loss.sum() / weight.sum()
    elif reduction == "sum":
        return loss.sum()
    return loss