from typing import Optional

import mlx.core as mx
import mlx.nn.layers as layers

def SoftmaxCrossEntropyLoss(scores: mx.array, labels: mx.array, weights: Optional[mx.array]=None, ignore_index=None, reduction="mean"):
    C = scores.shape[1]
    if ignore_index is not None: labels = mx.where(labels == ignore_index, C+1, labels)
    probs = layers.log_softmax(scores, 1)
    # loss = losses.cross_entropy(probs, labels, weights[labels, ....], reduction=reduction)
    mask = mx.expand_dims(labels, 1) == mx.arange(C).reshape([1, C] + [1] * (scores.ndim - 2))
    loss = (mask * -probs).sum(axis=1)
    if weights is not None:
        weights = weights[labels, ...]
        loss = loss * weights
    
    if reduction == "mean":
        if weights is None:
            loss = loss.sum() / mx.where(loss == 0, 0., 1.).sum()
        else:
            loss = loss.sum() / weights.sum()
    elif reduction == "sum":
        loss = loss.sum()
    return loss, probs
   