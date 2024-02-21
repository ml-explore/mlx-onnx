import math

import mlx.core as mx


def LRN(x: mx.array, size: int, alpha=0.0001, beta=0.75, bias=1.0):
    if x.ndim != 4:
        raise NotImplementedError("LRN only supports 4D tensors")
    square_sum = mx.zeros(x.shape).astype(x.dtype)
    minc = x.shape[1]
    c1 = int(math.floor((size - 1) / 2))
    c2 = int(math.ceil((size - 1) / 2)) + 1
    for c in range(x.shape[0]):
        begin = max(0, c - c1)
        end = min(minc, c + c2)
        square_sum[:, c, :, :] = mx.sum(x[:, begin:end, :, :] ** 2, axis=1)
    y = x / ((bias + (alpha / size) * square_sum) ** beta)
    return (y.astype(x.dtype),)
