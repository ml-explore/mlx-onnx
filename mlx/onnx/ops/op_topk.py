import mlx.core as mx

def TopK(x: mx.array, k: mx.array, axis=-1, largest=1, sorted=1):
    assert sorted == 1, "[TopK] Only sorted is supported"
    if isinstance(k, mx.array):
        k = k.item()
    if x.ndim == 2 and axis == 1:
        sample = mx.arange(x.shape[0])[:, None]
        if largest == 0:
            sorted_indices = mx.argpartition(x, kth=k - 1, axis=axis)
            sorted_indices = sorted_indices[:, :k]
            sorted_indices = sorted_indices[sample, mx.argsort(x[sample, sorted_indices])]
        else:
            sorted_indices = mx.argpartition(-x, kth=k-1, axis=axis)
            sorted_indices = sorted_indices[:, :k]
            sorted_indices = sorted_indices[sample, mx.argsort(-x[sample, sorted_indices])]
        sorted_distances = x[sample, sorted_indices]
        return (sorted_distances, sorted_indices.astype(mx.int64))
    
    if largest == 0:
        sorted_indices = mx.argsort(x, axis=axis)
        sorted_values = mx.sort(x, axis=axis)
    else:
        sorted_indices = mx.argsort(-x, axis=axis)
        sorted_values = -mx.sort(-x, axis=axis)
    ark = mx.arange(k)
    topk_sorted_indices = mx.take(sorted_indices, ark, axis=axis)
    topk_sorted_values = mx.take(sorted_values, ark, axis=axis)
    return topk_sorted_values, topk_sorted_indices.astype(mx.int64)
