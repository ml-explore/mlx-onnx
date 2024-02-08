import mlx.core as mx


def OneHot(indicies: mx.array, depth: mx.array, values: mx.array, axis=-1):
    if isinstance(values, mx.array):
        values = values.tolist()
    if isinstance(depth, mx.array):
        depth = depth.item()
    depth_range = mx.arange(depth)
    if axis < 0:
        axis = indicies.ndim + axis + 1
    ls = list(indicies.shape[0:axis])
    rs = list(indicies.shape[axis : indicies.ndim])
    new_shape = [1] * len(ls) + list(depth_range.shape) + [1] * len(rs)
    tgts = depth_range.reshape(new_shape)
    vals = (indicies % depth).reshape(ls + [1] + rs)
    return mx.where(tgts == vals, values[1], values[0])
