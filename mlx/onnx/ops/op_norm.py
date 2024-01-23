import mlx.core as mx
import mlx.nn.layers as layers

def layer_norm(x, axis=-1, eps=1e-5):
    means = mx.mean(x, axis=axis, keepdims=True)
    var = mx.var(x, axis=axis, keepdims=True)
    x = (x - means) * mx.rsqrt(var + eps)
    return x

def GroupNormalization(x: mx.array, scale: mx.array, bias: mx.array, num_groups: int, epsilon=1e-5):
    x_shape = x.shape
    x = x.reshape([x_shape[0], num_groups, -1])
    x = layer_norm(x, axis=-1, eps=epsilon)
    return (scale.reshape([-1, 1]) * x + bias.reshape([-1, 1])).reshape(x_shape)
    
def InstanceNormalization(x: mx.array, scale: mx.array, bias: mx.array, epsilon=1e-5):
    return scale.reshape([-1, 1, 1]) * layer_norm(x, axis=(2, 3), eps=epsilon) + bias.reshape([-1, 1, 1])

def LayerNormalization(
    x: mx.array, scale: mx.array, bias: mx.array, axis=-1, stash_type=1, epsilon=1e-5
):
    axis = [i for i in range(axis if axis >= 0 else x.ndim + axis, x.ndim)]
    mean = x.mean(axis=axis, keepdims=True)
    invstd = (((x - mean) ** 2).mean(axis=axis, keepdims=True) + epsilon).rsqrt()
    return scale * layer_norm(x, axis=axis, eps=epsilon) + bias, mean, invstd