import mlx.core as mx
from typing import Optional

def norm(x, axis=-1, eps=1e-5, mean: Optional[mx.array]=None, var: Optional[mx.array]=None):
    mean = mean if mean is not None else mx.mean(x, axis=axis, keepdims=True)
    var = var if var is not None else mx.rsqrt(mx.var(x, axis=axis, keepdims=True) + eps)
    return (x - mean) * var

def BatchNormalization(x: mx.array, scale: mx.array, bias: mx.array, input_mean: mx.array, input_var: mx.array, momentum=0.9, epsilon=1e-5, spatial=1):
    assert spatial == 1, "Spatial BatchNorm not supported"
    t_shape = [1, -1] + [1] * (x.ndim - 2)
    var = mx.rsqrt(input_var + epsilon)
    return norm(x, eps=epsilon, mean=input_mean.reshape(t_shape), var=var.reshape(t_shape)) * scale.reshape(t_shape) + bias.reshape(t_shape)

def GroupNormalization(x: mx.array, scale: mx.array, bias: mx.array, num_groups: int, epsilon=1e-5):
    x_shape = x.shape
    x = x.reshape([x_shape[0], num_groups, -1])
    x = norm(x, axis=-1, eps=epsilon)
    return (scale.reshape([-1, 1]) * x + bias.reshape([-1, 1])).reshape(x_shape)
    
def InstanceNormalization(x: mx.array, scale: mx.array, bias: mx.array, epsilon=1e-5):
    return scale.reshape([-1, 1, 1]) * norm(x, axis=(2, 3), eps=epsilon) + bias.reshape([-1, 1, 1])

def LayerNormalization(
    x: mx.array, scale: mx.array, bias: mx.array, axis=-1, stash_type=1, epsilon=1e-5
):
    axis = [i for i in range(axis if axis >= 0 else x.ndim + axis, x.ndim)]
    mean = x.mean(axis=axis, keepdims=True)
    invstd = (((x - mean) ** 2).mean(axis=axis, keepdims=True) + epsilon).rsqrt()
    return scale * norm(x, axis=axis, eps=epsilon) + bias, mean, invstd