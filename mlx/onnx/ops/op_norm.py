import mlx.core as mx
import mlx.nn.layers as layers

def InstanceNormalization(x: mx.array, scale: mx.array, bias: mx.array, epsilon=1e-5):
    t = layers.InstanceNorm(dims=0, eps=epsilon)
    setattr(t, "weight", scale.reshape([-1, 1, 1]))
    setattr(t, "bias", bias.reshape([-1, 1, 1]))
    return t(x, axis=(2, 3))

def LayerNormalization(
    x: mx.array, scale: mx.array, bias: mx.array, axis=-1, stash_type=1, epsilon=1e-5
):
    axis = [i for i in range(axis if axis >= 0 else x.ndim + axis, x.ndim)]
    t = layers.LayerNorm(dims=0, eps=epsilon)
    setattr(t, "weight", scale)
    setattr(t, "bias", bias)
    mean = x.mean(axis=axis, keepdims=True)
    invstd = (((x - mean) ** 2).mean(axis=axis, keepdims=True) + epsilon).rsqrt()
    return t(x, axis=axis), mean, invstd