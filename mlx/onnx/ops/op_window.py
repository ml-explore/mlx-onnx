import mlx.core as mx
from .helper import DTYPE_MAP
import math

def start(size, output_datatype, periodic):
    dtype = DTYPE_MAP[output_datatype]
    N_1 = size if periodic == 1 else size - 1
    return mx.arange(size, dtype=dtype), N_1

def HannWindow(size: mx.array, output_datatype=1, periodic=1):
    if isinstance(size, mx.array):
        size = size.item()
    ni, N_1 = start(size, output_datatype, periodic)
    res = mx.sin(ni * math.pi / N_1) ** 2
    return res.astype(DTYPE_MAP[output_datatype])

def BlackmanWindow(size: mx.array, output_datatype=1, periodic=1):
    if isinstance(size, mx.array):
        size = size.item()
    ni, N_1 = start(size, output_datatype, periodic)
    res = 0.42 - 0.5 * mx.cos(2 * math.pi * ni / N_1) + 0.08 * mx.cos(4 * math.pi * ni / N_1)
    return res.astype(DTYPE_MAP[output_datatype])

def HammingWindow(size: mx.array, output_datatype=1, periodic=1):
    if isinstance(size, mx.array):
        size = size.item()
    ni, N_1 = start(size, output_datatype, periodic)
    alpha = 25. / 46.
    res = alpha - mx.cos(2 * math.pi * ni / N_1) * (1 - alpha)
    return res.astype(DTYPE_MAP[output_datatype])