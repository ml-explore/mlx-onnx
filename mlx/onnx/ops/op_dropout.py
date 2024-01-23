from typing import Optional

import mlx.core as mx
import numpy as np


def Dropout(x: mx.array, ratio: int = 0.5, training_mode=0, seed: Optional[int] = None):
    assert training_mode == 0, "Training mode not supported yet"
    return x, mx.ones(x.shape, dtype=mx.bool_)
