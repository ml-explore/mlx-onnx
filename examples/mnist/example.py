# Copyright © 2024 Apple Inc.

import mlx.core as mx
import numpy as np
from onnx import hub
from PIL import Image

from mlx.onnx import MlxBackend

if __name__ == "__main__":
    x = (
        mx.array(np.asarray(Image.open("./nine.jpeg")))
        .reshape((1, 1, 28, 28))
        .astype(mx.float32)
    )
    model = hub.load("mnist")
    backend = MlxBackend(model)
    res = backend.run(x)
    print(f"It was a {mx.argmax(res[0]).item()}")
