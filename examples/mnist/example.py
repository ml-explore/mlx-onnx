import mlx.core as mx
from mlx.onnx import MlxBackend
from onnx import hub
from PIL import Image
import numpy as np

if __name__ == "__main__":
    x = mx.array(np.asarray(Image.open("./nine.jpeg"))).reshape((1, 1, 28, 28)).astype(mx.float32)
    model = hub.load("mnist")
    backend = MlxBackend(model)
    res = backend.run(x)
    print(f"It was a {mx.argmax(res[0]).item()}")
