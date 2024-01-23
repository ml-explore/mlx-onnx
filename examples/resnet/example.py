import onnx
from mlx.onnx import MlxBackend
import mlx.core as mx
from PIL import Image
import numpy as np

if __name__ == "__main__":
    img = Image.open("./hen.jpg")
    aspect_ratio = img.size[0] / img.size[1]
    img = img.resize((int(224*max(aspect_ratio,1.0)), int(224*max(1.0/aspect_ratio,1.0))))

    img = np.asarray(img, dtype=np.float32)
    img -= [127.0, 127.0, 127.0]
    img /= [128.0, 128.0, 128.0]

    img = img.transpose((2,0,1))
    _input = mx.expand_dims(mx.array(img).astype(mx.float32), 0)
    
    model = onnx.hub.load("resnet50")
    backend = MlxBackend(model)
    x = backend.run(_input)

    with open("./imagenet_labels.txt") as f:
        labels = [l.strip() for l in f.readlines()]
    out = x[0]
    print("Image containes a", labels[mx.argmax(out, axis=1).item()], mx.max(out).item())