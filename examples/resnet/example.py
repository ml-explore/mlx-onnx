import mlx.core as mx
import mlx.data as dx
import onnx

from mlx.onnx import MlxBackend


def run(image: str):
    dataset = (
        dx.buffer_from_vector([{"file_name": image.encode()}])
        .load_image("file_name", output_key="image")
        .image_resize_smallest_side("image", 256)
        .image_center_crop("image", 224, 224)
        .key_transform("image", lambda x: (x - 127.0) / 128.0)
    )
    with open("./imagenet_labels.txt") as f:
        labels = [l.strip() for l in f.readlines()]

    model = onnx.hub.load("resnet50")
    backend = MlxBackend(model)
    res = []
    for data in dataset:
        img = mx.array(data["image"]).transpose((2, 0, 1))
        img = mx.expand_dims(img, 0)
        x = backend.run(img)
        res.append((labels[mx.argmax(x[0], axis=1).item()], mx.max(x[0]).item()))
    return res


if __name__ == "__main__":
    for label, score in run("./car.jpg"):
        print("Image containes a", label, score)
