# MLX ONNX

MLX support for the Open Neural Network Exchange ([ONNX](https://onnx.ai/)) 

## Install

```shell
pip install mlx-onnx
```

## Usage

```python
from mlx.onnx import MlxBackend
from onnx import hub

model = hub.load("mnist")
backend = MlxBackend(model)
result = backend.run(...) # pass inputs to model
```

## Examples

- [ResNet](./examples/resnet/example.py)
- [Mnist](./examples/mnist/example.py)
