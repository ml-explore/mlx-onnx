# mlx-onnx
MLX support for the Open Neural Network Exchange ([ONNX](https://onnx.ai/)) 

## Usage
```python
from mlx.onnx import MlxBackend
from onnx import hub

model = hub.load("mnist")
backend = MlxBackend(model)
result = backend.run(...) # pass inputs to model
```

## Examples
[ResNet Example](./examples/resnet/example.py)
[Mnist Example](./examples/mnist/example.py)