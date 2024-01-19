from setuptools import setup

setup(
    name="mlx-onnx",
    version="0.0.1",
    description="MLX backend for onnx",
    install_requires=["mlx", "onnx"],
    extras_require={
        "test": ["numpy", "pytest"],
        "dev": ["pre-commit"],
    },
    packages=["mlx.onnx"],
)
