# Copyright © 2024 Apple Inc.

from setuptools import setup

setup(
    name="mlx-onnx",
    version="0.0.1",
    author="MLX Contributors",
    author_email="mlx@group.apple.com",
    description="MLX backend for ONNX",
    url="https://github.com/ml-explore/mlx-onnx",
    install_requires=["mlx", "onnx"],
    extras_require={
        "test": ["numpy", "pytest"],
        "dev": ["pre-commit"],
    },
    packages=["mlx.onnx"],
    python_requires=">=3.8",
)
