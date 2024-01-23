import io

import mlx.core as mx
import numpy as np


def ImageDecoder(x: mx.array, pixel_format="RGB"):
    try:
        import PIL.Image
    except ImportError as e:
        raise ImportError(
            "Pillow is required for ImageDecoder. Please install it with `pip install Pillow`"
        ) from e
    img = PIL.Image.open(io.BytesIO(bytes(x)))
    if pixel_format == "RGB":
        img = np.array(img)
    elif pixel_format == "BGR":
        img = np.array(img)[:, :, ::-1]
    elif pixel_format == "Grayscale":
        img = img.convert("L")
        img = np.array(img)
        img = np.expand_dims(img, axis=2)
    else:
        raise ValueError(f"Unsupported pixel format: {pixel_format}")

    return mx.array(img, dtype=mx.uint8)
