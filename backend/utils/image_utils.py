"""Image preprocessing and postprocessing for the compression pipeline."""

import io
import numpy as np
from numpy.typing import NDArray
from PIL import Image
import torch
from torch import Tensor

_TARGET_SIZE = 256


def preprocess_upload(file_bytes: bytes, target_size: int = _TARGET_SIZE) -> Tensor:
    """Convert uploaded file bytes to a model-ready tensor.

    Loads an image from raw bytes, converts to grayscale, resizes to
    target_size x target_size, and normalizes pixel values to [0, 1].

    Args:
        file_bytes: Raw image bytes (PNG or JPG).
        target_size: Output spatial dimension (default 256).

    Returns:
        Float tensor of shape (1, 1, target_size, target_size), values in [0, 1].

    Raises:
        ValueError: If the bytes cannot be decoded as an image.
    """
    try:
        image = Image.open(io.BytesIO(file_bytes))
    except Exception as exc:
        raise ValueError(f"Cannot decode image: {exc}") from exc

    image = image.convert("L")  # force single-channel grayscale
    image = image.resize((target_size, target_size), Image.LANCZOS)
    arr = np.array(image, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(arr).unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)
    return tensor


def tensor_to_numpy(tensor: Tensor) -> NDArray:
    """Convert a model output tensor to a 2D numpy array.

    Handles tensors of shape (1, 1, H, W), (1, H, W), or (H, W).

    Args:
        tensor: PyTorch tensor, values in [0, 1].

    Returns:
        2D numpy array (H, W), dtype float32, values in [0, 1].
    """
    return tensor.detach().cpu().numpy().squeeze().astype(np.float32)


def numpy_to_png_bytes(arr: NDArray) -> bytes:
    """Encode a [0, 1] float numpy array as PNG bytes.

    Args:
        arr: 2D grayscale array (H, W), values in [0, 1].

    Returns:
        PNG-encoded bytes.
    """
    uint8_arr = (np.clip(arr, 0.0, 1.0) * 255.0).astype(np.uint8)
    image = Image.fromarray(uint8_arr, mode="L")
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return buf.getvalue()
