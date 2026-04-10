"""Unit tests for image_utils.py."""

import io
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.image_utils import numpy_to_png_bytes, preprocess_upload, tensor_to_numpy


def _make_png_bytes(width: int = 64, height: int = 64, mode: str = "L") -> bytes:
    """Create a minimal in-memory PNG image."""
    arr = np.random.randint(0, 256, (height, width), dtype=np.uint8)
    image = Image.fromarray(arr, mode=mode)
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return buf.getvalue()


def _make_rgb_png_bytes() -> bytes:
    arr = np.random.randint(0, 256, (64, 64, 3), dtype=np.uint8)
    image = Image.fromarray(arr, mode="RGB")
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return buf.getvalue()


# --- preprocess_upload ---

def test_preprocess_grayscale_png_shape() -> None:
    png = _make_png_bytes()
    tensor = preprocess_upload(png)
    assert tensor.shape == (1, 1, 256, 256)


def test_preprocess_rgb_png_converts_to_grayscale() -> None:
    png = _make_rgb_png_bytes()
    tensor = preprocess_upload(png)
    assert tensor.shape == (1, 1, 256, 256)


def test_preprocess_value_range() -> None:
    png = _make_png_bytes()
    tensor = preprocess_upload(png)
    assert tensor.min().item() >= 0.0
    assert tensor.max().item() <= 1.0


def test_preprocess_dtype_float32() -> None:
    png = _make_png_bytes()
    tensor = preprocess_upload(png)
    assert tensor.dtype == torch.float32


def test_preprocess_invalid_bytes_raises() -> None:
    with pytest.raises(ValueError, match="Cannot decode image"):
        preprocess_upload(b"not an image")


def test_preprocess_empty_bytes_raises() -> None:
    with pytest.raises(ValueError):
        preprocess_upload(b"")


# --- tensor_to_numpy ---

def test_tensor_to_numpy_4d() -> None:
    t = torch.ones(1, 1, 256, 256)
    arr = tensor_to_numpy(t)
    assert arr.shape == (256, 256)


def test_tensor_to_numpy_dtype() -> None:
    t = torch.rand(1, 1, 256, 256)
    arr = tensor_to_numpy(t)
    assert arr.dtype == np.float32


# --- numpy_to_png_bytes ---

def test_numpy_to_png_bytes_roundtrip() -> None:
    arr = np.random.rand(256, 256).astype(np.float32)
    png_bytes = numpy_to_png_bytes(arr)
    image = Image.open(io.BytesIO(png_bytes))
    assert image.size == (256, 256)
    assert image.mode == "L"


def test_numpy_to_png_bytes_clamps_values() -> None:
    arr = np.array([[2.0, -1.0]], dtype=np.float32)
    png_bytes = numpy_to_png_bytes(arr)
    image = Image.open(io.BytesIO(png_bytes))
    pixels = np.array(image)
    assert pixels.max() == 255
    assert pixels.min() == 0
