"""Unit tests for metrics.py."""

import math
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.metrics import compression_ratio, psnr, ssim


def _random_image(seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.random((256, 256), dtype=np.float64).astype(np.float32)


# --- PSNR ---

def test_psnr_identical_images() -> None:
    img = _random_image()
    result = psnr(img, img)
    assert result == math.inf


def test_psnr_known_value() -> None:
    """MSE of 0.01 with data_range=1.0 -> PSNR = 10*log10(1/0.01) = 20 dB."""
    original = np.zeros((256, 256), dtype=np.float32)
    reconstructed = np.full((256, 256), 0.1, dtype=np.float32)  # MSE = 0.01
    result = psnr(original, reconstructed)
    assert abs(result - 20.0) < 0.01


def test_psnr_returns_float() -> None:
    img1, img2 = _random_image(0), _random_image(1)
    result = psnr(img1, img2)
    assert isinstance(result, float)


def test_psnr_positive() -> None:
    img1, img2 = _random_image(0), _random_image(1)
    assert psnr(img1, img2) > 0.0


# --- SSIM ---

def test_ssim_identical_images() -> None:
    img = _random_image()
    result = ssim(img, img)
    assert abs(result - 1.0) < 1e-6


def test_ssim_range() -> None:
    img1, img2 = _random_image(0), _random_image(1)
    result = ssim(img1, img2)
    assert 0.0 <= result <= 1.0


def test_ssim_returns_float() -> None:
    img1, img2 = _random_image(0), _random_image(1)
    assert isinstance(ssim(img1, img2), float)


# --- Compression ratio ---

@pytest.mark.parametrize(
    "latent_dim,expected",
    [
        (64, 256.0),   # 65536 / 256
        (128, 128.0),  # 65536 / 512
        (256, 64.0),   # 65536 / 1024
    ],
)
def test_compression_ratio_values(latent_dim: int, expected: float) -> None:
    result = compression_ratio(latent_dim)
    assert result == expected, f"Expected {expected} for latent_dim={latent_dim}, got {result}"


def test_compression_ratio_returns_float() -> None:
    assert isinstance(compression_ratio(128), float)
