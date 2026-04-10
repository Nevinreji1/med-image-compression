"""Image quality metrics for compression evaluation."""

import math
import numpy as np
from numpy.typing import NDArray
from skimage.metrics import structural_similarity as sk_ssim

_ORIGINAL_SIZE_BYTES = 256 * 256 * 1  # raw 8-bit grayscale pixels
_BYTES_PER_FLOAT32 = 4


def psnr(original: NDArray, reconstructed: NDArray, data_range: float = 1.0) -> float:
    """Compute Peak Signal-to-Noise Ratio.

    Args:
        original: Original image array, shape (H, W), values in [0, 1].
        reconstructed: Reconstructed image array, same shape.
        data_range: The data range of the input images (default 1.0).

    Returns:
        PSNR value in dB. Returns math.inf if images are identical.
    """
    mse = float(np.mean((original.astype(np.float64) - reconstructed.astype(np.float64)) ** 2))
    if mse == 0.0:
        return math.inf
    return float(10.0 * math.log10((data_range ** 2) / mse))


def ssim(original: NDArray, reconstructed: NDArray, data_range: float = 1.0) -> float:
    """Compute Structural Similarity Index for grayscale images.

    Args:
        original: 2D grayscale array (H, W), values in [0, 1].
        reconstructed: 2D grayscale array (H, W), same shape.
        data_range: The data range of the input images (default 1.0).

    Returns:
        SSIM score in [0, 1].
    """
    return float(sk_ssim(original, reconstructed, data_range=data_range))


def compression_ratio(latent_dim: int) -> float:
    """Compute the compression ratio for a given latent dimensionality.

    Original size: 256 * 256 * 1 bytes (8-bit grayscale, uncompressed).
    Compressed size: latent_dim * 4 bytes (float32 mean vector μ only).

    Args:
        latent_dim: Number of latent dimensions (e.g. 64, 128, 256).

    Returns:
        Ratio as a float (e.g. 128.0 for latent_dim=128).
    """
    compressed_size = latent_dim * _BYTES_PER_FLOAT32
    return float(_ORIGINAL_SIZE_BYTES / compressed_size)
