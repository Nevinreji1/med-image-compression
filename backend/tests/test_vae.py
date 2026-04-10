"""Unit tests for the ConvVAE model."""

import pytest
import torch

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from model.vae import ConvVAE


@pytest.mark.parametrize("latent_dim", [64, 128, 256])
def test_forward_shapes(latent_dim: int) -> None:
    model = ConvVAE(latent_dim=latent_dim)
    x = torch.randn(2, 1, 256, 256)
    recon, mu, logvar = model(x)
    assert recon.shape == (2, 1, 256, 256), f"recon shape {recon.shape}"
    assert mu.shape == (2, latent_dim), f"mu shape {mu.shape}"
    assert logvar.shape == (2, latent_dim), f"logvar shape {logvar.shape}"


@pytest.mark.parametrize("latent_dim", [64, 128, 256])
def test_encode_shapes(latent_dim: int) -> None:
    model = ConvVAE(latent_dim=latent_dim)
    x = torch.randn(3, 1, 256, 256)
    mu, logvar = model.encode(x)
    assert mu.shape == (3, latent_dim)
    assert logvar.shape == (3, latent_dim)


@pytest.mark.parametrize("latent_dim", [64, 128, 256])
def test_decode_shapes(latent_dim: int) -> None:
    model = ConvVAE(latent_dim=latent_dim)
    z = torch.randn(2, latent_dim)
    recon = model.decode(z)
    assert recon.shape == (2, 1, 256, 256)


def test_output_range() -> None:
    """Sigmoid output must be in [0, 1]."""
    model = ConvVAE(latent_dim=128)
    x = torch.randn(1, 1, 256, 256)
    recon, _, _ = model(x)
    assert recon.min().item() >= 0.0
    assert recon.max().item() <= 1.0


def test_eval_mode_deterministic() -> None:
    """In eval mode reparameterize returns mu, so two forward passes are identical."""
    model = ConvVAE(latent_dim=128)
    model.eval()
    x = torch.randn(1, 1, 256, 256)
    with torch.no_grad():
        recon1, _, _ = model(x)
        recon2, _, _ = model(x)
    assert torch.allclose(recon1, recon2), "eval mode must be deterministic"


def test_train_mode_stochastic() -> None:
    """In training mode two forward passes should (very likely) differ due to sampling."""
    model = ConvVAE(latent_dim=128)
    model.train()
    x = torch.randn(1, 1, 256, 256)
    recon1, _, _ = model(x)
    recon2, _, _ = model(x)
    # With 128-dim noise, exact equality is astronomically unlikely
    assert not torch.allclose(recon1, recon2), "training mode should be stochastic"


def test_latent_dim_attribute() -> None:
    for dim in [64, 128, 256]:
        model = ConvVAE(latent_dim=dim)
        assert model.latent_dim == dim
