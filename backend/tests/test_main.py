"""Integration tests for the FastAPI application."""

import io
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent.parent))

# Patch model loading before importing app so no .pth files are needed
with patch("main.models", {}):
    from main import app

from httpx import ASGITransport, AsyncClient


def _make_png_bytes(size: int = 64) -> bytes:
    arr = np.random.randint(0, 256, (size, size), dtype=np.uint8)
    image = Image.fromarray(arr, mode="L")
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return buf.getvalue()


def _make_mock_model(latent_dim: int = 128) -> MagicMock:
    """Create a mock ConvVAE that returns deterministic tensors."""
    mock = MagicMock()
    mock.latent_dim = latent_dim
    recon = torch.rand(1, 1, 256, 256)
    mu = torch.zeros(1, latent_dim)
    logvar = torch.zeros(1, latent_dim)
    mock.return_value = (recon, mu, logvar)
    mock.__call__ = lambda self, x: (recon, mu, logvar)
    return mock


@pytest.fixture
async def client():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        yield c


@pytest.mark.asyncio
async def test_health_endpoint(client) -> None:
    response = await client.get("/api/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert "models_loaded" in data


@pytest.mark.asyncio
async def test_compress_invalid_latent_dim(client) -> None:
    png = _make_png_bytes()
    response = await client.post(
        "/api/compress",
        data={"latent_dim": "999"},
        files={"file": ("test.png", png, "image/png")},
    )
    assert response.status_code == 400
    assert "latent_dim" in response.json()["detail"]


@pytest.mark.asyncio
async def test_compress_empty_file(client) -> None:
    import main
    with patch.object(main, "models", {128: _make_mock_model(128)}):
        response = await client.post(
            "/api/compress",
            data={"latent_dim": "128"},
            files={"file": ("empty.png", b"", "image/png")},
        )
    assert response.status_code == 400
    assert "empty" in response.json()["detail"].lower()


@pytest.mark.asyncio
async def test_compress_invalid_image_bytes(client) -> None:
    import main
    with patch.object(main, "models", {128: _make_mock_model(128)}):
        response = await client.post(
            "/api/compress",
            data={"latent_dim": "128"},
            files={"file": ("bad.png", b"not-an-image", "image/png")},
        )
    assert response.status_code == 400


@pytest.mark.asyncio
async def test_compress_model_not_loaded(client) -> None:
    import main
    with patch.object(main, "models", {}):
        png = _make_png_bytes()
        response = await client.post(
            "/api/compress",
            data={"latent_dim": "128"},
            files={"file": ("test.png", png, "image/png")},
        )
    assert response.status_code == 503


@pytest.mark.asyncio
async def test_compress_success(client) -> None:
    import main
    mock_model = _make_mock_model(128)
    # Make the mock callable (not MagicMock's default __call__)
    recon = torch.rand(1, 1, 256, 256)
    mu = torch.zeros(1, 128)
    logvar = torch.zeros(1, 128)

    def side_effect(x):
        return recon, mu, logvar

    mock_model.side_effect = side_effect

    with patch.object(main, "models", {128: mock_model}):
        png = _make_png_bytes(256)
        response = await client.post(
            "/api/compress",
            data={"latent_dim": "128"},
            files={"file": ("xray.png", png, "image/png")},
        )

    assert response.status_code == 200
    data = response.json()
    assert "reconstructed_b64" in data
    assert "psnr" in data
    assert "ssim" in data
    assert "original_size_bytes" in data
    assert "compressed_size_bytes" in data
    assert "compression_ratio" in data
    assert data["latent_dim"] == 128
    assert data["original_size_bytes"] == 65536
    assert data["compressed_size_bytes"] == 512
    assert data["compression_ratio"] == "128:1"
    # Verify base64 is valid
    import base64
    decoded = base64.b64decode(data["reconstructed_b64"])
    assert len(decoded) > 0
