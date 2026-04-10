"""FastAPI backend for Medical Image Compression VAE.

Loads three ConvVAE checkpoints (latent_dim=64, 128, 256) at startup,
exposes a compress endpoint, and serves the static SvelteKit frontend.
"""

import base64
import logging
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncGenerator

import torch
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from model.vae import ConvVAE
from utils.image_utils import numpy_to_png_bytes, preprocess_upload, tensor_to_numpy
from utils.metrics import compression_ratio, psnr, ssim

logger = logging.getLogger(__name__)

WEIGHTS_DIR = Path(__file__).parent / "weights"
FRONTEND_DIR = Path(__file__).parent.parent / "frontend" / "build"
VALID_LATENT_DIMS = frozenset({64, 128, 256})
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MAX_UPLOAD_BYTES = 10 * 1024 * 1024  # 10 MB

# Populated at startup; keyed by latent_dim integer
models: dict[int, ConvVAE] = {}


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Load all VAE checkpoints at application startup."""
    for dim in sorted(VALID_LATENT_DIMS):
        weight_path = WEIGHTS_DIR / f"weights_{dim}.pth"
        if weight_path.exists():
            model = ConvVAE(latent_dim=dim)
            state = torch.load(weight_path, map_location=DEVICE, weights_only=True)
            model.load_state_dict(state)
            model.to(DEVICE)
            model.eval()
            models[dim] = model
            logger.info("Loaded model latent_dim=%d from %s", dim, weight_path)
        else:
            logger.warning("Weights not found at %s — latent_dim=%d unavailable", weight_path, dim)

    if not models:
        logger.error(
            "No models loaded. Place weights_64.pth / weights_128.pth / weights_256.pth "
            "in backend/weights/ before starting the server."
        )

    yield

    models.clear()


app = FastAPI(title="Medical Image Compression VAE", version="1.0.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


@app.post("/api/compress")
async def compress(
    file: UploadFile = File(...),
    latent_dim: int = Form(...),
) -> JSONResponse:
    """Compress and reconstruct a medical image using the VAE.

    Args:
        file: Uploaded image (PNG or JPG).
        latent_dim: Latent space dimensionality — must be 64, 128, or 256.

    Returns:
        JSON with base64-encoded reconstructed PNG and quality metrics.
    """
    if latent_dim not in VALID_LATENT_DIMS:
        raise HTTPException(
            status_code=400,
            detail=f"latent_dim must be one of {sorted(VALID_LATENT_DIMS)}, got {latent_dim}",
        )

    if latent_dim not in models:
        raise HTTPException(
            status_code=503,
            detail=f"Model for latent_dim={latent_dim} is not loaded. "
                   "Check that the weights file exists in backend/weights/.",
        )

    file_bytes = await file.read()

    if len(file_bytes) == 0:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    if len(file_bytes) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="File too large (max 10 MB).")

    try:
        input_tensor = preprocess_upload(file_bytes)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    input_tensor = input_tensor.to(DEVICE)
    model = models[latent_dim]

    with torch.no_grad():
        recon_tensor, _mu, _logvar = model(input_tensor)

    original_np = tensor_to_numpy(input_tensor)
    recon_np = tensor_to_numpy(recon_tensor)
    recon_png = numpy_to_png_bytes(recon_np)

    original_size = 256 * 256 * 1
    compressed_size = latent_dim * 4
    ratio = compression_ratio(latent_dim)

    return JSONResponse({
        "reconstructed_b64": base64.b64encode(recon_png).decode(),
        "psnr": round(psnr(original_np, recon_np), 2),
        "ssim": round(ssim(original_np, recon_np), 4),
        "original_size_bytes": original_size,
        "compressed_size_bytes": compressed_size,
        "compression_ratio": f"{ratio:.0f}:1",
        "latent_dim": latent_dim,
    })


@app.get("/api/health")
async def health() -> JSONResponse:
    """Return server health and which models are loaded."""
    return JSONResponse({
        "status": "ok",
        "models_loaded": sorted(models.keys()),
    })


# Serve compiled SvelteKit frontend (only mounted when build exists)
if FRONTEND_DIR.exists():
    app.mount("/", StaticFiles(directory=str(FRONTEND_DIR), html=True), name="frontend")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
