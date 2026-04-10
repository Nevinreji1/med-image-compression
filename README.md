# Medical Image Compression VAE

A Convolutional Variational Autoencoder (VAE) that compresses chest X-ray images into compact latent representations and reconstructs them with measurable quality metrics.

## Architecture

```
Input (B, 1, 256, 256)
    │
    ▼  Encoder
    Conv2d(1→32, k=4, s=2)  + BN + LeakyReLU  →  (B, 32, 128, 128)
    Conv2d(32→64, k=4, s=2) + BN + LeakyReLU  →  (B, 64, 64, 64)
    Conv2d(64→128,k=4, s=2) + BN + LeakyReLU  →  (B, 128, 32, 32)
    Conv2d(128→256,k=4,s=2) + BN + LeakyReLU  →  (B, 256, 16, 16)
    Flatten → Linear → μ (latent_dim)
              Linear → logvar (latent_dim)
    │
    ▼  Reparameterize:  z = μ + σε,  ε ~ N(0,I)
    │
    ▼  Decoder
    Linear → reshape (B, 256, 16, 16)
    ConvT2d(256→128,k=4,s=2) + BN + LeakyReLU → (B, 128, 32, 32)
    ConvT2d(128→64, k=4,s=2) + BN + LeakyReLU → (B, 64, 64, 64)
    ConvT2d(64→32,  k=4,s=2) + BN + LeakyReLU → (B, 32, 128, 128)
    ConvT2d(32→1,   k=4,s=2) + Sigmoid         → (B, 1, 256, 256)
```

**Loss:** `MSE + β·KL + λ·(1 − SSIM)` where β=1.0, λ=0.1

**Three model variants:** latent_dim = 64, 128, 256

## Project Structure

```
vae-medical-compression/
├── backend/
│   ├── main.py               # FastAPI server
│   ├── model/
│   │   ├── vae.py            # ConvVAE class
│   │   └── train.py          # Standalone training script (Colab-ready)
│   ├── utils/
│   │   ├── metrics.py        # PSNR, SSIM, compression ratio
│   │   └── image_utils.py    # Preprocessing / postprocessing
│   ├── weights/              # Place .pth files here (gitignored)
│   ├── tests/                # pytest test suite
│   ├── pyproject.toml
│   └── requirements.txt
├── frontend/
│   ├── src/routes/+page.svelte  # Single-page UI
│   ├── svelte.config.js         # adapter-static
│   └── vite.config.ts           # /api proxy for dev
└── README.md
```

## Setup

### Backend

```bash
cd backend

# Install deps (creates .venv automatically)
uv sync --dev

# Start the server (port 8000)
uv run python main.py
```

Requires Python ≥ 3.10 and [uv](https://docs.astral.sh/uv/).

### Frontend

```bash
cd frontend

# Install deps
bun install
bun add -d @sveltejs/adapter-static tailwindcss @tailwindcss/vite

# Dev mode (proxies /api → localhost:8000)
bun run dev

# Production build
bun run build
```

## Training on Google Colab

1. Upload `backend/model/vae.py` and `backend/model/train.py` to Colab.

2. Set up the environment:

```python
!pip install torch torchvision pytorch-msssim Pillow numpy
!kaggle datasets download -d nih-chest-xrays/data --path ./data
!unzip -q ./data/data.zip -d ./data
```

3. Train all three variants:

```bash
!python train.py --latent_dim 64  --data_dir ./data/images --output_dir ./weights
!python train.py --latent_dim 128 --data_dir ./data/images --output_dir ./weights
!python train.py --latent_dim 256 --data_dir ./data/images --output_dir ./weights
```

4. Download `weights_64.pth`, `weights_128.pth`, `weights_256.pth` and place them in `backend/weights/`.

## Running the Full Stack

```bash
# 1. Build the frontend
cd frontend && bun run build

# 2. Start FastAPI (serves API + static frontend on port 8000)
cd backend && uv run python main.py
```

Then open http://localhost:8000.

## API

### POST /api/compress

**Request:** `multipart/form-data`
- `file` — PNG or JPG image
- `latent_dim` — integer: 64, 128, or 256

**Response:**
```json
{
  "reconstructed_b64": "<base64 PNG>",
  "psnr": 34.21,
  "ssim": 0.912,
  "original_size_bytes": 65536,
  "compressed_size_bytes": 512,
  "compression_ratio": "128:1",
  "latent_dim": 128
}
```

### GET /api/health

```json
{ "status": "ok", "models_loaded": [64, 128, 256] }
```

## Metrics

| Metric | Formula | Notes |
|--------|---------|-------|
| PSNR | 10·log₁₀(1/MSE) | Higher is better. Typical: 25–40 dB |
| SSIM | structural similarity | Range [0,1]. Typical: 0.7–0.95 |
| Compression ratio | 65536 / (latent_dim·4) | dim=64→256:1, dim=128→128:1, dim=256→64:1 |

## Tests

```bash
cd backend
uv run pytest tests/ -v
```
