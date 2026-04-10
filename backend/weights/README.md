# Model Weights

This directory contains pre-trained VAE model weights. The `.pth` files are excluded from version control via `.gitignore`.

## Available Weight Files

| File | Image Size | Description |
|------|------------|-------------|
| `weights_64.pth` | 64×64 | VAE trained on 64×64 medical images |
| `weights_128.pth` | 128×128 | VAE trained on 128×128 medical images |
| `weights_256.pth` | 256×256 | VAE trained on 256×256 medical images |

## Setup

Place the `.pth` weight files in this directory before running the backend. They can be obtained from the project maintainer or trained from scratch using the training scripts.
