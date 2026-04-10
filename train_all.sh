#!/usr/bin/env bash
# train_all.sh — Train VAE for all three latent dimensions.
#
# Usage:
#   ./train_all.sh <path/to/images>
#
# Example (after downloading NIH ChestX-ray14 in Colab):
#   ./train_all.sh ./data/images
#
# Weights are saved to backend/weights/ as weights_64.pth, weights_128.pth, weights_256.pth

set -euo pipefail

IMAGES_DIR="${1:-}"
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WEIGHTS_DIR="$PROJECT_ROOT/backend/weights"
BACKEND_DIR="$PROJECT_ROOT/backend"

if [[ -z "$IMAGES_DIR" ]]; then
  echo "Usage: $0 <path/to/images>" >&2
  exit 1
fi

if [[ ! -d "$IMAGES_DIR" ]]; then
  echo "Error: images directory not found: $IMAGES_DIR" >&2
  exit 1
fi

mkdir -p "$WEIGHTS_DIR"
cd "$BACKEND_DIR"

for DIM in 64 128 256; do
  echo ""
  echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
  echo "Training latent_dim=$DIM"
  echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
  uv run python model/train.py \
    --latent_dim "$DIM" \
    --data_dir   "$IMAGES_DIR" \
    --output_dir "$WEIGHTS_DIR"
done

echo ""
echo "All done. Weights:"
ls -lh "$WEIGHTS_DIR"/*.pth
