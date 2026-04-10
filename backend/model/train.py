"""Standalone training script for the Medical Compression VAE.

Designed to run on Google Colab or locally. Trains one model per
invocation; run three times to produce weights for latent_dim=64, 128, 256.

Usage:
    python train.py --latent_dim 128 --data_dir ./data/images --output_dir ./weights

Colab setup (paste into a cell before running):
    !pip install torch torchvision pytorch-msssim Pillow numpy
    !kaggle datasets download -d nih-chest-xrays/data --path ./data
    !unzip -q ./data/data.zip -d ./data

    # Then:
    !python train.py --latent_dim 64  --data_dir ./data/images
    !python train.py --latent_dim 128 --data_dir ./data/images
    !python train.py --latent_dim 256 --data_dir ./data/images
"""

# ── Colab setup block ─────────────────────────────────────────────────────────
# Uncomment and run this block in Colab before executing the script:
#
# import subprocess
# subprocess.run(["pip", "install", "pytorch-msssim"], check=True)
# ─────────────────────────────────────────────────────────────────────────────

import argparse
import sys
from pathlib import Path

# Allow importing vae.py from the same directory when run from Colab
sys.path.insert(0, str(Path(__file__).parent))

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import transforms
from PIL import Image
from pytorch_msssim import ssim as msssim

from vae import ConvVAE


# ── Dataset ──────────────────────────────────────────────────────────────────

class ChestXrayDataset(Dataset):
    """NIH ChestX-ray14 image dataset (unsupervised — labels ignored).

    Loads PNG/JPG images from a directory, converts to grayscale,
    resizes to target_size x target_size, and returns tensors in [0, 1].
    """

    _EXTENSIONS = {".png", ".jpg", ".jpeg"}

    def __init__(self, image_dir: Path, target_size: int = 256) -> None:
        self.paths = sorted(
            p for p in image_dir.iterdir()
            if p.suffix.lower() in self._EXTENSIONS
        )
        if not self.paths:
            raise FileNotFoundError(f"No images found in {image_dir}")
        self.transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            transforms.Resize((target_size, target_size)),
            transforms.ToTensor(),  # -> [0, 1] float tensor (1, H, W)
        ])

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int) -> Tensor:
        image = Image.open(self.paths[idx]).convert("RGB")  # PIL reads all formats
        return self.transform(image)


# ── Data loaders ─────────────────────────────────────────────────────────────

def create_dataloaders(
    dataset: Dataset,
    batch_size: int = 32,
    val_split: float = 0.2,
    num_workers: int = 2,
) -> tuple[DataLoader, DataLoader]:
    """Split dataset into train/val and return DataLoaders."""
    n_val = int(len(dataset) * val_split)  # type: ignore[arg-type]
    n_train = len(dataset) - n_val  # type: ignore[arg-type]
    generator = torch.Generator().manual_seed(42)
    train_set, val_set = random_split(dataset, [n_train, n_val], generator=generator)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    return train_loader, val_loader


# ── Loss ─────────────────────────────────────────────────────────────────────

def compute_loss(
    recon: Tensor,
    target: Tensor,
    mu: Tensor,
    logvar: Tensor,
    beta: float = 1.0,
    lam: float = 0.1,
) -> tuple[Tensor, dict[str, float]]:
    """Compute combined VAE loss: MSE + β*KL + λ*(1 - SSIM).

    Args:
        recon: Reconstructed image, shape (B, 1, H, W), values in [0, 1].
        target: Original image, same shape.
        mu: Latent mean, shape (B, latent_dim).
        logvar: Latent log-variance, shape (B, latent_dim).
        beta: Weight for the KL divergence term.
        lam: Weight for the SSIM perceptual term.

    Returns:
        Tuple of (total_loss_tensor, metrics_dict).
    """
    mse = F.mse_loss(recon, target, reduction="mean")
    kl = -0.5 * torch.mean(1.0 + logvar - mu.pow(2) - logvar.exp())
    ssim_val = msssim(recon, target, data_range=1.0, size_average=True)
    ssim_loss = 1.0 - ssim_val
    total = mse + beta * kl + lam * ssim_loss
    return total, {
        "mse": mse.item(),
        "kl": kl.item(),
        "ssim_loss": ssim_loss.item(),
        "total": total.item(),
    }


# ── Epoch routines ───────────────────────────────────────────────────────────

def train_one_epoch(
    model: ConvVAE,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    beta: float,
    lam: float,
) -> dict[str, float]:
    model.train()
    totals: dict[str, float] = {"mse": 0.0, "kl": 0.0, "ssim_loss": 0.0, "total": 0.0}
    for batch in loader:
        x = batch.to(device)
        optimizer.zero_grad()
        recon, mu, logvar = model(x)
        loss, metrics = compute_loss(recon, x, mu, logvar, beta, lam)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        for k in totals:
            totals[k] += metrics[k]
    n = len(loader)
    return {k: v / n for k, v in totals.items()}


def validate(
    model: ConvVAE,
    loader: DataLoader,
    device: torch.device,
    beta: float,
    lam: float,
) -> dict[str, float]:
    model.eval()
    totals: dict[str, float] = {"mse": 0.0, "kl": 0.0, "ssim_loss": 0.0, "total": 0.0}
    with torch.no_grad():
        for batch in loader:
            x = batch.to(device)
            recon, mu, logvar = model(x)
            _, metrics = compute_loss(recon, x, mu, logvar, beta, lam)
            for k in totals:
                totals[k] += metrics[k]
    n = len(loader)
    return {k: v / n for k, v in totals.items()}


# ── Main training loop ────────────────────────────────────────────────────────

def train(args: argparse.Namespace) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    image_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading images from {image_dir} …")
    dataset = ChestXrayDataset(image_dir)
    # Limit to first N images as specified
    if len(dataset) > args.max_images:
        from torch.utils.data import Subset
        dataset = Subset(dataset, list(range(args.max_images)))
    print(f"Dataset size: {len(dataset)} images")

    train_loader, val_loader = create_dataloaders(
        dataset, batch_size=args.batch_size, num_workers=args.num_workers
    )
    print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")

    model = ConvVAE(latent_dim=args.latent_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    best_val_loss = float("inf")
    output_path = output_dir / f"weights_{args.latent_dim}.pth"

    for epoch in range(1, args.epochs + 1):
        train_metrics = train_one_epoch(model, train_loader, optimizer, device, args.beta, args.lam)
        val_metrics = validate(model, val_loader, device, args.beta, args.lam)

        print(
            f"Epoch {epoch:3d}/{args.epochs} | "
            f"train_loss={train_metrics['total']:.4f} "
            f"(mse={train_metrics['mse']:.4f} kl={train_metrics['kl']:.4f}) | "
            f"val_loss={val_metrics['total']:.4f} "
            f"ssim_loss={val_metrics['ssim_loss']:.4f}"
        )

        if val_metrics["total"] < best_val_loss:
            best_val_loss = val_metrics["total"]
            torch.save(model.state_dict(), output_path)
            print(f"  ✓ Saved best checkpoint → {output_path} (val_loss={best_val_loss:.4f})")

    print(f"\nTraining complete. Best val loss: {best_val_loss:.4f}")
    print(f"Checkpoint saved to: {output_path}")


# ── CLI entry point ───────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Train Medical Compression VAE")
    parser.add_argument("--latent_dim", type=int, default=128, choices=[64, 128, 256],
                        help="Latent space dimensionality (default: 128)")
    parser.add_argument("--epochs", type=int, default=50,
                        help="Number of training epochs (default: 50)")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Training batch size (default: 32)")
    parser.add_argument("--lr", type=float, default=1e-3,
                        help="Adam learning rate (default: 1e-3)")
    parser.add_argument("--beta", type=float, default=1.0,
                        help="KL divergence weight β (default: 1.0)")
    parser.add_argument("--lam", type=float, default=0.1,
                        help="SSIM loss weight λ (default: 0.1)")
    parser.add_argument("--data_dir", type=str, default="./data/images",
                        help="Path to folder containing PNG/JPG images")
    parser.add_argument("--output_dir", type=str, default="./weights",
                        help="Directory to save checkpoint files")
    parser.add_argument("--max_images", type=int, default=1000,
                        help="Maximum number of images to use (default: 1000)")
    parser.add_argument("--num_workers", type=int, default=2,
                        help="DataLoader worker processes (default: 2)")
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
