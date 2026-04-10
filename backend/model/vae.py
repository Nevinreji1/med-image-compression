"""Convolutional Variational Autoencoder for medical image compression."""

import torch
import torch.nn as nn
from torch import Tensor


def _encoder_block(in_ch: int, out_ch: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, kernel_size=4, stride=2, padding=1, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.LeakyReLU(0.2, inplace=True),
    )


def _decoder_block(in_ch: int, out_ch: int) -> nn.Sequential:
    return nn.Sequential(
        nn.ConvTranspose2d(in_ch, out_ch, kernel_size=4, stride=2, padding=1, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.LeakyReLU(0.2, inplace=True),
    )


class ConvVAE(nn.Module):
    """Convolutional VAE for 256x256 single-channel (grayscale) images.

    Architecture:
        Encoder: 4x Conv2d (stride=2) blocks -> flatten -> mu/logvar heads
        Decoder: linear -> reshape -> 4x ConvTranspose2d blocks -> sigmoid

    Args:
        latent_dim: Dimensionality of the latent space (64, 128, or 256).
    """

    # Fixed spatial size after 4 stride-2 downsamples: 256 / 2^4 = 16
    _FEAT_CH = 256
    _FEAT_HW = 16
    _FEAT_FLAT = _FEAT_CH * _FEAT_HW * _FEAT_HW  # 65536

    def __init__(self, latent_dim: int = 128) -> None:
        super().__init__()
        self.latent_dim = latent_dim

        # Encoder conv stack: (B,1,256,256) -> (B,256,16,16)
        self.encoder_conv = nn.Sequential(
            _encoder_block(1, 32),    # -> (B,32,128,128)
            _encoder_block(32, 64),   # -> (B,64,64,64)
            _encoder_block(64, 128),  # -> (B,128,32,32)
            _encoder_block(128, 256), # -> (B,256,16,16)
        )

        # Latent heads
        self.fc_mu = nn.Linear(self._FEAT_FLAT, latent_dim)
        self.fc_logvar = nn.Linear(self._FEAT_FLAT, latent_dim)

        # Decoder input projection
        self.fc_decode = nn.Linear(latent_dim, self._FEAT_FLAT)

        # Decoder conv stack: (B,256,16,16) -> (B,1,256,256)
        self.decoder_conv = nn.Sequential(
            _decoder_block(256, 128), # -> (B,128,32,32)
            _decoder_block(128, 64),  # -> (B,64,64,64)
            _decoder_block(64, 32),   # -> (B,32,128,128)
            nn.ConvTranspose2d(32, 1, kernel_size=4, stride=2, padding=1),  # -> (B,1,256,256)
            nn.Sigmoid(),
        )

    def encode(self, x: Tensor) -> tuple[Tensor, Tensor]:
        """Encode input to latent distribution parameters.

        Args:
            x: Input tensor of shape (B, 1, 256, 256), values in [0, 1].

        Returns:
            Tuple of (mu, logvar), each shape (B, latent_dim).
        """
        h = self.encoder_conv(x)
        h = h.flatten(start_dim=1)
        logvar = self.fc_logvar(h).clamp(-4.0, 15.0)
        return self.fc_mu(h), logvar

    def reparameterize(self, mu: Tensor, logvar: Tensor) -> Tensor:
        """Sample latent vector via the reparameterization trick.

        During eval mode returns mu directly (deterministic compression).
        During training adds Gaussian noise for variational learning.

        Args:
            mu: Mean of latent distribution, shape (B, latent_dim).
            logvar: Log-variance of latent distribution, shape (B, latent_dim).

        Returns:
            Sampled latent vector z, shape (B, latent_dim).
        """
        if not self.training:
            return mu
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z: Tensor) -> Tensor:
        """Decode latent vector to image.

        Args:
            z: Latent vector of shape (B, latent_dim).

        Returns:
            Reconstructed image tensor of shape (B, 1, 256, 256), values in [0, 1].
        """
        h = self.fc_decode(z)
        h = h.view(-1, self._FEAT_CH, self._FEAT_HW, self._FEAT_HW)
        return self.decoder_conv(h)

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """Full forward pass.

        Args:
            x: Input tensor of shape (B, 1, 256, 256).

        Returns:
            Tuple of (reconstruction, mu, logvar).
            reconstruction: shape (B, 1, 256, 256), values in [0, 1].
            mu: shape (B, latent_dim).
            logvar: shape (B, latent_dim).
        """
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        return recon, mu, logvar
