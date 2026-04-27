"""
U-Net segmentation model for ultrasound pleural effusion detection.

Architecture:
    - Encoder: Contracting path with double convolutions and max pooling
    - Bottleneck: Double convolution
    - Decoder: Expansive path with transposed convolutions and skip connections
    - Output: Binary segmentation map (effusion vs background)
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class _DoubleConv(nn.Module):
    """Two consecutive Conv2d -> BatchNorm -> ReLU blocks."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class _Down(nn.Module):
    """Downscaling block: MaxPool followed by DoubleConv."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.pool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            _DoubleConv(in_channels, out_channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pool_conv(x)


class _Up(nn.Module):
    """Upscaling block: TransposedConv (or bilinear) followed by DoubleConv."""

    def __init__(self, in_channels: int, out_channels: int, bilinear: bool = True) -> None:
        super().__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
            self.conv = _DoubleConv(in_channels, out_channels)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = _DoubleConv(in_channels, out_channels)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        x1 = self.up(x1)
        # Pad x1 to match x2 spatial dimensions (handles odd input sizes)
        diff_h = x2.size(2) - x1.size(2)
        diff_w = x2.size(3) - x1.size(3)
        x1 = F.pad(x1, [diff_w // 2, diff_w - diff_w // 2, diff_h // 2, diff_h - diff_h // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class _OutConv(nn.Module):
    """1×1 convolution to map features to the desired number of classes."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class UNet(nn.Module):
    """
    U-Net for binary segmentation of pleural effusion in ultrasound images.

    Args:
        in_channels: Number of input image channels (1 for grayscale, 3 for RGB).
        num_classes: Number of segmentation classes (default 1 for binary mask).
        base_features: Number of feature maps in the first encoder block.
        bilinear: Use bilinear upsampling instead of transposed convolutions.

    Example::
        model = UNet(in_channels=1, num_classes=1)
        logits = model(torch.randn(1, 1, 256, 256))  # (1, 1, 256, 256)
    """

    def __init__(
        self,
        in_channels: int = 1,
        num_classes: int = 1,
        base_features: int = 64,
        bilinear: bool = True,
    ) -> None:
        super().__init__()
        f = base_features
        factor = 2 if bilinear else 1

        self.inc = _DoubleConv(in_channels, f)
        self.down1 = _Down(f, f * 2)
        self.down2 = _Down(f * 2, f * 4)
        self.down3 = _Down(f * 4, f * 8)
        self.down4 = _Down(f * 8, f * 16 // factor)

        self.up1 = _Up(f * 16, f * 8 // factor, bilinear)
        self.up2 = _Up(f * 8, f * 4 // factor, bilinear)
        self.up3 = _Up(f * 4, f * 2 // factor, bilinear)
        self.up4 = _Up(f * 2, f, bilinear)
        self.outc = _OutConv(f, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        return self.outc(x)

    def predict_mask(self, x: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
        """Return a binary mask tensor (values 0 or 1)."""
        with torch.no_grad():
            logits = self.forward(x)
            probs = torch.sigmoid(logits)
            return (probs >= threshold).float()


class SegmentationLoss(nn.Module):
    """
    Combined Binary Cross-Entropy + Dice loss, commonly used for medical image segmentation.

    Args:
        bce_weight: Weight for the BCE component (Dice weight = 1 - bce_weight).
        smooth: Smoothing term to avoid division by zero in Dice loss.
    """

    def __init__(self, bce_weight: float = 0.5, smooth: float = 1.0) -> None:
        super().__init__()
        self.bce_weight = bce_weight
        self.smooth = smooth
        self.bce = nn.BCEWithLogitsLoss()

    def _dice_loss(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = torch.sigmoid(logits)
        intersection = (probs * targets).sum(dim=(2, 3))
        union = probs.sum(dim=(2, 3)) + targets.sum(dim=(2, 3))
        dice = (2.0 * intersection + self.smooth) / (union + self.smooth)
        return 1.0 - dice.mean()

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce = self.bce(logits, targets)
        dice = self._dice_loss(logits, targets)
        return self.bce_weight * bce + (1.0 - self.bce_weight) * dice
