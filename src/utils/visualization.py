"""
Visualization utilities for segmentation results and ultrasound analysis reports.
"""
from __future__ import annotations

import io
import base64
from pathlib import Path
from typing import Optional, Tuple, Union

import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")  # non-interactive backend
import matplotlib.pyplot as plt
import torch


# ---------------------------------------------------------------------------
# Colour palette
# ---------------------------------------------------------------------------
EFFUSION_COLOR = (0, 120, 255)   # bright blue (BGR) – marks effusion region
CONTOUR_COLOR  = (0, 255, 80)    # green – contour border


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def tensor_to_numpy_image(tensor: torch.Tensor) -> np.ndarray:
    """
    Convert a (1, H, W) or (H, W) float tensor in [0,1] to uint8 (H, W) array.
    """
    if tensor.dim() == 3:
        tensor = tensor.squeeze(0)
    arr = tensor.detach().cpu().numpy()
    return (arr * 255).clip(0, 255).astype(np.uint8)


def overlay_mask(
    image: np.ndarray,
    mask: np.ndarray,
    alpha: float = 0.4,
    effusion_color: Tuple[int, int, int] = EFFUSION_COLOR,
    contour_color: Tuple[int, int, int] = CONTOUR_COLOR,
) -> np.ndarray:
    """
    Blend a binary segmentation mask onto a grayscale ultrasound image.

    Args:
        image: Grayscale (H, W) uint8 array.
        mask: Binary (H, W) uint8 array (values 0 or 1).
        alpha: Blend factor for the coloured overlay.
        effusion_color: BGR colour for the effusion region.
        contour_color: BGR colour for the contour border.

    Returns:
        (H, W, 3) uint8 BGR array.
    """
    # Convert grayscale to BGR
    bgr = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    # Create colour overlay
    overlay = bgr.copy()
    overlay[mask == 1] = effusion_color

    # Blend
    result = cv2.addWeighted(overlay, alpha, bgr, 1.0 - alpha, 0)

    # Draw contour
    mask_u8 = (mask * 255).astype(np.uint8)
    contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(result, contours, -1, contour_color, 2)

    return result


def compute_effusion_area(mask: np.ndarray, pixel_spacing_mm: float = 0.1) -> float:
    """
    Estimate the effusion area in mm² given a binary mask.

    Args:
        mask: Binary (H, W) array (values 0 or 1).
        pixel_spacing_mm: Physical size of one pixel in mm (approximate).

    Returns:
        Estimated area in mm².
    """
    pixel_count = int(mask.sum())
    return pixel_count * (pixel_spacing_mm ** 2)


def compute_effusion_ratio(mask: np.ndarray) -> float:
    """Return the fraction of image pixels classified as effusion."""
    total = mask.size
    if total == 0:
        return 0.0
    return float(mask.sum()) / total


def severity_from_ratio(ratio: float) -> str:
    """
    Map effusion area ratio to a clinical severity label.

    Thresholds are approximate and for educational/demo purposes only.
    """
    if ratio < 0.02:
        return "无明显积液 (No significant effusion)"
    elif ratio < 0.08:
        return "少量积液 (Small effusion)"
    elif ratio < 0.20:
        return "中量积液 (Moderate effusion)"
    else:
        return "大量积液 (Large effusion)"


def create_analysis_figure(
    original: np.ndarray,
    overlay: np.ndarray,
    mask: np.ndarray,
    title: str = "超声胸腔积液分析 / Pleural Effusion Analysis",
) -> plt.Figure:
    """
    Create a three-panel matplotlib figure for the analysis report.

    Panels: original image | overlay | binary mask
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(title, fontsize=14, fontweight="bold")

    axes[0].imshow(original, cmap="gray")
    axes[0].set_title("原始图像 / Original")
    axes[0].axis("off")

    axes[1].imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
    axes[1].set_title("分割叠加 / Segmentation Overlay")
    axes[1].axis("off")

    axes[2].imshow(mask, cmap="Blues")
    axes[2].set_title("积液掩码 / Effusion Mask")
    axes[2].axis("off")

    plt.tight_layout()
    return fig


def figure_to_base64(fig: plt.Figure, fmt: str = "png") -> str:
    """Render a matplotlib figure to a base64-encoded string."""
    buf = io.BytesIO()
    fig.savefig(buf, format=fmt, bbox_inches="tight", dpi=100)
    buf.seek(0)
    encoded = base64.b64encode(buf.read()).decode("utf-8")
    plt.close(fig)
    return encoded


def save_overlay(
    overlay: np.ndarray,
    path: Union[str, Path],
) -> None:
    """Save an overlay BGR image to disk."""
    cv2.imwrite(str(path), overlay)
