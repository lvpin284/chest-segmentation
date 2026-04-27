"""
Image preprocessing pipeline for ultrasound pleural effusion images.

Provides:
    - Normalization (CLAHE contrast enhancement, z-score)
    - Resizing and padding to a fixed input resolution
    - Data augmentation transforms for training
    - Tensor conversion utilities
"""
from __future__ import annotations

import io
from pathlib import Path
from typing import Tuple, Union

import cv2
import numpy as np
import torch
from PIL import Image
from torchvision import transforms


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DEFAULT_IMAGE_SIZE: Tuple[int, int] = (256, 256)  # (height, width)


# ---------------------------------------------------------------------------
# Low-level helpers
# ---------------------------------------------------------------------------

def load_image(source: Union[str, Path, bytes, np.ndarray]) -> np.ndarray:
    """
    Load an image from a file path, raw bytes, or an existing NumPy array.

    Returns:
        Grayscale NumPy array with dtype uint8 and shape (H, W).
    """
    if isinstance(source, np.ndarray):
        img = source
    elif isinstance(source, (str, Path)):
        img = cv2.imread(str(source), cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise FileNotFoundError(f"Cannot load image from path: {source}")
    elif isinstance(source, bytes):
        arr = np.frombuffer(source, dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise ValueError("Cannot decode image from bytes.")
    else:
        raise TypeError(f"Unsupported source type: {type(source)}")

    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img


def apply_clahe(image: np.ndarray, clip_limit: float = 2.0, tile_size: int = 8) -> np.ndarray:
    """
    Apply CLAHE (Contrast Limited Adaptive Histogram Equalization).

    Enhances local contrast in ultrasound images which often have low dynamic range.
    """
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_size, tile_size))
    return clahe.apply(image)


def resize_with_padding(
    image: np.ndarray,
    target_size: Tuple[int, int] = DEFAULT_IMAGE_SIZE,
) -> np.ndarray:
    """
    Resize image to *target_size* while preserving aspect ratio, then zero-pad.

    Args:
        image: Grayscale (H, W) uint8 array.
        target_size: (height, width) tuple.

    Returns:
        Grayscale (target_h, target_w) uint8 array.
    """
    th, tw = target_size
    h, w = image.shape[:2]
    scale = min(tw / w, th / h)
    new_w, new_h = int(w * scale), int(h * scale)

    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    padded = np.zeros((th, tw), dtype=np.uint8)
    y_offset = (th - new_h) // 2
    x_offset = (tw - new_w) // 2
    padded[y_offset : y_offset + new_h, x_offset : x_offset + new_w] = resized
    return padded


def normalize(image: np.ndarray) -> np.ndarray:
    """Normalize pixel values to [0, 1] float32."""
    return image.astype(np.float32) / 255.0


def to_tensor(image: np.ndarray) -> torch.Tensor:
    """Convert a (H, W) float32 NumPy array to a (1, H, W) float32 tensor."""
    return torch.from_numpy(image).unsqueeze(0)


# ---------------------------------------------------------------------------
# High-level pipeline classes
# ---------------------------------------------------------------------------

class UltrasoundPreprocessor:
    """
    Full preprocessing pipeline for a single ultrasound image.

    Steps:
        1. Load from file / bytes / array
        2. Apply CLAHE contrast enhancement
        3. Resize with aspect-ratio-preserving padding
        4. Normalize to [0, 1]
        5. Convert to (1, H, W) float32 tensor

    Args:
        image_size: Target (height, width) for the output tensor.
        apply_clahe: Whether to apply CLAHE enhancement (default True).
        clip_limit: CLAHE clip limit.
        tile_size: CLAHE tile grid size.

    Example::
        preprocessor = UltrasoundPreprocessor()
        tensor = preprocessor("scan.png")  # shape (1, 256, 256)
    """

    def __init__(
        self,
        image_size: Tuple[int, int] = DEFAULT_IMAGE_SIZE,
        apply_clahe: bool = True,
        clip_limit: float = 2.0,
        tile_size: int = 8,
    ) -> None:
        self.image_size = image_size
        self._apply_clahe = apply_clahe
        self.clip_limit = clip_limit
        self.tile_size = tile_size

    def __call__(
        self, source: Union[str, Path, bytes, np.ndarray]
    ) -> torch.Tensor:
        img = load_image(source)
        if self._apply_clahe:
            img = apply_clahe(img, self.clip_limit, self.tile_size)
        img = resize_with_padding(img, self.image_size)
        img_f = normalize(img)
        return to_tensor(img_f)

    def preprocess_batch(
        self, sources: list
    ) -> torch.Tensor:
        """Process a list of sources and stack into a (N, 1, H, W) batch tensor."""
        tensors = [self(s) for s in sources]
        return torch.stack(tensors, dim=0)


class TrainingAugmentor:
    """
    Augmentation pipeline used during training.

    Applies random horizontal flip, rotation, brightness/contrast jitter,
    and Gaussian noise to increase dataset diversity.

    Args:
        image_size: Target (height, width).
        flip_prob: Probability of horizontal flip.
        rotation_degrees: Max rotation angle in degrees.
        noise_std: Standard deviation of additive Gaussian noise (after [0,1] normalization).

    Example::
        augmentor = TrainingAugmentor()
        tensor = augmentor("scan.png")
    """

    def __init__(
        self,
        image_size: Tuple[int, int] = DEFAULT_IMAGE_SIZE,
        flip_prob: float = 0.5,
        rotation_degrees: float = 10.0,
        noise_std: float = 0.02,
    ) -> None:
        self.preprocessor = UltrasoundPreprocessor(image_size=image_size)
        self.flip_prob = flip_prob
        self.rotation_degrees = rotation_degrees
        self.noise_std = noise_std

        self._transform = transforms.Compose([
            transforms.RandomHorizontalFlip(p=flip_prob),
            transforms.RandomRotation(degrees=rotation_degrees),
        ])

    def __call__(
        self, source: Union[str, Path, bytes, np.ndarray]
    ) -> torch.Tensor:
        tensor = self.preprocessor(source)  # (1, H, W)
        # Convert to PIL for torchvision transforms
        pil = transforms.ToPILImage()(tensor)
        pil = self._transform(pil)
        tensor = transforms.ToTensor()(pil)  # back to (1, H, W) float32
        if self.noise_std > 0:
            tensor = tensor + torch.randn_like(tensor) * self.noise_std
            tensor = tensor.clamp(0.0, 1.0)
        return tensor
