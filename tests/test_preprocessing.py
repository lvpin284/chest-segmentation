"""Tests for the image preprocessing pipeline."""
import io
import numpy as np
import pytest
import torch
from PIL import Image

from src.preprocessing.image_processor import (
    UltrasoundPreprocessor,
    TrainingAugmentor,
    apply_clahe,
    load_image,
    normalize,
    resize_with_padding,
    to_tensor,
)


def _make_gray_array(h: int = 200, w: int = 300) -> np.ndarray:
    """Return a synthetic grayscale uint8 array."""
    rng = np.random.default_rng(42)
    return rng.integers(0, 256, (h, w), dtype=np.uint8)


def _make_png_bytes(h: int = 200, w: int = 300) -> bytes:
    """Return a synthetic image encoded as PNG bytes."""
    arr = _make_gray_array(h, w)
    img = Image.fromarray(arr, mode="L")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


class TestLoadImage:
    def test_from_numpy_array(self):
        arr = _make_gray_array()
        result = load_image(arr)
        assert result.shape == (200, 300)
        assert result.dtype == np.uint8

    def test_from_bytes(self):
        raw = _make_png_bytes()
        result = load_image(raw)
        assert result.ndim == 2
        assert result.dtype == np.uint8

    def test_unsupported_type_raises(self):
        with pytest.raises(TypeError):
            load_image(12345)  # type: ignore

    def test_bad_bytes_raises(self):
        with pytest.raises(ValueError):
            load_image(b"not_an_image")


class TestApplyClahe:
    def test_output_shape_unchanged(self):
        arr = _make_gray_array()
        result = apply_clahe(arr)
        assert result.shape == arr.shape

    def test_output_dtype_uint8(self):
        arr = _make_gray_array()
        result = apply_clahe(arr)
        assert result.dtype == np.uint8


class TestResizeWithPadding:
    def test_output_size(self):
        arr = _make_gray_array(200, 300)
        result = resize_with_padding(arr, (256, 256))
        assert result.shape == (256, 256)

    def test_output_size_non_square(self):
        arr = _make_gray_array(100, 400)
        result = resize_with_padding(arr, (128, 512))
        assert result.shape == (128, 512)

    def test_output_dtype(self):
        arr = _make_gray_array()
        result = resize_with_padding(arr, (128, 128))
        assert result.dtype == np.uint8


class TestNormalize:
    def test_range_is_zero_to_one(self):
        arr = _make_gray_array()
        result = normalize(arr)
        assert result.min() >= 0.0
        assert result.max() <= 1.0

    def test_dtype_float32(self):
        arr = _make_gray_array()
        result = normalize(arr)
        assert result.dtype == np.float32


class TestToTensor:
    def test_shape(self):
        arr = normalize(_make_gray_array())
        t = to_tensor(arr)
        assert t.shape == (1, 200, 300)

    def test_dtype(self):
        arr = normalize(_make_gray_array())
        t = to_tensor(arr)
        assert t.dtype == torch.float32


class TestUltrasoundPreprocessor:
    def test_output_tensor_shape(self):
        preprocessor = UltrasoundPreprocessor(image_size=(128, 128))
        raw = _make_png_bytes()
        tensor = preprocessor(raw)
        assert tensor.shape == (1, 128, 128)

    def test_output_range(self):
        preprocessor = UltrasoundPreprocessor(image_size=(64, 64))
        raw = _make_png_bytes()
        tensor = preprocessor(raw)
        assert tensor.min().item() >= 0.0
        assert tensor.max().item() <= 1.0

    def test_batch_processing(self):
        preprocessor = UltrasoundPreprocessor(image_size=(64, 64))
        sources = [_make_png_bytes() for _ in range(4)]
        batch = preprocessor.preprocess_batch(sources)
        assert batch.shape == (4, 1, 64, 64)

    def test_no_clahe_option(self):
        preprocessor = UltrasoundPreprocessor(image_size=(64, 64), apply_clahe=False)
        raw = _make_png_bytes()
        tensor = preprocessor(raw)
        assert tensor.shape == (1, 64, 64)


class TestTrainingAugmentor:
    def test_output_shape(self):
        augmentor = TrainingAugmentor(image_size=(64, 64))
        raw = _make_png_bytes()
        tensor = augmentor(raw)
        assert tensor.shape == (1, 64, 64)

    def test_output_clipped(self):
        augmentor = TrainingAugmentor(image_size=(64, 64), noise_std=0.1)
        raw = _make_png_bytes()
        tensor = augmentor(raw)
        assert tensor.min().item() >= 0.0
        assert tensor.max().item() <= 1.0
