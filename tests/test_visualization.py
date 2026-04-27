"""Tests for visualization utilities."""
import io
import numpy as np
import pytest
import matplotlib
matplotlib.use("Agg")

from src.utils.visualization import (
    compute_effusion_area,
    compute_effusion_ratio,
    create_analysis_figure,
    figure_to_base64,
    overlay_mask,
    severity_from_ratio,
    tensor_to_numpy_image,
)
import torch


def _make_gray(h=256, w=256):
    rng = np.random.default_rng(0)
    return rng.integers(0, 256, (h, w), dtype=np.uint8)


def _make_mask(h=256, w=256, fill_ratio=0.1):
    mask = np.zeros((h, w), dtype=np.uint8)
    n = int(h * w * fill_ratio)
    idx = np.arange(n)
    mask.flat[idx] = 1
    return mask


class TestTensorToNumpyImage:
    def test_3d_tensor(self):
        t = torch.ones(1, 64, 64)
        arr = tensor_to_numpy_image(t)
        assert arr.shape == (64, 64)
        assert arr.dtype == np.uint8

    def test_2d_tensor(self):
        t = torch.zeros(64, 64)
        arr = tensor_to_numpy_image(t)
        assert arr.shape == (64, 64)


class TestOverlayMask:
    def test_output_shape_and_dtype(self):
        img = _make_gray()
        mask = _make_mask(fill_ratio=0.1)
        result = overlay_mask(img, mask)
        assert result.shape == (256, 256, 3)
        assert result.dtype == np.uint8

    def test_empty_mask(self):
        img = _make_gray()
        mask = np.zeros((256, 256), dtype=np.uint8)
        result = overlay_mask(img, mask)
        assert result.shape == (256, 256, 3)


class TestComputeMetrics:
    def test_effusion_ratio_zero_mask(self):
        mask = np.zeros((100, 100), dtype=np.uint8)
        assert compute_effusion_ratio(mask) == 0.0

    def test_effusion_ratio_full_mask(self):
        mask = np.ones((100, 100), dtype=np.uint8)
        assert compute_effusion_ratio(mask) == pytest.approx(1.0)

    def test_effusion_area_proportional(self):
        mask = np.ones((100, 100), dtype=np.uint8)
        area = compute_effusion_area(mask, pixel_spacing_mm=1.0)
        assert area == pytest.approx(100 * 100 * 1.0)

    def test_effusion_area_pixel_spacing(self):
        mask = np.ones((10, 10), dtype=np.uint8)
        area_1 = compute_effusion_area(mask, pixel_spacing_mm=1.0)
        area_2 = compute_effusion_area(mask, pixel_spacing_mm=2.0)
        assert area_2 == pytest.approx(area_1 * 4)


class TestSeverityFromRatio:
    @pytest.mark.parametrize("ratio,expected_substr", [
        (0.0, "无明显积液"),
        (0.01, "无明显积液"),
        (0.05, "少量积液"),
        (0.10, "中量积液"),
        (0.25, "大量积液"),
    ])
    def test_severity_labels(self, ratio, expected_substr):
        result = severity_from_ratio(ratio)
        assert expected_substr in result, f"Expected '{expected_substr}' in '{result}'"


class TestFigureGeneration:
    def test_figure_to_base64_returns_string(self):
        img = _make_gray()
        mask = _make_mask(fill_ratio=0.05)
        import cv2
        overlay = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        fig = create_analysis_figure(img, overlay, mask)
        b64 = figure_to_base64(fig)
        assert isinstance(b64, str)
        assert len(b64) > 100  # non-trivial output

    def test_figure_has_three_panels(self):
        img = _make_gray()
        mask = _make_mask()
        import cv2
        overlay = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        fig = create_analysis_figure(img, overlay, mask)
        assert len(fig.axes) == 3
        import matplotlib.pyplot as plt
        plt.close(fig)
