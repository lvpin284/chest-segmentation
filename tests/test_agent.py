"""
Tests for the PleuralEffusionAgent (no LLM / no weights required).

These tests exercise the full analysis pipeline using only the rule-based
path (LLM = None) and a randomly-initialised U-Net, so no external
dependencies or model files are needed.
"""
import io
import numpy as np
import pytest
from PIL import Image

from src.agent.pleural_effusion_agent import (
    AnalysisReport,
    PleuralEffusionAgent,
    SegmentationEngine,
)


def _make_png_bytes(h: int = 128, w: int = 128) -> bytes:
    rng = np.random.default_rng(7)
    arr = rng.integers(0, 256, (h, w), dtype=np.uint8)
    img = Image.fromarray(arr, mode="L")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


class TestSegmentationEngine:
    def test_run_returns_correct_shapes(self):
        engine = SegmentationEngine(image_size=(128, 128))
        raw = _make_png_bytes()
        mask, prob_map = engine.run(raw)
        assert mask.shape == (128, 128), f"Unexpected mask shape: {mask.shape}"
        assert prob_map.shape == (128, 128), f"Unexpected prob_map shape: {prob_map.shape}"

    def test_mask_binary_values(self):
        engine = SegmentationEngine(image_size=(64, 64))
        raw = _make_png_bytes()
        mask, _ = engine.run(raw)
        unique_vals = set(mask.flatten().tolist())
        assert unique_vals.issubset({0, 1}), f"Non-binary mask values: {unique_vals}"

    def test_prob_map_range(self):
        engine = SegmentationEngine(image_size=(64, 64))
        raw = _make_png_bytes()
        _, prob_map = engine.run(raw)
        assert prob_map.min() >= 0.0
        assert prob_map.max() <= 1.0


class TestPleuralEffusionAgent:
    def _make_agent(self):
        """Create agent with no weights and no LLM (fully offline)."""
        return PleuralEffusionAgent(
            weights_path=None,
            llm=None,
            image_size=(64, 64),
        )

    def test_analyze_returns_report_instance(self):
        agent = self._make_agent()
        raw = _make_png_bytes()
        report = agent.analyze(raw, image_label="test_scan")
        assert isinstance(report, AnalysisReport)

    def test_report_fields_populated(self):
        agent = self._make_agent()
        raw = _make_png_bytes()
        report = agent.analyze(raw, image_label="test_scan")
        assert report.image_path == "test_scan"
        assert isinstance(report.severity, str)
        assert len(report.severity) > 0
        assert 0.0 <= report.effusion_ratio <= 1.0
        assert report.effusion_area_mm2 >= 0.0
        assert isinstance(report.findings, str)
        assert isinstance(report.recommendations, str)
        assert isinstance(report.overlay_b64, str)
        assert len(report.overlay_b64) > 100

    def test_report_to_dict(self):
        agent = self._make_agent()
        raw = _make_png_bytes()
        report = agent.analyze(raw)
        d = report.to_dict()
        expected_keys = {
            "image_path", "severity", "effusion_ratio",
            "effusion_area_mm2", "findings", "recommendations", "overlay_b64",
        }
        assert expected_keys.issubset(set(d.keys()))

    def test_report_str_representation(self):
        agent = self._make_agent()
        raw = _make_png_bytes()
        report = agent.analyze(raw)
        s = str(report)
        assert "积液" in s or "effusion" in s.lower()

    def test_numpy_array_input(self):
        agent = self._make_agent()
        arr = np.random.randint(0, 256, (128, 128), dtype=np.uint8)
        report = agent.analyze(arr)
        assert isinstance(report, AnalysisReport)
