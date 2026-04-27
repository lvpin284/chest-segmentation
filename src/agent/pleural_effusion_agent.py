"""
超声胸腔积液智能分析Agent
Ultrasound Pleural Effusion Intelligent Analysis Agent

This module implements a LangChain-based agent with specialist tools:

  1. segment_effusion  – run U-Net segmentation on a provided image path/bytes
  2. analyze_severity  – compute area metrics and severity grade from a mask
  3. generate_report   – produce a structured clinical report
  4. explain_findings  – call an LLM to interpret findings in plain language

The agent can be driven by any LLM that supports tool / function calling
(e.g. OpenAI GPT-4o, Anthropic Claude, local Ollama models).

Quick start::
    from src.agent.pleural_effusion_agent import PleuralEffusionAgent
    agent = PleuralEffusionAgent()              # uses env var OPENAI_API_KEY
    result = agent.analyze("path/to/scan.png")
    print(result["report"])
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch

from langchain_core.messages import HumanMessage, SystemMessage

from ..models.unet import UNet
from ..preprocessing.image_processor import UltrasoundPreprocessor
from ..utils.visualization import (
    compute_effusion_area,
    compute_effusion_ratio,
    create_analysis_figure,
    figure_to_base64,
    overlay_mask,
    severity_from_ratio,
    tensor_to_numpy_image,
)


# ---------------------------------------------------------------------------
# Default model weights path
# ---------------------------------------------------------------------------
_DEFAULT_WEIGHTS: Optional[str] = os.getenv("UNET_WEIGHTS_PATH", None)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class SegmentationResult:
    """Holds raw segmentation outputs for a single image."""
    image_path: str
    mask: np.ndarray           # binary (H, W) uint8
    probability_map: np.ndarray  # float32 (H, W) in [0, 1]
    effusion_ratio: float
    effusion_area_mm2: float
    severity: str
    overlay_b64: str           # base64-encoded PNG of the overlay figure


@dataclass
class AnalysisReport:
    """Structured report returned by the agent."""
    image_path: str
    severity: str
    effusion_ratio: float
    effusion_area_mm2: float
    findings: str              # LLM-generated clinical findings
    recommendations: str       # LLM-generated recommendations
    overlay_b64: str
    raw_metrics: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "image_path": self.image_path,
            "severity": self.severity,
            "effusion_ratio": round(self.effusion_ratio, 4),
            "effusion_area_mm2": round(self.effusion_area_mm2, 2),
            "findings": self.findings,
            "recommendations": self.recommendations,
            "overlay_b64": self.overlay_b64,
        }

    def __str__(self) -> str:
        lines = [
            "=" * 60,
            "  超声胸腔积液分析报告 / Pleural Effusion Analysis Report",
            "=" * 60,
            f"图像路径 / Image   : {self.image_path}",
            f"严重程度 / Severity: {self.severity}",
            f"积液比例 / Ratio   : {self.effusion_ratio:.2%}",
            f"积液面积 / Area    : {self.effusion_area_mm2:.1f} mm²",
            "",
            "发现 / Findings:",
            self.findings,
            "",
            "建议 / Recommendations:",
            self.recommendations,
            "=" * 60,
        ]
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# SegmentationEngine – wraps the U-Net model
# ---------------------------------------------------------------------------

class SegmentationEngine:
    """
    Thin wrapper around the U-Net model that handles device placement,
    weight loading, and inference.

    Args:
        weights_path: Path to a ``torch.save``-d state-dict file.
                      If *None*, the model runs with random weights
                      (useful for development / testing).
        device: "cuda", "mps", or "cpu".
        image_size: (H, W) expected by the model.
        threshold: Sigmoid probability threshold for the binary mask.
    """

    def __init__(
        self,
        weights_path: Optional[str] = None,
        device: Optional[str] = None,
        image_size: Tuple[int, int] = (256, 256),
        threshold: float = 0.5,
    ) -> None:
        self.device = torch.device(
            device
            or ("cuda" if torch.cuda.is_available() else
                "mps" if torch.backends.mps.is_available() else "cpu")
        )
        self.threshold = threshold
        self.preprocessor = UltrasoundPreprocessor(image_size=image_size)
        self.model = UNet(in_channels=1, num_classes=1).to(self.device)
        self.model.eval()

        if weights_path and Path(weights_path).exists():
            state = torch.load(weights_path, map_location=self.device)
            self.model.load_state_dict(state)

    @torch.no_grad()
    def run(
        self, source: Union[str, Path, bytes, np.ndarray]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Run segmentation on a single image.

        Returns:
            mask: Binary (H, W) uint8 array.
            prob_map: Float32 (H, W) probability array.
        """
        tensor = self.preprocessor(source).unsqueeze(0).to(self.device)  # (1,1,H,W)
        logits = self.model(tensor)
        prob_map = torch.sigmoid(logits).squeeze().cpu().numpy().astype(np.float32)
        mask = (prob_map >= self.threshold).astype(np.uint8)
        return mask, prob_map


# ---------------------------------------------------------------------------
# PleuralEffusionAgent
# ---------------------------------------------------------------------------

class PleuralEffusionAgent:
    """
    Intelligent agent for ultrasound pleural effusion analysis.

    The agent orchestrates:
        1. Image segmentation via a U-Net model.
        2. Quantitative metric computation.
        3. LLM-based clinical report generation (optional).

    When no LLM API key is configured the agent still performs segmentation
    and returns structured metrics without the natural language report.

    Args:
        weights_path: Path to U-Net weights file.
        llm: A LangChain ``BaseChatModel`` instance.
                If *None*, the agent will try to create an ``OpenAI`` client
                using the ``OPENAI_API_KEY`` environment variable.
        device: Torch device string.
        image_size: (H, W) for the segmentation model.
        threshold: Sigmoid threshold for binary mask.
        pixel_spacing_mm: Physical pixel spacing for area estimation.

    Example::
        agent = PleuralEffusionAgent()
        report = agent.analyze("ultrasound.png")
        print(report)
    """

    _SYSTEM_PROMPT = (
        "你是一名超声影像AI助手，专注于胸腔积液的检测与分析。\n"
        "You are an ultrasound AI assistant specialising in pleural effusion detection and analysis.\n\n"
        "根据提供的超声影像分析指标，用中英文双语提供简洁的临床描述和建议。\n"
        "Based on the provided ultrasound image metrics, provide a concise bilingual (Chinese/English) "
        "clinical description and recommendation.\n\n"
        "重要提示: 你的输出仅供参考，不能替代专业医师的诊断。\n"
        "IMPORTANT: Your output is for reference only and does not replace professional medical diagnosis."
    )

    def __init__(
        self,
        weights_path: Optional[str] = _DEFAULT_WEIGHTS,
        llm=None,
        device: Optional[str] = None,
        image_size: Tuple[int, int] = (256, 256),
        threshold: float = 0.5,
        pixel_spacing_mm: float = 0.1,
    ) -> None:
        self.engine = SegmentationEngine(
            weights_path=weights_path,
            device=device,
            image_size=image_size,
            threshold=threshold,
        )
        self.pixel_spacing_mm = pixel_spacing_mm
        self._llm = llm or self._try_create_llm()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def analyze(
        self,
        source: Union[str, Path, bytes, np.ndarray],
        image_label: str = "input_image",
    ) -> AnalysisReport:
        """
        Full analysis pipeline for a single ultrasound image.

        Args:
            source: Image file path, raw bytes, or NumPy array.
            image_label: Label used in the report (e.g. filename).

        Returns:
            :class:`AnalysisReport` with all metrics and the LLM report.
        """
        seg = self._segment(source, str(image_label))
        findings, recommendations = self._generate_text(seg)

        return AnalysisReport(
            image_path=seg.image_path,
            severity=seg.severity,
            effusion_ratio=seg.effusion_ratio,
            effusion_area_mm2=seg.effusion_area_mm2,
            findings=findings,
            recommendations=recommendations,
            overlay_b64=seg.overlay_b64,
        )

    # ------------------------------------------------------------------
    # Internal pipeline steps
    # ------------------------------------------------------------------

    def _segment(
        self,
        source: Union[str, Path, bytes, np.ndarray],
        label: str,
    ) -> SegmentationResult:
        """Run the U-Net segmentation and compute metrics."""
        from ..preprocessing.image_processor import load_image, apply_clahe, resize_with_padding, normalize

        # Original image for display
        raw = load_image(source)
        raw_clahe = apply_clahe(raw)
        raw_resized = resize_with_padding(raw_clahe, self.engine.preprocessor.image_size)

        # Segmentation
        mask, prob_map = self.engine.run(source)

        # Metrics
        ratio = compute_effusion_ratio(mask)
        area = compute_effusion_area(mask, self.pixel_spacing_mm)
        severity = severity_from_ratio(ratio)

        # Overlay figure
        ov = overlay_mask(raw_resized, mask)
        fig = create_analysis_figure(raw_resized, ov, mask)
        b64 = figure_to_base64(fig)

        return SegmentationResult(
            image_path=label,
            mask=mask,
            probability_map=prob_map,
            effusion_ratio=ratio,
            effusion_area_mm2=area,
            severity=severity,
            overlay_b64=b64,
        )

    def _generate_text(
        self, seg: SegmentationResult
    ) -> Tuple[str, str]:
        """Use the LLM to generate findings and recommendations."""
        if self._llm is None:
            findings = self._rule_based_findings(seg)
            recommendations = self._rule_based_recommendations(seg)
            return findings, recommendations

        prompt_text = (
            f"超声胸腔积液分析结果 / Pleural effusion analysis results:\n"
            f"- 严重程度 / Severity: {seg.severity}\n"
            f"- 积液占比 / Effusion ratio: {seg.effusion_ratio:.2%}\n"
            f"- 估计面积 / Estimated area: {seg.effusion_area_mm2:.1f} mm²\n\n"
            "请提供:\n"
            "1. 简洁的影像学描述 (中英文)\n"
            "2. 临床建议 (中英文)\n\n"
            "Please provide:\n"
            "1. A concise radiological description (bilingual)\n"
            "2. Clinical recommendations (bilingual)"
        )
        try:
            messages = [
                SystemMessage(content=self._SYSTEM_PROMPT),
                HumanMessage(content=prompt_text),
            ]
            response = self._llm.invoke(messages)
            text = response.content

            # Simple split: try to find "建议" or "Recommendation" as separator
            if "建议" in text or "Recommendation" in text:
                for sep in ["临床建议", "Recommendations:", "建议:", "建议："]:
                    if sep in text:
                        parts = text.split(sep, 1)
                        return parts[0].strip(), (sep + parts[1]).strip()
            return text, "请结合临床实际情况判断。/ Please correlate with clinical findings."
        except Exception as exc:  # noqa: BLE001
            findings = self._rule_based_findings(seg)
            return findings, f"LLM unavailable ({exc}). {self._rule_based_recommendations(seg)}"

    @staticmethod
    def _rule_based_findings(seg: SegmentationResult) -> str:
        return (
            f"超声图像显示: {seg.severity}。\n"
            f"积液区域占图像面积的 {seg.effusion_ratio:.2%}，"
            f"估算面积约 {seg.effusion_area_mm2:.1f} mm²。\n\n"
            f"Ultrasound image shows: {seg.severity}.\n"
            f"Effusion region occupies {seg.effusion_ratio:.2%} of the image area, "
            f"estimated area ≈ {seg.effusion_area_mm2:.1f} mm²."
        )

    @staticmethod
    def _rule_based_recommendations(seg: SegmentationResult) -> str:
        ratio = seg.effusion_ratio
        if ratio < 0.02:
            return (
                "建议: 无需特殊处理，定期随访。\n"
                "Recommendation: No specific treatment required; regular follow-up."
            )
        elif ratio < 0.08:
            return (
                "建议: 结合临床症状决定是否需要进一步检查或穿刺引流。\n"
                "Recommendation: Correlate with clinical symptoms to determine "
                "whether further investigation or drainage is required."
            )
        elif ratio < 0.20:
            return (
                "建议: 建议进行胸腔穿刺并送检积液，明确病因。\n"
                "Recommendation: Thoracocentesis with fluid analysis is recommended "
                "to clarify the aetiology."
            )
        else:
            return (
                "建议: 大量积液，建议紧急会诊，必要时行胸腔引流。\n"
                "Recommendation: Large effusion – urgent consultation recommended; "
                "chest drain placement may be required."
            )

    # ------------------------------------------------------------------
    # LLM factory
    # ------------------------------------------------------------------

    @staticmethod
    def _try_create_llm():
        """Attempt to create a ChatOpenAI instance; return None on failure."""
        try:
            from langchain_openai import ChatOpenAI  # type: ignore
            api_key = os.getenv("OPENAI_API_KEY", "")
            if not api_key:
                return None
            return ChatOpenAI(model="gpt-4o", temperature=0.2)
        except Exception:  # noqa: BLE001
            return None
