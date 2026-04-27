"""
Demo script: 超声胸腔积液分析Agent
Ultrasound Pleural Effusion Analysis Agent – Demo

Usage::
    # Analyze a real ultrasound image
    python examples/demo.py --image path/to/scan.png

    # Run with a synthetic image (no actual image needed)
    python examples/demo.py --synthetic

    # Save the overlay figure
    python examples/demo.py --synthetic --save-overlay output/overlay.png
"""
from __future__ import annotations

import argparse
import base64
import io
import os
import sys
from pathlib import Path

import numpy as np

# Ensure src/ is importable when running the script directly
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.agent.pleural_effusion_agent import PleuralEffusionAgent


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def generate_synthetic_scan(h: int = 256, w: int = 256, seed: int = 0) -> np.ndarray:
    """
    Generate a synthetic ultrasound-like image (for demo purposes).

    The image contains a circular hypoechoic region mimicking fluid.
    """
    rng = np.random.default_rng(seed)
    base = rng.integers(30, 80, (h, w), dtype=np.uint8)
    # Add speckle noise typical of ultrasound
    noise = rng.integers(0, 40, (h, w), dtype=np.uint8)
    img = np.clip(base.astype(int) + noise.astype(int), 0, 255).astype(np.uint8)
    # Add a dark ellipse to simulate pleural effusion (hypoechoic fluid)
    center = (w // 3, h * 2 // 3)
    axes = (w // 6, h // 8)
    import cv2
    cv2.ellipse(img, center, axes, 0, 0, 360, 20, -1)
    return img


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="超声胸腔积液Agent演示 / Pleural Effusion Agent Demo"
    )
    parser.add_argument("--image", type=str, default=None, help="Path to ultrasound image")
    parser.add_argument("--synthetic", action="store_true", help="Use a synthetic test image")
    parser.add_argument("--save-overlay", type=str, default=None, help="Path to save overlay PNG")
    args = parser.parse_args()

    if args.image is None and not args.synthetic:
        parser.print_help()
        print("\n⚠  Please provide --image <path> or use --synthetic\n")
        sys.exit(1)

    # Prepare input
    if args.synthetic:
        print("📡 Generating synthetic ultrasound image …")
        source = generate_synthetic_scan()
        label = "synthetic_scan"
    else:
        source = args.image
        label = Path(args.image).name

    # Build agent
    print("🤖 Initialising PleuralEffusionAgent …")
    agent = PleuralEffusionAgent()

    # Run analysis
    print("🔍 Analysing image …")
    report = agent.analyze(source, image_label=label)

    # Print report
    print("\n" + str(report))

    # Optionally save overlay
    if args.save_overlay:
        out_path = Path(args.save_overlay)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        img_bytes = base64.b64decode(report.overlay_b64)
        with open(out_path, "wb") as f:
            f.write(img_bytes)
        print(f"\n✅ Overlay figure saved to: {out_path}")


if __name__ == "__main__":
    main()
