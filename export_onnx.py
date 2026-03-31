from __future__ import annotations

from pathlib import Path

import torch
from ultralytics import YOLO

WEIGHTS = Path("runs/detect/runs/train/yolo26s-vehicle-opt/weights/best.pt")
IMGSZ = 640
OPSET = 13
DYNAMIC = False
SIMPLIFY = False
HALF = False


def resolve_device() -> str | int:
    return 0 if torch.cuda.is_available() else "cpu"


def main() -> None:
    if not WEIGHTS.exists():
        raise FileNotFoundError(
            f"Weights not found: {WEIGHTS}. Please run train.py first."
        )

    device = resolve_device()
    if HALF and device == "cpu":
        raise ValueError("HALF=True requires GPU. Set HALF=False or ensure CUDA is available.")

    print(f"Export device: {device}")
    model = YOLO(str(WEIGHTS))
    onnx_path = model.export(
        format="onnx",
        imgsz=IMGSZ,
        opset=OPSET,
        device=device,
        dynamic=DYNAMIC,
        simplify=SIMPLIFY,
        half=HALF,
    )
    print(f"ONNX exported to: {onnx_path}")


if __name__ == "__main__":
    main()
