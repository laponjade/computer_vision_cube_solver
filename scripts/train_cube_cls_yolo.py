#!/usr/bin/env python3
"""
Train an Ultralytics YOLO *classification* model on a folder with `train/` and `val/`.

Example::

    python scripts/train_cube_cls_yolo.py \\
      --data datasets/colors_data/augmented \\
      --model yolo26s-cls.pt \\
      --epochs 100 \\
      --imgsz 224
"""

from __future__ import annotations

import argparse
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser(description="Train YOLO classification (e.g. yolo26s-cls.pt).")
    ap.add_argument(
        "--data",
        type=Path,
        default=Path("datasets/colors_data/augmented"),
        help="Dataset root containing train/ and val/ class subfolders.",
    )
    ap.add_argument("--model", type=str, default="yolo26s-cls.pt", help="Ultralytics cls checkpoint name.")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument(
        "--patience",
        type=int,
        default=25,
        help="Early stopping: epochs without val improvement before stop (Ultralytics).",
    )
    ap.add_argument("--imgsz", type=int, default=224)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--project", type=str, default="runs/cls")
    ap.add_argument("--name", type=str, default="cube_colors")
    ap.add_argument(
        "--device",
        type=str,
        default=None,
        help="Ultralytics device, e.g. 0 or cpu. Default: Ultralytics auto.",
    )
    args = ap.parse_args()

    data = args.data.resolve()
    if not (data / "train").is_dir():
        raise SystemExit(f"Missing {data / 'train'}")
    if not (data / "val").is_dir():
        raise SystemExit(f"Missing {data / 'val'}")

    train_classes = [p.name for p in (data / "train").iterdir() if p.is_dir()]
    if not train_classes:
        raise SystemExit(f"No class folders under {data / 'train'}")

    from ultralytics import YOLO

    model = YOLO(args.model)
    train_kw: dict = {
        "data": str(data),
        "epochs": int(args.epochs),
        "patience": int(args.patience),
        "imgsz": int(args.imgsz),
        "batch": int(args.batch),
        "project": args.project,
        "name": args.name,
    }
    if args.device:
        train_kw["device"] = args.device

    model.train(**train_kw)


if __name__ == "__main__":
    main()
