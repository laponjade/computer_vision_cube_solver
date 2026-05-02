#!/usr/bin/env python3
"""
Train Ultralytics **YOLO26s-cls** on a classification folder (`train/` + `val/`).

Defaults target the synthetic cube-color dataset produced by
`scripts/build_synthetic_cube_color_dataset.py`.

If your `ultralytics` build does not ship `yolo26s-cls.pt`, install/update ultralytics or pass
`--model yolo11s-cls.pt` (or another `*-cls.pt` your version supports).

Example (from repo root)::

    python scripts/train_yolo26s_cls.py --device 0

Override data path or run name::

    python scripts/train_yolo26s_cls.py \\
      --data datasets/colors_data/augmented \\
      --name my_run \\
      --epochs 150 \\
      --patience 20 \\
      --device mps
"""

from __future__ import annotations

import argparse
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser(description="Train YOLO26s-cls on a cube color classification dataset.")
    ap.add_argument(
        "--data",
        type=Path,
        default=Path("datasets/synthetic_cube_colors/augmented"),
        help="Dataset root with train/ and val/ (default: synthetic cube colors).",
    )
    ap.add_argument(
        "--model",
        type=str,
        default="yolo26s-cls.pt",
        help="Ultralytics classification weights (default: yolo26s-cls.pt).",
    )
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument(
        "--patience",
        type=int,
        default=25,
        help="Early stopping: stop after this many epochs without val improvement (Ultralytics).",
    )
    ap.add_argument("--imgsz", type=int, default=224)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--project", type=str, default="runs/cls")
    ap.add_argument(
        "--name",
        type=str,
        default="yolo26s_synthetic_cube",
        help="Run name under --project (weights in project/name/weights/).",
    )
    ap.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device: 0, cpu, mps, etc. Omit for Ultralytics default.",
    )
    args = ap.parse_args()

    data = args.data.resolve()
    if not (data / "train").is_dir():
        raise SystemExit(f"Missing {data / 'train'}. Build data first (e.g. build_synthetic_cube_color_dataset.py).")
    if not (data / "val").is_dir():
        raise SystemExit(f"Missing {data / 'val'}")

    classes = [p.name for p in (data / "train").iterdir() if p.is_dir()]
    if not classes:
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

    print(f"Training {args.model} on {data} ({len(classes)} classes)")
    model.train(**train_kw)


if __name__ == "__main__":
    main()
