#!/usr/bin/env python3
"""
Fine-tune Ultralytics YOLO26s-OBB on the converted Rubik dataset.

1) Builds labels via labelme_to_yolo_obb.py (unless --skip-convert).
2) Runs YOLO(..., task='obb').train(...).

Example:
  .venv/bin/python scripts/train_yolo26s_obb.py --epochs 100 --batch 4 --imgsz 640
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument(
        "--model",
        type=str,
        default=str(root / "models" / "yolo26s-obb.pt"),
        help="Base checkpoint (downloaded once by Ultralytics if missing)",
    )
    parser.add_argument(
        "--data",
        type=str,
        default=str(root / "datasets" / "rubik_obb" / "cube_obb.yaml"),
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Training device: cpu | mps (Apple Silicon GPU) | 0 (first CUDA GPU), etc.",
    )
    parser.add_argument("--project", type=str, default=str(root / "runs" / "obb"))
    parser.add_argument("--name", type=str, default="rubik_cube")
    parser.add_argument("--face-only", action="store_true", help="Train single-class `face`")
    parser.add_argument("--skip-convert", action="store_true")
    args = parser.parse_args()

    if not args.skip_convert:
        cmd = [
            sys.executable,
            str(root / "scripts" / "labelme_to_yolo_obb.py"),
            "--images-root",
            str(root / "images"),
            "--out",
            str(root / "datasets" / "rubik_obb"),
        ]
        if args.face_only:
            cmd.append("--face-only")
        print("Running:", " ".join(cmd))
        r = subprocess.run(cmd, cwd=str(root))
        if r.returncode != 0:
            return r.returncode

    data_path = Path(args.data)
    if not data_path.is_file():
        print(f"Missing dataset yaml: {data_path}", file=sys.stderr)
        return 2

    from ultralytics import YOLO

    model = YOLO(args.model, task="obb")
    model.train(
        data=str(data_path),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        project=args.project,
        name=args.name,
        exist_ok=True,
    )
    print("Best weights typically at:", Path(args.project) / args.name / "weights" / "best.pt")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
