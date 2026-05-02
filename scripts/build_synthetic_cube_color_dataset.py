#!/usr/bin/env python3
"""
Build a 6-class *synthetic* cube-color dataset (no web scraping).

Creates solid-color PNG seeds using the same canonical BGR references as the backend Lab
palette (`backend/app/services/vision.py`), then runs `augment_cube_cls_dataset.py` with
`--color-safe` so hue/sat jitter stays mild (better for swatch-style training).

Output layout::

    ROOT/
      raw/<class>/*.png          # seed images (one dominant color per file)
      augmented/train|val/...   # Ultralytics classification layout (JPEG)

Example::

    python scripts/build_synthetic_cube_color_dataset.py \\
      --root datasets/synthetic_cube_colors \\
      --seeds-per-class 24 \\
      --aug-per-image 12 \\
      --img-size 256

If ROOT/augmented already exists and is non-empty, delete it first (or pick a new --root).
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np

# BGR (OpenCV), aligned with `_REF_BGR` in `backend/app/services/vision.py`
SYNTHETIC_BGR: dict[str, tuple[int, int, int]] = {
    "white": (220, 220, 220),
    "yellow": (0, 220, 220),
    "red": (0, 0, 220),
    "orange": (0, 140, 255),
    "blue": (220, 0, 0),
    "green": (0, 220, 0),
}

CLASS_ORDER = ("white", "yellow", "red", "orange", "blue", "green")


def _write_png(path: Path, bgr: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, buf = cv2.imencode(".png", bgr)
    if not ok:
        raise RuntimeError(f"Failed to encode PNG: {path}")
    buf.tofile(str(path))


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Generate synthetic solid-color cube PNGs and augment into train/val."
    )
    ap.add_argument(
        "--root",
        type=Path,
        default=Path("datasets/synthetic_cube_colors"),
        help="Output root (creates raw/ and augmented/).",
    )
    ap.add_argument("--seeds-per-class", type=int, default=20, help="Number of seed PNGs per class.")
    ap.add_argument(
        "--img-size",
        type=int,
        default=256,
        help="Square size (pixels) for each seed image.",
    )
    ap.add_argument(
        "--aug-per-image",
        type=int,
        default=10,
        help="Augmented JPEGs per seed in train split (passed to augment script).",
    )
    ap.add_argument("--val-ratio", type=float, default=0.12)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--skip-augment",
        action="store_true",
        help="Only write raw/ seeds; do not run augmentation.",
    )
    args = ap.parse_args()

    root: Path = args.root
    raw = root / "raw"
    aug_out = root / "augmented"

    if raw.exists() and any(raw.iterdir()):
        raise SystemExit(f"Refusing to overwrite non-empty raw folder: {raw}")
    if aug_out.exists() and any(aug_out.iterdir()):
        raise SystemExit(f"Refusing to overwrite non-empty augmented folder: {aug_out}")

    rng = np.random.default_rng(int(args.seed))
    h = w = int(np.clip(int(args.img_size), 64, 512))
    n = max(1, int(args.seeds_per_class))

    for cls in CLASS_ORDER:
        b0, g0, r0 = SYNTHETIC_BGR[cls]
        for i in range(n):
            # Tiny per-seed drift so seeds are not bitwise-identical (helps val split / diversity).
            db = int(rng.integers(-10, 11))
            dg = int(rng.integers(-10, 11))
            dr = int(rng.integers(-10, 11))
            b = int(np.clip(b0 + db, 0, 255))
            g = int(np.clip(g0 + dg, 0, 255))
            r = int(np.clip(r0 + dr, 0, 255))
            img = np.zeros((h, w, 3), dtype=np.uint8)
            img[:, :] = (b, g, r)
            # Mild spatial noise (still one dominant color class).
            noise = rng.normal(0.0, 2.2, img.shape).astype(np.float32)
            img = np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)
            out = raw / cls / f"seed_{i:03d}.png"
            _write_png(out, img)

    print(f"Wrote {n * len(CLASS_ORDER)} seed PNGs under {raw.resolve()}")

    if args.skip_augment:
        print("Skipped augmentation (--skip-augment). Run augment script manually when ready.")
        return

    aug_script = Path(__file__).resolve().parent / "augment_cube_cls_dataset.py"
    cmd = [
        sys.executable,
        str(aug_script),
        "--input",
        str(raw),
        "--output",
        str(aug_out),
        "--aug-per-image",
        str(int(args.aug_per_image)),
        "--val-ratio",
        str(float(args.val_ratio)),
        "--seed",
        str(int(args.seed)),
        "--color-safe",
    ]
    print("Running:", " ".join(cmd))
    subprocess.check_call(cmd, cwd=Path.cwd())
    print(f"Done. Train with: data={aug_out.resolve()}")


if __name__ == "__main__":
    main()
