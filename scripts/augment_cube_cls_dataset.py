#!/usr/bin/env python3
"""
Offline augmentation for image *classification*: each immediate subfolder of INPUT is one class.

Example::

    INPUT/
      white/*.jpg
      blue/*.jpg
      ...

Or your layout::

    datasets/colors_data/
      <any_class_name>/*.png

Class labels for training are the **folder names** (copied to train/ and val/).

Writes Ultralytics-style classification layout::

    OUTPUT/
      train/<class>/*.jpg
      val/<class>/*.jpg   (originals only, optional split)

Augmentations (OpenCV, applied with random parameters per synthetic sample):
  - Brightness & contrast (critical for color)
  - Hue / saturation / value jitter in HSV
  - Random crop (scale) + resize back to original size
  - Random horizontal flip, optional vertical flip, random in-plane rotation
  - Gaussian blur and Gaussian noise

Example::

    python scripts/augment_cube_cls_dataset.py \\
      --input datasets/colors_data \\
      --output datasets/colors_data/augmented \\
      --aug-per-image 8 \\
      --val-ratio 0.15 \\
      --seed 42
"""

from __future__ import annotations

import argparse
import random
import shutil
from functools import partial
from pathlib import Path

import cv2
import numpy as np

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def discover_class_dirs(root: Path) -> list[str]:
    """Each non-hidden immediate subdirectory is a class name."""
    if not root.is_dir():
        return []
    names: list[str] = []
    for p in sorted(root.iterdir()):
        if p.is_dir() and not p.name.startswith("."):
            names.append(p.name)
    return names


def _list_images(class_dir: Path) -> list[Path]:
    if not class_dir.is_dir():
        return []
    out: list[Path] = []
    for p in sorted(class_dir.iterdir()):
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS:
            out.append(p)
    return out


def _read_bgr(path: Path) -> np.ndarray:
    img = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"Could not read image: {path}")
    return img


def _write_jpg(path: Path, bgr: np.ndarray, quality: int = 92) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, buf = cv2.imencode(".jpg", bgr, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    if not ok:
        raise RuntimeError(f"Failed to encode JPEG: {path}")
    buf.tofile(str(path))


def augment_brightness_contrast(bgr: np.ndarray, rng: random.Random) -> np.ndarray:
    alpha = rng.uniform(0.82, 1.18)  # contrast
    beta = rng.uniform(-32.0, 32.0)  # brightness
    out = cv2.convertScaleAbs(bgr, alpha=alpha, beta=beta)
    return out


def augment_hsv_jitter(bgr: np.ndarray, rng: random.Random, *, color_safe: bool = False) -> np.ndarray:
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV).astype(np.float32)
    h, s, v = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]

    # Strong jitter shifts hue across class boundaries (bad for solid swatches). --color-safe uses mild ranges.
    if color_safe:
        dh = rng.uniform(-2.5, 2.5)
        sf = rng.uniform(0.9, 1.1)
        vf = rng.uniform(0.9, 1.1)
    else:
        dh = rng.uniform(-12.0, 12.0)  # OpenCV H is 0..179
        sf = rng.uniform(0.75, 1.28)
        vf = rng.uniform(0.82, 1.18)
    h = (h + dh) % 180.0

    s = np.clip(s * sf, 0, 255)

    v = np.clip(v * vf, 0, 255)

    hsv[:, :, 0] = h
    hsv[:, :, 1] = s
    hsv[:, :, 2] = v
    out = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
    return out


def augment_random_crop_resize(bgr: np.ndarray, rng: random.Random) -> np.ndarray:
    h, w = bgr.shape[:2]
    if h < 8 or w < 8:
        return bgr
    scale = rng.uniform(0.65, 1.0)
    ch = max(4, int(round(h * scale)))
    cw = max(4, int(round(w * scale)))
    y0 = rng.randint(0, max(0, h - ch))
    x0 = rng.randint(0, max(0, w - cw))
    crop = bgr[y0 : y0 + ch, x0 : x0 + cw]
    return cv2.resize(crop, (w, h), interpolation=cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR)


def augment_flip(bgr: np.ndarray, rng: random.Random) -> np.ndarray:
    out = bgr
    if rng.random() < 0.5:
        out = cv2.flip(out, 1)
    if rng.random() < 0.2:
        out = cv2.flip(out, 0)
    return out


def augment_rotation(bgr: np.ndarray, rng: random.Random) -> np.ndarray:
    if rng.random() > 0.65:
        return bgr
    h, w = bgr.shape[:2]
    angle = rng.uniform(-28.0, 28.0)
    m = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), angle, 1.0)
    return cv2.warpAffine(
        bgr,
        m,
        (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT101,
    )


def augment_blur(bgr: np.ndarray, rng: random.Random) -> np.ndarray:
    if rng.random() > 0.35:
        return bgr
    k = rng.choice((3, 5, 7))
    return cv2.GaussianBlur(bgr, (k, k), 0)


def augment_noise(bgr: np.ndarray, rng: random.Random) -> np.ndarray:
    if rng.random() > 0.35:
        return bgr
    sigma = rng.uniform(2.0, 10.0)
    noise = np.random.normal(0.0, sigma, bgr.shape).astype(np.float32)
    x = bgr.astype(np.float32) + noise
    return np.clip(x, 0, 255).astype(np.uint8)


def apply_augmentation_pipeline(
    bgr: np.ndarray, rng: random.Random, *, color_safe: bool = False
) -> np.ndarray:
    """
    Compose augmentations in a semi-random order so each synthetic sample differs.
    """
    x = bgr.copy()

    hsv_fn = partial(augment_hsv_jitter, color_safe=color_safe)
    # Order shuffle for variety (keeps all techniques present across the dataset).
    steps = [
        augment_brightness_contrast,
        hsv_fn,
        augment_random_crop_resize,
        augment_flip,
        augment_rotation,
        augment_blur,
        augment_noise,
    ]
    rng.shuffle(steps)
    for fn in steps:
        x = fn(x, rng)
    return x


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Augment a classification dataset: each subfolder of --input is one class."
    )
    ap.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Root folder; each immediate subfolder is a class (e.g. datasets/colors_data).",
    )
    ap.add_argument("--output", type=Path, required=True, help="Output root (creates train/ and val/).")
    ap.add_argument("--aug-per-image", type=int, default=6, help="Synthetic images per source train image.")
    ap.add_argument(
        "--val-ratio",
        type=float,
        default=0.12,
        help="Fraction of source images per class held out as val (originals only).",
    )
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--include-train-originals",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Copy original images into train/ for images not in val split.",
    )
    ap.add_argument("--jpeg-quality", type=int, default=92)
    ap.add_argument(
        "--color-safe",
        action="store_true",
        help="Mild HSV hue/sat/value jitter (better for solid-color / swatch classifiers).",
    )
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    rng = random.Random(args.seed)
    inp: Path = args.input
    out: Path = args.output

    if not inp.is_dir():
        raise SystemExit(f"Input not found: {inp}")

    classes = discover_class_dirs(inp)
    if not classes:
        raise SystemExit(
            f"No class subfolders found under {inp}. Add one folder per class (e.g. white/, blue/)."
        )

    out_train = out / "train"
    out_val = out / "val"
    if out.exists() and any(out.iterdir()):
        raise SystemExit(f"Output directory is not empty. Remove or choose a new path: {out}")

    total_aug = 0
    total_val = 0
    total_tr_orig = 0

    for cls in classes:
        files = _list_images(inp / cls)
        if not files:
            raise SystemExit(f"No images found in {inp / cls}")

        rng.shuffle(files)
        n_val = int(round(len(files) * float(np.clip(args.val_ratio, 0.0, 0.9))))
        val_files = set(files[:n_val])
        train_files = files[n_val:]
        if not train_files:
            raise SystemExit(
                f"Class {cls!r}: all {len(files)} image(s) went to val (--val-ratio too high?). "
                f"Lower --val-ratio or add more images."
            )

        for p in val_files:
            dst = out_val / cls / f"{p.stem}{p.suffix.lower()}"
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(p, dst)
            total_val += 1

        stem_counter = 0
        for p in train_files:
            if args.include_train_originals:
                dst = out_train / cls / f"orig_{p.stem}{p.suffix.lower()}"
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(p, dst)
                total_tr_orig += 1

            bgr = _read_bgr(p)
            for i in range(max(0, args.aug_per_image)):
                stem_counter += 1
                aug = apply_augmentation_pipeline(bgr, rng, color_safe=bool(args.color_safe))
                dst = out_train / cls / f"aug_{p.stem}_{stem_counter:05d}.jpg"
                _write_jpg(dst, aug, quality=args.jpeg_quality)
                total_aug += 1

    print(
        f"Done.\n"
        f"  classes ({len(classes)}): {', '.join(classes)}\n"
        f"  val images (originals): {total_val}\n"
        f"  train originals:      {total_tr_orig}\n"
        f"  train augmented:      {total_aug}\n"
        f"  output: {out.resolve()}\n"
        f"\nTrain with Ultralytics classification, e.g.:\n"
        f"  yolo classify train data={out.resolve()} model=yolo11s-cls.pt epochs=100 imgsz=224\n"
        f"(Use the exact *_cls.pt name supported by your installed ultralytics package.)"
    )


if __name__ == "__main__":
    main()
