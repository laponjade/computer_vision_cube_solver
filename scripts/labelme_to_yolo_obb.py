#!/usr/bin/env python3
"""
Convert LabelMe JSON (rectangle = two opposite corners) under images/{train,valid,test}
into Ultralytics YOLO-OBB label files.

OBB line format (normalized 0–1):
  class x1 y1 x2 y2 x3 y3 x4 y4
Corners are axis-aligned: top-left, top-right, bottom-right, bottom-left.

See: https://docs.ultralytics.com/datasets/obb/
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

try:
    import yaml
except ImportError:
    print("Install PyYAML: pip install pyyaml", file=sys.stderr)
    raise

logger = logging.getLogger(__name__)

# Match README / dataset convention (stable class ids)
CLASS_TO_ID: dict[str, int] = {
    "face": 0,
    "red_tile": 1,
    "white_tile": 2,
    "blue_tile": 3,
    "orange_tile": 4,
    "green_tile": 5,
    "yellow_tile": 6,
}
ID_TO_NAME: dict[int, str] = {v: k for k, v in CLASS_TO_ID.items()}


def rectangle_to_obb_line(
    cls: int, x1: float, y1: float, x2: float, y2: float, iw: int, ih: int
) -> str:
    xmin, xmax = min(x1, x2), max(x1, x2)
    ymin, ymax = min(y1, y2), max(y1, y2)
    # TL, TR, BR, BL
    corners = [
        (xmin, ymin),
        (xmax, ymin),
        (xmax, ymax),
        (xmin, ymax),
    ]
    parts = [str(cls)]
    for x, y in corners:
        parts.append(f"{x / iw:.6f}")
        parts.append(f"{y / ih:.6f}")
    return " ".join(parts)


def convert_annotation(json_path: Path, label_out: Path, face_only: bool) -> int:
    data = json.loads(json_path.read_text(encoding="utf-8"))
    ih = int(data["imageHeight"])
    iw = int(data["imageWidth"])
    lines: list[str] = []
    for shape in data.get("shapes", []):
        if shape.get("shape_type") != "rectangle":
            logger.warning("Skip non-rectangle in %s: %s", json_path.name, shape.get("shape_type"))
            continue
        pts = shape.get("points") or []
        if len(pts) != 2:
            logger.warning("Skip bad points in %s", json_path.name)
            continue
        label = shape.get("label", "")
        if label not in CLASS_TO_ID:
            logger.warning("Unknown label %r in %s — skipped", label, json_path.name)
            continue
        if face_only and label != "face":
            continue
        cls = 0 if face_only else CLASS_TO_ID[label]
        x1, y1 = float(pts[0][0]), float(pts[0][1])
        x2, y2 = float(pts[1][0]), float(pts[1][1])
        lines.append(rectangle_to_obb_line(cls, x1, y1, x2, y2, iw, ih))
    label_out.parent.mkdir(parents=True, exist_ok=True)
    label_out.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    return len(lines)


def symlink_image(src: Path, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() or dest.is_symlink():
        dest.unlink()
    dest.symlink_to(src.resolve())


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--images-root",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "images",
        help="Directory containing train/, valid/, test/ with Cube_*.jpg + .json",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "datasets" / "rubik_obb",
        help="YOLO dataset root (images + labels + cube_obb.yaml)",
    )
    parser.add_argument(
        "--face-only",
        action="store_true",
        help="Export only the `face` class as id 0 (single-class dataset)",
    )
    parser.add_argument(
        "--copy-images",
        action="store_true",
        help="Copy JPGs instead of symlinking (large disk use)",
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    splits = ["train", "valid", "test"]
    total_boxes = 0
    total_images = 0

    for split in splits:
        src_dir = args.images_root / split
        if not src_dir.is_dir():
            logger.warning("Missing split folder: %s", src_dir)
            continue
        for json_path in sorted(src_dir.glob("Cube_*.json")):
            stem = json_path.stem
            img_path = src_dir / f"{stem}.jpg"
            if not img_path.is_file():
                logger.warning("No image for %s — skip", json_path.name)
                continue
            out_img = args.out / "images" / split / f"{stem}.jpg"
            out_lbl = args.out / "labels" / split / f"{stem}.txt"
            if args.copy_images:
                import shutil

                shutil.copy2(img_path, out_img)
            else:
                symlink_image(img_path, out_img)
            n = convert_annotation(json_path, out_lbl, face_only=args.face_only)
            total_boxes += n
            total_images += 1

    nc = 1 if args.face_only else len(CLASS_TO_ID)
    names = {0: "face"} if args.face_only else {i: ID_TO_NAME[i] for i in range(nc)}

    yaml_path = args.out / "cube_obb.yaml"
    cfg = {
        "path": str(args.out.resolve()),
        "train": "images/train",
        "val": "images/valid",
        "test": "images/test",
        "nc": nc,
        "names": names,
    }
    yaml_path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")

    logger.info("Wrote %s", yaml_path)
    logger.info("Images: %s, boxes: %s (face_only=%s)", total_images, total_boxes, args.face_only)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
