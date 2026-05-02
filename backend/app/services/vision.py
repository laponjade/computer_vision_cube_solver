"""
Cube vision pipeline (single face):

- **Face localization**: YOLO (detect or OBB) when weights are available, otherwise a contour-based fallback.
- **Color extraction**: strategies selected via `VISION_COLOR_MODE`:
  - `model`: run YOLO on the warped face and infer per-tile colors from class labels
  - `lab`: classic OpenCV heuristic using Lab (a/b) distance to canonical color references
  - `dl`: sticker crops + HSV guess; optional `VISION_DL_CLS_WEIGHTS` runs YOLO-cls per crop (test only).
    Use `VISION_MODEL_CLASSIFY_STRICT=true` to use only the classifier for sticker `color` (skip HSV).
"""

from __future__ import annotations

import logging
import os
import time
import base64
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import cv2
import numpy as np

from app.schemas.vision import FaceGridResult, StickerCell

if TYPE_CHECKING:
    from ultralytics import YOLO

logger = logging.getLogger(__name__)


def default_face_weights_path() -> Path:
    """First `models/cube_face.pt` walking up from this file (repo root or Docker `/app`)."""
    here = Path(__file__).resolve()
    for root in here.parents:
        models_dir = root / "models"
        if models_dir.is_dir():
            return models_dir / "cube_face.pt"
    return Path("models") / "cube_face.pt"


def _repo_root_from_here() -> Path:
    """
    Best-effort repo root resolution for local/dev.
    We treat the first parent that contains `runs/` or `models/` as the root.
    """
    here = Path(__file__).resolve()
    for root in here.parents:
        if (root / "runs").is_dir() or (root / "models").is_dir():
            return root
    return Path.cwd()


def _auto_weights() -> tuple[str | None, Literal["detect", "obb"]]:
    """
    Choose weights when YOLO_FACE_WEIGHTS isn't provided.

    Preference order:
    - `models/cube_face.pt` (detect)
    - `runs/obb/rubik_cube_from_best/weights/best.pt` (obb) if present
    - newest `runs/obb/*/weights/best.pt` (obb)
    """
    models_path = default_face_weights_path()
    if models_path.is_file():
        return str(models_path), "detect"

    root = _repo_root_from_here()
    preferred = root / "runs" / "obb" / "rubik_cube_from_best" / "weights" / "best.pt"
    if preferred.is_file():
        return str(preferred), "obb"

    runs_obb = root / "runs" / "obb"
    if runs_obb.is_dir():
        candidates = list(runs_obb.glob("*/weights/best.pt"))
        candidates = [p for p in candidates if p.is_file()]
        if candidates:
            newest = max(candidates, key=lambda p: p.stat().st_mtime)
            return str(newest), "obb"

    return None, "detect"

# --- Lab heuristic reference palette -------------------------------------------------
#
# Canonical sticker colors (BGR) tuned for typical cube plastics under neutral light.
# We convert these to Lab and classify each sticker by nearest reference in Lab a/b.
_REF_BGR = np.array(
    [
        [220, 220, 220],  # white
        [0, 220, 220],  # yellow
        [0, 0, 220],  # red
        [0, 140, 255],  # orange
        [220, 0, 0],  # blue
        [0, 220, 0],  # green
    ],
    dtype=np.float64,
)
_COLOR_NAMES: tuple[str, ...] = ("white", "yellow", "red", "orange", "blue", "green")
_REF_LAB = cv2.cvtColor(_REF_BGR.reshape(1, 6, 3).astype(np.uint8), cv2.COLOR_BGR2LAB).reshape(
    6, 3
).astype(np.float64)
# Use only a/b channels for distance (less sensitive to exposure).
_REF_AB = _REF_LAB[:, 1:3]


@dataclass
class _Bbox:
    x1: float
    y1: float
    x2: float
    y2: float
    confidence: float


def _softmax_neg_distances(dists: np.ndarray, sigma: float = 18.0) -> tuple[int, float]:
    """
    Convert distances to a soft confidence.

    Role: map "how close is this sticker to each reference color" into a probability-like score
    so the UI can display confidence and we can down-weight heuristic fallbacks.
    """
    logits = -((dists.astype(np.float64)) ** 2) / (2 * sigma**2)
    logits = logits - np.max(logits)
    exp = np.exp(logits)
    probs = exp / (np.sum(exp) + 1e-12)
    idx = int(np.argmax(probs))
    return idx, float(probs[idx])


def _robust_bgr_cell(roi: np.ndarray) -> np.ndarray:
    """Return a robust per-cell BGR statistic (median) to reduce glare sensitivity."""
    if roi.size == 0:
        return np.array([128.0, 128.0, 128.0])
    # Median is much less sensitive to specular highlights/glare than mean.
    flat = roi.reshape(-1, 3).astype(np.float64)
    return np.median(flat, axis=0)


def _center_patch(roi: np.ndarray, ratio: float = 0.55) -> np.ndarray:
    """
    Crop to the center region of a ROI.

    Role: avoid sampling tile borders / grid lines / background, which often pushes colors toward
    neutral (white/gray) and hurts discrimination.
    """
    if roi.size == 0:
        return roi
    h, w = roi.shape[:2]
    if h < 2 or w < 2:
        return roi
    r = float(np.clip(ratio, 0.15, 1.0))
    ph = max(1, int(round(h * r)))
    pw = max(1, int(round(w * r)))
    y1 = (h - ph) // 2
    x1 = (w - pw) // 2
    return roi[y1 : y1 + ph, x1 : x1 + pw]


def _allow_white_from_bgr(
    bgr_med: np.ndarray,
    *,
    min_v: float,
    max_c: float,
) -> bool:
    """
    Decide whether "white" should be considered for this sticker.

    Role: prevent white from dominating when colors are low-chroma or washed out.
    We only allow white when the patch is bright enough AND sufficiently low-chroma.
    """
    b = float(bgr_med[0])
    g = float(bgr_med[1])
    r = float(bgr_med[2])
    v = (b + g + r) / 3.0  # brightness proxy
    c = max(b, g, r) - min(b, g, r)  # chroma proxy
    return (v >= min_v) and (c <= max_c)


def _gray_world_white_balance(bgr: np.ndarray) -> np.ndarray:
    """
    Simple per-image white balance: scale channels so their means match.
    Helps separate red/orange/yellow when camera auto-WB drifts warm/cool.
    """
    img = bgr.astype(np.float32)
    means = img.reshape(-1, 3).mean(axis=0)
    mean_all = float(np.mean(means) + 1e-6)
    scale = mean_all / (means + 1e-6)
    out = img * scale.reshape(1, 1, 3)
    return np.clip(out, 0, 255).astype(np.uint8)


def _yolo_cls_top1(r0) -> tuple[str, float]:
    """
    Read top-1 class name and confidence from an Ultralytics *classification* result.

    Role: optional test path for VISION_DL_CLS_WEIGHTS without touching HSV logic.
    """
    probs = getattr(r0, "probs", None)
    if probs is None:
        return "", 0.0
    i = int(probs.top1)
    names = getattr(r0, "names", None)
    raw = ""
    if isinstance(names, dict):
        raw = str(names.get(i, "") or "")
    elif isinstance(names, (list, tuple)) and 0 <= i < len(names):
        raw = str(names[i] or "")
    conf_t = getattr(probs, "top1conf", None)
    if conf_t is None:
        try:
            import torch

            cf = float(torch.nn.functional.softmax(probs.data, dim=0).max().item())
        except Exception:
            cf = 0.0
    else:
        cf = float(conf_t.item()) if hasattr(conf_t, "item") else float(conf_t)
    if cf > 1.0001:
        cf = min(1.0, cf / 100.0)
    return raw, min(1.0, max(0.0, cf))


def _classify_hsv_color(bgr_patch: np.ndarray) -> tuple[str | None, float]:
    """
    HSV range-based sticker color classifier.

    Role: given a sticker patch (BGR), predict one of the 6 cube colors using OpenCV HSV masks.
    Returns (color_name_or_none, confidence_0_to_1) where confidence is the fraction of pixels
    that match the winning mask (or a white confidence proxy).
    """
    if bgr_patch.size == 0:
        return None, 0.0

    hsv = cv2.cvtColor(bgr_patch, cv2.COLOR_BGR2HSV)
    h = hsv[:, :, 0]
    s = hsv[:, :, 1]
    v = hsv[:, :, 2]

    # White gate in HSV-space: low saturation + high value.
    white_s_max = int(os.getenv("VISION_HSV_WHITE_S_MAX", "55"))
    white_v_min = int(os.getenv("VISION_HSV_WHITE_V_MIN", "150"))
    is_white = (s <= white_s_max) & (v >= white_v_min)
    white_ratio = float(np.mean(is_white))

    # If it strongly looks white, return it early.
    white_ratio_min = float(os.getenv("VISION_HSV_WHITE_RATIO_MIN", "0.45"))
    if white_ratio >= white_ratio_min:
        return "white", min(1.0, max(0.0, white_ratio))

    # Color masks (OpenCV hue 0..179).
    # These defaults are meant as a starting point; tune if needed.
    def mask_range(h_lo: int, h_hi: int, s_min: int, v_min: int) -> np.ndarray:
        return (h >= h_lo) & (h <= h_hi) & (s >= s_min) & (v >= v_min)

    s_min = int(os.getenv("VISION_HSV_S_MIN", "55"))
    v_min = int(os.getenv("VISION_HSV_V_MIN", "40"))

    # Tunable hue bands (OpenCV hue 0..179). Orange often collapses into red under warm lighting,
    # so we keep a tighter low-red band and a slightly wider orange band by default.
    red1_hi = int(os.getenv("VISION_HSV_RED1_HI", "7"))
    orange_lo = int(os.getenv("VISION_HSV_ORANGE_LO", "8"))
    orange_hi = int(os.getenv("VISION_HSV_ORANGE_HI", "24"))
    yellow_lo = int(os.getenv("VISION_HSV_YELLOW_LO", "25"))
    yellow_hi = int(os.getenv("VISION_HSV_YELLOW_HI", "38"))
    green_lo = int(os.getenv("VISION_HSV_GREEN_LO", "39"))
    green_hi = int(os.getenv("VISION_HSV_GREEN_HI", "85"))
    blue_lo = int(os.getenv("VISION_HSV_BLUE_LO", "86"))
    blue_hi = int(os.getenv("VISION_HSV_BLUE_HI", "135"))

    # Red wraps around (0..red1_hi and 170..179)
    red1 = mask_range(0, red1_hi, s_min, v_min)
    red2 = mask_range(170, 179, s_min, v_min)
    red = red1 | red2

    orange = mask_range(orange_lo, orange_hi, s_min, v_min)
    yellow = mask_range(yellow_lo, yellow_hi, s_min, v_min)
    green = mask_range(green_lo, green_hi, s_min, v_min)
    blue = mask_range(blue_lo, blue_hi, s_min, v_min)

    scores: dict[str, float] = {
        "red": float(np.mean(red)),
        "orange": float(np.mean(orange)),
        "yellow": float(np.mean(yellow)),
        "green": float(np.mean(green)),
        "blue": float(np.mean(blue)),
    }
    color, conf = max(scores.items(), key=lambda kv: kv[1])
    min_ratio = float(os.getenv("VISION_HSV_MIN_RATIO", "0.08"))
    if conf < min_ratio:
        return None, float(conf)

    # Orange-vs-red tie-breaker: if "red" wins but the average hue of saturated pixels falls
    # into the orange band, remap to orange. This fixes common warm-light failure cases.
    if color == "red":
        sat_mask = (s >= s_min) & (v >= v_min)
        if np.any(sat_mask):
            h_vals = h[sat_mask].astype(np.float32)
            h_mean = float(np.mean(h_vals))
            if float(orange_lo) <= h_mean <= float(orange_hi):
                return "orange", float(conf)
    return color, float(conf)

def _name_to_color(name: str) -> str | None:
    """
    Map model class names to our canonical 6 color names.
    Supports names like: red, red_tile(s), blue_tiles, white_tile, etc.
    """
    n = name.strip().lower()
    # normalize common dataset naming
    for suffix in ("_tiles", "_tile", "-tiles", "-tile", " tiles", " tile"):
        if n.endswith(suffix):
            n = n[: -len(suffix)]
            break
    n = n.replace(" ", "_")
    if "white" in n:
        return "white"
    if "yellow" in n:
        return "yellow"
    if "orange" in n:
        return "orange"
    if "red" in n:
        return "red"
    if "blue" in n:
        return "blue"
    if "green" in n:
        return "green"
    return None


def _grid_from_model_detections(
    warped_bgr: np.ndarray,
    r0,
    grid_size: int,
    face_class_id: int,
    conf_threshold: float,
    *,
    fill_missing_with_lab: bool = True,
) -> list[list[StickerCell]] | None:
    """
    Model-driven color extraction on the warped face.

    Role: run a YOLO result (already computed on the warped face) through a grid assignment step:
    - assign each detection to a (row, col) based on its center
    - keep only the highest-confidence detection per cell
    - ignore detections that correspond to the `face` class

    This requires the model to include classes for tile colors (red/green/blue/white/yellow/orange).
    """
    names = getattr(getattr(r0, "names", None), "copy", lambda: getattr(r0, "names", None))()
    # ultralytics exposes names on Result (dict or list) OR on model; handle both later.
    if names is None:
        return None

    h, w = warped_bgr.shape[:2]
    if h < 8 or w < 8:
        return None
    cell_w = w / float(grid_size)
    cell_h = h / float(grid_size)

    best: dict[tuple[int, int], tuple[str, float]] = {}

    def put_cell(cx: float, cy: float, cls_id: int, conf: float) -> None:
        if conf < conf_threshold or cls_id == face_class_id:
            return
        cls_name = None
        if isinstance(names, dict):
            cls_name = names.get(int(cls_id))
        elif isinstance(names, (list, tuple)):
            if 0 <= int(cls_id) < len(names):
                cls_name = names[int(cls_id)]
        if not cls_name:
            return
        color = _name_to_color(str(cls_name))
        if color is None:
            return
        c = int(np.clip(cx / cell_w, 0, grid_size - 1))
        r = int(np.clip(cy / cell_h, 0, grid_size - 1))
        key = (r, c)
        prev = best.get(key)
        if prev is None or conf > prev[1]:
            best[key] = (color, float(conf))

    # OBB path
    obb = getattr(r0, "obb", None)
    if obb is not None and len(obb) > 0:
        cls_arr = obb.cls.cpu().numpy().astype(int).ravel()
        conf_arr = obb.conf.cpu().numpy().astype(np.float64).ravel()
        # xywhr: cx,cy,w,h,rot
        xywhr = obb.xywhr.cpu().numpy().astype(np.float64)
        for (cx, cy, *_), cls_id, conf in zip(xywhr, cls_arr, conf_arr, strict=False):
            put_cell(float(cx), float(cy), int(cls_id), float(conf))
    else:
        # Detect path
        boxes = getattr(r0, "boxes", None)
        if boxes is None or len(boxes) == 0:
            return None
        cls_arr = boxes.cls.cpu().numpy().astype(int).ravel()
        conf_arr = boxes.conf.cpu().numpy().astype(np.float64).ravel()
        xyxy = boxes.xyxy.cpu().numpy().astype(np.float64)
        for (x1, y1, x2, y2), cls_id, conf in zip(xyxy, cls_arr, conf_arr, strict=False):
            put_cell(float((x1 + x2) / 2.0), float((y1 + y2) / 2.0), int(cls_id), float(conf))

    if not best:
        return None

    # If Lab filling is disabled, we still return a full grid, but leave missing cells as `color=None`.
    # This is useful to evaluate sticker-color predictions produced by the model itself.
    if not fill_missing_with_lab:
        grid: list[list[StickerCell]] = []
        for r in range(grid_size):
            row: list[StickerCell] = []
            for c in range(grid_size):
                y1, y2 = int(r * cell_h), int((r + 1) * cell_h)
                x1, x2 = int(c * cell_w), int((c + 1) * cell_w)
                cell_roi = _center_patch(warped_bgr[y1:y2, x1:x2])
                bgr_med = _robust_bgr_cell(cell_roi)
                if (r, c) in best:
                    color, conf = best[(r, c)]
                    row.append(
                        StickerCell(
                            color=color,  # type: ignore[arg-type]
                            confidence=float(conf),
                            bgr_mean=(float(bgr_med[0]), float(bgr_med[1]), float(bgr_med[2])),
                        )
                    )
                else:
                    row.append(
                        StickerCell(
                            color=None,
                            confidence=0.0,
                            bgr_mean=(float(bgr_med[0]), float(bgr_med[1]), float(bgr_med[2])),
                        )
                    )
            grid.append(row)
        return grid

    grid: list[list[StickerCell]] = []
    for r in range(grid_size):
        row: list[StickerCell] = []
        for c in range(grid_size):
            if (r, c) in best:
                color, conf = best[(r, c)]
                # keep bgr_mean consistent with schema (sample from cell area)
                y1, y2 = int(r * cell_h), int((r + 1) * cell_h)
                x1, x2 = int(c * cell_w), int((c + 1) * cell_w)
                cell_roi = _center_patch(warped_bgr[y1:y2, x1:x2])
                bgr_med = _robust_bgr_cell(cell_roi)
                row.append(
                    StickerCell(
                        color=color,  # type: ignore[arg-type]
                        confidence=float(conf),
                        bgr_mean=(float(bgr_med[0]), float(bgr_med[1]), float(bgr_med[2])),
                    )
                )
            else:
                # fallback per-cell heuristic for missing detections
                y1, y2 = int(r * cell_h), int((r + 1) * cell_h)
                x1, x2 = int(c * cell_w), int((c + 1) * cell_w)
                cell = _center_patch(warped_bgr[y1:y2, x1:x2])
                bgr_med = _robust_bgr_cell(cell)
                med_u8 = np.clip(bgr_med, 0, 255).astype(np.uint8).reshape(1, 1, 3)
                lab = cv2.cvtColor(med_u8, cv2.COLOR_BGR2LAB).reshape(3).astype(np.float64)
                ab = lab[1:3]
                dists = np.linalg.norm(_REF_AB - ab, axis=1)

                # Apply the same white gate for heuristic fill so white doesn't dominate.
                min_v = float(os.getenv("VISION_WHITE_GATE_MIN_V", "155"))
                max_c = float(os.getenv("VISION_WHITE_GATE_MAX_C", "45"))
                allow_white = _allow_white_from_bgr(bgr_med, min_v=min_v, max_c=max_c)
                if allow_white:
                    idx, conf = _softmax_neg_distances(dists)
                    name = _COLOR_NAMES[idx]
                else:
                    idx_sub, conf = _softmax_neg_distances(dists[1:])
                    idx = 1 + idx_sub
                    name = _COLOR_NAMES[idx]
                row.append(
                    StickerCell(
                        color=name,  # type: ignore[arg-type]
                        confidence=float(conf) * 0.35,  # mark as weak
                        bgr_mean=(float(bgr_med[0]), float(bgr_med[1]), float(bgr_med[2])),
                    )
                )
        grid.append(row)
    return grid


def extract_color_grid(
    face_bgr: np.ndarray,
    grid_size: int = 3,
    inner_margin_ratio: float = 0.08,
) -> list[list[StickerCell]]:
    """
    Lab-heuristic color extraction on the warped face.

    Role: split the warped face into a `grid_size x grid_size` grid, compute a robust color
    statistic per cell, and classify it by distance to canonical references in Lab (a/b only).
    """
    # Normalize lighting/white-balance before per-cell stats.
    face_bgr = _gray_world_white_balance(face_bgr)
    h, w = face_bgr.shape[:2]
    side = min(h, w)
    y0 = (h - side) // 2
    x0 = (w - side) // 2
    square = face_bgr[y0 : y0 + side, x0 : x0 + side]
    m = int(side * inner_margin_ratio)
    inner = square[m : side - m, m : side - m]
    if inner.size == 0:
        inner = square
    ih, iw = inner.shape[:2]
    cell_h = ih // grid_size
    cell_w = iw // grid_size
    grid: list[list[StickerCell]] = []
    for r in range(grid_size):
        row: list[StickerCell] = []
        for c in range(grid_size):
            y1, y2 = r * cell_h, (r + 1) * cell_h
            x1, x2 = c * cell_w, (c + 1) * cell_w
            cell = _center_patch(inner[y1:y2, x1:x2])
            bgr_med = _robust_bgr_cell(cell)
            med_u8 = np.clip(bgr_med, 0, 255).astype(np.uint8).reshape(1, 1, 3)
            lab = cv2.cvtColor(med_u8, cv2.COLOR_BGR2LAB).reshape(3).astype(np.float64)
            ab = lab[1:3]
            dists = np.linalg.norm(_REF_AB - ab, axis=1)

            # "White gate": only consider white when the patch is bright + low-chroma.
            # Otherwise, exclude white from candidates so it can't steal weakly-saturated colors.
            min_v = float(os.getenv("VISION_WHITE_GATE_MIN_V", "155"))
            max_c = float(os.getenv("VISION_WHITE_GATE_MAX_C", "45"))
            allow_white = _allow_white_from_bgr(bgr_med, min_v=min_v, max_c=max_c)
            if allow_white:
                idx, conf = _softmax_neg_distances(dists)
                name = _COLOR_NAMES[idx]
            else:
                # Exclude white (index 0) from candidates.
                idx_sub, conf = _softmax_neg_distances(dists[1:])
                idx = 1 + idx_sub
                name = _COLOR_NAMES[idx]
            row.append(
                StickerCell(
                    color=name,  # type: ignore[arg-type]
                    confidence=conf,
                    bgr_mean=(float(bgr_med[0]), float(bgr_med[1]), float(bgr_med[2])),
                )
            )
        grid.append(row)
    return grid


def _order_quad_pts(pts: np.ndarray) -> np.ndarray:
    """Geometry helper. Role: order 4 points as TL, TR, BR, BL for stable warping."""
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1).reshape(-1)
    tl = pts[np.argmin(s)]
    br = pts[np.argmax(s)]
    tr = pts[np.argmin(diff)]
    bl = pts[np.argmax(diff)]
    return np.array([tl, tr, br, bl], dtype=np.float32)


def _warp_quad_to_square(bgr: np.ndarray, quad: np.ndarray, out_size: int) -> np.ndarray:
    """Perspective helper. Role: warp an arbitrary face quad into a square image."""
    dst = np.array(
        [[0, 0], [out_size - 1, 0], [out_size - 1, out_size - 1], [0, out_size - 1]],
        dtype=np.float32,
    )
    h = cv2.getPerspectiveTransform(quad.astype(np.float32), dst)
    return cv2.warpPerspective(bgr, h, (out_size, out_size))


def detect_face_contour(bgr: np.ndarray) -> tuple[np.ndarray, float] | None:
    """
    Find the largest roughly-square quadrilateral (cube face outline).
    Returns (4x2 float32 quad in image coords, quality score 0..1).
    Tries a few edge thresholds so webcams with soft contrast still yield a quad.
    """
    gray0 = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray0 = clahe.apply(gray0)
    gray0 = cv2.GaussianBlur(gray0, (5, 5), 0)
    h_img, w_img = bgr.shape[:2]
    img_area = float(h_img * w_img)
    best: tuple[float, np.ndarray] | None = None

    for low, high in ((30, 90), (40, 120), (20, 60)):
        edges = cv2.Canny(gray0, low, high)
        edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1)
        contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            peri = cv2.arcLength(cnt, True)
            if peri < 80:
                continue
            approx = cv2.approxPolyDP(cnt, 0.022 * peri, True)
            if len(approx) != 4 or not cv2.isContourConvex(approx):
                continue
            pts = approx.reshape(4, 2).astype(np.float32)
            area = cv2.contourArea(pts)
            if area < img_area * 0.03:
                continue
            rect = cv2.minAreaRect(pts)
            (_, _), (rw, rh) = rect
            if rw < 1 or rh < 1:
                continue
            ar = max(rw, rh) / min(rw, rh)
            if ar > 1.45:
                continue
            squareness = area / (rw * rh + 1e-6)
            score = float(min(1.0, squareness * (1.0 / ar)))
            if best is None or score > best[0]:
                best = (score, _order_quad_pts(pts))

    if best is None:
        return None
    score, quad = best
    return quad, max(0.15, min(1.0, score))


def _bbox_from_quad(quad: np.ndarray) -> _Bbox:
    xs, ys = quad[:, 0], quad[:, 1]
    return _Bbox(float(np.min(xs)), float(np.min(ys)), float(np.max(xs)), float(np.max(ys)), 0.0)


def _bbox_from_xyxy(xyxy: np.ndarray, conf: float) -> _Bbox:
    x1, y1, x2, y2 = [float(x) for x in xyxy.flatten()[:4]]
    return _Bbox(x1, y1, x2, y2, conf)


def _expand_square_bbox(bb: _Bbox, w: int, h: int, pad: float = 0.02) -> tuple[int, int, int, int]:
    """Crop helper. Role: expand and square a bbox while clamping to image bounds."""
    cx = (bb.x1 + bb.x2) / 2
    cy = (bb.y1 + bb.y2) / 2
    side = max(bb.x2 - bb.x1, bb.y2 - bb.y1) * (1 + pad)
    half = side / 2
    x1 = int(max(0, cx - half))
    y1 = int(max(0, cy - half))
    x2 = int(min(w, cx + half))
    y2 = int(min(h, cy + half))
    if x2 <= x1 or y2 <= y1:
        return int(bb.x1), int(bb.y1), int(bb.x2), int(bb.y2)
    return x1, y1, x2, y2


def _crop_square_resize(bgr: np.ndarray, bb: _Bbox, out_size: int) -> np.ndarray:
    """Crop helper. Role: crop around bbox and normalize to a square `out_size` image."""
    h, w = bgr.shape[:2]
    x1, y1, x2, y2 = _expand_square_bbox(bb, w, h)
    crop = bgr[y1:y2, x1:x2]
    if crop.size == 0:
        return cv2.resize(bgr, (out_size, out_size))
    side = min(crop.shape[0], crop.shape[1])
    ch, cw = crop.shape[:2]
    cy0 = (ch - side) // 2
    cx0 = (cw - side) // 2
    square = crop[cy0 : cy0 + side, cx0 : cx0 + side]
    return cv2.resize(square, (out_size, out_size))


def _obb_face_quad_and_conf(
    r0,
    face_class_id: int,
    conf_threshold: float,
) -> tuple[np.ndarray, float] | None:
    """
    From an OBB predict result, pick the highest-confidence `face` class detection
    and return ordered 4x2 quad (float32) + confidence.
    """
    obb = getattr(r0, "obb", None)
    if obb is None or len(obb) == 0:
        return None
    cls_t = obb.cls
    conf_t = obb.conf
    if cls_t is None or conf_t is None:
        return None
    cls_arr = cls_t.cpu().numpy().astype(int).ravel()
    conf_arr = conf_t.cpu().numpy().astype(np.float64).ravel()
    mask = (cls_arr == face_class_id) & (conf_arr >= conf_threshold)
    if not np.any(mask):
        return None
    idxs = np.where(mask)[0]
    j = int(idxs[int(np.argmax(conf_arr[mask]))])
    corners = obb.xyxyxyxy[j].cpu().numpy().reshape(4, 2).astype(np.float32)
    quad = _order_quad_pts(corners)
    return quad, float(conf_arr[j])


class _ColorExtractor:
    """
    Strategy interface for extracting a `grid_size x grid_size` sticker grid from a warped face.

    Role: keep `VisionPipeline.analyze_bgr()` structured by isolating color-readout logic from
    face localization and warping.
    """

    def extract(self, warped_bgr: np.ndarray) -> tuple[list[list[StickerCell]], str | None]:
        raise NotImplementedError


class _LabColorExtractor(_ColorExtractor):
    """Color-readout strategy using OpenCV + Lab heuristic (no model tile classes required)."""

    def __init__(self, grid_size: int = 3) -> None:
        self.grid_size = grid_size

    def extract(self, warped_bgr: np.ndarray) -> tuple[list[list[StickerCell]], str | None]:
        return extract_color_grid(warped_bgr, grid_size=self.grid_size), None


class _DLCropExtractor(_ColorExtractor):
    """
    DL mode: per-sticker crops + optional HSV; optional YOLO-cls (VISION_DL_CLS_WEIGHTS).

    Role: crops for UI. By default `color`/`confidence` come from HSV and cls fills `cls_*`.
    When `VISION_MODEL_CLASSIFY_STRICT=true`, skip HSV for sticker colors and use only the
    classifier for `color`/`confidence` (requires valid cls weights).
    """

    def __init__(
        self,
        grid_size: int = 3,
        cls_yolo=None,
        cls_imgsz: int = 224,
        *,
        classify_strict: bool = False,
    ) -> None:
        self.grid_size = grid_size
        self.cls_yolo = cls_yolo
        self.cls_imgsz = int(np.clip(int(cls_imgsz), 32, 640))
        self.classify_strict = classify_strict

    def extract(self, warped_bgr: np.ndarray) -> tuple[list[list[StickerCell]], str | None]:
        # Crop size sent to UI (square JPEG). Keep small to avoid huge payloads.
        crop_size = int(os.getenv("VISION_DL_CROP_SIZE", "96"))
        crop_size = int(np.clip(crop_size, 32, 256))
        jpeg_q = int(os.getenv("VISION_DL_JPEG_QUALITY", "80"))
        jpeg_q = int(np.clip(jpeg_q, 40, 95))

        h, w = warped_bgr.shape[:2]
        cell_w = w / float(self.grid_size)
        cell_h = h / float(self.grid_size)

        cells: list[tuple[np.ndarray, np.ndarray, str | None, float, str | None]] = []
        for r in range(self.grid_size):
            for c in range(self.grid_size):
                y1, y2 = int(r * cell_h), int((r + 1) * cell_h)
                x1, x2 = int(c * cell_w), int((c + 1) * cell_w)
                patch = _center_patch(warped_bgr[y1:y2, x1:x2])
                bgr_med = _robust_bgr_cell(patch)
                if self.classify_strict:
                    hsv_color, hsv_conf = None, 0.0
                else:
                    hsv_color, hsv_conf = _classify_hsv_color(patch)

                patch_rs = cv2.resize(patch, (crop_size, crop_size), interpolation=cv2.INTER_AREA)
                ok, buf = cv2.imencode(".jpg", patch_rs, [int(cv2.IMWRITE_JPEG_QUALITY), jpeg_q])
                crop_b64 = base64.b64encode(buf.tobytes()).decode("ascii") if ok else None

                cls_patch = cv2.resize(
                    patch, (self.cls_imgsz, self.cls_imgsz), interpolation=cv2.INTER_AREA
                )
                cells.append((bgr_med, cls_patch, hsv_color, float(hsv_conf), crop_b64))

        cls_rows: list[list[tuple[str, float, str | None]]] | None = None
        if self.cls_yolo is not None:
            cls_rows = []
            for r in range(self.grid_size):
                cls_row: list[tuple[str, float, str | None]] = []
                for c in range(self.grid_size):
                    idx = r * self.grid_size + c
                    _, cls_patch, _, _, _ = cells[idx]
                    try:
                        cr = self.cls_yolo.predict(
                            source=cls_patch,
                            imgsz=self.cls_imgsz,
                            verbose=False,
                        )[0]
                        raw, cconf = _yolo_cls_top1(cr)
                        mapped = _name_to_color(raw) if raw else None
                        cls_row.append((raw, cconf, mapped))
                    except Exception:
                        cls_row.append(("", 0.0, None))
                cls_rows.append(cls_row)

        grid: list[list[StickerCell]] = []
        idx = 0
        for r in range(self.grid_size):
            row: list[StickerCell] = []
            for c in range(self.grid_size):
                bgr_med, _, hsv_color, hsv_conf, crop_b64 = cells[idx]
                idx += 1
                cls_color = None
                cls_confidence = 0.0
                cls_label_raw: str | None = None
                if cls_rows is not None:
                    raw, cconf, mapped = cls_rows[r][c]
                    cls_label_raw = raw[:128] if raw else None
                    cls_confidence = float(cconf)
                    cls_color = mapped  # type: ignore[assignment]

                if self.classify_strict:
                    row.append(
                        StickerCell(
                            color=cls_color,  # type: ignore[arg-type]
                            confidence=float(cls_confidence),
                            bgr_mean=(float(bgr_med[0]), float(bgr_med[1]), float(bgr_med[2])),
                            crop_jpeg_b64=crop_b64,
                            cls_color=cls_color,  # type: ignore[arg-type]
                            cls_confidence=cls_confidence,
                            cls_label_raw=cls_label_raw,
                        )
                    )
                else:
                    row.append(
                        StickerCell(
                            color=hsv_color,  # type: ignore[arg-type]
                            confidence=float(hsv_conf),
                            bgr_mean=(float(bgr_med[0]), float(bgr_med[1]), float(bgr_med[2])),
                            crop_jpeg_b64=crop_b64,
                            cls_color=cls_color,  # type: ignore[arg-type]
                            cls_confidence=cls_confidence,
                            cls_label_raw=cls_label_raw,
                        )
                    )
            grid.append(row)
        if self.classify_strict:
            if self.cls_yolo is None:
                msg = (
                    "DL mode: VISION_MODEL_CLASSIFY_STRICT=true but no classifier loaded "
                    "(set VISION_DL_CLS_WEIGHTS to a valid .pt). HSV skipped."
                )
            else:
                msg = (
                    "DL mode: classifier-only (VISION_MODEL_CLASSIFY_STRICT=true; "
                    "HSV skipped for sticker colors)."
                )
        else:
            msg = "DL mode: per-sticker crops + HSV range color guess (debug)."
            if self.cls_yolo is not None:
                msg += " YOLO-cls test head enabled (VISION_DL_CLS_WEIGHTS)."
        return grid, msg


class _ModelColorExtractor(_ColorExtractor):
    """
    Color-readout strategy using a YOLO model on the warped face.

    Role: if the YOLO weights include tile color classes, we use detections to populate the grid.
    If no tile detections are found (or something fails), we fall back to the Lab heuristic.
    """

    def __init__(
        self,
        yolo,
        yolo_conf: float,
        face_class_id: int,
        grid_size: int = 3,
        *,
        allow_lab_fallback: bool = True,
    ) -> None:
        self.yolo = yolo
        self.yolo_conf = yolo_conf
        self.face_class_id = face_class_id
        self.grid_size = grid_size
        self.allow_lab_fallback = allow_lab_fallback

    def extract(self, warped_bgr: np.ndarray) -> tuple[list[list[StickerCell]], str | None]:
        try:
            face_results = self.yolo.predict(
                source=warped_bgr,
                imgsz=640,
                conf=max(0.05, self.yolo_conf * 0.6),
                verbose=False,
            )
            fr0 = face_results[0]
            model_grid = _grid_from_model_detections(
                warped_bgr,
                fr0,
                grid_size=self.grid_size,
                face_class_id=int(self.face_class_id),
                conf_threshold=max(0.05, self.yolo_conf * 0.6),
                fill_missing_with_lab=self.allow_lab_fallback,
            )
            if model_grid is not None:
                if not self.allow_lab_fallback:
                    return model_grid, "Strict model mode: showing only model-detected tiles (missing = null)."
                return model_grid, None
            if not self.allow_lab_fallback:
                return [], "Strict model mode: no tile detections parsed from the model output."
            return (
                extract_color_grid(warped_bgr, grid_size=self.grid_size),
                "Used Lab heuristic (no tile classes detected).",
            )
        except Exception:
            if not self.allow_lab_fallback:
                return [], "Strict model mode: model tile detection crashed (no Lab fallback used)."
            return (
                extract_color_grid(warped_bgr, grid_size=self.grid_size),
                "Used Lab heuristic (model grid failed).",
            )


class VisionPipeline:
    """
    End-to-end pipeline for a single camera frame.

    Structure:
    - **Face localization + warping**: YOLO (detect/OBB) or contour fallback -> `warped` square face image.
    - **Color extraction**: delegated to a `_ColorExtractor` strategy chosen by `VISION_COLOR_MODE`.
    """

    def __init__(
        self,
        yolo_weights: str | None = None,
        yolo_conf: float = 0.28,
        contour_fallback: bool = True,
        warp_size: int = 360,
        yolo_task: Literal["detect", "obb"] = "detect",
        obb_face_class_id: int = 0,
    ) -> None:
        self.yolo_conf = yolo_conf
        self.contour_fallback = contour_fallback
        self.warp_size = warp_size
        self.yolo_task: Literal["detect", "obb"] = yolo_task
        self.obb_face_class_id = obb_face_class_id
        self._yolo: YOLO | None = None
        self._yolo_cls: YOLO | None = None
        mode_raw = os.getenv("VISION_COLOR_MODE", "model").strip().lower()
        if mode_raw == "lab":
            self.color_mode: Literal["model", "lab", "dl"] = "lab"
        elif mode_raw in ("dl", "deep", "deep_learning"):
            self.color_mode = "dl"
        else:
            self.color_mode = "model"
        weights = (yolo_weights or os.getenv("YOLO_FACE_WEIGHTS") or "").strip()
        path = Path(weights).expanduser() if weights else None
        if path is not None and path.is_file():
            from ultralytics import YOLO as YOLO_Cls

            kwargs: dict = {}
            if yolo_task == "obb":
                kwargs["task"] = "obb"
            self._yolo = YOLO_Cls(str(path), **kwargs)
            logger.info("Loaded YOLO (%s) weights from %s", yolo_task, path)
        else:
            logger.info(
                "YOLO_FACE_WEIGHTS not set or not found (%s); contour fallback %s",
                str(path) if path is not None else "(unset)",
                "enabled" if contour_fallback else "disabled",
            )

        # Choose the color-readout strategy once (keeps `analyze_bgr` simpler).
        if self.color_mode == "dl":
            cls_w = (os.getenv("VISION_DL_CLS_WEIGHTS") or "").strip()
            cls_path = Path(cls_w).expanduser() if cls_w else None
            if cls_path is not None and cls_path.is_file():
                from ultralytics import YOLO as YOLO_Cls2

                self._yolo_cls = YOLO_Cls2(str(cls_path), task="classify")
                logger.info("Loaded YOLO-cls (test) weights from %s", cls_path)
            elif cls_w:
                logger.warning("VISION_DL_CLS_WEIGHTS set but file not found: %s", cls_w)
            cls_imgsz = int(os.getenv("VISION_DL_CLS_IMGSZ", "224"))
            classify_strict = os.getenv("VISION_MODEL_CLASSIFY_STRICT", "false").strip().lower() in (
                "1",
                "true",
                "yes",
            )
            self._color_extractor = _DLCropExtractor(
                grid_size=3,
                cls_yolo=self._yolo_cls,
                cls_imgsz=cls_imgsz,
                classify_strict=classify_strict,
            )
        elif self.color_mode == "model" and self._yolo is not None:
            # Testing switch: allow you to evaluate sticker-color performance of the model itself.
            # When enabled, we do NOT fill missing cells with the Lab heuristic and we do NOT
            # fall back to Lab if the model yields no usable tile grid.
            strict_model = os.getenv("VISION_MODEL_STRICT", "false").strip().lower() in (
                "1",
                "true",
                "yes",
            )
            self._color_extractor: _ColorExtractor = _ModelColorExtractor(
                yolo=self._yolo,
                yolo_conf=self.yolo_conf,
                face_class_id=int(self.obb_face_class_id),
                grid_size=3,
                allow_lab_fallback=not strict_model,
            )
        else:
            # Either explicitly requested `lab`, or model weights are unavailable.
            self._color_extractor = _LabColorExtractor(grid_size=3)

    def analyze_bgr(self, bgr: np.ndarray) -> FaceGridResult:
        """
        Analyze a single BGR frame.

        Role: detect/warp the cube face and return a structured 3×3 sticker grid with confidence.
        """
        t0 = time.perf_counter()
        h, w = bgr.shape[:2]
        method: Literal["yolo", "yolo_obb", "contour_fallback"] = "contour_fallback"
        face_conf = 0.0
        bbox: tuple[float, float, float, float] | None = None
        warped: np.ndarray | None = None
        msg: str | None = None

        if self._yolo is not None:
            results = self._yolo.predict(
                source=bgr,
                imgsz=640,
                conf=self.yolo_conf,
                verbose=False,
            )
            r0 = results[0]
            if self.yolo_task == "obb":
                quad_conf = _obb_face_quad_and_conf(
                    r0, self.obb_face_class_id, self.yolo_conf
                )
                if quad_conf is not None:
                    quad, face_conf = quad_conf
                    bb = _bbox_from_quad(quad)
                    bbox = (bb.x1, bb.y1, bb.x2, bb.y2)
                    method = "yolo_obb"
                    warped = _warp_quad_to_square(bgr, quad, self.warp_size)
            else:
                boxes = getattr(r0, "boxes", None)
                if boxes is not None and len(boxes) > 0:
                    # Pick the best detection for the face class (not “any class”).
                    cls_arr = boxes.cls.cpu().numpy().astype(int).ravel()
                    conf_arr = boxes.conf.cpu().numpy().astype(np.float64).ravel()
                    mask = cls_arr == int(self.obb_face_class_id)
                    if np.any(mask):
                        idxs = np.where(mask)[0]
                        best_i = int(idxs[int(np.argmax(conf_arr[mask]))])
                        xyxy = boxes.xyxy[best_i].cpu().numpy()
                        face_conf = float(conf_arr[best_i])
                        bb = _bbox_from_xyxy(xyxy, face_conf)
                        bbox = (bb.x1, bb.y1, bb.x2, bb.y2)
                        method = "yolo"
                        warped = _crop_square_resize(bgr, bb, self.warp_size)

        if warped is None and self.contour_fallback:
            quad_conf = detect_face_contour(bgr)
            if quad_conf is not None:
                quad, qscore = quad_conf
                method = "contour_fallback"
                face_conf = qscore
                bb = _bbox_from_quad(quad)
                bbox = (bb.x1, bb.y1, bb.x2, bb.y2)
                warped = _warp_quad_to_square(bgr, quad, self.warp_size)
                if self._yolo is not None:
                    msg = (
                        "YOLO (OBB or detect) found no usable face above threshold; "
                        "used contour fallback."
                    )

        if warped is None:
            elapsed_ms = (time.perf_counter() - t0) * 1000
            return FaceGridResult(
                face_detected=False,
                face_confidence=0.0,
                detection_method=method,
                bbox_xyxy=None,
                grid=[],
                inference_ms=round(elapsed_ms, 2),
                message=msg
                or (
                    "No face detected. Set YOLO_FACE_WEIGHTS and YOLO_TASK (detect|obb), "
                    "or improve lighting/contrast for contour fallback."
                ),
            )

        grid, color_msg = self._color_extractor.extract(warped)
        if color_msg:
            msg = (msg + " " if msg else "") + color_msg
        elapsed_ms = (time.perf_counter() - t0) * 1000
        dl_sticker_color_source: Literal["hsv", "cls"] | None = None
        if self.color_mode == "dl":
            ex = self._color_extractor
            if isinstance(ex, _DLCropExtractor):
                dl_sticker_color_source = "cls" if ex.classify_strict else "hsv"
        return FaceGridResult(
            face_detected=True,
            face_confidence=round(face_conf, 4),
            detection_method=method,
            bbox_xyxy=bbox,
            grid=grid,
            inference_ms=round(elapsed_ms, 2),
            message=msg,
            dl_sticker_color_source=dl_sticker_color_source,
        )


_pipeline: VisionPipeline | None = None


def get_pipeline() -> VisionPipeline:
    global _pipeline
    if _pipeline is None:
        weights_env = os.getenv("YOLO_FACE_WEIGHTS")
        weights_env = weights_env.strip() if weights_env else None
        auto_w, auto_task = _auto_weights()
        weights = weights_env or auto_w

        conf = float(os.getenv("YOLO_CONF_THRESHOLD", "0.28"))
        fallback = os.getenv("VISION_CONTOUR_FALLBACK", "true").lower() in (
            "1",
            "true",
            "yes",
        )
        warp = int(os.getenv("VISION_WARP_SIZE", "360"))
        task_raw = os.getenv("YOLO_TASK")
        task_raw = task_raw.strip().lower() if task_raw else ""
        if task_raw in ("detect", "obb"):
            yolo_task = "obb" if task_raw == "obb" else "detect"
        else:
            yolo_task = auto_task if weights_env is None else "detect"
        face_cls = int(os.getenv("YOLO_OBB_FACE_CLASS_ID", "0"))
        _pipeline = VisionPipeline(
            yolo_weights=weights,
            yolo_conf=conf,
            contour_fallback=fallback,
            warp_size=warp,
            yolo_task=yolo_task,
            obb_face_class_id=face_cls,
        )
    return _pipeline


def decode_image_bytes(data: bytes) -> np.ndarray:
    arr = np.frombuffer(data, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Could not decode image bytes as BGR")
    return img
