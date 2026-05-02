"""API schemas for single-face vision."""

from typing import Literal

from pydantic import BaseModel, Field

ColorName = Literal["white", "yellow", "red", "orange", "blue", "green"]


class StickerCell(BaseModel):
    # Testing/debugging: `color` may be null when evaluating model-only detection
    # (e.g. strict mode without Lab fallback).
    color: ColorName | None = None
    confidence: float = Field(
        ge=0.0,
        le=1.0,
        description="Confidence for the predicted color; 0 when color is null",
    )
    bgr_mean: tuple[float, float, float]
    crop_jpeg_b64: str | None = Field(
        default=None,
        description="Optional base64-encoded JPEG crop of the sticker cell (for DL mode / debugging)",
    )
    # Optional YOLO-cls test head (VISION_DL_CLS_WEIGHTS); does not replace HSV fields above.
    cls_color: ColorName | None = Field(
        default=None,
        description="YOLO-cls predicted color when test weights are loaded",
    )
    cls_confidence: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="YOLO-cls top-1 confidence when test weights are loaded",
    )
    cls_label_raw: str | None = Field(
        default=None,
        description="YOLO-cls top-1 class name before mapping to canonical colors",
    )


class FaceGridResult(BaseModel):
    face_detected: bool
    face_confidence: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="YOLO box confidence or contour quality score",
    )
    detection_method: Literal["yolo", "yolo_obb", "contour_fallback"]
    bbox_xyxy: tuple[float, float, float, float] | None = None
    grid: list[list[StickerCell]] = Field(
        default_factory=list,
        description="3 rows x 3 columns, row-major from top-left of the face",
    )
    inference_ms: float = 0.0
    message: str | None = None
    # DL mode only: how sticker `color`/`confidence` were produced (HSV vs classifier-only).
    dl_sticker_color_source: Literal["hsv", "cls"] | None = None
