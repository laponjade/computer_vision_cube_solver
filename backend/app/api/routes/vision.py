"""Single-face live frame analysis."""

import logging

from fastapi import APIRouter, File, HTTPException, UploadFile

from app.schemas.vision import FaceGridResult
from app.services.vision import decode_image_bytes, get_pipeline

logger = logging.getLogger(__name__)

router = APIRouter()


@router.post("/analyze-frame", response_model=FaceGridResult)
async def analyze_frame(file: UploadFile = File(...)) -> FaceGridResult:
    """
    Accept a JPEG/PNG frame from the client camera, return face bbox (if any),
    3x3 sticker colors with per-cell confidence, and timing.
    """
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Expected an image/* upload")
    data = await file.read()
    if not data:
        raise HTTPException(status_code=400, detail="Empty body")
    try:
        bgr = decode_image_bytes(data)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e

    try:
        return get_pipeline().analyze_bgr(bgr)
    except Exception:
        logger.exception("vision analyze_frame failed")
        raise HTTPException(status_code=500, detail="Vision processing failed") from None
