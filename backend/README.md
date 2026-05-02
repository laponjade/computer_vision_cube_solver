# Backend

FastAPI service for cube vision, state building, Kociemba solving, and verification.

## Run (development)

```bash
cd backend
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

API docs: http://localhost:8000/docs

## Vision (single face)

- `POST /vision/analyze-frame` — multipart field `file` (JPEG/PNG). Returns JSON: face bbox, `detection_method` (`yolo`, `yolo_obb`, or `contour_fallback`), 3×3 `grid` with per-sticker `color` + `confidence`, and `inference_ms`.
- **Detect model:** set `YOLO_FACE_WEIGHTS` to a `.pt` with axis-aligned boxes (e.g. single-class `cube_face`). Keep `YOLO_TASK=detect` (default).
- **OBB model (e.g. trained `yolo26s-obb`):** set `YOLO_FACE_WEIGHTS` to `best.pt`, set `YOLO_TASK=obb`, and `YOLO_OBB_FACE_CLASS_ID` to the class index of `face` in your data yaml (usually `0`). The API uses the OBB quad to warp the face, then runs the same 3×3 color readout.
- If weights are missing or no face is found, `VISION_CONTOUR_FALLBACK=true` (default) uses OpenCV contour fallback.
- **`VISION_COLOR_MODE=dl`:** returns per-sticker `crop_jpeg_b64` plus HSV-based `color` / `confidence`. Optional test-only YOLO-cls: set `VISION_DL_CLS_WEIGHTS` to a trained `best.pt` / `last.pt` (classification); responses include `cls_color`, `cls_confidence`, `cls_label_raw`. Optional `VISION_DL_CLS_IMGSZ` (default `224`). Set **`VISION_MODEL_CLASSIFY_STRICT=true`** to skip HSV for sticker colors and use only the classifier for `color` / `confidence` (requires valid `VISION_DL_CLS_WEIGHTS`). Response field **`dl_sticker_color_source`** is `hsv` or `cls`.
