# Computer Vision Cube Solver

Monorepo for a 3x3 Rubik's Cube solver: camera capture, YOLO-based vision, Kociemba solving, verification, and a Next.js UI.

## Layout

| Path | Purpose |
|------|---------|
| `backend/` | FastAPI service: vision, cube state, Kociemba, verification |
| `frontend/` | Next.js UI: guided capture, solution, verification |
| `infra/docker/` | Dockerfiles for backend and frontend |
| `datasets/` | Training / evaluation images (not committed by default) |
| `models/` | YOLO weights and exports (not committed by default) |
| `docker-compose.yml` | Local orchestration |
| `scripts/download_hf_rubik_retinanet.py` | Snapshot [HF Rubik RetinaNet demo repo](https://huggingface.co/itsyuimorii/model-rubik-cube-recognition-with-tensorflow.js-retinanet) into `third_party/` (no pretrained weights shipped there; optional Gradio demo) |
| `scripts/labelme_to_yolo_obb.py` | Build Ultralytics **YOLO-OBB** labels under `datasets/rubik_obb/` from `images/{train,valid,test}` LabelMe JSON |
| `scripts/train_yolo26s_obb.py` | Fine-tune **`yolo26s-obb.pt`** (place under `models/` or let Ultralytics download) on `datasets/rubik_obb/cube_obb.yaml` |

### Train YOLO26s-OBB on your cube images

1. Put LabelMe pairs `Cube_xxxxx.jpg` + `Cube_xxxxx.json` in `images/train`, `images/valid`, and optionally `images/test` (same layout as now).
2. Ensure base weights exist: `models/yolo26s-obb.pt` (run once: `.venv/bin/python -c "from ultralytics import YOLO; YOLO('yolo26s-obb.pt')"` then move the file into `models/` if needed).
3. Build dataset + YAML: `.venv/bin/python scripts/labelme_to_yolo_obb.py`  
   - Single-class **face** only: add `--face-only`.
4. Train: `.venv/bin/python scripts/train_yolo26s_obb.py --epochs 100 --batch 4 --imgsz 640 --device mps` on **Apple Silicon** (M3), or `--device 0` on **NVIDIA CUDA**, or `--device cpu` if needed.
5. Weights appear under `runs/obb/<run_name>/weights/best.pt` (ignored by git).

**Live camera UI with your OBB `best.pt`:** set backend env `YOLO_FACE_WEIGHTS` to that file, `YOLO_TASK=obb`, restart uvicorn, then open the frontend `/capture` page (same as before).

## Quick start (local, after dependencies are wired)

1. Copy `.env.example` to `.env` and adjust.
2. Backend: create a venv, install `backend/requirements.txt`, run uvicorn (see `backend/README.md` when added).
3. Frontend: `npm install` in `frontend/`, then `npm run dev`.
4. Or: `docker compose up` once Dockerfiles are finalized.

## License

Add your license here.
