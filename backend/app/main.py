"""FastAPI entrypoint for the cube solver backend."""

from pathlib import Path

from dotenv import load_dotenv

# Load `.env` so YOLO_FACE_WEIGHTS, YOLO_TASK, etc. apply when running uvicorn from `backend/`.
_backend_dir = Path(__file__).resolve().parent.parent
_repo_root = _backend_dir.parent
for _env_path in (Path.cwd() / ".env", _backend_dir / ".env", _repo_root / ".env"):
    if _env_path.is_file():
        load_dotenv(_env_path, override=True)

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes import session, vision

app = FastAPI(title="Cube Solver API", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(session.router, prefix="/session", tags=["session"])
app.include_router(vision.router, prefix="/vision", tags=["vision"])


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}
