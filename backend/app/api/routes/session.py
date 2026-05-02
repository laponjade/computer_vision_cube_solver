"""Session routes: multi-face capture + solve."""

from __future__ import annotations

import logging
from typing import Literal

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from app.services.cube_state import CubeStateError, build_facelet_state_from_faces
from app.services.solver import SolveError, solve_kociemba

logger = logging.getLogger(__name__)

router = APIRouter()

FaceName = Literal["U", "R", "F", "D", "L", "B"]


class SolveRequest(BaseModel):
    faces: dict[FaceName, object] = Field(
        description="Captured faces keyed by U/R/F/D/L/B. Each value is a FaceGridResult-like object."
    )


class SolveResponse(BaseModel):
    state: str = Field(description="54-character kociemba facelet string over URFDLB")
    solution: str = Field(description="Kociemba solution move string")
    color_to_face: dict[str, str] = Field(
        description="Mapping from observed center colors to face letters"
    )


@router.post("/solve", response_model=SolveResponse)
def solve(req: SolveRequest) -> SolveResponse:
    try:
        state, mapping = build_facelet_state_from_faces(faces=req.faces)  # type: ignore[arg-type]
        sol = solve_kociemba(state)
        return SolveResponse(state=state, solution=sol, color_to_face=mapping)
    except CubeStateError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    except SolveError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    except Exception:
        logger.exception("session solve failed")
        raise HTTPException(status_code=500, detail="Solve failed") from None
