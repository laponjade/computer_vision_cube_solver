"""Build and validate 54-sticker cube state from 6 captured faces.

We assume the client captures faces in fixed order and labels them as:
U, R, F, D, L, B (standard kociemba face order).

Each face is a 3x3 grid of canonical colors:
white, yellow, red, orange, blue, green.

We convert those colors into a 54-character facelet string using the center
stickers as the mapping from color -> face letter (U/R/F/D/L/B). This is the
standard way to build the kociemba state string from observed colors.
"""

from __future__ import annotations

from collections import Counter
from typing import Literal

ColorName = Literal["white", "yellow", "red", "orange", "blue", "green"]
FaceName = Literal["U", "R", "F", "D", "L", "B"]


class CubeStateError(ValueError):
    """Raised when the 6-face capture cannot be converted to a valid cube state."""


def _require_grid(grid: list[list[dict]] | list[list[object]], face: FaceName) -> None:
    if not isinstance(grid, list) or len(grid) != 3:
        raise CubeStateError(f"{face}: expected 3 rows")
    for r in grid:
        if not isinstance(r, list) or len(r) != 3:
            raise CubeStateError(f"{face}: expected 3 columns per row")


def _cell_color(cell: object) -> str | None:
    # Works with pydantic models (attr) and plain dicts.
    if isinstance(cell, dict):
        return cell.get("color")
    return getattr(cell, "color", None)


def _grid_colors(face: FaceName, grid) -> list[ColorName]:
    _require_grid(grid, face)
    out: list[ColorName] = []
    for r in range(3):
        for c in range(3):
            col = _cell_color(grid[r][c])
            if col is None:
                raise CubeStateError(f"{face}: sticker ({r},{c}) is null/unknown")
            out.append(col)  # type: ignore[arg-type]
    return out


def _center_color(face: FaceName, grid) -> ColorName:
    _require_grid(grid, face)
    col = _cell_color(grid[1][1])
    if col is None:
        raise CubeStateError(f"{face}: center sticker is null/unknown")
    return col  # type: ignore[return-value]


def build_facelet_state_from_faces(
    *,
    faces: dict[FaceName, object],
) -> tuple[str, dict[str, str]]:
    """
    Convert 6 faces to a kociemba facelet string.

    Returns:
      - state: 54-char string over letters URFDLB
      - mapping: dict mapping color -> face letter (derived from centers)
    """
    missing = [f for f in ("U", "R", "F", "D", "L", "B") if f not in faces]
    if missing:
        raise CubeStateError(f"Missing faces: {', '.join(missing)}")

    # Extract centers to build a color -> face-letter mapping.
    centers: dict[FaceName, ColorName] = {}
    for f in ("U", "R", "F", "D", "L", "B"):
        face_obj = faces[f]  # could be pydantic model/dict
        grid = face_obj["grid"] if isinstance(face_obj, dict) else getattr(face_obj, "grid", None)
        if grid is None:
            raise CubeStateError(f"{f}: missing grid")
        centers[f] = _center_color(f, grid)

    # Ensure 6 unique center colors.
    inv = Counter(centers.values())
    dup = [c for c, n in inv.items() if n > 1]
    if dup:
        raise CubeStateError(
            "Center colors must be unique (each face has a different center). "
            f"Duplicates: {', '.join(dup)}"
        )

    color_to_face_letter: dict[str, str] = {centers[f]: f for f in centers}

    # Build state in URFDLB order, each face row-major.
    state_letters: list[str] = []
    all_colors: list[str] = []
    for f in ("U", "R", "F", "D", "L", "B"):
        face_obj = faces[f]
        grid = face_obj["grid"] if isinstance(face_obj, dict) else getattr(face_obj, "grid", None)
        cols = _grid_colors(f, grid)
        all_colors.extend(cols)
        for col in cols:
            state_letters.append(color_to_face_letter[str(col)])

    # Basic sanity: 54 stickers and 9 of each face letter.
    if len(state_letters) != 54:
        raise CubeStateError(f"Expected 54 stickers, got {len(state_letters)}")
    letters_count = Counter(state_letters)
    for f in ("U", "R", "F", "D", "L", "B"):
        if letters_count.get(f, 0) != 9:
            raise CubeStateError(
                f"Invalid distribution for {f}: expected 9 stickers, got {letters_count.get(f, 0)}"
            )

    # Also validate that each color appears 9 times (common early failure signal).
    colors_count = Counter(all_colors)
    for col in ("white", "yellow", "red", "orange", "blue", "green"):
        if colors_count.get(col, 0) != 9:
            raise CubeStateError(
                f"Invalid color count for {col}: expected 9, got {colors_count.get(col, 0)}"
            )

    return "".join(state_letters), color_to_face_letter
