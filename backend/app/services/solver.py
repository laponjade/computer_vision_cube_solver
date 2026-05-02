"""Kociemba solver integration."""

from __future__ import annotations

import kociemba


class SolveError(RuntimeError):
    """Raised when the cube state is invalid or unsolvable."""


def solve_kociemba(state: str) -> str:
    """
    Solve a cube from a 54-character facelet state string (URFDLB letters).
    Returns a move sequence like: "R U R' U' ..."
    """
    if not isinstance(state, str) or len(state) != 54:
        raise SolveError("State must be a 54-character string")
    try:
        sol = kociemba.solve(state)
    except Exception as e:
        # kociemba throws generic Exceptions; try to return a more helpful message.
        verify_fn = getattr(kociemba, "verify", None)
        if callable(verify_fn):
            try:
                v = verify_fn(state)
                if isinstance(v, str) and v and v.strip().upper() != "OK":
                    raise SolveError(f"{v.strip()} (state={state})") from e
            except SolveError:
                raise
            except Exception:
                pass
        raise SolveError(f"{str(e) or 'kociemba solve failed'} (state={state})") from e
    if not isinstance(sol, str) or not sol.strip():
        raise SolveError("Empty solution")
    return sol.strip()
