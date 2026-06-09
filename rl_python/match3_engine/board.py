from __future__ import annotations

from typing import List, Optional, Tuple

from .cells import Cell, clone_cell
from .constants import ROWS, COLS


Board = List[List[Optional[Cell]]]


def cell_key(r: int, c: int) -> str:
    return f"{r},{c}"


def create_empty_board() -> Board:
    return [[None for _ in range(COLS)] for _ in range(ROWS)]


def clone_board(board: Board) -> Board:
    return [[clone_cell(cell) for cell in row] for row in board]


def in_bounds(r: int, c: int) -> bool:
    return 0 <= r < ROWS and 0 <= c < COLS


def cells_equal(a: Optional[Cell], b: Optional[Cell]) -> bool:
    if a is None and b is None:
        return True
    if a is None or b is None:
        return False
    if a.kind != b.kind or a.shape != b.shape or a.level != b.level:
        return False
    if bool(a.frozen) != bool(b.frozen):
        return False
    a_pt = getattr(a, "powerup_type", "")
    b_pt = getattr(b, "powerup_type", "")
    return a_pt == b_pt
