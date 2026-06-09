from __future__ import annotations

import random

from .board import Board
from .cells import create_cell
from .constants import ROWS, COLS, INITIAL_FROZEN_RATIO
from .match import find_matches


def apply_gravity_and_refill(board: Board, rng: random.Random) -> None:
    for c in range(COLS):
        stack = []
        for r in range(ROWS - 1, -1, -1):
            if board[r][c]:
                stack.append(board[r][c])
        for r in range(ROWS - 1, -1, -1):
            idx = ROWS - 1 - r
            board[r][c] = stack[idx] if idx < len(stack) else create_cell(rng)


def reshuffle_board(board: Board, rng: random.Random, max_attempts: int = 200) -> bool:
    for _ in range(max_attempts):
        for r in range(ROWS):
            for c in range(COLS):
                board[r][c] = create_cell(rng)
        if not find_matches(board):
            return True
    return False


def freeze_random_cells(board: Board, rng: random.Random, ratio: float = INITIAL_FROZEN_RATIO) -> None:
    positions = [{"r": r, "c": c} for r in range(ROWS) for c in range(COLS)]
    total = int(ROWS * COLS * ratio)
    rng.shuffle(positions)
    for i in range(total):
        p = positions[i]
        cell = board[p["r"]][p["c"]]
        if cell:
            cell.frozen = True
