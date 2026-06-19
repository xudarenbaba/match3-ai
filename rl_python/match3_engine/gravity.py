from __future__ import annotations

import random
from typing import Optional

from .board import Board
from .cells import create_cell, create_initial_cell
from .constants import ROWS, COLS, INITIAL_FROZEN_RATIO
from .match import find_matches


def apply_gravity_and_refill(
    board: Board, rng: random.Random, layout: Optional[list] = None
) -> None:
    """应用重力并补充新格子。layout 为 None 时视为全 1（向后兼容）。"""
    for c in range(COLS):
        # 收集该列活跃格中非空的格子（从底部到顶部）
        stack = []
        for r in range(ROWS - 1, -1, -1):
            if layout and not layout[r][c]:
                continue  # void 格跳过
            if board[r][c]:
                stack.append(board[r][c])
        # 重新填回活跃格（底部已有格子，顶部补新格）
        stack_idx = 0
        for r in range(ROWS - 1, -1, -1):
            if layout and not layout[r][c]:
                board[r][c] = None  # void 格保持 None
                continue
            board[r][c] = stack[stack_idx] if stack_idx < len(stack) else create_cell(rng)
            stack_idx += 1


def reshuffle_board(
    board: Board, rng: random.Random, layout: Optional[list] = None, max_attempts: int = 200
) -> bool:
    """重新洗牌棋盘，只处理活跃格。"""
    for _ in range(max_attempts):
        for r in range(ROWS):
            for c in range(COLS):
                if layout and not layout[r][c]:
                    board[r][c] = None  # void 格永远为 None
                else:
                    board[r][c] = create_initial_cell(rng)  # 初始只生成 L1，无道具（对齐 JS）
        if not find_matches(board):
            return True
    return False


def freeze_random_cells(
    board: Board, rng: random.Random, ratio: float = INITIAL_FROZEN_RATIO, layout: Optional[list] = None
) -> None:
    """随机冻结活跃格中的一部分格子。"""
    positions = []
    for r in range(ROWS):
        for c in range(COLS):
            if layout and not layout[r][c]:
                continue  # void 格不冻结
            positions.append({"r": r, "c": c})
    total = int(len(positions) * ratio)
    rng.shuffle(positions)
    for i in range(total):
        p = positions[i]
        cell = board[p["r"]][p["c"]]
        if cell:
            cell.frozen = True
