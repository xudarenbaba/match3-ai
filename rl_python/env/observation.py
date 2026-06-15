from __future__ import annotations

import numpy as np

from match3_engine.constants import ROWS, COLS, SHAPES, POWERUP_TYPES
from match3_engine.game import GameState

BOARD_CHANNELS = 28
GLOBAL_DIM = 15


def build_observation(state: GameState) -> dict:
    board = np.zeros((BOARD_CHANNELS, ROWS, COLS), dtype=np.float32)
    target_set = set(state.target_shapes)

    for r in range(ROWS):
        for c in range(COLS):
            cell = state.board[r][c]
            if cell is None:
                continue
            if cell.kind == "normal":
                si = SHAPES.index(cell.shape)
                if 1 <= cell.level <= 3:
                    board[si * 3 + (cell.level - 1), r, c] = 1.0
            elif cell.kind == "powerup":
                pi = POWERUP_TYPES.index(cell.powerup_type)
                board[12 + pi, r, c] = 1.0
            if cell.frozen:
                board[15, r, c] = 1.0
            if cell.shape in target_set:
                si = SHAPES.index(cell.shape)
                if cell.kind == "normal" and 1 <= cell.level <= 3:
                    board[16 + si * 3 + (cell.level - 1), r, c] = 1.0
                elif cell.kind == "powerup":
                    # 道具格视为最高进度，标记到 L3 通道
                    board[16 + si * 3 + 2, r, c] = 1.0

    steps_left = max(0, state.total_steps - state.steps_used)
    global_vec = np.zeros(GLOBAL_DIM, dtype=np.float32)
    global_vec[0] = state.steps_used / state.total_steps
    global_vec[1] = steps_left / state.total_steps
    global_vec[2] = min(1.0, state.score / 5000.0)
    global_vec[3] = min(1.0, state.chain_score_total / 3000.0)
    for i, shape in enumerate(SHAPES):
        global_vec[4 + i] = state.task_scores.get(shape, 0) / 4.0
        global_vec[8 + i] = 1.0 if shape in target_set else 0.0
    target_progress = [state.task_scores.get(s, 0) / 4.0 for s in state.target_shapes]
    global_vec[12] = min(target_progress) if target_progress else 0.0
    global_vec[13] = 1.0 if state.won else 0.0
    global_vec[14] = (state.last_action + 1) / 180.0

    return {"board": board, "global": global_vec}
