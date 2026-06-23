from __future__ import annotations

import numpy as np

from match3_engine.constants import ROWS, COLS, SHAPES, POWERUP_TYPES
from match3_engine.game import GameState

# 单帧 board 通道数（28 形状/道具/冻结 + 1 layout_mask + 1 row道具通道 = 30）
# 新增 powerup_row 通道（原 column/bomb/color 占 12-14，现 row/column/bomb/color 占 12-15）
BOARD_CHANNELS = 30
# 帧堆叠数（当前帧 + 前 N-1 帧）
FRAME_STACK = 3
# 堆叠后送入网络的 board 通道数
STACKED_BOARD_CHANNELS = BOARD_CHANNELS * FRAME_STACK  # 90
# global: 原 15 维 + 解冻任务进度(15) + 是否有解冻任务(16) = 17
GLOBAL_DIM = 17


def build_observation(state: GameState) -> dict:
    """构建单帧观测，board shape=(30,10,10)，global shape=(15,)。
    通道说明（30 个）：
      0-11  : shape×level 一热编码（4 shape × 3 level）
      12-15 : 道具类型（row/column/bomb/color）
      16    : 冻结标志
      17-28 : 目标 shape×level 一热（与通道 0-11 相同结构，仅标目标 shape）
      29    : layout_mask（1=活跃格，0=void 格）
    """
    board = np.zeros((BOARD_CHANNELS, ROWS, COLS), dtype=np.float32)
    target_set = set(state.target_shapes)
    layout = state.layout  # 可能为 None（全 1 满格）

    for r in range(ROWS):
        for c in range(COLS):
            # 通道 29：layout_mask
            if layout is None or layout[r][c]:
                board[29, r, c] = 1.0
            else:
                # void 格：其他通道均为 0，layout_mask 也为 0
                continue

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
                board[16, r, c] = 1.0
            if cell.shape in target_set:
                si = SHAPES.index(cell.shape)
                if cell.kind == "normal" and 1 <= cell.level <= 3:
                    board[17 + si * 3 + (cell.level - 1), r, c] = 1.0
                elif cell.kind == "powerup":
                    board[17 + si * 3 + 2, r, c] = 1.0

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
    # 无形状任务时视为已满足（1.0），避免空 min 误导网络
    global_vec[12] = min(target_progress) if target_progress else 1.0
    global_vec[13] = 1.0 if state.won else 0.0
    global_vec[14] = (state.last_action + 1) / 280.0  # 归一化用 MAX_ACTIONS=280
    # ── 解冻任务 ──────────────────────────────────────────────────
    ut = getattr(state, "unfreeze_target", 0)
    uc = getattr(state, "unfreeze_count", 0)
    global_vec[15] = min(1.0, uc / ut) if ut > 0 else 1.0  # 解冻进度（无任务=1.0已满足）
    global_vec[16] = 1.0 if ut > 0 else 0.0                # 是否有解冻任务

    return {"board": board, "global": global_vec}


def stack_observations(frames: list[dict]) -> dict:
    """将多帧单帧观测堆叠为网络输入。

    frames: 长度为 FRAME_STACK 的列表，index 0 为最旧帧，index -1 为最新帧。
    若历史帧不足，用零帧补齐最旧位置。
    返回 board shape=(87,10,10)，global 取最新帧的 (15,)。
    """
    while len(frames) < FRAME_STACK:
        frames = [{"board": np.zeros((BOARD_CHANNELS, ROWS, COLS), dtype=np.float32),
                   "global": np.zeros(GLOBAL_DIM, dtype=np.float32)}] + frames
    frames = frames[-FRAME_STACK:]
    stacked_board = np.concatenate([f["board"] for f in frames], axis=0)  # (87,10,10)
    return {"board": stacked_board, "global": frames[-1]["global"]}
