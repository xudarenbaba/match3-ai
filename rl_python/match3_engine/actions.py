from __future__ import annotations

import random
from typing import Optional

import numpy as np

from .board import Board, clone_board, in_bounds
from .cells import is_powerup, is_frozen
from .constants import ROWS, COLS, MAX_ACTIONS, SWAP_ACTIONS
from .resolver import try_swap

DIRS = [(0, 1), (0, -1), (1, 0), (-1, 0)]

# 捏爆动作起始编号：0-89 横向交换, 90-179 纵向交换, 180-279 捏爆
POP_OFFSET = SWAP_ACTIONS  # 180


def encode_swap(fr: dict, to: dict) -> int:
    if fr["r"] == to["r"] and fr["c"] + 1 == to["c"]:
        return fr["r"] * 9 + fr["c"]
    if fr["c"] == to["c"] and fr["r"] + 1 == to["r"]:
        return 90 + fr["r"] * 10 + fr["c"]
    if fr["r"] == to["r"] and fr["c"] - 1 == to["c"]:
        return fr["r"] * 9 + to["c"]
    if fr["c"] == to["c"] and fr["r"] - 1 == to["r"]:
        return 90 + to["r"] * 10 + to["c"]
    return -1


def encode_pop(r: int, c: int) -> int:
    return POP_OFFSET + r * COLS + c


def decode_action(action: int) -> dict | None:
    """解码动作为统一的 move 结构：
      交换 → {"type":"swap", "from":{r,c}, "to":{r,c}}
      捏爆 → {"type":"pop", "r":r, "c":c}
    """
    if action < 0 or action >= MAX_ACTIONS:
        return None
    if action < 90:
        r = action // 9
        c = action % 9
        return {"type": "swap", "from": {"r": r, "c": c}, "to": {"r": r, "c": c + 1}}
    if action < POP_OFFSET:
        idx = action - 90
        r = idx // 10
        c = idx % 10
        if r >= ROWS - 1:
            return None
        return {"type": "swap", "from": {"r": r, "c": c}, "to": {"r": r + 1, "c": c}}
    # 捏爆段
    idx = action - POP_OFFSET
    r = idx // COLS
    c = idx % COLS
    return {"type": "pop", "r": r, "c": c}


def get_adjacent_swaps(board: Board, layout: Optional[list] = None) -> list:
    """返回相邻可交换格对。layout 为 None 时视为全 1；void 格与冰冻格的 swap 被排除。"""
    swaps = []
    seen = set()
    for r in range(ROWS):
        for c in range(COLS):
            if layout and not layout[r][c]:
                continue
            cell = board[r][c]
            if not cell or is_frozen(cell):  # 冰冻格不可移动
                continue
            for dr, dc in DIRS:
                nr, nc = r + dr, c + dc
                if not in_bounds(nr, nc):
                    continue
                if layout and not layout[nr][nc]:
                    continue
                ncell = board[nr][nc]
                if not ncell or is_frozen(ncell):  # 目标冰冻格不可交换
                    continue
                key = (
                    f"{r},{c}-{nr},{nc}"
                    if r < nr or (r == nr and c < nc)
                    else f"{nr},{nc}-{r},{c}"
                )
                if key in seen:
                    continue
                seen.add(key)
                swaps.append({"from": {"r": r, "c": c}, "to": {"r": nr, "c": nc}})
    return swaps


def _can_pop(board: Board, r: int, c: int, layout: Optional[list]) -> bool:
    """捏爆有效性：普通格 且 非冰冻 且 非空 且 非道具 且 活跃格。"""
    if layout and not layout[r][c]:
        return False
    cell = board[r][c]
    if not cell:
        return False
    if cell.kind != "normal":  # 道具格不可捏
        return False
    if is_frozen(cell):        # 冰冻格不可捏
        return False
    return True


def build_action_mask(board: Board, layout: Optional[list] = None) -> np.ndarray:
    mask = np.zeros(MAX_ACTIONS, dtype=np.float32)
    effective_found = False

    # ── 交换段（0-179）：仅「有效交换」（能消除/触发道具/得分）────
    for swap in get_adjacent_swaps(board, layout):
        idx = encode_swap(swap["from"], swap["to"])
        if idx < 0:
            continue
        test_board = clone_board(board)
        result = try_swap(test_board, random.Random(0), swap["from"], swap["to"], layout)
        if result.get("had_match") or result.get("used_powerup") or result.get("total_score", 0) > 0:
            mask[idx] = 1.0
            effective_found = True

    # ── 捏爆段（180-279）：对每个可捏普通格标记有效 ────────────────
    for r in range(ROWS):
        for c in range(COLS):
            if _can_pop(board, r, c, layout):
                mask[encode_pop(r, c)] = 1.0
                effective_found = True

    # 若不存在任何有效动作，则退化为任意相邻交换，确保环境不会卡死。
    if not effective_found:
        for swap in get_adjacent_swaps(board, layout):
            idx = encode_swap(swap["from"], swap["to"])
            if idx >= 0:
                mask[idx] = 1.0
    return mask


def action_to_move(board: Board, action: int, layout: Optional[list] = None) -> dict | None:
    """把动作编号解析为可执行 move，并校验合法性（含冰冻锁定、捏爆条件）。"""
    move = decode_action(action)
    if move is None:
        return None
    mask = build_action_mask(board, layout)
    if mask[action] < 0.5:
        return None
    return move
