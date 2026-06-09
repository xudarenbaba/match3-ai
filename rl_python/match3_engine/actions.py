from __future__ import annotations

import random

import numpy as np

from .board import Board, clone_board, in_bounds
from .constants import ROWS, COLS, MAX_ACTIONS
from .resolver import try_swap

DIRS = [(0, 1), (0, -1), (1, 0), (-1, 0)]


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


def decode_action(action: int) -> dict | None:
    if action < 0 or action >= MAX_ACTIONS:
        return None
    if action < 90:
        r = action // 9
        c = action % 9
        return {"from": {"r": r, "c": c}, "to": {"r": r, "c": c + 1}}
    idx = action - 90
    r = idx // 10
    c = idx % 10
    if r >= ROWS - 1:
        return None
    return {"from": {"r": r, "c": c}, "to": {"r": r + 1, "c": c}}


def get_adjacent_swaps(board: Board) -> list:
    swaps = []
    seen = set()
    for r in range(ROWS):
        for c in range(COLS):
            if not board[r][c]:
                continue
            for dr, dc in DIRS:
                nr, nc = r + dr, c + dc
                if not in_bounds(nr, nc) or not board[nr][nc]:
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


def build_action_mask(board: Board) -> np.ndarray:
    mask = np.zeros(MAX_ACTIONS, dtype=np.float32)
    effective_found = False

    for swap in get_adjacent_swaps(board):
        idx = encode_swap(swap["from"], swap["to"])
        if idx < 0:
            continue
        test_board = clone_board(board)
        result = try_swap(test_board, random.Random(0), swap["from"], swap["to"])
        if result.get("had_match") or result.get("used_powerup") or result.get("total_score", 0) > 0:
            mask[idx] = 1.0
            effective_found = True

    # 若不存在有效交换，则退化为任意相邻交换，确保环境不会卡死。
    if not effective_found:
        for swap in get_adjacent_swaps(board):
            idx = encode_swap(swap["from"], swap["to"])
            if idx >= 0:
                mask[idx] = 1.0
    return mask


def swap_from_action(board: Board, action: int) -> dict | None:
    swap = decode_action(action)
    if swap is None:
        return None
    mask = build_action_mask(board)
    if mask[action] < 0.5:
        return None
    return swap
