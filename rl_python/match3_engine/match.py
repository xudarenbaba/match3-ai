from __future__ import annotations

from typing import Dict, List, Optional, Tuple
import random

from .board import Board, cell_key, in_bounds
from .cells import create_normal_cell
from .constants import ROWS, COLS


def find_matches(board: Board) -> List[dict]:
    matches: List[dict] = []

    for r in range(ROWS):
        c = 0
        while c < COLS:
            cell = board[r][c]
            if not cell or cell.kind != "normal" or cell.frozen:
                c += 1
                continue
            end = c + 1
            while end < COLS:
                nxt = board[r][end]
                if (
                    not nxt
                    or nxt.kind != "normal"
                    or nxt.frozen
                    or nxt.shape != cell.shape
                    or nxt.level != cell.level
                ):
                    break
                end += 1
            if end - c >= 3:
                matches.append({"cells": [{"r": r, "c": i} for i in range(c, end)], "shape": cell.shape, "level": cell.level})
            c = end

    for c in range(COLS):
        r = 0
        while r < ROWS:
            cell = board[r][c]
            if not cell or cell.kind != "normal" or cell.frozen:
                r += 1
                continue
            end = r + 1
            while end < ROWS:
                nxt = board[end][c]
                if (
                    not nxt
                    or nxt.kind != "normal"
                    or nxt.frozen
                    or nxt.shape != cell.shape
                    or nxt.level != cell.level
                ):
                    break
                end += 1
            if end - r >= 3:
                matches.append({"cells": [{"r": i, "c": c} for i in range(r, end)], "shape": cell.shape, "level": cell.level})
            r = end

    if len(matches) <= 1:
        return matches

    parent = list(range(len(matches)))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def unite(a: int, b: int) -> None:
        parent[find(a)] = find(b)

    pos_to_indices: Dict[str, List[int]] = {}
    for i, m in enumerate(matches):
        for p in m["cells"]:
            k = cell_key(p["r"], p["c"])
            pos_to_indices.setdefault(k, []).append(i)
    for indices in pos_to_indices.values():
        for j in range(1, len(indices)):
            unite(indices[0], indices[j])

    groups: Dict[int, dict] = {}
    for i, m in enumerate(matches):
        root = find(i)
        if root not in groups:
            groups[root] = {"shape": m["shape"], "level": m["level"], "cells": []}
        g = groups[root]
        seen = {(p["r"], p["c"]) for p in g["cells"]}
        for p in m["cells"]:
            key = (p["r"], p["c"])
            if key not in seen:
                g["cells"].append(p)
                seen.add(key)

    return [g for g in groups.values() if len(g["cells"]) >= 3]


def _unfreeze_adjacent(board: Board, matches: List[dict]) -> List[dict]:
    dirs = [(1, 0), (-1, 0), (0, 1), (0, -1)]
    adj = set()
    for m in matches:
        for p in m["cells"]:
            for dr, dc in dirs:
                nr, nc = p["r"] + dr, p["c"] + dc
                if in_bounds(nr, nc):
                    adj.add(cell_key(nr, nc))
    unfrozen = []
    for k in adj:
        r, c = map(int, k.split(","))
        cell = board[r][c]
        if cell and cell.frozen:
            cell.frozen = False
            unfrozen.append({"r": r, "c": c})
    return unfrozen


def pick_merge_positions(matches: List[dict], swap_from: Optional[dict], swap_to: Optional[dict]) -> List[dict]:
    positions = []
    for m in matches:
        cell_set = {cell_key(p["r"], p["c"]) for p in m["cells"]}
        chosen = None
        for pos in [swap_to, swap_from]:
            if pos and cell_key(pos["r"], pos["c"]) in cell_set:
                chosen = pos
                break
        if chosen is None:
            chosen = max(m["cells"], key=lambda p: (p["r"], p["c"]))
        positions.append(chosen)
    return positions


def apply_merges(board: Board, matches: List[dict], merge_positions: List[dict], rng: random.Random) -> dict:
    score = 0
    special_gained: Dict[str, int] = {}
    cleared_by_shape: Dict[str, int] = {}  # 每种 shape 本轮被消除（清空）的格子数
    cleared = set()
    results: Dict[str, Optional[object]] = {}
    unfrozen = _unfreeze_adjacent(board, matches)

    for m, pos in zip(matches, merge_positions):
        n = len(m["cells"])
        level = m["level"]
        shape = m["shape"]
        k = cell_key(pos["r"], pos["c"])
        if level == 1:
            score += n
            results[k] = create_normal_cell(rng, shape, 2)
            # L1 合并：n 格全部清空后升级到合并位，合并位变为 L2（仍存在），其余 n-1 格被消除
            cleared_by_shape[shape] = cleared_by_shape.get(shape, 0) + (n - 1)
        elif level == 2:
            score += n * 2
            results[k] = create_normal_cell(rng, shape, 3)
            # L2 合并：合并位变为 L3，其余 n-1 格被消除
            cleared_by_shape[shape] = cleared_by_shape.get(shape, 0) + (n - 1)
        else:
            score += n * 3
            special_gained[shape] = special_gained.get(shape, 0) + 1
            results[k] = None
            # L3 合并：n 格全部消除（合并位也消失，产生任务分）
            cleared_by_shape[shape] = cleared_by_shape.get(shape, 0) + n
        for p in m["cells"]:
            if p["r"] == pos["r"] and p["c"] == pos["c"]:
                continue
            cleared.add(cell_key(p["r"], p["c"]))

    for k in cleared:
        r, c = map(int, k.split(","))
        board[r][c] = None
    for k, val in results.items():
        r, c = map(int, k.split(","))
        board[r][c] = val

    return {"score": score, "special_gained": special_gained,
            "cleared_by_shape": cleared_by_shape, "unfrozen": unfrozen}
