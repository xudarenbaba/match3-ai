from __future__ import annotations

from typing import List, Optional

from .board import Board, cell_key, in_bounds
from .constants import ROWS, COLS


def unfreeze_targets(board: Board, targets: List[dict]) -> List[dict]:
    out = []
    for p in targets:
        cell = board[p["r"]][p["c"]]
        if cell and cell.frozen:
            cell.frozen = False
            out.append({"r": p["r"], "c": p["c"]})
    return out


def powerup_targets(
    board: Board,
    power_pos: dict,
    partner_pos: dict,
    layout: Optional[list] = None,
) -> List[dict]:
    """计算道具作用范围。layout 为 None 时视为全 1；void 格不进入目标列表。"""
    power_cell = board[power_pos["r"]][power_pos["c"]]
    partner_cell = board[partner_pos["r"]][partner_pos["c"]]
    targets = set()

    if not power_cell or power_cell.kind != "powerup":
        return []

    def is_active(r: int, c: int) -> bool:
        if not in_bounds(r, c):
            return False
        if layout and not layout[r][c]:
            return False
        return True

    if power_cell.powerup_type == "column":
        for r in range(ROWS):
            if is_active(r, power_pos["c"]):
                targets.add(cell_key(r, power_pos["c"]))
    elif power_cell.powerup_type == "bomb":
        for dr in range(-1, 2):
            for dc in range(-1, 2):
                nr, nc = power_pos["r"] + dr, power_pos["c"] + dc
                if is_active(nr, nc):
                    targets.add(cell_key(nr, nc))
    elif power_cell.powerup_type == "color":
        target_shape = partner_cell.shape if partner_cell else None
        if target_shape:
            for r in range(ROWS):
                for c in range(COLS):
                    cell = board[r][c]
                    if is_active(r, c) and cell and cell.shape == target_shape:
                        targets.add(cell_key(r, c))

    if is_active(power_pos["r"], power_pos["c"]):
        targets.add(cell_key(power_pos["r"], power_pos["c"]))

    return [{"r": int(k.split(",")[0]), "c": int(k.split(",")[1])} for k in targets]
