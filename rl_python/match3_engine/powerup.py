from __future__ import annotations

from typing import List, Optional, Tuple

from .board import Board, cell_key, in_bounds
from .cells import NormalCell
from .constants import ROWS, COLS


def unfreeze_targets(board: Board, targets: List[dict]) -> List[dict]:
    out = []
    for p in targets:
        cell = board[p["r"]][p["c"]]
        if cell and cell.frozen:
            cell.frozen = False
            out.append({"r": p["r"], "c": p["c"]})
    return out


def _upgrade_targets(
    board: Board,
    targets: List[dict],
    target_shapes: Optional[List[str]] = None,
) -> Tuple[dict, dict]:
    """
    对 targets 中每个格子的等级 +1：
      - 原 L1 → L2，保留在原位
      - 原 L2 → 升为 L3（完成目标），格子设为 None（消失）
    返回：
      special_gained: {shape: count}  —— 升级到 L3 消失的目标 shape 计数
      cleared_by_shape: {shape: count}  —— 消失的格子（升到 L3 的）
    """
    special_gained: dict = {}
    cleared_by_shape: dict = {}

    for p in targets:
        r, c = p["r"], p["c"]
        cell = board[r][c]
        if cell is None or cell.kind != "normal":
            continue
        if cell.frozen:
            cell.frozen = False  # 顺带解冻

        if cell.level == 1:
            cell.level = 2
        elif cell.level == 2:
            # 升为 L3 → 视为完成目标，格子消失
            shape = cell.shape
            board[r][c] = None
            cleared_by_shape[shape] = cleared_by_shape.get(shape, 0) + 1
            if target_shapes is None or shape in target_shapes:
                special_gained[shape] = special_gained.get(shape, 0) + 1
        # L3 普通格或道具格：不处理

    return special_gained, cleared_by_shape


def powerup_upgrade_targets(
    board: Board,
    power_pos: dict,
    partner_pos: dict,
    layout: Optional[list] = None,
    target_shapes: Optional[List[str]] = None,
) -> Tuple[List[dict], dict, dict]:
    """
    执行道具升级效果：
      row   → 该行所有其他普通格 +1 级
      column → 该列所有其他普通格 +1 级
      color  → 全图与 partner 同形状的所有普通格 +1 级
      bomb   → 九宫格内所有格子直接消除（保留原逻辑）

    返回：(升级目标列表, special_gained, cleared_by_shape)
    bomb 类型返回的 targets 是待消除格列表（兼容旧逻辑）。
    """
    power_cell = board[power_pos["r"]][power_pos["c"]]
    partner_cell = board[partner_pos["r"]][partner_pos["c"]]

    if not power_cell or power_cell.kind != "powerup":
        return [], {}, {}

    def is_active(r: int, c: int) -> bool:
        if not in_bounds(r, c):
            return False
        if layout and not layout[r][c]:
            return False
        return True

    pt = power_cell.powerup_type

    if pt == "row":
        targets = [
            {"r": power_pos["r"], "c": c}
            for c in range(COLS)
            if is_active(power_pos["r"], c) and c != power_pos["c"]
            and board[power_pos["r"]][c] is not None
            and board[power_pos["r"]][c].kind == "normal"
        ]
        # 道具本身消失
        board[power_pos["r"]][power_pos["c"]] = None
        special_gained, cleared_by_shape = _upgrade_targets(board, targets, target_shapes)
        return targets, special_gained, cleared_by_shape

    elif pt == "column":
        targets = [
            {"r": r, "c": power_pos["c"]}
            for r in range(ROWS)
            if is_active(r, power_pos["c"]) and r != power_pos["r"]
            and board[r][power_pos["c"]] is not None
            and board[r][power_pos["c"]].kind == "normal"
        ]
        board[power_pos["r"]][power_pos["c"]] = None
        special_gained, cleared_by_shape = _upgrade_targets(board, targets, target_shapes)
        return targets, special_gained, cleared_by_shape

    elif pt == "color":
        target_shape = partner_cell.shape if partner_cell else None
        if not target_shape:
            return [], {}, {}
        targets = [
            {"r": r, "c": c}
            for r in range(ROWS)
            for c in range(COLS)
            if is_active(r, c)
            and (r != power_pos["r"] or c != power_pos["c"])
            and board[r][c] is not None
            and board[r][c].kind == "normal"
            and board[r][c].shape == target_shape
        ]
        board[power_pos["r"]][power_pos["c"]] = None
        special_gained, cleared_by_shape = _upgrade_targets(board, targets, target_shapes)
        return targets, special_gained, cleared_by_shape

    elif pt == "bomb":
        # 炸弹保持原逻辑：九宫格直接消除
        bomb_targets = []
        for dr in range(-1, 2):
            for dc in range(-1, 2):
                nr, nc = power_pos["r"] + dr, power_pos["c"] + dc
                if is_active(nr, nc):
                    bomb_targets.append({"r": nr, "c": nc})
        unfreeze_targets(board, bomb_targets)
        special_gained_bomb: dict = {}
        cleared_bomb: dict = {}
        for p in bomb_targets:
            cell = board[p["r"]][p["c"]]
            if cell:
                shape = cell.shape
                cleared_bomb[shape] = cleared_bomb.get(shape, 0) + 1
                if target_shapes and shape in target_shapes and cell.kind == "normal" and cell.level >= 2:
                    special_gained_bomb[shape] = special_gained_bomb.get(shape, 0) + 1
                board[p["r"]][p["c"]] = None
        return bomb_targets, special_gained_bomb, cleared_bomb

    return [], {}, {}
