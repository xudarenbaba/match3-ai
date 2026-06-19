from __future__ import annotations

import random
from typing import Optional

from .board import Board, clone_board
from .cells import is_powerup
from .match import find_matches, pick_merge_positions, apply_merges
from .gravity import apply_gravity_and_refill
from .powerup import powerup_upgrade_targets


def swap_cells(board: Board, fr: dict, to: dict) -> None:
    board[fr["r"]][fr["c"]], board[to["r"]][to["c"]] = board[to["r"]][to["c"]], board[fr["r"]][fr["c"]]


def resolve_board(
    board: Board,
    rng: random.Random,
    swap_from: Optional[dict] = None,
    swap_to: Optional[dict] = None,
    layout: Optional[list] = None,
    target_shapes: Optional[list] = None,
) -> dict:
    total_score = 0
    chain_score = 0
    is_first = True
    special_gained: dict = {}
    cleared_by_shape: dict = {}
    merge_events: list = []
    had_match = False

    while True:
        matches = find_matches(board)
        if not matches:
            break
        had_match = True
        merge_positions = pick_merge_positions(
            matches,
            swap_from if is_first else None,
            swap_to if is_first else None,
        )
        merged = apply_merges(board, matches, merge_positions, rng)
        total_score += merged["score"]
        if not is_first:
            chain_score += merged["score"]
        for shape, n in merged["special_gained"].items():
            special_gained[shape] = special_gained.get(shape, 0) + n
        for shape, n in merged["cleared_by_shape"].items():
            cleared_by_shape[shape] = cleared_by_shape.get(shape, 0) + n
        merge_events.extend(merged.get("merge_events", []))
        apply_gravity_and_refill(board, rng, layout)
        is_first = False

    return {
        "total_score": total_score,
        "chain_score": chain_score,
        "special_gained": special_gained,
        "cleared_by_shape": cleared_by_shape,
        "merge_events": merge_events,
        "had_match": had_match,
        "used_powerup": False,
        "task_from_powerup": {},
    }


def _resolve_powerup_swap(
    board: Board,
    rng: random.Random,
    fr: dict,
    to: dict,
    layout: Optional[list] = None,
    target_shapes: Optional[list] = None,
) -> dict:
    first = board[fr["r"]][fr["c"]]
    second = board[to["r"]][to["c"]]
    powerups = []
    if is_powerup(first):
        powerups.append({"pos": fr, "partner": to, "cell": first})
    if is_powerup(second):
        powerups.append({"pos": to, "partner": fr, "cell": second})

    if not powerups:
        return {
            "total_score": 0,
            "chain_score": 0,
            "special_gained": {},
            "cleared_by_shape": {},
            "had_match": False,
            "used_powerup": False,
            "task_from_powerup": {},
        }

    total_score = 0
    combined_special: dict = {}
    combined_cleared: dict = {}
    triggered = False

    for pw in powerups:
        _targets, sp_gained, cleared = powerup_upgrade_targets(
            board, pw["pos"], pw["partner"], layout, target_shapes
        )
        if not _targets and pw["cell"].powerup_type not in ("row", "column", "color", "bomb"):
            continue
        triggered = True

        # bomb 逻辑：直接统计消除格数（_upgrade_targets 已经把格子删了）
        # row/column/color 逻辑：升级后 L2→L3 消失的格子也已经删了
        for shape, n in sp_gained.items():
            combined_special[shape] = combined_special.get(shape, 0) + n
        for shape, n in cleared.items():
            combined_cleared[shape] = combined_cleared.get(shape, 0) + n
        total_score += sum(cleared.values())

    if not triggered:
        return {
            "total_score": 0,
            "chain_score": 0,
            "special_gained": {},
            "cleared_by_shape": {},
            "had_match": False,
            "used_powerup": False,
            "task_from_powerup": {},
        }

    # 升级/消除后重力补位
    apply_gravity_and_refill(board, rng, layout)

    # 补位后可能触发新的普通三连
    chain = resolve_board(board, rng, fr, to, layout, target_shapes)
    total_score += chain["total_score"]

    for shape, n in chain["special_gained"].items():
        combined_special[shape] = combined_special.get(shape, 0) + n
    for shape, n in chain["cleared_by_shape"].items():
        combined_cleared[shape] = combined_cleared.get(shape, 0) + n

    return {
        "total_score": total_score,
        "chain_score": chain["total_score"],
        "special_gained": combined_special,
        "cleared_by_shape": combined_cleared,
        "merge_events": chain.get("merge_events", []),
        "had_match": chain["had_match"] or True,
        "used_powerup": True,
        # 任务分统一只走 special_gained，task_from_powerup 置空避免重复计分
        "task_from_powerup": {},
    }


def try_swap(
    board: Board,
    rng: random.Random,
    fr: dict,
    to: dict,
    layout: Optional[list] = None,
    target_shapes: Optional[list] = None,
) -> dict:
    swap_cells(board, fr, to)
    power_res = _resolve_powerup_swap(board, rng, fr, to, layout, target_shapes)
    if power_res["used_powerup"]:
        return power_res
    normal_res = resolve_board(board, rng, fr, to, layout, target_shapes)
    normal_res["task_from_powerup"] = {}
    return normal_res
