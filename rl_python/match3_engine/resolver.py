from __future__ import annotations

import random
from typing import Optional

from .board import Board, clone_board
from .cells import is_powerup
from .match import find_matches, pick_merge_positions, apply_merges
from .gravity import apply_gravity_and_refill
from .powerup import powerup_targets, unfreeze_targets


def swap_cells(board: Board, fr: dict, to: dict) -> None:
    board[fr["r"]][fr["c"]], board[to["r"]][to["c"]] = board[to["r"]][to["c"]], board[fr["r"]][fr["c"]]


def resolve_board(
    board: Board,
    rng: random.Random,
    swap_from: Optional[dict] = None,
    swap_to: Optional[dict] = None,
    layout: Optional[list] = None,
) -> dict:
    total_score = 0
    chain_score = 0
    is_first = True
    special_gained: dict = {}
    cleared_by_shape: dict = {}
    merge_events: list = []  # 收集所有合并事件，供 reward 使用
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
    board: Board, rng: random.Random, fr: dict, to: dict, layout: Optional[list] = None
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
            "had_match": False,
            "used_powerup": False,
            "task_from_powerup": {},
        }

    total_score = 0
    task_from_powerup: dict = {}
    triggered = False

    for pw in powerups:
        targets = powerup_targets(board, pw["pos"], pw["partner"], layout)
        unfreeze_targets(board, targets)
        removed = len(targets)
        if removed == 0:
            continue
        triggered = True
        for p in targets:
            board[p["r"]][p["c"]] = None
        total_score += removed
        if removed >= 9:
            task_from_powerup[pw["cell"].shape] = task_from_powerup.get(pw["cell"].shape, 0) + 1

    if not triggered:
        return {
            "total_score": 0,
            "chain_score": 0,
            "special_gained": {},
            "had_match": False,
            "used_powerup": False,
            "task_from_powerup": {},
        }

    apply_gravity_and_refill(board, rng, layout)
    chain = resolve_board(board, rng, fr, to, layout)
    total_score += chain["total_score"]
    return {
        "total_score": total_score,
        "chain_score": chain["total_score"],
        "special_gained": chain["special_gained"],
        "cleared_by_shape": chain.get("cleared_by_shape", {}),
        "had_match": chain["had_match"] or True,
        "used_powerup": True,
        "task_from_powerup": task_from_powerup,
    }


def try_swap(
    board: Board, rng: random.Random, fr: dict, to: dict, layout: Optional[list] = None
) -> dict:
    swap_cells(board, fr, to)
    power_res = _resolve_powerup_swap(board, rng, fr, to, layout)
    if power_res["used_powerup"]:
        return power_res
    normal_res = resolve_board(board, rng, fr, to, layout)
    normal_res["task_from_powerup"] = {}
    return normal_res
