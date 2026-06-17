from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from .board import Board, clone_board, create_empty_board
from .constants import SHAPES, MAX_STEPS, TASK_TARGET
from .gravity import reshuffle_board, freeze_random_cells
from .layouts import get_layout, pick_layout, LAYOUT_POOL
from .resolver import try_swap


@dataclass
class GameState:
    board: Board
    score: int = 0
    chain_score_total: int = 0
    task_scores: Dict[str, int] = field(default_factory=lambda: {s: 0 for s in SHAPES})
    target_shapes: List[str] = field(default_factory=list)
    task_target: int = TASK_TARGET
    total_steps: int = MAX_STEPS
    steps_used: int = 0
    last_action: int = -1
    won: bool = False
    over: bool = False
    layout: Optional[list] = None        # 10×10 的 0/1 列表，None 表示全 1
    layout_name: str = "full"


def pick_two_shapes(rng: random.Random) -> List[str]:
    copy_shapes = SHAPES.copy()
    rng.shuffle(copy_shapes)
    return copy_shapes[:2]


def create_game_state(
    rng: random.Random,
    *,
    total_steps: int = MAX_STEPS,
    task_target_shapes: Optional[List[str]] = None,
    task_target: int = TASK_TARGET,
    freeze: bool = True,
    frozen_ratio: float = 0.12,
    layout_name: Optional[str] = None,
    curriculum_level: int = 3,
) -> GameState:
    target_shapes = task_target_shapes or pick_two_shapes(rng)

    # 布局选择
    pool = LAYOUT_POOL.get(curriculum_level, LAYOUT_POOL[3])
    chosen_layout_name = layout_name or pick_layout(rng, pool)
    layout = get_layout(chosen_layout_name)

    board = create_empty_board()
    reshuffle_board(board, rng, layout)
    if freeze:
        freeze_random_cells(board, rng, frozen_ratio, layout)

    return GameState(
        board=board,
        target_shapes=target_shapes,
        task_target=task_target,
        total_steps=total_steps,
        task_scores={s: 0 for s in SHAPES},
        layout=layout,
        layout_name=chosen_layout_name,
    )


def snapshot_state(state: GameState) -> GameState:
    return GameState(
        board=clone_board(state.board),
        score=state.score,
        chain_score_total=state.chain_score_total,
        task_scores=dict(state.task_scores),
        target_shapes=list(state.target_shapes),
        task_target=state.task_target,
        total_steps=state.total_steps,
        steps_used=state.steps_used,
        last_action=state.last_action,
        won=state.won,
        over=state.over,
        layout=state.layout,
        layout_name=state.layout_name,
    )


def check_victory(state: GameState, task_target: Optional[int] = None) -> bool:
    target = task_target if task_target is not None else state.task_target
    return all(state.task_scores.get(s, 0) >= target for s in state.target_shapes)


def _apply_task_progress(state: GameState, result: dict) -> None:
    for shape, n in result.get("special_gained", {}).items():
        state.task_scores[shape] = state.task_scores.get(shape, 0) + n
    for shape, n in result.get("task_from_powerup", {}).items():
        state.task_scores[shape] = state.task_scores.get(shape, 0) + n


def execute_move(state: GameState, rng: random.Random, fr: dict, to: dict) -> dict:
    if state.over:
        return {"ok": False, "reason": "game over"}
    result = try_swap(state.board, rng, fr, to, state.layout)
    result["last_action_before"] = state.last_action
    state.score += result["total_score"]
    state.chain_score_total += result["chain_score"]
    _apply_task_progress(state, result)
    state.steps_used += 1
    state.last_action = int(result.get("action_index", -1))
    if check_victory(state):
        state.won = True
        state.over = True
    elif state.steps_used >= state.total_steps:
        state.over = True
    return {"ok": True, "result": result}
