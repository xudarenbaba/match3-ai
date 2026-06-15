from __future__ import annotations

from match3_engine.constants import TASK_TARGET
from match3_engine.game import GameState

REWARD = {
    "score_scale": 0.01,
    "chain_scale": 0.005,
    "task_delta": 2.0,
    "empty_swap": -0.2,
    "step_cost": -0.03,
    "reverse_swap_penalty": -0.25,
    "win_bonus": 50.0,
    "steps_left_bonus": 0.2,
    "lose_penalty": -10.0,
    "task_deficit_penalty": -3.0,
}


def compute_reward(prev: GameState, result: dict, nxt: GameState) -> float:
    r = 0.0
    r += result.get("total_score", 0) * REWARD["score_scale"]
    r += result.get("chain_score", 0) * REWARD["chain_scale"]

    for shape in nxt.target_shapes:
        delta = nxt.task_scores.get(shape, 0) - prev.task_scores.get(shape, 0)
        r += delta * REWARD["task_delta"]

    if not result.get("had_match") and not result.get("used_powerup"):
        r += REWARD["empty_swap"]
    r += REWARD["step_cost"]
    last_action_before = int(result.get("last_action_before", -1))
    curr_action = int(result.get("action_index", -1))
    if curr_action >= 0 and last_action_before >= 0 and curr_action == last_action_before:
        r += REWARD["reverse_swap_penalty"]

    if nxt.won:
        r += REWARD["win_bonus"]
        r += max(0, nxt.total_steps - nxt.steps_used) * REWARD["steps_left_bonus"]
    elif nxt.over:
        r += REWARD["lose_penalty"]
        deficit = sum(max(0, nxt.task_target - nxt.task_scores.get(s, 0)) for s in nxt.target_shapes)
        r += deficit * REWARD["task_deficit_penalty"]
    return r
