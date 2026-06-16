from __future__ import annotations

from match3_engine.constants import TASK_TARGET
from match3_engine.game import GameState

REWARD = {
    # ── 密集信号 ──────────────────────────────────────────────────
    # 每消除 1 格目标色块（无论等级）即时奖励。
    # L1/L2 消除约产生 2-3 格，可得 0.6-0.9；L3×3 消除产生 task_delta，
    # 此时 cleared_by_shape 也会计 3 格（+0.9），与 task_delta(+2.0) 叠加，
    # 强化"消目标 L3"这个最终动作。
    "target_clear_per_cell": 0.3,

    # 每消除 1 格非目标色块的轻微惩罚，制造相对偏好。
    # 量级约为 target_clear 的 1/6，不会压制正常消除。
    "non_target_clear_per_cell": -0.05,

    # ── 稀疏任务信号 ──────────────────────────────────────────────
    # L3×3 → special_gained → task_score +1 时触发。
    "task_delta": 2.0,

    # ── 基础分数信号（保留但降权，避免和密集信号重叠过度） ──────────
    "score_scale": 0.005,   # 从 0.01 降到 0.005
    "chain_scale": 0.005,

    # ── 步数惩罚 ──────────────────────────────────────────────────
    "empty_swap": -0.2,
    "step_cost": -0.03,
    "reverse_swap_penalty": -0.25,

    # ── 终局信号 ──────────────────────────────────────────────────
    "win_bonus": 50.0,
    "steps_left_bonus": 0.2,
    "lose_penalty": -10.0,
    "task_deficit_penalty": -3.0,
}


def compute_reward(prev: GameState, result: dict, nxt: GameState) -> float:
    r = 0.0

    # ── 基础分数 ──────────────────────────────────────────────────
    r += result.get("total_score", 0) * REWARD["score_scale"]
    r += result.get("chain_score", 0) * REWARD["chain_scale"]

    # ── 密集目标色块奖励：每消除 1 格即时给分 ─────────────────────
    target_set = set(nxt.target_shapes)
    cleared_by_shape = result.get("cleared_by_shape", {})
    for shape, n in cleared_by_shape.items():
        if shape in target_set:
            r += n * REWARD["target_clear_per_cell"]
        else:
            r += n * REWARD["non_target_clear_per_cell"]

    # ── 稀疏任务分：L3 合并完成 task_score +1 ─────────────────────
    for shape in nxt.target_shapes:
        delta = nxt.task_scores.get(shape, 0) - prev.task_scores.get(shape, 0)
        r += delta * REWARD["task_delta"]

    # ── 步数惩罚 ──────────────────────────────────────────────────
    if not result.get("had_match") and not result.get("used_powerup"):
        r += REWARD["empty_swap"]
    r += REWARD["step_cost"]
    last_action_before = int(result.get("last_action_before", -1))
    curr_action = int(result.get("action_index", -1))
    if curr_action >= 0 and last_action_before >= 0 and curr_action == last_action_before:
        r += REWARD["reverse_swap_penalty"]

    # ── 终局信号 ──────────────────────────────────────────────────
    if nxt.won:
        r += REWARD["win_bonus"]
        r += max(0, nxt.total_steps - nxt.steps_used) * REWARD["steps_left_bonus"]
    elif nxt.over:
        r += REWARD["lose_penalty"]
        deficit = sum(max(0, nxt.task_target - nxt.task_scores.get(s, 0)) for s in nxt.target_shapes)
        r += deficit * REWARD["task_deficit_penalty"]

    return r
