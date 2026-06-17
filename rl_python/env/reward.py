from __future__ import annotations

from match3_engine.constants import TASK_TARGET
from match3_engine.game import GameState

REWARD = {
    # ── 密集信号 ──────────────────────────────────────────────────
    "target_clear_per_cell": 0.3,
    "non_target_clear_per_cell": -0.05,

    # ── L2 目标格消除额外奖励（在 target_clear 基础上叠加）────────
    # L2 目标三连消是当前唯一能直接得任务分的操作，需要和 L1 目标消除
    # 拉开明显差距，让模型优先学会「凑L2再消」而不是随手消L1
    "target_L2_merge_bonus": 1.5,

    # ── 稀疏任务信号 ──────────────────────────────────────────────
    "task_delta": 3.0,

    # ── 连消道具奖励：4连/5连消触发生成道具 ─────────────────────
    # 生成 column 道具（4连消）
    "combo_powerup_column": 0.4,
    # 生成 color 道具（5连+消）
    "combo_powerup_color": 0.6,

    # ── 基础分数信号 ──────────────────────────────────────────────
    "score_scale": 0.005,
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

    # ── 密集目标色块奖励 ─────────────────────────────────────────
    target_set = set(nxt.target_shapes)
    cleared_by_shape = result.get("cleared_by_shape", {})
    for shape, n in cleared_by_shape.items():
        if shape in target_set:
            r += n * REWARD["target_clear_per_cell"]
        else:
            r += n * REWARD["non_target_clear_per_cell"]

    # ── 稀疏任务分 ────────────────────────────────────────────────
    for shape in nxt.target_shapes:
        delta = nxt.task_scores.get(shape, 0) - prev.task_scores.get(shape, 0)
        r += delta * REWARD["task_delta"]

    # ── L2 目标格消除额外奖励 + 连消道具生成奖励 ────────────────
    # merge_events 包含每次合并的 level/shape/result_cell
    for event in result.get("merge_events", []):
        # L2 目标格消除额外奖励（在上方 cleared_by_shape 通用奖励基础上叠加）
        if event.get("level") == 2 and event.get("match", {}).get("shape") in target_set:
            r += REWARD["target_L2_merge_bonus"]
        # 道具生成奖励
        rc = event.get("result_cell")
        if rc and getattr(rc, "kind", None) == "powerup":
            pt = getattr(rc, "powerup_type", "")
            if pt == "column":
                r += REWARD["combo_powerup_column"]
            elif pt == "color":
                r += REWARD["combo_powerup_color"]

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
