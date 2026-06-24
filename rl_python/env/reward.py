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

    # ── 多连消奖励：让 4连/5连明显优于 3连 ──────────────────────
    # 解决模型无法从 action mask（0/1）中感知连消数的问题，
    # 显式加权让 4连 vs 3连 差距从 0.7 扩大到 ~2.0
    "match_size_bonus_4": 1.0,   # 4连消额外加（目标或非目标均计）
    "match_size_bonus_5": 2.0,   # 5连+消额外加

    # ── 稀疏任务信号 ──────────────────────────────────────────────
    "task_delta": 3.0,

    # ── 解冻任务信号 ──────────────────────────────────────────────
    # 每解冻 1 个冰壳 +1.0（0.5→1.0：让触发解冻的合并与非解冻合并拉开更大差距，
    # 使档位判断边界更清晰，解冻信号在训练中也更强）
    "unfreeze_reward": 1.0,

    # ── 捏爆成本：防滥用 ─────────────────────────────────────────
    # 捏爆本身无即时正收益（捏掉的格子不计消除分），只有改善下一步局面才划算。
    # 固定额外成本让盲目捏爆亏分，战略价值靠 value function 多步信用分配体现。
    "pop_cost": -0.1,

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
    # 终局奖励尺度持续压低：win_bonus=12 时回报量级 ~40，follow 项仍占评分 ~68%，
    # value 噪声足以逆转即时奖励的正确排序（如 L2目标四连 vs 普通三连差值被淹没）。
    # 压到 win_bonus=6 → 回报量级 ~20，value 误差/即时奖励 比值降低一半，
    # 使 lookahead 的 W·r_imm 项占比从 ~32% 提升到 ~50%，即时奖励主导决策。
    "win_bonus": 6.0,
    "steps_left_bonus": 0.02,
    "lose_penalty": -2.0,
    "task_deficit_penalty": -0.5,
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

    # ── 解冻奖励 ──────────────────────────────────────────────────
    r += result.get("unfrozen_count", 0) * REWARD["unfreeze_reward"]

    # ── L2 目标格消除额外奖励 + 多连消奖励 + 道具生成奖励 ────────
    # merge_events 包含每次合并的 level/shape/count/result_cell
    for event in result.get("merge_events", []):
        n = event.get("count", 0)
        level = event.get("level", 0)
        shape = event.get("match", {}).get("shape")

        # L2 目标格消除额外奖励
        if level == 2 and shape in target_set:
            r += REWARD["target_L2_merge_bonus"]

        # 多连消奖励（4连/5连+，不区分目标与否）
        if n == 4:
            r += REWARD["match_size_bonus_4"]
        elif n >= 5:
            r += REWARD["match_size_bonus_5"]

        # 道具生成奖励
        rc = event.get("result_cell")
        if rc and getattr(rc, "kind", None) == "powerup":
            pt = getattr(rc, "powerup_type", "")
            if pt == "column":
                r += REWARD["combo_powerup_column"]
            elif pt == "color":
                r += REWARD["combo_powerup_color"]

    # ── 步数惩罚 ──────────────────────────────────────────────────
    is_pop = bool(result.get("is_pop"))
    if is_pop:
        # 捏爆有专属成本，不叠加 empty_swap（捏爆本就不要求形成消除）
        r += REWARD["pop_cost"]
    elif not result.get("had_match") and not result.get("used_powerup"):
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
        # 失败缺口 = 形状任务缺口 + 解冻任务缺口
        shape_deficit = sum(max(0, nxt.task_target - nxt.task_scores.get(s, 0)) for s in nxt.target_shapes)
        unfreeze_deficit = max(0, nxt.unfreeze_target - nxt.unfreeze_count)
        r += (shape_deficit + unfreeze_deficit) * REWARD["task_deficit_penalty"]

    return r
