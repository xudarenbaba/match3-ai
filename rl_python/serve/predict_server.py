#!/usr/bin/env python3
"""加载训练好的 MaskablePPO 模型，为浏览器提供走棋推理 API。

推理策略：2-step lookahead
  1. 从模型拿当前帧 top-K 候选动作（按 logits 排序）
  2. 对每个候选，用 Python 引擎模拟执行一步，得到即时奖励和下一棋盘状态
  3. 对下一状态再跑一次模型 value head 估值
  4. 选择 (即时奖励 + GAMMA × 下一步价值) 最大的动作返回
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import random
import sys
from http.server import BaseHTTPRequestHandler, HTTPServer

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import numpy as np
import torch
from sb3_contrib import MaskablePPO

from env.observation import build_observation, stack_observations, BOARD_CHANNELS, ROWS, GLOBAL_DIM
from env.reward import compute_reward
from match3_engine.actions import (
    build_action_mask, decode_action, POP_OFFSET, get_adjacent_swaps, encode_swap,
)
from match3_engine.game import snapshot_state, execute_move
from serve.state_codec import json_to_game_state

MODEL: MaskablePPO | None = None
NON_DETERMINISTIC = False

# lookahead 超参数
LOOKAHEAD_GAMMA = 0.99       # 折扣因子（与训练保持一致）
# 即时奖励权重：value head 估值量级远大于单步即时奖励，若直接相加会被 V 噪声淹没。
# 放大即时奖励权重，让「4连>3连」等正确的单步信号主导决策。
LOOKAHEAD_REWARD_WEIGHT = 8.0
# lookahead 模拟会触发 gravity 随机补充新格子。用固定种子让同一局面的评分可复现、
# 推理结果稳定（否则同一局面多次请求可能给出不同动作）。
LOOKAHEAD_SEED = 12345


# ── 档位阈值常量 ─────────────────────────────────────────────────
# 用于 _tier_of 的档位判断，避免 follow 量级主导决策导致优先级越权。
# P8 铺垫/捏爆的增量 follow 阈值：需超过此值才算「解锁了下一步有意义消除」
# 取值对应目标L1三连的 W*r ≈ 4.6，即捏爆/铺垫后下一步至少能达目标三连水平
TIER_P8_INCREMENT_THRESHOLD = 4.0


def _classify_result(result: dict, target_shapes: list) -> dict:
    """从 execute_move 的 result 中提取档位判断所需的特征。

    返回：
        has_match     : 当步产生消除
        task_gained   : 形状任务分增量（L2三连 → special_gained 有目标 shape）
        max_count     : 当步最大单次消除连数
        has_target    : 当步消除格中含目标格
        has_powerup_gen: 当步生成了道具（4连/5连结果）
        unfrozen      : 当步解冻格数
    """
    target_set = set(target_shapes)
    merge_events = result.get("merge_events", [])
    special_gained = result.get("special_gained", {})

    has_match = bool(result.get("had_match"))
    task_gained = sum(v for k, v in special_gained.items() if k in target_set)
    max_count = max((e.get("count", 0) for e in merge_events), default=0)
    has_target = any(
        e.get("match", {}).get("shape") in target_set for e in merge_events
    )
    has_powerup_gen = any(
        getattr(e.get("result_cell"), "kind", None) == "powerup"
        for e in merge_events
    )
    unfrozen = int(result.get("unfrozen_count", 0))

    return {
        "has_match": has_match,
        "task_gained": task_gained,
        "max_count": max_count,
        "has_target": has_target,
        "has_powerup_gen": has_powerup_gen,
        "unfrozen": unfrozen,
    }


def _tier_of(feat: dict, is_pop: bool, pop_increment: float = 0.0) -> int:
    """根据档位特征返回优先级档位编号（1=最高，10=最低）。

    优先级定义（严格不越权，存在高档则只选高档）：
      P1  目标L2三连（直接完成形状任务分）
      P2  目标L1五连+道具
      P3  目标L1四连+道具
      P4  解冻≥2格的任意消除
      P5  解冻1格 + 目标格消除
      P6  解冻1格的非目标格消除
      P7  目标格L1三连（无解冻）
      P8  铺垫/捏爆（下一步有意义消除，increment > 阈值）
      P9  非目标格消除（无解冻）
      P10 无意义捏爆/无消除交换
    """
    if is_pop:
        # 捏爆当步无消除，价值全靠 increment
        if pop_increment > TIER_P8_INCREMENT_THRESHOLD:
            return 8
        return 10

    hm = feat["has_match"]
    if not hm:
        return 10  # 无消除交换（空挥），归 P10

    tg = feat["task_gained"]
    mc = feat["max_count"]
    ht = feat["has_target"]
    hp = feat["has_powerup_gen"]
    uf = feat["unfrozen"]

    # P1：形状任务分增量（L2三连触发 special_gained）
    if tg > 0:
        return 1

    # P2：目标格五连+生成道具
    if mc >= 5 and ht and hp:
        return 2

    # P3：目标格四连+生成道具
    if mc == 4 and ht and hp:
        return 3

    # P4：解冻≥2格
    if uf >= 2:
        return 4

    # P5：解冻1格 + 目标格
    if uf == 1 and ht:
        return 5

    # P6：解冻1格（非目标格亦可）
    if uf == 1:
        return 6

    # P7：目标格消除（无解冻，可含四连/五连但未生成道具的边缘情况）
    if ht:
        return 7

    # P9：非目标格消除（无解冻）
    return 9


def _get_value(obs: dict) -> float:
    """用模型的 value head 估计当前局面价值。"""
    board_t = torch.tensor(obs["board"][None], dtype=torch.float32)
    global_t = torch.tensor(obs["global"][None], dtype=torch.float32)
    obs_tensor = {
        "board": board_t.to(MODEL.device),
        "global": global_t.to(MODEL.device),
    }
    with torch.no_grad():
        value = MODEL.policy.predict_values(obs_tensor)
    return float(value.squeeze().cpu().numpy())


def _immediate_reward(prev_state, result: dict, next_state) -> float:
    """复用训练时的奖励函数，计算即时奖励。"""
    return compute_reward(prev_state, result, next_state)


def _best_followup_pruned(after_state, cols: set, frame_t: dict, frame_t1: dict) -> float:
    """枚举 after_state 中「涉及 cols±1 列」的有效交换，返回最优 1-step 值
    `max_swap( W·r' + γ·V(s'') )`。返回 -inf 表示无任何消除机会。

    动作（交换/捏爆）只改变其涉及的列（及下落），新消除机会只可能在这些列附近，
    故按列剪枝枚举，大幅减少计算量同时不漏掉被解锁的大消除/解冻机会。
    """
    near = set()
    for c in cols:
        near.update({c - 1, c, c + 1})
    best = -float("inf")
    for sw in get_adjacent_swaps(after_state.board, after_state.layout):
        if sw["from"]["c"] not in near and sw["to"]["c"] not in near:
            continue
        sr = random.Random(LOOKAHEAD_SEED)
        ns = snapshot_state(after_state)
        ns.last_action = after_state.last_action
        er = execute_move(ns, sr, {"type": "swap", "from": sw["from"], "to": sw["to"]})
        if not er["ok"] or not er["result"].get("had_match"):
            continue  # 只看能形成消除的交换
        er["result"]["action_index"] = encode_swap(sw["from"], sw["to"])
        r = _immediate_reward(after_state, er["result"], ns)
        v = _get_value(stack_observations([frame_t, frame_t1, build_observation(ns)]))
        cand = LOOKAHEAD_REWARD_WEIGHT * r + LOOKAHEAD_GAMMA * v
        if cand > best:
            best = cand
    return best


def _affected_cols(move: dict) -> set:
    """动作影响的列：交换涉及两列，捏爆涉及一列。"""
    if move["type"] == "pop":
        return {move["c"]}
    return {move["from"]["c"], move["to"]["c"]}


def _score_action(
    state, prev_frames: list, f_curr: dict, move: dict, action_idx: int,
    pop_baseline: float = -float("inf"),
) -> float:
    """统一的对称 2-step 评分（交换与捏爆同一基准）。

    对捏爆候选：只计算捏爆「真正新解锁」的增量价值。
        increment = max(0, follow_after - pop_baseline)
    当捏爆没有改善局面时（follow_after ≤ baseline），score ≈ W·r_pop（纯负），
    必然输给任何有正即时收益的交换，杜绝「蹭本来就有的机会」。

    pop_baseline：捏爆前 col±1 范围内本来就有的最优后续价值，由 lookahead_select
    一次性预算并传入，避免每个捏爆候选重复计算（性能优化）。
    """
    sim_rng = random.Random(LOOKAHEAD_SEED)
    ss = snapshot_state(state)
    ss.last_action = state.last_action
    er = execute_move(ss, sim_rng, move)
    if not er["ok"]:
        return -float("inf")
    er["result"]["action_index"] = action_idx
    r = _immediate_reward(state, er["result"], ss)
    f_next = build_observation(ss)
    cols = _affected_cols(move)
    follow_after = _best_followup_pruned(ss, cols, f_curr, f_next)

    if move["type"] == "pop":
        if pop_baseline == -float("inf"):
            # 原局面该列无消除机会：捏爆可用 follow_after 完整计分
            increment = follow_after if follow_after != -float("inf") else 0.0
        else:
            increment = max(0.0, follow_after - pop_baseline) if follow_after != -float("inf") else 0.0
        return LOOKAHEAD_REWARD_WEIGHT * r + LOOKAHEAD_GAMMA * increment
    else:
        if follow_after == -float("inf"):
            v = _get_value(stack_observations(prev_frames + [f_next]))
            return LOOKAHEAD_REWARD_WEIGHT * r + LOOKAHEAD_GAMMA * v
        return LOOKAHEAD_REWARD_WEIGHT * r + LOOKAHEAD_GAMMA * follow_after


def lookahead_select(state, obs: dict, mask: np.ndarray) -> int:
    """硬档位 + follow 精细化的 lookahead 动作选择。

    执行流程：
      1. 模拟所有有效动作，提取即时结果特征（不含 follow）
      2. 按 P1-P10 档位分组
      3. 只在最高档（编号最小）内，计算 W·r + γ·follow 精细排序，选最优
      4. 严格不越权：高档候选存在时，低档候选绝对不参与竞争

    捏爆专项：
      - 当步无消除，tier 由 pop_increment（follow_after - baseline）决定
      - increment > 阈值 → P8（铺垫），否则 → P10
      - P8 内按 increment 大小排序，不引入 follow 的 value 噪声
    """
    valid_indices = np.where(mask > 0)[0]
    if len(valid_indices) == 0:
        action, _ = MODEL.predict(obs, action_masks=mask.astype(bool), deterministic=True)
        return int(action)

    prev_frames = last_two_frames(obs)  # [f_{t-1}, f_t]
    f_curr = prev_frames[1]             # f_t

    # ── 预计算捏爆基线（每列只算一次）───────────────────────────────
    pop_baselines: dict[int, float] = {}
    for action_idx in valid_indices:
        if action_idx < POP_OFFSET:
            continue
        mv = decode_action(int(action_idx))
        if mv is None:
            continue
        c = mv["c"]
        if c not in pop_baselines:
            pop_baselines[c] = _best_followup_pruned(state, {c}, f_curr, f_curr)

    # ── 第一遍：模拟所有动作，取即时结果，确定档位 ──────────────────
    candidates: list[dict] = []  # {action_idx, move, tier, result, r_imm, increment}

    for action_idx in valid_indices:
        move = decode_action(int(action_idx))
        if move is None:
            continue

        sim_rng = random.Random(LOOKAHEAD_SEED)
        ss = snapshot_state(state)
        ss.last_action = state.last_action
        er = execute_move(ss, sim_rng, move)
        if not er["ok"]:
            continue
        er["result"]["action_index"] = int(action_idx)
        r_imm = _immediate_reward(state, er["result"], ss)
        feat = _classify_result(er["result"], state.target_shapes)
        is_pop = move["type"] == "pop"

        # 捏爆需要算 increment
        pop_increment = 0.0
        if is_pop:
            c = move["c"]
            baseline = pop_baselines.get(c, -float("inf"))
            # follow_after：捏爆后该列消除机会（只算能消除的）
            f_next = build_observation(ss)
            follow_after = _best_followup_pruned(ss, {c}, f_curr, f_next)
            if baseline == -float("inf"):
                pop_increment = follow_after if follow_after != -float("inf") else 0.0
            else:
                pop_increment = max(0.0, follow_after - baseline) if follow_after != -float("inf") else 0.0

        tier = _tier_of(feat, is_pop, pop_increment)
        candidates.append({
            "action_idx": int(action_idx),
            "move": move,
            "tier": tier,
            "result_state": ss,   # 已模拟后的状态，供 follow 计算复用
            "r_imm": r_imm,
            "feat": feat,
            "is_pop": is_pop,
            "pop_increment": pop_increment,
        })

    if not candidates:
        return int(valid_indices[0])

    # ── 第二遍：找最高档，只在该档内精细排序 ────────────────────────
    best_tier = min(c["tier"] for c in candidates)
    top_candidates = [c for c in candidates if c["tier"] == best_tier]

    if len(top_candidates) == 1:
        return top_candidates[0]["action_idx"]

    # 同档内精细排序
    best_action = top_candidates[0]["action_idx"]
    best_score = -float("inf")

    for cand in top_candidates:
        if cand["is_pop"]:
            # P8 捏爆：按 increment 大小排，不引入 follow value 噪声
            score = cand["pop_increment"]
        else:
            # 交换类：W·r_imm + γ·follow
            cols = _affected_cols(cand["move"])
            f_next = build_observation(cand["result_state"])
            follow = _best_followup_pruned(cand["result_state"], cols, f_curr, f_next)
            if follow == -float("inf"):
                v = _get_value(stack_observations(prev_frames + [f_next]))
                score = LOOKAHEAD_REWARD_WEIGHT * cand["r_imm"] + LOOKAHEAD_GAMMA * v
            else:
                score = LOOKAHEAD_REWARD_WEIGHT * cand["r_imm"] + LOOKAHEAD_GAMMA * follow

        if score > best_score:
            best_score = score
            best_action = cand["action_idx"]

    return best_action


def last_two_frames(stacked_obs: dict) -> list[dict]:
    """从堆叠 obs (FRAME_STACK 帧) 中取出最近两帧 [f_{t-1}, f_t] 作为单帧 dict。

    用于 lookahead 构造 V(s') 的输入：[f_{t-1}, f_t, f_{t+1}]，保持 3 帧真实历史。
    global 向量取最新帧（stacked_obs 里只保留了最新帧的 global）。
    """
    board_stacked = stacked_obs["board"]  # (BOARD_CHANNELS*FRAME_STACK, 10, 10)
    f_prev = board_stacked[BOARD_CHANNELS:2 * BOARD_CHANNELS]   # 倒数第二帧
    f_curr = board_stacked[-BOARD_CHANNELS:]                    # 最新帧
    g = stacked_obs["global"]
    return [{"board": f_prev, "global": g}, {"board": f_curr, "global": g}]


def obs_single_from_stacked(stacked_obs: dict) -> dict:
    """从堆叠 obs 中取出最新一帧（最后 BOARD_CHANNELS 个通道）。"""
    board_stacked = stacked_obs["board"]
    board_last = board_stacked[-BOARD_CHANNELS:]
    return {"board": board_last, "global": stacked_obs["global"]}


class PredictHandler(BaseHTTPRequestHandler):
    def _send_cors(self) -> None:
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")

    def _json_response(self, code: int, payload: dict) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self._send_cors()
        self.end_headers()
        self.wfile.write(body)

    def do_OPTIONS(self) -> None:
        self.send_response(204)
        self._send_cors()
        self.end_headers()

    def do_GET(self) -> None:
        if self.path != "/health":
            self.send_response(404)
            self.end_headers()
            return
        self._json_response(200, {"ok": True, "model_loaded": MODEL is not None})

    def do_POST(self) -> None:
        if self.path != "/predict":
            self.send_response(404)
            self.end_headers()
            return
        if MODEL is None:
            self._json_response(503, {"error": "model not loaded"})
            return

        length = int(self.headers.get("Content-Length", 0))
        try:
            payload = json.loads(self.rfile.read(length))
            state = json_to_game_state(payload)

            # 从前端传来的 frameHistory 重建帧堆叠 obs
            frame_history_raw = payload.get("frameHistory", [])
            frames = []
            for fh in frame_history_raw:
                board_flat = fh.get("board", [])
                global_flat = fh.get("global", [])
                board_arr = np.array(board_flat, dtype=np.float32).reshape(BOARD_CHANNELS, ROWS, ROWS)
                global_arr = np.array(global_flat, dtype=np.float32).reshape(GLOBAL_DIM)
                frames.append({"board": board_arr, "global": global_arr})

            # 若前端没有传历史（兼容旧版），退化为单帧重复堆叠
            if not frames:
                frames = [build_observation(state)]

            obs = stack_observations(frames)
            mask = build_action_mask(state.board, state.layout)

            # ── 2-step lookahead 选择动作 ─────────────────────────
            action = lookahead_select(state, obs, mask)

            if mask[action] < 0.5:
                valid = np.where(mask > 0)[0]
                if len(valid) == 0:
                    self._json_response(400, {"error": "no valid actions"})
                    return
                action = int(valid[0])

            move = decode_action(action)
            if move is None:
                self._json_response(400, {"error": "invalid action decode"})
                return

            resp = {"action": action, "type": move["type"],
                    "reason": f"RL 策略（2步前瞻，动作 #{action}）"}
            if move["type"] == "pop":
                resp["r"] = move["r"]
                resp["c"] = move["c"]
            else:
                resp["from"] = move["from"]
                resp["to"] = move["to"]
            self._json_response(200, resp)
        except Exception as exc:
            self._json_response(500, {"error": str(exc)})

    def log_message(self, format: str, *args) -> None:
        return


def main() -> None:
    global MODEL, NON_DETERMINISTIC
    parser = argparse.ArgumentParser(description="RL 推理服务")
    parser.add_argument(
        "--model",
        type=str,
        default="runs/ppo_match3/final_model",
        help="模型路径（可带或不带 .zip）",
    )
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument(
        "--stochastic",
        action="store_true",
        help="启用非确定性采样推理，降低重复动作概率",
    )
    args = parser.parse_args()

    model_path = args.model
    if not model_path.endswith(".zip"):
        model_path = model_path + ".zip"
    if not os.path.isabs(model_path):
        model_path = os.path.join(ROOT, model_path)
    if not os.path.isfile(model_path):
        print(f"错误：找不到模型文件 {model_path}")
        sys.exit(1)

    MODEL = MaskablePPO.load(model_path)
    NON_DETERMINISTIC = bool(args.stochastic)
    print(f"已加载模型: {model_path}")
    print(f"推理模式: 全评估对称 2-step lookahead (gamma={LOOKAHEAD_GAMMA}, W={LOOKAHEAD_REWARD_WEIGHT})")
    server = HTTPServer((args.host, args.port), PredictHandler)
    print(f"推理服务: http://{args.host}:{args.port}")
    print("  GET  /health")
    print("  POST /predict")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n已停止")


if __name__ == "__main__":
    main()
