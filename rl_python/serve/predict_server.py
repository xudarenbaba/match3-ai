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
from match3_engine.actions import build_action_mask, decode_action, POP_OFFSET
from match3_engine.game import snapshot_state, execute_move
from serve.state_codec import json_to_game_state

MODEL: MaskablePPO | None = None
NON_DETERMINISTIC = False

# lookahead 超参数
LOOKAHEAD_TOP_K = 12         # 第一层展开候选动作数
LOOKAHEAD_GAMMA = 0.99       # 折扣因子（与训练保持一致）
# 即时奖励权重：value head 估值量级远大于单步即时奖励，若直接相加会被 V 噪声淹没。
# 放大即时奖励权重，让「4连>3连」等正确的单步信号主导决策。
LOOKAHEAD_REWARD_WEIGHT = 8.0
# lookahead 模拟会触发 gravity 随机补充新格子。用固定种子让同一局面的评分可复现、
# 推理结果稳定（否则同一局面多次请求可能给出不同动作）。
LOOKAHEAD_SEED = 12345


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


def _best_followup_value(after_state, frame_t: dict, frame_t1: dict) -> float:
    """枚举 after_state 的所有有效交换，返回最优 1-step lookahead 值
    `max_swap( W·r' + γ·V(s'') )`。frame_t/frame_t1 是 after_state 的前两帧历史。

    这是 Bellman 最优 V(s') ≈ max_a' Q(s', a') 的真实估计，用于替代 value head
    对 after_state 的直接估值——避免 value 对某些局面（尤其捏爆后局面）的高估。
    第二层只考虑交换，不再套捏爆，避免捏爆套捏爆。
    """
    fmask = build_action_mask(after_state.board, after_state.layout)
    best = -float("inf")
    for sa in np.where(fmask > 0)[0]:
        if sa >= POP_OFFSET:
            continue
        sw = decode_action(int(sa))
        if sw is None:
            continue
        sr = random.Random(LOOKAHEAD_SEED)
        ns = snapshot_state(after_state)
        ns.last_action = after_state.last_action
        er = execute_move(ns, sr, sw)
        if not er["ok"]:
            continue
        er["result"]["action_index"] = int(sa)
        r = _immediate_reward(after_state, er["result"], ns)
        v = _get_value(stack_observations([frame_t, frame_t1, build_observation(ns)]))
        cand = LOOKAHEAD_REWARD_WEIGHT * r + LOOKAHEAD_GAMMA * v
        if cand > best:
            best = cand
    if best == -float("inf"):
        # 无任何有效交换：退化为 after_state 自身 value
        best = _get_value(stack_observations([frame_t, frame_t1, frame_t1]))
    return best


def _score_action(state, prev_frames: list, f_curr: dict, move: dict, action_idx: int) -> float:
    """统一的 2-step 评分（交换与捏爆对称）：

        score = W·r(s,a) + γ · max_a'( W·r(s',a') + γ·V(s'') )

    交换与捏爆都展开下一层，用同一基准比较。捏爆无即时收益（r 含 pop_cost 为负），
    只有当「捏爆后最优下一步」明显优于「直接交换及其后续」时才会胜出——从而实现
    「捏爆必须为下一步创造大消除/道具机会」的语义，杜绝随意捏爆。
    """
    sim_rng = random.Random(LOOKAHEAD_SEED)
    ss = snapshot_state(state)
    ss.last_action = state.last_action
    er = execute_move(ss, sim_rng, move)
    if not er["ok"]:
        return -float("inf")
    er["result"]["action_index"] = action_idx
    r = _immediate_reward(state, er["result"], ss)  # 交换得分 或 捏爆成本(负)
    f_next = build_observation(ss)
    v_la = _best_followup_value(ss, f_curr, f_next)
    return LOOKAHEAD_REWARD_WEIGHT * r + LOOKAHEAD_GAMMA * v_la


def lookahead_select(state, obs: dict, mask: np.ndarray) -> int:
    """
    统一 2-step lookahead 动作选择：
      所有候选（交换/捏爆）均评估 `W·r + γ·max_followup`，对称公平。
      捏爆只有在「捏爆后下一步真能大消除/触发道具」时才会胜出。
    """
    # ── 拿 action distribution logits ──────────────────────────────
    board_t = torch.tensor(obs["board"][None], dtype=torch.float32)
    global_t = torch.tensor(obs["global"][None], dtype=torch.float32)
    obs_tensor = {
        "board": board_t.to(MODEL.device),
        "global": global_t.to(MODEL.device),
    }
    mask_np = mask.astype(bool)[None]

    with torch.no_grad():
        dist = MODEL.policy.get_distribution(obs_tensor, action_masks=mask_np)
        logits = dist.distribution.logits.squeeze(0).cpu().numpy()

    valid_indices = np.where(mask > 0)[0]
    if len(valid_indices) == 0:
        action, _ = MODEL.predict(obs, action_masks=mask.astype(bool), deterministic=True)
        return int(action)

    valid_logits = logits[valid_indices]
    k = min(LOOKAHEAD_TOP_K, len(valid_indices))
    top_local = np.argsort(valid_logits)[::-1][:k]
    top_k_actions = valid_indices[top_local]

    prev_frames = last_two_frames(obs)  # [f_{t-1}, f_t]
    f_curr = prev_frames[1]             # f_t

    best_action = int(top_k_actions[0])
    best_score = -float("inf")

    for action_idx in top_k_actions:
        move = decode_action(int(action_idx))
        if move is None:
            continue
        total = _score_action(state, prev_frames, f_curr, move, int(action_idx))
        if total > best_score:
            best_score = total
            best_action = int(action_idx)

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
    global MODEL, NON_DETERMINISTIC, LOOKAHEAD_TOP_K
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
    parser.add_argument(
        "--top-k",
        type=int,
        default=LOOKAHEAD_TOP_K,
        help=f"lookahead 展开的候选动作数（默认 {LOOKAHEAD_TOP_K}）",
    )
    args = parser.parse_args()

    LOOKAHEAD_TOP_K = args.top_k

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
    print(f"推理模式: 2-step lookahead (top-k={LOOKAHEAD_TOP_K}, gamma={LOOKAHEAD_GAMMA})")
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
