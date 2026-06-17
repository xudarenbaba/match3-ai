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
from match3_engine.actions import build_action_mask, decode_action
from match3_engine.game import snapshot_state, execute_move
from serve.state_codec import json_to_game_state

MODEL: MaskablePPO | None = None
NON_DETERMINISTIC = False

# lookahead 超参数
LOOKAHEAD_TOP_K = 8      # 第一层展开候选动作数
LOOKAHEAD_GAMMA = 0.99   # 折扣因子（与训练保持一致）
_RNG = random.Random()   # 仅供 lookahead 模拟使用


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


def lookahead_select(state, obs: dict, mask: np.ndarray) -> int:
    """
    2-step lookahead 动作选择：
      - 获取当前 top-K 候选动作（按 action logits 降序）
      - 对每个候选模拟一步，计算即时奖励 + gamma * V(s')
      - 返回得分最高的动作
    """
    # ── 拿 action distribution logits ──────────────────────────────
    board_t = torch.tensor(obs["board"][None], dtype=torch.float32)
    global_t = torch.tensor(obs["global"][None], dtype=torch.float32)
    obs_tensor = {
        "board": board_t.to(MODEL.device),
        "global": global_t.to(MODEL.device),
    }
    # get_distribution 期望 numpy 数组，shape (n_envs, n_actions)
    mask_np = mask.astype(bool)[None]  # (1, 180)

    with torch.no_grad():
        dist = MODEL.policy.get_distribution(obs_tensor, action_masks=mask_np)
        logits = dist.distribution.logits.squeeze(0).cpu().numpy()  # (180,)

    # 只考虑有效动作，取 top-K
    valid_indices = np.where(mask > 0)[0]
    if len(valid_indices) == 0:
        # 无有效动作，回退到模型直接预测
        action, _ = MODEL.predict(obs, action_masks=mask.astype(bool), deterministic=True)
        return int(action)

    valid_logits = logits[valid_indices]
    k = min(LOOKAHEAD_TOP_K, len(valid_indices))
    top_local = np.argsort(valid_logits)[::-1][:k]
    top_k_actions = valid_indices[top_local]

    best_action = int(top_k_actions[0])
    best_score = -float("inf")

    for action_idx in top_k_actions:
        swap = decode_action(int(action_idx))
        if swap is None:
            continue

        # ── 模拟执行一步 ──────────────────────────────────────────
        sim_state = snapshot_state(state)
        sim_state.last_action = state.last_action
        exec_result = execute_move(sim_state, _RNG, swap["from"], swap["to"])
        if not exec_result["ok"]:
            continue
        step_result = exec_result["result"]
        step_result["action_index"] = int(action_idx)

        # ── 即时奖励 ──────────────────────────────────────────────
        r_imm = _immediate_reward(state, step_result, sim_state)

        # ── 下一状态的价值估计 ────────────────────────────────────
        next_obs_single = build_observation(sim_state)
        # 用当前帧堆叠：把新帧追加到旧帧历史末尾
        next_obs = stack_observations([obs_single_from_stacked(obs), next_obs_single])
        v_next = _get_value(next_obs)

        total = r_imm + LOOKAHEAD_GAMMA * v_next

        if total > best_score:
            best_score = total
            best_action = int(action_idx)

    return best_action


def obs_single_from_stacked(stacked_obs: dict) -> dict:
    """从堆叠 obs 中取出最新一帧（最后 BOARD_CHANNELS 个通道）。"""
    board_stacked = stacked_obs["board"]  # (87, 10, 10)
    board_last = board_stacked[-BOARD_CHANNELS:]  # (29, 10, 10)
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

            swap = decode_action(action)
            if swap is None:
                self._json_response(400, {"error": "invalid action decode"})
                return

            self._json_response(
                200,
                {
                    "action": action,
                    "from": swap["from"],
                    "to": swap["to"],
                    "reason": f"RL 策略（2步前瞻，动作 #{action}）",
                },
            )
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
