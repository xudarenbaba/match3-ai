#!/usr/bin/env python3
"""加载训练好的 MaskablePPO 模型，为浏览器提供走棋推理 API。"""

from __future__ import annotations

import argparse
import json
import os
import sys
from http.server import BaseHTTPRequestHandler, HTTPServer

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import numpy as np
from sb3_contrib import MaskablePPO

from env.observation import build_observation
from match3_engine.actions import build_action_mask, decode_action
from serve.state_codec import json_to_game_state

MODEL: MaskablePPO | None = None
NON_DETERMINISTIC = False


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
            obs = build_observation(state)
            mask = build_action_mask(state.board)
            action, _ = MODEL.predict(obs, action_masks=mask.astype(bool), deterministic=not NON_DETERMINISTIC)
            action = int(action)

            if mask[action] < 0.5:
                valid = np.where(mask > 0)[0]
                if len(valid) == 0:
                    self._json_response(400, {"error": "no valid actions"})
                    return
                action = int(valid[0])

            if state.last_action >= 0 and action == state.last_action:
                valid = np.where(mask > 0)[0]
                alternatives = [int(a) for a in valid if int(a) != state.last_action]
                if alternatives:
                    action = int(alternatives[0])

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
                    "reason": f"RL 策略（动作 #{action}）",
                },
            )
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
