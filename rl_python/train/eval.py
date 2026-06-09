#!/usr/bin/env python3
"""评估训练好的模型或随机策略基线。"""

from __future__ import annotations

import argparse
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import numpy as np
from sb3_contrib import MaskablePPO
from env.match3_env import Match3Env


def run_episodes(env: Match3Env, model=None, episodes: int = 50) -> dict:
    wins = 0
    scores = []
    returns = []

    for ep in range(episodes):
        obs, info = env.reset(seed=ep)
        done = False
        ep_return = 0.0
        while not done:
            if model is None:
                mask = info["action_mask"]
                valid = np.where(mask > 0)[0]
                action = int(np.random.choice(valid))
            else:
                action, _ = model.predict(obs, action_masks=env.action_masks(), deterministic=True)
                action = int(action)

            obs, reward, terminated, truncated, info = env.step(action)
            ep_return += reward
            done = terminated or truncated

        wins += int(info.get("won", False))
        scores.append(info.get("score", 0))
        returns.append(ep_return)

    return {
        "episodes": episodes,
        "win_rate": wins / episodes,
        "avg_score": float(np.mean(scores)),
        "avg_return": float(np.mean(returns)),
    }


def main():
    parser = argparse.ArgumentParser(description="评估消消乐 RL 模型")
    parser.add_argument("--model", type=str, default=None, help="模型路径（不含 .zip）")
    parser.add_argument("--curriculum", type=int, default=3, choices=[1, 2, 3])
    parser.add_argument("--episodes", type=int, default=50)
    args = parser.parse_args()

    env = Match3Env(curriculum_level=args.curriculum)

    if args.model:
        model = MaskablePPO.load(args.model)
        label = args.model
    else:
        model = None
        label = "random"

    stats = run_episodes(env, model, args.episodes)
    print(f"策略: {label}")
    print(f"局数: {stats['episodes']}")
    print(f"胜率: {stats['win_rate']:.1%}")
    print(f"平均分: {stats['avg_score']:.1f}")
    print(f"平均回报: {stats['avg_return']:.2f}")


if __name__ == "__main__":
    main()
