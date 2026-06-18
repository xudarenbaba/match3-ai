#!/usr/bin/env python3
"""使用 MaskablePPO 训练消消乐策略。"""

from __future__ import annotations

import argparse
import os
import sys

# 将 rl_python 根目录加入 path
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import gymnasium as gym
import numpy as np
from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv

from env.match3_env import Match3Env


def mask_fn(env: gym.Env) -> np.ndarray:
    return env.unwrapped.action_masks()


def make_env(curriculum_level: int, seed: int = 0):
    def _init():
        env = Match3Env(curriculum_level=curriculum_level, seed=seed)
        env = ActionMasker(env, mask_fn)
        return env

    return _init


def main():
    parser = argparse.ArgumentParser(description="训练消消乐 RL 策略 (MaskablePPO)")
    parser.add_argument("--curriculum", type=int, default=3, choices=[1, 2, 3], help="课程难度 1-3")
    parser.add_argument("--timesteps", type=int, default=500_000, help="总训练步数")
    parser.add_argument("--n-envs", type=int, default=8, help="并行环境数")
    parser.add_argument("--save-dir", type=str, default="runs/ppo_match3", help="模型与日志目录")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(os.path.join(args.save_dir, "checkpoints"), exist_ok=True)

    env = make_vec_env(
        make_env(args.curriculum, args.seed),
        n_envs=args.n_envs,
        vec_env_cls=SubprocVecEnv if args.n_envs > 1 else None,
        seed=args.seed,
    )

    eval_env = ActionMasker(Match3Env(curriculum_level=args.curriculum, seed=args.seed + 1000), mask_fn)

    policy_kwargs = dict(
        net_arch=dict(pi=[256, 128], vf=[256, 128]),
    )

    model = MaskablePPO(
        "MultiInputPolicy",
        env,
        verbose=1,
        tensorboard_log=os.path.join(args.save_dir, "tb"),
        learning_rate=3e-4,
        n_steps=2048,    # 1024 → 2048：更长的 GAE 窗口，win_bonus 信号传播更远
        batch_size=512,  # 256 → 512：配合 n_steps 增大
        gamma=0.99,
        ent_coef=0.03,   # 0.01 → 0.03：增加探索，避免早收敛到次优策略
        policy_kwargs=policy_kwargs,
        seed=args.seed,
    )

    checkpoint_cb = CheckpointCallback(
        save_freq=max(10_000 // args.n_envs, 1),
        save_path=os.path.join(args.save_dir, "checkpoints"),
        name_prefix="ppo_match3",
    )
    eval_cb = EvalCallback(
        eval_env,
        best_model_save_path=os.path.join(args.save_dir, "best"),
        log_path=os.path.join(args.save_dir, "eval"),
        eval_freq=max(20_000 // args.n_envs, 1),
        n_eval_episodes=20,
        deterministic=True,
    )

    model.learn(total_timesteps=args.timesteps, callback=[checkpoint_cb, eval_cb])
    final_path = os.path.join(args.save_dir, "final_model")
    model.save(final_path)
    print(f"训练完成，模型已保存至: {final_path}.zip")


if __name__ == "__main__":
    main()
