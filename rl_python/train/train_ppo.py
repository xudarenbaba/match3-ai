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
from train.features import Match3CnnExtractor


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
    parser.add_argument("--timesteps", type=int, default=2_500_000, help="总训练步数（CNN 需更多步收敛，推荐 ≥2.5M）")
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
        features_extractor_class=Match3CnnExtractor,
        features_extractor_kwargs=dict(features_dim=256),
        net_arch=dict(pi=[128], vf=[128]),
    )

    model = MaskablePPO(
        "MultiInputPolicy",
        env,
        verbose=1,
        tensorboard_log=os.path.join(args.save_dir, "tb"),
        learning_rate=1e-4,  # 3e-4 → 1e-4：更小 LR，让 value head 精细收敛而不震荡
        n_steps=2048,
        batch_size=512,
        gamma=0.99,
        ent_coef=0.005,  # 0.01 → 0.005：策略进一步减少探索，利用已有知识收敛
        vf_coef=2.0,     # 1.0 → 2.0：大幅加强 value head 学习权重，是核心改动
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
