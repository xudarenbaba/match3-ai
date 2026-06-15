from __future__ import annotations

import random
from typing import Optional, Tuple

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from match3_engine.actions import build_action_mask, swap_from_action, decode_action
from match3_engine.constants import MAX_ACTIONS, MAX_STEPS, TASK_TARGET, SHAPES, ROWS
from match3_engine.game import create_game_state, execute_move, snapshot_state
from env.observation import build_observation, BOARD_CHANNELS, GLOBAL_DIM
from env.reward import compute_reward


class Match3Env(gym.Env):
    """消消乐 Gymnasium 环境，支持课程学习。"""

    metadata = {"render_modes": ["human"]}

    def __init__(self, curriculum_level: int = 3, seed: Optional[int] = None):
        super().__init__()
        self.curriculum_level = curriculum_level
        self._rng = random.Random(seed)
        self.state = None

        self.observation_space = spaces.Dict(
            {
                "board": spaces.Box(0.0, 1.0, shape=(BOARD_CHANNELS, ROWS, ROWS), dtype=np.float32),
                "global": spaces.Box(0.0, 1.0, shape=(GLOBAL_DIM,), dtype=np.float32),
            }
        )
        self.action_space = spaces.Discrete(MAX_ACTIONS)

    def _curriculum_options(self) -> dict:
        if self.curriculum_level <= 1:
            return {"total_steps": 150, "task_target": 2, "freeze": False}
        if self.curriculum_level == 2:
            return {"total_steps": 120, "task_target": 3, "freeze": True}
        return {"total_steps": MAX_STEPS, "task_target": TASK_TARGET, "freeze": True}

    def _target_shapes(self) -> list:
        copy_shapes = SHAPES.copy()
        self._rng.shuffle(copy_shapes)
        if self.curriculum_level <= 1:
            return [copy_shapes[0]]
        return copy_shapes[:2]

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        if seed is not None:
            self._rng = random.Random(seed)
        opts = self._curriculum_options()
        self.state = create_game_state(
            self._rng,
            total_steps=opts["total_steps"],
            task_target_shapes=self._target_shapes(),
            task_target=opts["task_target"],
            freeze=opts["freeze"],
        )
        obs = build_observation(self.state)
        mask = build_action_mask(self.state.board)
        return obs, {"action_mask": mask}

    def step(self, action: int) -> Tuple[dict, float, bool, bool, dict]:
        prev = snapshot_state(self.state)
        swap = swap_from_action(self.state.board, int(action))
        if swap is None:
            mask = build_action_mask(self.state.board)
            valid_idx = np.where(mask > 0)[0]
            if len(valid_idx) == 0:
                obs = build_observation(self.state)
                return obs, -1.0, True, False, {"action_mask": mask, "invalid": True}
            action = int(valid_idx[0])
            swap = decode_action(action)
            if swap is None:
                obs = build_observation(self.state)
                return obs, -1.0, True, False, {"action_mask": mask, "invalid": True}

        move = execute_move(self.state, self._rng, swap["from"], swap["to"])
        result = move["result"]
        result["action_index"] = int(action)
        reward = compute_reward(prev, result, self.state)

        obs = build_observation(self.state)
        mask = build_action_mask(self.state.board)
        info = {
            "action_mask": mask,
            "won": self.state.won,
            "score": self.state.score,
            "steps_used": self.state.steps_used,
        }
        return obs, reward, self.state.over, False, info

    def action_masks(self) -> np.ndarray:
        return build_action_mask(self.state.board).astype(bool)
