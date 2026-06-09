"""基础引擎与环境冒烟测试。"""

import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import random
import numpy as np

from match3_engine.game import create_game_state, execute_move
from match3_engine.actions import build_action_mask, get_adjacent_swaps
from env.match3_env import Match3Env
from env.observation import build_observation, BOARD_CHANNELS, GLOBAL_DIM


def test_create_and_step():
    rng = random.Random(42)
    state = create_game_state(rng)
    swaps = get_adjacent_swaps(state.board)
    assert len(swaps) > 0
    mask = build_action_mask(state.board)
    assert mask.sum() > 0
    swap = swaps[0]
    res = execute_move(state, rng, swap["from"], swap["to"])
    assert res["ok"]


def test_observation_shape():
    rng = random.Random(0)
    state = create_game_state(rng)
    obs = build_observation(state)
    assert obs["board"].shape == (BOARD_CHANNELS, 10, 10)
    assert obs["global"].shape == (GLOBAL_DIM,)


def test_env_episode():
    env = Match3Env(curriculum_level=1, seed=123)
    obs, info = env.reset()
    assert "board" in obs and "global" in obs
    assert info["action_mask"].sum() > 0
    for _ in range(20):
        mask = info["action_mask"]
        valid = np.where(mask > 0)[0]
        action = int(valid[0])
        obs, reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            break
    assert True


if __name__ == "__main__":
    test_create_and_step()
    test_observation_shape()
    test_env_episode()
    print("all tests passed")
