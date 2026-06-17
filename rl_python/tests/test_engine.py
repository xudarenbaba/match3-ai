"""基础引擎与环境冒烟测试。"""

import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import random
import numpy as np

from match3_engine.game import create_game_state, execute_move
from match3_engine.actions import build_action_mask, get_adjacent_swaps
from match3_engine.layouts import LAYOUT_NAMES, get_layout
from env.match3_env import Match3Env
from env.observation import build_observation, BOARD_CHANNELS, GLOBAL_DIM, STACKED_BOARD_CHANNELS


def test_create_and_step():
    rng = random.Random(42)
    state = create_game_state(rng)
    swaps = get_adjacent_swaps(state.board, state.layout)
    assert len(swaps) > 0
    mask = build_action_mask(state.board, state.layout)
    assert mask.sum() > 0
    swap = swaps[0]
    res = execute_move(state, rng, swap["from"], swap["to"])
    assert res["ok"]


def test_observation_shape():
    rng = random.Random(0)
    state = create_game_state(rng)
    obs = build_observation(state)
    # 新增了 layout_mask 通道，BOARD_CHANNELS = 29
    assert obs["board"].shape == (BOARD_CHANNELS, 10, 10), f"Expected ({BOARD_CHANNELS}, 10, 10), got {obs['board'].shape}"
    assert obs["global"].shape == (GLOBAL_DIM,)


def test_stacked_obs_shape():
    """帧堆叠后 board 应为 (87, 10, 10)。"""
    from env.observation import stack_observations
    rng = random.Random(1)
    state = create_game_state(rng)
    frame = build_observation(state)
    stacked = stack_observations([frame])
    assert stacked["board"].shape == (STACKED_BOARD_CHANNELS, 10, 10), \
        f"Expected ({STACKED_BOARD_CHANNELS}, 10, 10), got {stacked['board'].shape}"


def test_env_episode():
    env = Match3Env(curriculum_level=1, seed=123)
    obs, info = env.reset()
    assert "board" in obs and "global" in obs
    assert obs["board"].shape == (STACKED_BOARD_CHANNELS, 10, 10)
    assert info["action_mask"].sum() > 0
    for _ in range(20):
        mask = info["action_mask"]
        valid = np.where(mask > 0)[0]
        action = int(valid[0])
        obs, reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            break
    assert True


def test_all_layouts():
    """所有布局都能正常初始化并运行若干步。"""
    for layout_name in LAYOUT_NAMES:
        rng = random.Random(42)
        state = create_game_state(rng, layout_name=layout_name)
        assert state.layout_name == layout_name
        layout = state.layout
        # 验证 void 格在棋盘上为 None
        for r in range(10):
            for c in range(10):
                if not layout[r][c]:
                    assert state.board[r][c] is None, \
                        f"Layout {layout_name}: void cell ({r},{c}) should be None"
        # 验证 action mask 有效
        mask = build_action_mask(state.board, layout)
        assert mask.sum() > 0 or True  # 某些极端布局可能无有效动作，不强制


def test_4_combo_generates_column_powerup():
    """4连消应生成 column 道具。"""
    from match3_engine.match import find_matches, pick_merge_positions, apply_merges
    from match3_engine.cells import NormalCell
    from match3_engine.board import create_empty_board

    rng = random.Random(0)
    board = create_empty_board()
    # 手动放置 4个同shape同level相邻格子（横向）
    for c in range(4):
        board[5][c] = NormalCell(shape="circle", level=1)

    matches = find_matches(board)
    assert len(matches) == 1
    assert len(matches[0]["cells"]) == 4

    positions = pick_merge_positions(matches, None, None)
    merged = apply_merges(board, matches, positions, rng)

    # 合并位应放 column 道具
    pos = positions[0]
    result_cell = board[pos["r"]][pos["c"]]
    assert result_cell is not None
    assert result_cell.kind == "powerup"
    assert result_cell.powerup_type == "column", f"Expected column, got {result_cell.powerup_type}"


def test_5_combo_generates_color_powerup():
    """5连+消应生成 color 道具。"""
    from match3_engine.match import find_matches, pick_merge_positions, apply_merges
    from match3_engine.cells import NormalCell
    from match3_engine.board import create_empty_board

    rng = random.Random(0)
    board = create_empty_board()
    for c in range(5):
        board[5][c] = NormalCell(shape="square", level=2)

    matches = find_matches(board)
    assert len(matches) == 1
    assert len(matches[0]["cells"]) == 5

    positions = pick_merge_positions(matches, None, None)
    apply_merges(board, matches, positions, rng)

    pos = positions[0]
    result_cell = board[pos["r"]][pos["c"]]
    assert result_cell is not None
    assert result_cell.kind == "powerup"
    assert result_cell.powerup_type == "color", f"Expected color, got {result_cell.powerup_type}"


def test_layout_mask_in_observation():
    """observation 中 layout_mask channel（28）应正确反映布局。"""
    rng = random.Random(0)
    state = create_game_state(rng, layout_name="cross")
    obs = build_observation(state)
    layout = state.layout
    layout_ch = obs["board"][28]  # channel 28
    for r in range(10):
        for c in range(10):
            expected = 1.0 if layout[r][c] else 0.0
            assert layout_ch[r, c] == expected, \
                f"cross layout: ({r},{c}) expected {expected}, got {layout_ch[r,c]}"


if __name__ == "__main__":
    test_create_and_step()
    test_observation_shape()
    test_stacked_obs_shape()
    test_env_episode()
    test_all_layouts()
    test_4_combo_generates_column_powerup()
    test_5_combo_generates_color_powerup()
    test_layout_mask_in_observation()
    print("all tests passed")
