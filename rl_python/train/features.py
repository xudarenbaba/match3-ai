"""自定义特征提取器：用小卷积处理 10×10 棋盘的空间结构。

默认 MultiInputPolicy 会把 board (C,10,10) 直接 Flatten 成 C*100 维向量，
丢失空间结构，难以识别「局部 3 连/4 连机会」这类模式。

Match3CnnExtractor 对 board 用两层 3×3 卷积（padding=1 保持 10×10 尺寸）
提取空间特征，再与 global 向量拼接。
"""

from __future__ import annotations

import gymnasium as gym
import torch
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


class Match3CnnExtractor(BaseFeaturesExtractor):
    """
    Dict 观测特征提取器：
      board  (C,10,10) → Conv3x3(C→32) → ReLU → Conv3x3(32→64) → ReLU → Flatten → Linear
      global (G,)      → 原样拼接
    输出维度 = features_dim（board 分支） + G（global 分支）
    """

    def __init__(self, observation_space: gym.spaces.Dict, features_dim: int = 256):
        # features_dim 这里指 board 卷积分支输出维度；总输出在下方 _features_dim 重设
        super().__init__(observation_space, features_dim=1)

        board_space = observation_space["board"]
        global_space = observation_space["global"]
        c, h, w = board_space.shape  # (C,10,10)
        g_dim = int(global_space.shape[0])

        self.cnn = nn.Sequential(
            nn.Conv2d(c, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Flatten(),
        )

        # 动态推断卷积输出维度
        with torch.no_grad():
            sample = torch.zeros(1, c, h, w)
            n_flatten = self.cnn(sample).shape[1]

        self.linear = nn.Sequential(
            nn.Linear(n_flatten, features_dim),
            nn.ReLU(),
        )

        # 最终输出维度 = board 分支 features_dim + global 原始维度
        self._features_dim = features_dim + g_dim

    def forward(self, observations: dict) -> torch.Tensor:
        board = observations["board"]
        global_vec = observations["global"]
        board_feat = self.linear(self.cnn(board))
        return torch.cat([board_feat, global_vec], dim=1)
