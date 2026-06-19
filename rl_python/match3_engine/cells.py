from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Optional
import random

from .constants import SHAPES, POWERUP_TYPES, POWERUP_SPAWN_RATE


@dataclass
class NormalCell:
    kind: str = "normal"
    shape: str = "circle"
    level: int = 1
    frozen: bool = False


@dataclass
class PowerCell:
    kind: str = "powerup"
    shape: str = "circle"
    powerup_type: str = "column"
    level: int = 3
    frozen: bool = False


Cell = NormalCell | PowerCell


def create_normal_cell(rng: random.Random, shape: Optional[str] = None, level: Optional[int] = None) -> NormalCell:
    if shape is None:
        shape = rng.choice(SHAPES)
    if level is None:
        r = rng.random()
        level = 1 if r < 0.6 else (2 if r < 0.9 else 3)
    return NormalCell(shape=shape, level=level)


def create_powerup_cell(
    rng: random.Random, shape: Optional[str] = None, powerup_type: Optional[str] = None
) -> PowerCell:
    if shape is None:
        shape = rng.choice(SHAPES)
    if powerup_type is None:
        powerup_type = rng.choice(POWERUP_TYPES)
    return PowerCell(shape=shape, powerup_type=powerup_type)


def create_cell(rng: random.Random) -> Cell:
    """顶部掉落补充用：2.5% 概率道具，否则只生成等级 1 普通格（对齐 JS createCell）。"""
    if rng.random() < POWERUP_SPAWN_RATE:
        return create_powerup_cell(rng)
    return create_normal_cell(rng, level=1)


def create_initial_cell(rng: random.Random) -> Cell:
    """初始化棋盘用：只生成等级 1 的普通格，不生成道具（对齐 JS createInitialCell）。"""
    return create_normal_cell(rng, level=1)


def is_powerup(cell: Optional[Cell]) -> bool:
    return bool(cell and cell.kind == "powerup")


def is_frozen(cell: Optional[Cell]) -> bool:
    return bool(cell and cell.frozen)


def clone_cell(cell: Optional[Cell]) -> Optional[Cell]:
    if cell is None:
        return None
    return replace(cell)
