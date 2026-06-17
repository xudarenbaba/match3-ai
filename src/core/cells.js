import { SHAPES, POWERUP_TYPES, POWERUP_SPAWN_RATE } from './constants.js';

export function createNormalCell(shape = null, level = null) {
  if (shape == null) {
    shape = SHAPES[Math.floor(Math.random() * SHAPES.length)];
  }
  if (level == null) {
    const r = Math.random();
    level = r < 0.6 ? 1 : r < 0.9 ? 2 : 3;
  }
  return { kind: 'normal', shape, level, frozen: false };
}

export function createPowerupCell(shape = null, powerupType = null) {
  if (shape == null) {
    shape = SHAPES[Math.floor(Math.random() * SHAPES.length)];
  }
  if (powerupType == null) {
    powerupType = POWERUP_TYPES[Math.floor(Math.random() * POWERUP_TYPES.length)];
  }
  return { kind: 'powerup', shape, powerupType, level: 3, frozen: false };
}

export function createCell() {
  if (Math.random() < POWERUP_SPAWN_RATE) return createPowerupCell();
  // 顶部落下的新格子只能是等级 1，形状随机
  return createNormalCell(null, 1);
}

// 初始化棋盘时用：只生成等级 1 的普通格，不生成道具
export function createInitialCell() {
  return createNormalCell(null, 1);
}

export function isPowerup(cell) {
  return Boolean(cell && cell.kind === 'powerup');
}

export function isFrozen(cell) {
  return Boolean(cell && cell.frozen);
}
