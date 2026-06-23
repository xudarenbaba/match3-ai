export const ROWS = 10;
export const COLS = 10;
export const SHAPES = ['circle', 'square', 'triangle', 'star'];
export const SHAPE_NAMES = {
  circle: '圆形',
  square: '方块',
  triangle: '三角形',
  star: '五角星',
};
export const POWERUP_TYPES = ['row', 'column', 'bomb', 'color'];
export const MAX_STEPS = 100;
export const INITIAL_FROZEN_RATIO = 0.12;
export const POWERUP_SPAWN_RATE = 0.025;
export const TASK_TARGET = 4;
// 动作空间：0-89 横向交换 + 90-179 纵向交换 + 180-279 捏爆(pop)
export const SWAP_ACTIONS = 180;
export const MAX_ACTIONS = 280;

/** @typedef {{ kind: 'normal', shape: string, level: number, frozen: boolean }} NormalCell */
/** @typedef {{ kind: 'powerup', shape: string, powerupType: 'column'|'bomb'|'color', level: 3, frozen: boolean }} PowerCell */
/** @typedef {NormalCell | PowerCell} Cell */
/** @typedef {Cell | null} BoardCell */
