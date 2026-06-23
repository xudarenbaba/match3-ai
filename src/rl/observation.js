import { ROWS, COLS, SHAPES, POWERUP_TYPES } from '../core/constants.js';

// 单帧 board 通道数（28 形状/道具/冻结 + 1 row道具 + 1 layout_mask = 30）
// 通道布局与 rl_python/env/observation.py 严格对齐：
//   0-11  : shape×level 一热（4 shape × 3 level）
//   12-15 : 道具类型（row/column/bomb/color）
//   16    : 冻结标志
//   17-28 : 目标 shape×level 一热（仅目标 shape）
//   29    : layout_mask
const BOARD_CHANNELS = 30;
// 帧堆叠数，与 Python 侧 FRAME_STACK 保持一致
export const FRAME_STACK = 3;
// 堆叠后的 board 通道数
export const STACKED_BOARD_CHANNELS = BOARD_CHANNELS * FRAME_STACK; // 90
// global: 原 15 维 + 解冻任务进度(15) + 是否有解冻任务(16) = 17
const GLOBAL_DIM = 17;

export const OBS_BOARD_CHANNELS = STACKED_BOARD_CHANNELS;
export const OBS_GLOBAL_DIM = GLOBAL_DIM;

/**
 * 构建单帧棋盘观测。
 * 通道说明（30 个）：
 *   0-11  : shape×level 一热编码（4 shape × 3 level）
 *   12-15 : 道具类型（row/column/bomb/color）
 *   16    : 冻结标志
 *   17-28 : 目标 shape×level 一热（仅目标 shape）
 *   29    : layout_mask（1=活跃格，0=void 格）
 */
export function buildObservation(state) {
  const board = new Array(BOARD_CHANNELS * ROWS * COLS).fill(0);
  const targetSet = new Set(state.targetShapes);
  const layout = state.layout || null;

  for (let r = 0; r < ROWS; r++) {
    for (let c = 0; c < COLS; c++) {
      const base = r * COLS + c;

      // 通道 29：layout_mask
      const isActive = !layout || layout[r][c];
      if (isActive) {
        board[29 * ROWS * COLS + base] = 1;
      } else {
        continue; // void 格其他通道均为 0
      }

      const cell = state.board[r][c];
      if (!cell) continue;

      if (cell.kind === 'normal') {
        const shapeIdx = SHAPES.indexOf(cell.shape);
        if (shapeIdx >= 0 && cell.level >= 1 && cell.level <= 3) {
          board[(shapeIdx * 3 + (cell.level - 1)) * ROWS * COLS + base] = 1;
        }
      } else if (cell.kind === 'powerup') {
        const pIdx = POWERUP_TYPES.indexOf(cell.powerupType);
        if (pIdx >= 0) {
          board[(12 + pIdx) * ROWS * COLS + base] = 1;
        }
      }

      if (cell.frozen) {
        board[16 * ROWS * COLS + base] = 1;
      }

      if (targetSet.has(cell.shape)) {
        const si = SHAPES.indexOf(cell.shape);
        if (si >= 0) {
          if (cell.kind === 'normal' && cell.level >= 1 && cell.level <= 3) {
            board[(17 + si * 3 + (cell.level - 1)) * ROWS * COLS + base] = 1;
          } else if (cell.kind === 'powerup') {
            board[(17 + si * 3 + 2) * ROWS * COLS + base] = 1;
          }
        }
      }
    }
  }

  const stepsLeft = Math.max(0, state.totalSteps - state.stepsUsed);
  const global = new Array(GLOBAL_DIM).fill(0);
  global[0] = state.stepsUsed / state.totalSteps;
  global[1] = stepsLeft / state.totalSteps;
  global[2] = Math.min(1, state.score / 5000);
  global[3] = Math.min(1, state.chainScoreTotal / 3000);
  SHAPES.forEach((shape, i) => {
    global[4 + i] = (state.taskScores[shape] || 0) / 4;
    global[8 + i] = targetSet.has(shape) ? 1 : 0;
  });
  const targetProgress = state.targetShapes.map((s) => (state.taskScores[s] || 0) / 4);
  // 无形状任务时视为已满足（1.0）
  global[12] = targetProgress.length > 0 ? Math.min(...targetProgress) : 1;
  global[13] = state.won ? 1 : 0;
  global[14] = ((state.lastAction ?? -1) + 1) / 280; // 归一化用 MAX_ACTIONS=280
  // 解冻任务
  const ut = state.unfreezeTarget ?? 0;
  const uc = state.unfreezeCount ?? 0;
  global[15] = ut > 0 ? Math.min(1, uc / ut) : 1;
  global[16] = ut > 0 ? 1 : 0;

  return { board, global };
}
