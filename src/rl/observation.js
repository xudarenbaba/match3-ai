import { ROWS, COLS, SHAPES, POWERUP_TYPES } from '../core/constants.js';

// 单帧 board 通道数
const BOARD_CHANNELS = 28;
// 帧堆叠数，与 Python 侧 FRAME_STACK 保持一致
export const FRAME_STACK = 3;
// 堆叠后的 board 通道数
export const STACKED_BOARD_CHANNELS = BOARD_CHANNELS * FRAME_STACK; // 84
const GLOBAL_DIM = 15;

export const OBS_BOARD_CHANNELS = STACKED_BOARD_CHANNELS;
export const OBS_GLOBAL_DIM = GLOBAL_DIM;

/**
 * 构建单帧棋盘观测，返回普通 Array（便于 JSON 序列化传给推理服务）。
 * board: 长度 BOARD_CHANNELS*ROWS*COLS 的数组
 * global: 长度 GLOBAL_DIM 的数组
 */
export function buildObservation(state) {
  const board = new Array(BOARD_CHANNELS * ROWS * COLS).fill(0);
  const targetSet = new Set(state.targetShapes);

  for (let r = 0; r < ROWS; r++) {
    for (let c = 0; c < COLS; c++) {
      const cell = state.board[r][c];
      const base = r * COLS + c;
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
        board[15 * ROWS * COLS + base] = 1;
      }

      if (targetSet.has(cell.shape)) {
        const si = SHAPES.indexOf(cell.shape);
        if (si >= 0) {
          if (cell.kind === 'normal' && cell.level >= 1 && cell.level <= 3) {
            board[(16 + si * 3 + (cell.level - 1)) * ROWS * COLS + base] = 1;
          } else if (cell.kind === 'powerup') {
            // 道具格视为最高进度，标记到 L3 通道
            board[(16 + si * 3 + 2) * ROWS * COLS + base] = 1;
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
  global[12] = targetProgress.length > 0 ? Math.min(...targetProgress) : 0;
  global[13] = state.won ? 1 : 0;
  global[14] = ((state.lastAction ?? -1) + 1) / 180;

  return { board, global };
}
