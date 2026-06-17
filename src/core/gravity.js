import { ROWS, COLS, INITIAL_FROZEN_RATIO } from './constants.js';
import { createCell, createInitialCell } from './cells.js';
import { findMatches } from './match.js';

/**
 * 应用重力并补充新格子。
 * layout: 10×10 的 0/1 数组（1=活跃，0=void），为 null 时视为全 1（向后兼容）
 */
export function applyGravityAndRefill(board, layout = null) {
  for (let c = 0; c < COLS; c++) {
    // 收集该列活跃格中非空的格子（从底部到顶部）
    const stack = [];
    for (let r = ROWS - 1; r >= 0; r--) {
      if (layout && !layout[r][c]) continue; // void 格跳过
      if (board[r][c]) stack.push(board[r][c]);
    }
    // 重新填回活跃格（底部已有格子，顶部补新格）
    let stackIdx = 0;
    for (let r = ROWS - 1; r >= 0; r--) {
      if (layout && !layout[r][c]) {
        board[r][c] = null; // void 格保持 null
        continue;
      }
      board[r][c] = stackIdx < stack.length ? stack[stackIdx++] : createCell();
    }
  }
}

/**
 * 重新洗牌棋盘，只处理活跃格。
 * layout: 10×10 的 0/1 数组，为 null 时视为全 1
 */
export function reshuffleBoard(board, layout = null, maxAttempts = 200) {
  for (let attempt = 0; attempt < maxAttempts; attempt++) {
    for (let r = 0; r < ROWS; r++) {
      for (let c = 0; c < COLS; c++) {
        if (layout && !layout[r][c]) {
          board[r][c] = null; // void 格永远为 null
        } else {
          board[r][c] = createInitialCell();
        }
      }
    }
    if (findMatches(board).length === 0) return true;
  }
  return false;
}

/**
 * 随机冻结活跃格中的一部分格子。
 * layout: 10×10 的 0/1 数组，为 null 时视为全 1
 */
export function freezeRandomCells(board, ratio = INITIAL_FROZEN_RATIO, layout = null) {
  const positions = [];
  for (let r = 0; r < ROWS; r++) {
    for (let c = 0; c < COLS; c++) {
      if (layout && !layout[r][c]) continue; // void 格不冻结
      positions.push({ r, c });
    }
  }
  const total = Math.floor(positions.length * ratio);
  for (let i = 0; i < positions.length; i++) {
    const j = i + Math.floor(Math.random() * (positions.length - i));
    const tmp = positions[i];
    positions[i] = positions[j];
    positions[j] = tmp;
  }
  for (let i = 0; i < total; i++) {
    const { r, c } = positions[i];
    if (board[r][c]) board[r][c].frozen = true;
  }
}
