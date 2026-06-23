import { ROWS, COLS, MAX_ACTIONS, SWAP_ACTIONS } from '../core/constants.js';
import { inBounds, cloneBoard } from '../core/board.js';
import { isFrozen } from '../core/cells.js';
import { trySwap } from '../core/resolver.js';

const DIRS = [
  [0, 1],
  [0, -1],
  [1, 0],
  [-1, 0],
];

// 捏爆动作起始编号：0-89 横向交换, 90-179 纵向交换, 180-279 捏爆
const POP_OFFSET = SWAP_ACTIONS; // 180

/** 固定顺序：先 90 个水平交换，再 90 个垂直交换 */
export function encodeSwap(from, to) {
  if (from.r === to.r && from.c + 1 === to.c) return from.r * 9 + from.c;
  if (from.c === to.c && from.r + 1 === to.r) return 90 + from.r * 10 + from.c;
  if (from.r === to.r && from.c - 1 === to.c) return from.r * 9 + to.c;
  if (from.c === to.c && from.r - 1 === to.r) return 90 + to.r * 10 + to.c;
  return -1;
}

export function encodePop(r, c) {
  return POP_OFFSET + r * COLS + c;
}

/**
 * 解码动作为统一 move 结构：
 *   交换 → { type:'swap', from:{r,c}, to:{r,c} }
 *   捏爆 → { type:'pop', r, c }
 */
export function decodeAction(action) {
  if (action < 0 || action >= MAX_ACTIONS) return null;
  if (action < 90) {
    const r = Math.floor(action / 9);
    const c = action % 9;
    return { type: 'swap', from: { r, c }, to: { r, c: c + 1 } };
  }
  if (action < POP_OFFSET) {
    const idx = action - 90;
    const r = Math.floor(idx / 10);
    const c = idx % 10;
    if (r >= ROWS - 1) return null;
    return { type: 'swap', from: { r, c }, to: { r: r + 1, c } };
  }
  const idx = action - POP_OFFSET;
  const r = Math.floor(idx / COLS);
  const c = idx % COLS;
  return { type: 'pop', r, c };
}

/**
 * 获取所有相邻可交换格对。冰冻格与 void 格的 swap 被排除。
 */
export function getAdjacentSwaps(board, layout = null) {
  const swaps = [];
  const seen = new Set();
  for (let r = 0; r < ROWS; r++) {
    for (let c = 0; c < COLS; c++) {
      if (layout && !layout[r][c]) continue;
      const cell = board[r][c];
      if (!cell || isFrozen(cell)) continue; // 冰冻格不可移动
      for (const [dr, dc] of DIRS) {
        const nr = r + dr;
        const nc = c + dc;
        if (!inBounds(nr, nc)) continue;
        if (layout && !layout[nr][nc]) continue;
        const ncell = board[nr][nc];
        if (!ncell || isFrozen(ncell)) continue; // 目标冰冻格不可交换
        const key =
          r < nr || (r === nr && c < nc)
            ? `${r},${c}-${nr},${nc}`
            : `${nr},${nc}-${r},${c}`;
        if (seen.has(key)) continue;
        seen.add(key);
        swaps.push({ from: { r, c }, to: { r: nr, c: nc } });
      }
    }
  }
  return swaps;
}

/** 捏爆有效性：普通格 且 非冰冻 且 非空 且 非道具 且 活跃格。 */
function canPop(board, r, c, layout) {
  if (layout && !layout[r][c]) return false;
  const cell = board[r][c];
  if (!cell) return false;
  if (cell.kind !== 'normal') return false; // 道具格不可捏
  if (isFrozen(cell)) return false; // 冰冻格不可捏
  return true;
}

export function buildActionMask(board, layout = null) {
  const mask = new Float32Array(MAX_ACTIONS);
  let effectiveFound = false;

  // 交换段（0-179）
  for (const swap of getAdjacentSwaps(board, layout)) {
    const idx = encodeSwap(swap.from, swap.to);
    if (idx < 0) continue;
    const test = trySwap(cloneBoard(board), swap.from, swap.to, {}, layout);
    if (test.hadMatch || test.usedPowerup || (test.totalScore || 0) > 0) {
      mask[idx] = 1;
      effectiveFound = true;
    }
  }

  // 捏爆段（180-279）
  for (let r = 0; r < ROWS; r++) {
    for (let c = 0; c < COLS; c++) {
      if (canPop(board, r, c, layout)) {
        mask[encodePop(r, c)] = 1;
        effectiveFound = true;
      }
    }
  }

  // 退化：无有效动作时允许任意相邻交换
  if (!effectiveFound) {
    for (const swap of getAdjacentSwaps(board, layout)) {
      const idx = encodeSwap(swap.from, swap.to);
      if (idx >= 0) mask[idx] = 1;
    }
  }
  return mask;
}

/** 把动作编号解析为可执行 move，校验合法性（含冰冻锁定、捏爆条件）。 */
export function actionToMove(board, action, layout = null) {
  const move = decodeAction(action);
  if (!move) return null;
  const mask = buildActionMask(board, layout);
  if (!mask[action]) return null;
  return move;
}
