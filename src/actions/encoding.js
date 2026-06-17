import { ROWS, COLS, MAX_ACTIONS } from '../core/constants.js';
import { inBounds } from '../core/board.js';

const DIRS = [
  [0, 1],
  [0, -1],
  [1, 0],
  [-1, 0],
];

/** 固定顺序：先 90 个水平交换，再 90 个垂直交换 */
export function encodeSwap(from, to) {
  if (from.r === to.r && from.c + 1 === to.c) {
    return from.r * 9 + from.c;
  }
  if (from.c === to.c && from.r + 1 === to.r) {
    return 90 + from.r * 10 + from.c;
  }
  if (from.r === to.r && from.c - 1 === to.c) {
    return from.r * 9 + to.c;
  }
  if (from.c === to.c && from.r - 1 === to.r) {
    return 90 + to.r * 10 + to.c;
  }
  return -1;
}

export function decodeAction(action) {
  if (action < 0 || action >= MAX_ACTIONS) return null;
  if (action < 90) {
    const r = Math.floor(action / 9);
    const c = action % 9;
    return { from: { r, c }, to: { r, c: c + 1 } };
  }
  const idx = action - 90;
  const r = Math.floor(idx / 10);
  const c = idx % 10;
  if (r >= ROWS - 1) return null;
  return { from: { r, c }, to: { r: r + 1, c } };
}

/**
 * 获取所有相邻可交换格对。
 * layout: 10×10 的 0/1 数组，为 null 时视为全 1；void 格的 swap 被排除。
 */
export function getAdjacentSwaps(board, layout = null) {
  const swaps = [];
  const seen = new Set();
  for (let r = 0; r < ROWS; r++) {
    for (let c = 0; c < COLS; c++) {
      if (layout && !layout[r][c]) continue; // void 格
      if (!board[r][c]) continue;
      for (const [dr, dc] of DIRS) {
        const nr = r + dr;
        const nc = c + dc;
        if (!inBounds(nr, nc)) continue;
        if (layout && !layout[nr][nc]) continue; // 目标 void 格
        if (!board[nr][nc]) continue;
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

export function buildActionMask(board, layout = null) {
  const mask = new Float32Array(MAX_ACTIONS);
  const swaps = getAdjacentSwaps(board, layout);
  for (const swap of swaps) {
    const idx = encodeSwap(swap.from, swap.to);
    if (idx >= 0) mask[idx] = 1;
  }
  return mask;
}

export function swapFromAction(board, action, layout = null) {
  const swap = decodeAction(action);
  if (!swap) return null;
  // 检查两个格子是否都是活跃格
  if (layout) {
    if (!layout[swap.from.r][swap.from.c] || !layout[swap.to.r][swap.to.c]) return null;
  }
  const mask = buildActionMask(board, layout);
  if (!mask[action]) return null;
  return swap;
}
