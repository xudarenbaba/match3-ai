import { ROWS, COLS } from './constants.js';

export function cellKey(r, c) {
  return `${r},${c}`;
}

export function createEmptyBoard() {
  return Array.from({ length: ROWS }, () => Array(COLS).fill(null));
}

export function cloneBoard(board) {
  return board.map((row) => row.map((c) => (c ? { ...c } : null)));
}

export function inBounds(r, c) {
  return r >= 0 && r < ROWS && c >= 0 && c < COLS;
}

/** 统计棋盘上冰冻格数量（用于解冻任务计数：操作前后差值=本步解冻数）。 */
export function countFrozen(board) {
  let n = 0;
  for (let r = 0; r < ROWS; r++) {
    for (let c = 0; c < COLS; c++) {
      const cell = board[r][c];
      if (cell && cell.frozen) n += 1;
    }
  }
  return n;
}

export function cellsEqual(a, b) {
  if (!a && !b) return true;
  if (!a || !b) return false;
  return (
    a.kind === b.kind &&
    a.shape === b.shape &&
    a.level === b.level &&
    Boolean(a.frozen) === Boolean(b.frozen) &&
    (a.powerupType || '') === (b.powerupType || '')
  );
}

export function boardDiffChangedCells(prevBoard, nextBoard) {
  const changed = [];
  for (let r = 0; r < ROWS; r++) {
    for (let c = 0; c < COLS; c++) {
      if (!cellsEqual(prevBoard[r][c], nextBoard[r][c])) changed.push({ r, c });
    }
  }
  return changed;
}
