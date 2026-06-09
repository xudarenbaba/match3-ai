import { ROWS, COLS, INITIAL_FROZEN_RATIO } from './constants.js';
import { createCell } from './cells.js';
import { findMatches } from './match.js';

export function applyGravityAndRefill(board) {
  for (let c = 0; c < COLS; c++) {
    const stack = [];
    for (let r = ROWS - 1; r >= 0; r--) {
      if (board[r][c]) stack.push(board[r][c]);
    }
    for (let r = ROWS - 1; r >= 0; r--) {
      const idx = ROWS - 1 - r;
      board[r][c] = idx < stack.length ? stack[idx] : createCell();
    }
  }
}

export function reshuffleBoard(board, maxAttempts = 200) {
  for (let attempt = 0; attempt < maxAttempts; attempt++) {
    for (let r = 0; r < ROWS; r++) {
      for (let c = 0; c < COLS; c++) board[r][c] = createCell();
    }
    if (findMatches(board).length === 0) return true;
  }
  return false;
}

export function freezeRandomCells(board, ratio = INITIAL_FROZEN_RATIO) {
  const positions = [];
  for (let r = 0; r < ROWS; r++) {
    for (let c = 0; c < COLS; c++) {
      positions.push({ r, c });
    }
  }
  const total = Math.floor(ROWS * COLS * ratio);
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
