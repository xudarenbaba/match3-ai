import { ROWS, COLS } from './constants.js';
import { cellKey, inBounds } from './board.js';

export function unfreezeTargets(board, targets) {
  const out = [];
  for (const p of targets) {
    const cell = board[p.r][p.c];
    if (cell && cell.frozen) {
      cell.frozen = false;
      out.push({ r: p.r, c: p.c });
    }
  }
  return out;
}

export function powerupTargets(board, powerPos, partnerPos) {
  const powerCell = board[powerPos.r][powerPos.c];
  const partnerCell = board[partnerPos.r][partnerPos.c];
  const targets = new Set();

  if (!powerCell || powerCell.kind !== 'powerup') return [];

  if (powerCell.powerupType === 'column') {
    for (let r = 0; r < ROWS; r++) targets.add(cellKey(r, powerPos.c));
  } else if (powerCell.powerupType === 'bomb') {
    for (let dr = -1; dr <= 1; dr++) {
      for (let dc = -1; dc <= 1; dc++) {
        const nr = powerPos.r + dr;
        const nc = powerPos.c + dc;
        if (inBounds(nr, nc)) targets.add(cellKey(nr, nc));
      }
    }
  } else if (powerCell.powerupType === 'color') {
    const targetShape = partnerCell?.shape;
    if (targetShape) {
      for (let r = 0; r < ROWS; r++) {
        for (let c = 0; c < COLS; c++) {
          if (board[r][c]?.shape === targetShape) targets.add(cellKey(r, c));
        }
      }
    }
  }

  targets.add(cellKey(powerPos.r, powerPos.c));
  return Array.from(targets).map((k) => {
    const [r, c] = k.split(',').map(Number);
    return { r, c };
  });
}
