import { ROWS, COLS } from './constants.js';
import { inBounds } from './board.js';

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

/**
 * 对 targets 中每个普通格升级 +1：
 *   L1 → L2（保留）
 *   L2 → L3（消失，计任务分）
 *
 * 返回 { specialGained, clearedByShape, upgraded, cleared }
 *   upgraded: 升到 L2 的格子坐标列表（用于动画）
 *   cleared:  升到 L3 消失的格子坐标列表（用于动画）
 */
function upgradeTargets(board, targets, targetShapes = null) {
  const specialGained = {};
  const clearedByShape = {};
  const upgraded = [];
  const cleared = [];

  for (const p of targets) {
    const cell = board[p.r][p.c];
    if (!cell || cell.kind !== 'normal') continue;
    if (cell.frozen) cell.frozen = false;

    if (cell.level === 1) {
      cell.level = 2;
      upgraded.push({ r: p.r, c: p.c });
    } else if (cell.level === 2) {
      // 升为 L3 → 完成目标，格子消失
      const { shape } = cell;
      board[p.r][p.c] = null;
      clearedByShape[shape] = (clearedByShape[shape] || 0) + 1;
      cleared.push({ r: p.r, c: p.c });
      if (!targetShapes || targetShapes.includes(shape)) {
        specialGained[shape] = (specialGained[shape] || 0) + 1;
      }
    }
    // L3 普通格或道具格：不处理
  }

  return { specialGained, clearedByShape, upgraded, cleared };
}

/**
 * 执行道具升级效果，直接修改 board。
 *
 * row   → 该行所有其他普通格 +1 级
 * column → 该列所有其他普通格 +1 级
 * color  → 全图与 partnerCell 同形状的所有普通格 +1 级
 * bomb   → 九宫格直接消除（原逻辑）
 *
 * 返回 { targets, specialGained, clearedByShape, upgraded, cleared }
 */
export function applyPowerupEffect(board, powerPos, partnerPos, layout = null, targetShapes = null) {
  const powerCell = board[powerPos.r][powerPos.c];
  const partnerCell = board[partnerPos.r][partnerPos.c];

  if (!powerCell || powerCell.kind !== 'powerup') {
    return { targets: [], specialGained: {}, clearedByShape: {}, upgraded: [], cleared: [] };
  }

  const isActive = (r, c) => {
    if (!inBounds(r, c)) return false;
    if (layout && !layout[r][c]) return false;
    return true;
  };

  const pt = powerCell.powerupType;

  if (pt === 'row') {
    const targets = [];
    for (let c = 0; c < COLS; c++) {
      if (c === powerPos.c) continue;
      if (!isActive(powerPos.r, c)) continue;
      const cell = board[powerPos.r][c];
      if (cell && cell.kind === 'normal') targets.push({ r: powerPos.r, c });
    }
    board[powerPos.r][powerPos.c] = null; // 道具消失
    const result = upgradeTargets(board, targets, targetShapes);
    return { targets, ...result };
  }

  if (pt === 'column') {
    const targets = [];
    for (let r = 0; r < ROWS; r++) {
      if (r === powerPos.r) continue;
      if (!isActive(r, powerPos.c)) continue;
      const cell = board[r][powerPos.c];
      if (cell && cell.kind === 'normal') targets.push({ r, c: powerPos.c });
    }
    board[powerPos.r][powerPos.c] = null;
    const result = upgradeTargets(board, targets, targetShapes);
    return { targets, ...result };
  }

  if (pt === 'color') {
    const targetShape = partnerCell?.shape;
    if (!targetShape) return { targets: [], specialGained: {}, clearedByShape: {}, upgraded: [], cleared: [] };
    const targets = [];
    for (let r = 0; r < ROWS; r++) {
      for (let c = 0; c < COLS; c++) {
        if (r === powerPos.r && c === powerPos.c) continue;
        if (!isActive(r, c)) continue;
        const cell = board[r][c];
        if (cell && cell.kind === 'normal' && cell.shape === targetShape) targets.push({ r, c });
      }
    }
    board[powerPos.r][powerPos.c] = null;
    const result = upgradeTargets(board, targets, targetShapes);
    return { targets, ...result };
  }

  if (pt === 'bomb') {
    // 炸弹：九宫格直接消除，保留原逻辑
    const targets = [];
    for (let dr = -1; dr <= 1; dr++) {
      for (let dc = -1; dc <= 1; dc++) {
        const nr = powerPos.r + dr;
        const nc = powerPos.c + dc;
        if (isActive(nr, nc)) targets.push({ r: nr, c: nc });
      }
    }
    unfreezeTargets(board, targets);
    const specialGained = {};
    const clearedByShape = {};
    const cleared = [];
    for (const p of targets) {
      const cell = board[p.r][p.c];
      if (cell) {
        clearedByShape[cell.shape] = (clearedByShape[cell.shape] || 0) + 1;
        cleared.push({ r: p.r, c: p.c });
        if (cell.kind === 'normal' && cell.level >= 2 && (!targetShapes || targetShapes.includes(cell.shape))) {
          specialGained[cell.shape] = (specialGained[cell.shape] || 0) + 1;
        }
        board[p.r][p.c] = null;
      }
    }
    return { targets, specialGained, clearedByShape, upgraded: [], cleared };
  }

  return { targets: [], specialGained: {}, clearedByShape: {}, upgraded: [], cleared: [] };
}
