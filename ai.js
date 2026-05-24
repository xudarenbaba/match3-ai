/**
 * AI：贪心 + 2 步向前看（Beam Search）
 */

import {
  cloneBoard,
  getValidSwaps,
  simulateSwap,
  SHAPES,
  SHAPE_NAMES,
} from './game.js';

const SPECIAL_TARGET = 4;

/**
 * 局面评估（用于第二步潜力 &  tie-break）
 * @param {import('./game.js').BoardCell[][]} board
 * @param {string[]} targetShapes
 * @param {Record<string, number>} currentSpecial
 */
export function evaluateBoard(board, targetShapes, currentSpecial) {
  let score = 0;
  const rows = board.length;
  const cols = board[0].length;

  for (const shape of targetShapes) {
    const need = Math.max(0, SPECIAL_TARGET - (currentSpecial[shape] || 0));
    if (need <= 0) continue;

    let countL1 = 0, countL2 = 0, countL3 = 0;

    for (let r = 0; r < rows; r++) {
      for (let c = 0; c < cols; c++) {
        const cell = board[r][c];
        if (!cell || cell.shape !== shape) continue;
        if (cell.level === 1) countL1++;
        else if (cell.level === 2) countL2++;
        else countL3++;
      }
    }

    // 各等级数量价值（等级越高越有价值）
    score += countL3 * 90 * need;
    score += countL2 * 30 * need;
    score += countL1 * 10 * need;

    // 直接相邻同级别对：为下一步合并做好铺垫
    score += clusterBonus(board, shape) * 20 * need;

    // 差一格就能凑 3 连的潜力（核心改进）
    score += nearMissPotential(board, shape, rows, cols) * 35 * need;
  }

  // 非目标图形略降权
  for (const shape of SHAPES) {
    if (targetShapes.includes(shape)) continue;
    let n = 0;
    for (let r = 0; r < rows; r++) {
      for (let c = 0; c < cols; c++) {
        if (board[r][c]?.shape === shape) n++;
      }
    }
    score += n * 0.5;
  }

  return score;
}

/**
 * 计算目标图形"差一格就能凑 3 连"的数量
 * 场景：同行/列上，同形状同等级出现 ≥2 个，且恰好缺一个相邻位置
 */
function nearMissPotential(board, shape, rows, cols) {
  let potential = 0;

  // 横向扫描：每行找同形状同等级的位置，看有没有差一格的 3 连机会
  for (let r = 0; r < rows; r++) {
    const byLevel = { 1: [], 2: [], 3: [] };
    for (let c = 0; c < cols; c++) {
      const cell = board[r][c];
      if (cell?.shape === shape) byLevel[cell.level].push(c);
    }
    for (const cols_arr of Object.values(byLevel)) {
      if (cols_arr.length < 2) continue;
      // 检查任意两个列位置，看是否构成"差一格"结构
      for (let i = 0; i < cols_arr.length; i++) {
        for (let j = i + 1; j < cols_arr.length; j++) {
          const gap = cols_arr[j] - cols_arr[i];
          if (gap === 1 || gap === 2) potential += 1;  // 连续2个 或 中间差1格
        }
      }
    }
  }

  // 纵向扫描
  for (let c = 0; c < cols; c++) {
    const byLevel = { 1: [], 2: [], 3: [] };
    for (let r = 0; r < rows; r++) {
      const cell = board[r][c];
      if (cell?.shape === shape) byLevel[cell.level].push(r);
    }
    for (const rows_arr of Object.values(byLevel)) {
      if (rows_arr.length < 2) continue;
      for (let i = 0; i < rows_arr.length; i++) {
        for (let j = i + 1; j < rows_arr.length; j++) {
          const gap = rows_arr[j] - rows_arr[i];
          if (gap === 1 || gap === 2) potential += 1;
        }
      }
    }
  }

  return potential;
}

function clusterBonus(board, shape) {
  let bonus = 0;
  const rows = board.length;
  const cols = board[0].length;
  const dirs = [[0, 1], [1, 0]];

  for (let r = 0; r < rows; r++) {
    for (let c = 0; c < cols; c++) {
      const a = board[r][c];
      if (!a || a.shape !== shape) continue;
      for (const [dr, dc] of dirs) {
        const nr = r + dr;
        const nc = c + dc;
        if (nr >= rows || nc >= cols) continue;
        const b = board[nr][nc];
        if (b && b.shape === shape && b.level === a.level) bonus += 1;
      }
    }
  }
  return bonus;
}

/**
 * 评估一次交换的即时收益 + 一步后续潜力
 */
function scoreSwap(board, swap, targetShapes, currentSpecial) {
  const sim = cloneBoard(board);
  const result = simulateSwap(sim, swap.from, swap.to);
  if (!result.success) return -Infinity;

  let value = result.totalScore * 10;

  // 目标图形特殊分权重极高
  for (const shape of targetShapes) {
    const gained = result.specialGained[shape] || 0;
    const cur = currentSpecial[shape] || 0;
    const need = SPECIAL_TARGET - cur;
    if (gained > 0) {
      value += gained * 5000;
      if (cur + gained >= SPECIAL_TARGET) value += 20000;
    }
    // 接近通关时，任何 toward 目标 shape 的 2/3 级合并也有价值
    if (need > 0 && need <= 2) value += (result.specialGained[shape] || 0) * 3000;
  }

  value += result.chainScore * 12;

  // 2 步向前看：先按快速估分排序，再取前 60 个，找最优后手
  const afterBoard = sim;
  const followUps = getValidSwaps(afterBoard);

  // 快速估分：优先排列目标图形相关的后手
  followUps.sort((a, b) => {
    const qa = quickRank(afterBoard, a, targetShapes);
    const qb = quickRank(afterBoard, b, targetShapes);
    return qb - qa;
  });

  let bestFollow = 0;
  for (const fu of followUps.slice(0, 60)) {
    const fuSim = cloneBoard(afterBoard);
    const fuRes = simulateSwap(fuSim, fu.from, fu.to);
    if (!fuRes.success) continue;
    let fuVal = fuRes.totalScore * 8 + fuRes.chainScore * 10;
    for (const shape of targetShapes) {
      fuVal += (fuRes.specialGained[shape] || 0) * 2000;
    }
    if (fuVal > bestFollow) bestFollow = fuVal;
  }
  value += bestFollow * 0.55;  // 提升后手权重

  value += evaluateBoard(afterBoard, targetShapes, {
    ...currentSpecial,
    ...mergeSpecial(currentSpecial, result.specialGained),
  });

  return value;
}

/**
 * 快速粗排：判断某次交换是否涉及目标图形
 */
function quickRank(board, swap, targetShapes) {
  let rank = 0;
  const cells = [swap.from, swap.to];
  for (const { r, c } of cells) {
    const cell = board[r][c];
    if (!cell) continue;
    if (targetShapes.includes(cell.shape)) rank += cell.level * 10;
    else rank += 1;
  }
  return rank;
}

function mergeSpecial(base, gained) {
  const out = { ...base };
  Object.entries(gained || {}).forEach(([k, v]) => {
    out[k] = (out[k] || 0) + v;
  });
  return out;
}

/**
 * 选择最优交换
 * @returns {{ from, to, value, reason: string } | null}
 */
export function findBestMove(state) {
  const swaps = getValidSwaps(state.board);
  if (swaps.length === 0) return null;

  let best = null;
  let bestVal = -Infinity;

  for (const swap of swaps) {
    const val = scoreSwap(
      state.board,
      swap,
      state.targetShapes,
      state.specialScores
    );
    if (val > bestVal) {
      bestVal = val;
      best = { from: swap.from, to: swap.to, value: val };
    }
  }

  if (!best) return null;

  const fromLabel = `(${best.from.r + 1},${best.from.c + 1})`;
  const toLabel = `(${best.to.r + 1},${best.to.c + 1})`;
  const targets = state.targetShapes
    .map((s) => SHAPE_NAMES[s])
    .join('、');

  return {
    ...best,
    reason: `优先推进目标图形【${targets}】的特殊积分，预估收益 ${Math.round(bestVal)}`,
  };
}
