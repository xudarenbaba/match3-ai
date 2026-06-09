/**
 * AI 选步（v2）
 * - 候选动作：全部相邻交换（允许无合并）
 * - 目标：在有限步数内最大化分数并优先完成任务
 */

import {
  SHAPES,
  SHAPE_NAMES,
  cloneBoard,
  getAdjacentSwaps,
  simulateSwap,
  isPowerup,
} from './game.js';

const TASK_TARGET = 4;

function stateTaskNeed(state, shape) {
  return Math.max(0, TASK_TARGET - (state.taskScores[shape] || 0));
}

function urgency(state) {
  if (state.totalSteps <= 0) return 1;
  return Math.min(1, state.stepsUsed / state.totalSteps);
}

function quickRank(board, swap, state) {
  let rank = 0;
  const a = board[swap.from.r][swap.from.c];
  const b = board[swap.to.r][swap.to.c];
  const cells = [a, b];

  cells.forEach((cell) => {
    if (!cell) return;
    const isTarget = state.targetShapes.includes(cell.shape);
    rank += isTarget ? 12 : 1;
    rank += (cell.level || 1) * (isTarget ? 3 : 1);
    if (isPowerup(cell)) rank += 140;
  });

  // 目标图形紧急度越高，相关操作越加权
  for (const shape of state.targetShapes) {
    const need = stateTaskNeed(state, shape);
    if (need > 0) {
      if (a?.shape === shape) rank += 20 * need;
      if (b?.shape === shape) rank += 20 * need;
    }
  }

  return rank;
}

function evaluateBoard(board, state) {
  let score = 0;
  const rows = board.length;
  const cols = board[0].length;

  for (const shape of state.targetShapes) {
    const need = stateTaskNeed(state, shape);
    if (need <= 0) continue;

    let l1 = 0;
    let l2 = 0;
    let l3 = 0;
    let power = 0;
    let cluster = 0;

    for (let r = 0; r < rows; r++) {
      for (let c = 0; c < cols; c++) {
        const cell = board[r][c];
        if (!cell || cell.shape !== shape) continue;
        if (cell.kind === 'powerup') {
          power += 1;
          continue;
        }
        if (cell.level === 1) l1 += 1;
        else if (cell.level === 2) l2 += 1;
        else l3 += 1;

        const right = c + 1 < cols ? board[r][c + 1] : null;
        const down = r + 1 < rows ? board[r + 1][c] : null;
        if (right && right.kind === 'normal' && right.shape === shape && right.level === cell.level) cluster += 1;
        if (down && down.kind === 'normal' && down.shape === shape && down.level === cell.level) cluster += 1;
      }
    }

    score += l1 * 6 * need;
    score += l2 * 16 * need;
    score += l3 * 30 * need;
    score += power * 80 * need;
    score += cluster * 14 * need;
  }

  // 轻微偏好总体有可操作素材
  for (const shape of SHAPES) {
    if (state.targetShapes.includes(shape)) continue;
    let n = 0;
    for (let r = 0; r < rows; r++) {
      for (let c = 0; c < cols; c++) {
        if (board[r][c]?.shape === shape) n += 1;
      }
    }
    score += n * 0.2;
  }

  return score;
}

function scoreSwap(board, swap, state) {
  const sim = cloneBoard(board);
  const result = simulateSwap(sim, swap.from, swap.to);
  if (!result) return -Infinity;

  const u = urgency(state);
  let value = 0;

  // 即时分与连锁
  value += result.totalScore * (8 + u * 6);
  value += result.chainScore * (10 + u * 5);

  // 任务奖励：三级合并累计、道具大消除触发任务
  for (const shape of state.targetShapes) {
    const need = stateTaskNeed(state, shape);
    if (need <= 0) continue;
    const level3Inc = result.specialGained?.[shape] || 0;
    const powerTask = result.taskFromPowerup?.[shape] || 0;
    if (level3Inc > 0) value += level3Inc * 1300 * need;
    if (powerTask > 0) value += powerTask * 2200 * need;

    // 临近达成任务时更激进
    if ((state.taskScores[shape] || 0) + powerTask >= TASK_TARGET) {
      value += 9000;
    }
  }

  // 使用道具奖励
  const usedPowerup = result.steps?.some((s) => s.type === 'powerup');
  if (usedPowerup) value += 900;

  // 后手潜力（两步）
  const followUps = getAdjacentSwaps(sim);
  followUps.sort((a, b) => quickRank(sim, b, state) - quickRank(sim, a, state));
  let bestFollow = 0;
  for (const fu of followUps.slice(0, 40)) {
    const fuSim = cloneBoard(sim);
    const fuRes = simulateSwap(fuSim, fu.from, fu.to);
    if (!fuRes) continue;
    let v = fuRes.totalScore * 7 + fuRes.chainScore * 8;
    for (const shape of state.targetShapes) {
      v += (fuRes.specialGained?.[shape] || 0) * 900;
      v += (fuRes.taskFromPowerup?.[shape] || 0) * 1600;
    }
    if (fuRes.steps?.some((s) => s.type === 'powerup')) v += 600;
    if (v > bestFollow) bestFollow = v;
  }
  value += bestFollow * 0.5;

  value += evaluateBoard(sim, state) * (0.5 + (1 - u) * 0.6);
  return value;
}

export function findBestMove(state) {
  if (state.over) return null;
  const swaps = getAdjacentSwaps(state.board);
  if (swaps.length === 0) return null;

  swaps.sort((a, b) => quickRank(state.board, b, state) - quickRank(state.board, a, state));
  const candidates = swaps.slice(0, 60);

  let best = null;
  let bestVal = -Infinity;
  for (const swap of candidates) {
    const val = scoreSwap(state.board, swap, state);
    if (val > bestVal) {
      bestVal = val;
      best = { from: swap.from, to: swap.to, value: val };
    }
  }
  if (!best) return null;

  const targets = state.targetShapes.map((s) => SHAPE_NAMES[s]).join('、');
  return {
    ...best,
    reason: `优先推进目标【${targets}】任务并兼顾后手，预估收益 ${Math.round(bestVal)}`,
  };
}
