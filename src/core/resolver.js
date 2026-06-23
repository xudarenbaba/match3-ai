import { cloneBoard, countFrozen } from './board.js';
import { isPowerup } from './cells.js';
import { findMatches, pickMergePositions, applyMerges } from './match.js';
import { applyGravityAndRefill } from './gravity.js';
import { applyPowerupEffect } from './powerup.js';

export function swapCells(board, from, to) {
  const tmp = board[from.r][from.c];
  board[from.r][from.c] = board[to.r][to.c];
  board[to.r][to.c] = tmp;
}

export function resolveBoard(board, swapFrom = null, swapTo = null, options = {}, layout = null, targetShapes = null) {
  const captureBoards = Boolean(options.captureBoards);
  let totalScore = 0;
  let chainScore = 0;
  let isFirstMerge = true;
  const specialGained = {};
  const steps = [];
  let hadMatch = false;

  while (true) {
    const matches = findMatches(board);
    if (matches.length === 0) break;
    hadMatch = true;

    const mergePositions = pickMergePositions(
      matches,
      isFirstMerge ? swapFrom : null,
      isFirstMerge ? swapTo : null
    );
    const merged = applyMerges(board, matches, mergePositions);
    totalScore += merged.score;
    if (!isFirstMerge) chainScore += merged.score;

    Object.entries(merged.specialGained).forEach(([shape, n]) => {
      specialGained[shape] = (specialGained[shape] || 0) + n;
    });

    steps.push({
      type: 'merge',
      score: merged.score,
      mergeEvents: merged.mergeEvents,
      specialGained: { ...merged.specialGained },
      unfrozen: merged.unfrozenByAdjacency,
      boardAfterMerge: captureBoards ? cloneBoard(board) : null,
    });
    applyGravityAndRefill(board, layout);
    steps.push({
      type: 'refill',
      boardAfterRefill: captureBoards ? cloneBoard(board) : null,
    });
    isFirstMerge = false;
  }

  return { totalScore, chainScore, specialGained, steps, hadMatch };
}

function resolvePowerupSwap(board, from, to, captureBoards = false, layout = null, targetShapes = null) {
  const first = board[from.r][from.c];
  const second = board[to.r][to.c];
  const powerups = [];
  if (isPowerup(first)) powerups.push({ pos: from, partner: to, cell: first });
  if (isPowerup(second)) powerups.push({ pos: to, partner: from, cell: second });

  if (powerups.length === 0) {
    return { usedPowerup: false, totalScore: 0, chainScore: 0, specialGained: {}, taskFromPowerup: {}, steps: [], hadMatch: false };
  }

  const combinedSpecial = {};
  const powerEvents = [];
  let totalScore = 0;
  let anyTriggered = false;

  for (const pw of powerups) {
    const { targets, specialGained, clearedByShape, upgraded, cleared } =
      applyPowerupEffect(board, pw.pos, pw.partner, layout, targetShapes);

    // 行/列/同道具：targets 可能为空（空行/列），但只要是道具就算触发
    anyTriggered = true;

    Object.entries(specialGained).forEach(([shape, n]) => {
      combinedSpecial[shape] = (combinedSpecial[shape] || 0) + n;
    });
    // bomb 或升级消除的格子才计分
    totalScore += cleared.length + Object.values(clearedByShape).reduce((a, b) => a + b, 0);

    powerEvents.push({
      powerupType: pw.cell.powerupType,
      shape: pw.cell.shape,
      targets,
      upgraded,
      cleared,
      triggerAt: pw.pos,
    });
  }

  if (!anyTriggered) {
    return { usedPowerup: false, totalScore: 0, chainScore: 0, specialGained: {}, taskFromPowerup: {}, steps: [], hadMatch: false };
  }

  const steps = [];
  steps.push({
    type: 'powerup',
    events: powerEvents,
    score: totalScore,
    boardAfterPowerup: captureBoards ? cloneBoard(board) : null,
  });

  applyGravityAndRefill(board, layout);
  steps.push({
    type: 'refill',
    boardAfterRefill: captureBoards ? cloneBoard(board) : null,
  });

  const chain = resolveBoard(board, from, to, { captureBoards }, layout, targetShapes);
  totalScore += chain.totalScore;

  Object.entries(chain.specialGained).forEach(([shape, n]) => {
    combinedSpecial[shape] = (combinedSpecial[shape] || 0) + n;
  });

  steps.push(...chain.steps);

  return {
    usedPowerup: true,
    totalScore,
    chainScore: chain.totalScore,
    specialGained: combinedSpecial,
    // 任务分统一只走 specialGained，taskFromPowerup 置空避免重复计分
    taskFromPowerup: {},
    steps,
    hadMatch: chain.hadMatch || true,
  };
}

export function trySwap(board, from, to, options = {}, layout = null, targetShapes = null) {
  const beforeFrozen = countFrozen(board);
  swapCells(board, from, to);
  const afterSwapBoard = options.captureBoards ? cloneBoard(board) : null;

  const powerRes = resolvePowerupSwap(board, from, to, options.captureBoards, layout, targetShapes);
  if (powerRes.usedPowerup) {
    return { success: true, afterSwapBoard, unfrozenCount: beforeFrozen - countFrozen(board), isPop: false, ...powerRes };
  }

  const normalRes = resolveBoard(board, from, to, options, layout, targetShapes);
  return {
    success: true,
    afterSwapBoard,
    taskFromPowerup: {},
    unfrozenCount: beforeFrozen - countFrozen(board),
    isPop: false,
    ...normalRes,
  };
}

/**
 * 捏爆：直接清空一个普通非冰冻格，不解冻相邻冰壳。
 * 清空后重力补位并结算连锁（连锁消除按正常规则解冻、计任务分）。
 */
export function popCell(board, r, c, options = {}, layout = null, targetShapes = null) {
  const cell = board[r][c];
  if (!cell || cell.kind !== 'normal' || cell.frozen) {
    return {
      success: false, usedPowerup: false, totalScore: 0, chainScore: 0,
      specialGained: {}, taskFromPowerup: {}, steps: [], hadMatch: false,
      unfrozenCount: 0, isPop: true, popped: false,
    };
  }

  const beforeFrozen = countFrozen(board);
  board[r][c] = null; // 清空，不触发相邻解冻
  const steps = [];
  if (options.captureBoards) {
    steps.push({ type: 'pop', popped: { r, c }, boardAfterPop: cloneBoard(board) });
  }
  applyGravityAndRefill(board, layout);
  if (options.captureBoards) {
    steps.push({ type: 'refill', boardAfterRefill: cloneBoard(board) });
  }

  const chain = resolveBoard(board, null, null, options, layout, targetShapes);
  steps.push(...chain.steps);

  return {
    success: true,
    usedPowerup: false,
    totalScore: chain.totalScore,
    chainScore: chain.chainScore,
    specialGained: chain.specialGained,
    taskFromPowerup: {},
    steps,
    hadMatch: chain.hadMatch,
    // 捏爆本身不解冻；解冻仅来自连锁消除
    unfrozenCount: beforeFrozen - countFrozen(board),
    isPop: true,
    popped: true,
  };
}

export function simulateSwap(board, from, to, options = {}, layout = null, targetShapes = null) {
  const sim = cloneBoard(board);
  const result = trySwap(sim, from, to, options, layout, targetShapes);
  if (options.includeFinalBoard) return { ...result, finalBoard: cloneBoard(sim) };
  return result;
}

/** 捏爆的模拟版本（前端预演用）。 */
export function simulatePop(board, r, c, options = {}, layout = null, targetShapes = null) {
  const sim = cloneBoard(board);
  const result = popCell(sim, r, c, options, layout, targetShapes);
  if (options.includeFinalBoard) return { ...result, finalBoard: cloneBoard(sim) };
  return result;
}
