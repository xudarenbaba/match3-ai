import { cloneBoard } from './board.js';
import { isPowerup } from './cells.js';
import { findMatches, pickMergePositions, applyMerges } from './match.js';
import { applyGravityAndRefill } from './gravity.js';
import { powerupTargets, unfreezeTargets } from './powerup.js';

export function swapCells(board, from, to) {
  const tmp = board[from.r][from.c];
  board[from.r][from.c] = board[to.r][to.c];
  board[to.r][to.c] = tmp;
}

export function resolveBoard(board, swapFrom = null, swapTo = null, options = {}) {
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
    applyGravityAndRefill(board);
    steps.push({
      type: 'refill',
      boardAfterRefill: captureBoards ? cloneBoard(board) : null,
    });
    isFirstMerge = false;
  }

  return { totalScore, chainScore, specialGained, steps, hadMatch };
}

function resolvePowerupSwap(board, from, to, captureBoards = false) {
  const powerEvents = [];
  const taskFromPowerup = {};
  let totalScore = 0;
  let chainScore = 0;
  const steps = [];

  const first = board[from.r][from.c];
  const second = board[to.r][to.c];
  const powerups = [];
  if (isPowerup(first)) powerups.push({ pos: from, partner: to, cell: first });
  if (isPowerup(second)) powerups.push({ pos: to, partner: from, cell: second });

  if (powerups.length === 0) {
    return {
      usedPowerup: false,
      totalScore: 0,
      chainScore: 0,
      specialGained: {},
      taskFromPowerup,
      steps,
      hadMatch: false,
    };
  }

  for (const pw of powerups) {
    const targets = powerupTargets(board, pw.pos, pw.partner);
    const unfrozen = unfreezeTargets(board, targets);
    const removedCount = targets.length;
    if (removedCount === 0) continue;

    targets.forEach((p) => {
      board[p.r][p.c] = null;
    });
    totalScore += removedCount;
    if (removedCount >= 9) {
      taskFromPowerup[pw.cell.shape] = (taskFromPowerup[pw.cell.shape] || 0) + 1;
    }

    powerEvents.push({
      powerupType: pw.cell.powerupType,
      shape: pw.cell.shape,
      removedCount,
      targets,
      unfrozen,
      triggerAt: pw.pos,
    });
  }

  if (powerEvents.length === 0) {
    return {
      usedPowerup: false,
      totalScore: 0,
      chainScore: 0,
      specialGained: {},
      taskFromPowerup,
      steps,
      hadMatch: false,
    };
  }

  steps.push({
    type: 'powerup',
    events: powerEvents,
    score: powerEvents.reduce((acc, e) => acc + e.removedCount, 0),
    boardAfterPowerup: captureBoards ? cloneBoard(board) : null,
  });

  applyGravityAndRefill(board);
  steps.push({
    type: 'refill',
    boardAfterRefill: captureBoards ? cloneBoard(board) : null,
  });

  const chain = resolveBoard(board, from, to, { captureBoards });
  totalScore += chain.totalScore;
  chainScore += chain.totalScore;

  steps.push(...chain.steps);
  return {
    usedPowerup: true,
    totalScore,
    chainScore,
    specialGained: chain.specialGained,
    taskFromPowerup,
    steps,
    hadMatch: chain.hadMatch || true,
  };
}

export function trySwap(board, from, to, options = {}) {
  swapCells(board, from, to);
  const afterSwapBoard = options.captureBoards ? cloneBoard(board) : null;

  const powerRes = resolvePowerupSwap(board, from, to, options.captureBoards);
  if (powerRes.usedPowerup) {
    return {
      success: true,
      afterSwapBoard,
      ...powerRes,
    };
  }

  const normalRes = resolveBoard(board, from, to, options);
  return {
    success: true,
    afterSwapBoard,
    taskFromPowerup: {},
    ...normalRes,
  };
}

export function simulateSwap(board, from, to, options = {}) {
  const sim = cloneBoard(board);
  const result = trySwap(sim, from, to, options);
  if (options.includeFinalBoard) {
    return { ...result, finalBoard: cloneBoard(sim) };
  }
  return result;
}
