import { SHAPES, MAX_STEPS, INITIAL_FROZEN_RATIO } from './constants.js';
import { cloneBoard, createEmptyBoard } from './board.js';
import { reshuffleBoard, freezeRandomCells } from './gravity.js';
import { trySwap } from './resolver.js';
import { encodeSwap } from '../actions/encoding.js';
import { getLayout, pickLayout, LAYOUT_POOL } from './layouts.js';

export function pickTwoShapes() {
  const copy = [...SHAPES];
  const a = copy.splice(Math.floor(Math.random() * copy.length), 1)[0];
  const b = copy[Math.floor(Math.random() * copy.length)];
  return [a, b];
}

/**
 * 创建新游戏状态。
 * options.layoutName: 指定布局名称，不传则随机选取
 * options.curriculumLevel: 1/2/3，影响布局池，默认 3
 */
export function createGameState(options = {}) {
  const targetShapes = options.targetShapes || pickTwoShapes();

  // 布局选择
  const curriculumLevel = options.curriculumLevel ?? 3;
  const pool = LAYOUT_POOL[curriculumLevel] || LAYOUT_POOL[3];
  const layoutName = options.layoutName || pickLayout(pool);
  const layout = getLayout(layoutName);

  const board = createEmptyBoard();
  reshuffleBoard(board, layout);
  if (options.freeze !== false) {
    freezeRandomCells(board, options.frozenRatio ?? INITIAL_FROZEN_RATIO, layout);
  }
  return {
    board,
    layout,
    layoutName,
    score: 0,
    chainScoreTotal: 0,
    taskScores: Object.fromEntries(SHAPES.map((s) => [s, 0])),
    targetShapes,
    totalSteps: options.totalSteps ?? MAX_STEPS,
    stepsUsed: 0,
    lastAction: -1,
    won: false,
    over: false,
    history: [],
  };
}

function applyTaskProgress(state, result) {
  Object.entries(result.specialGained || {}).forEach(([shape, n]) => {
    state.taskScores[shape] = (state.taskScores[shape] || 0) + n;
  });
  Object.entries(result.taskFromPowerup || {}).forEach(([shape, n]) => {
    state.taskScores[shape] = (state.taskScores[shape] || 0) + n;
  });
}

export function checkVictory(state, taskTarget = 4) {
  return state.targetShapes.every((s) => (state.taskScores[s] || 0) >= taskTarget);
}

export function executeMove(state, from, to) {
  if (state.over) return { ok: false, reason: '本局已结束' };
  const result = trySwap(state.board, from, to, {}, state.layout);

  state.score += result.totalScore;
  state.chainScoreTotal += result.chainScore;
  applyTaskProgress(state, result);
  state.stepsUsed += 1;
  state.lastAction = encodeSwap(from, to);
  state.history.push({ from, to, ...result });

  if (checkVictory(state)) {
    state.won = true;
    state.over = true;
  } else if (state.stepsUsed >= state.totalSteps) {
    state.over = true;
  }

  return { ok: true, result };
}

export function commitPreparedMove(state, from, to, preparedResult, finalBoard) {
  if (!preparedResult || !finalBoard) return { ok: false, reason: '预演数据不完整' };
  if (state.over) return { ok: false, reason: '本局已结束' };

  state.board = cloneBoard(finalBoard);
  state.score += preparedResult.totalScore;
  state.chainScoreTotal += preparedResult.chainScore;
  applyTaskProgress(state, preparedResult);
  state.stepsUsed += 1;
  state.lastAction = encodeSwap(from, to);
  state.history.push({ from, to, ...preparedResult });

  if (checkVictory(state)) {
    state.won = true;
    state.over = true;
  } else if (state.stepsUsed >= state.totalSteps) {
    state.over = true;
  }

  return { ok: true, result: preparedResult };
}
