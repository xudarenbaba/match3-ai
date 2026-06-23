import { SHAPES, MAX_STEPS, INITIAL_FROZEN_RATIO } from './constants.js';
import { cloneBoard, createEmptyBoard } from './board.js';
import { reshuffleBoard, freezeRandomCells } from './gravity.js';
import { trySwap, popCell } from './resolver.js';
import { encodeSwap, encodePop } from '../actions/encoding.js';
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
  // null/undefined = 默认随机 2 个形状；[] = 明确无形状任务（仅解冻任务）
  const targetShapes = options.targetShapes == null ? pickTwoShapes() : [...options.targetShapes];

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
    // 解冻任务：需解冻 unfreezeTarget 个冰壳；0 = 无解冻任务
    unfreezeTarget: options.unfreezeTarget ?? 0,
    unfreezeCount: 0,
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
  // 形状任务与解冻任务可组合，均满足才胜利。空任务视为已满足。
  const shapeOk =
    state.targetShapes.length === 0 ||
    state.targetShapes.every((s) => (state.taskScores[s] || 0) >= taskTarget);
  const unfreezeOk =
    (state.unfreezeTarget ?? 0) === 0 || (state.unfreezeCount ?? 0) >= state.unfreezeTarget;
  return shapeOk && unfreezeOk;
}

/** 执行一个 move（统一接口）：{type:'swap',from,to} 或 {type:'pop',r,c} */
export function executeMove(state, move) {
  if (state.over) return { ok: false, reason: '本局已结束' };
  let result;
  let actionIdx;
  if (move.type === 'pop') {
    result = popCell(state.board, move.r, move.c, {}, state.layout, state.targetShapes);
    actionIdx = encodePop(move.r, move.c);
  } else {
    result = trySwap(state.board, move.from, move.to, {}, state.layout, state.targetShapes);
    actionIdx = encodeSwap(move.from, move.to);
  }

  state.score += result.totalScore;
  state.chainScoreTotal += result.chainScore;
  applyTaskProgress(state, result);
  state.unfreezeCount = (state.unfreezeCount || 0) + (result.unfrozenCount || 0);
  state.stepsUsed += 1;
  state.lastAction = actionIdx;
  state.history.push({ move, ...result });

  if (checkVictory(state)) {
    state.won = true;
    state.over = true;
  } else if (state.stepsUsed >= state.totalSteps) {
    state.over = true;
  }

  return { ok: true, result };
}

export function simulateSwapWithTargets(board, from, to, options = {}, layout = null, targetShapes = null) {
  const sim = cloneBoard(board);
  const result = trySwap(sim, from, to, options, layout, targetShapes);
  if (options.includeFinalBoard) return { ...result, finalBoard: cloneBoard(sim) };
  return result;
}

export function commitPreparedMove(state, move, preparedResult, finalBoard) {
  if (!preparedResult || !finalBoard) return { ok: false, reason: '预演数据不完整' };
  if (state.over) return { ok: false, reason: '本局已结束' };

  state.board = cloneBoard(finalBoard);
  state.score += preparedResult.totalScore;
  state.chainScoreTotal += preparedResult.chainScore;
  applyTaskProgress(state, preparedResult);
  state.unfreezeCount = (state.unfreezeCount || 0) + (preparedResult.unfrozenCount || 0);
  state.stepsUsed += 1;
  state.lastAction = move.type === 'pop' ? encodePop(move.r, move.c) : encodeSwap(move.from, move.to);
  state.history.push({ move, ...preparedResult });

  if (checkVictory(state)) {
    state.won = true;
    state.over = true;
  } else if (state.stepsUsed >= state.totalSteps) {
    state.over = true;
  }

  return { ok: true, result: preparedResult };
}
