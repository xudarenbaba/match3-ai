/**
 * DOM 渲染与动画
 */

import {
  ROWS,
  COLS,
  SHAPES,
  SHAPE_NAMES,
  createGameState,
  cloneBoard,
  simulateSwap,
  commitPreparedMove,
  boardDiffChangedCells,
  isPowerup,
  isFrozen,
} from '../core/index.js';
import { checkRlServer, findRlMove, resetFrameHistory } from '../ai/rl-policy.js';

const gridEl = () => document.getElementById('grid');
const scoreEl = () => document.getElementById('score');
const chainEl = () => document.getElementById('chain-score');
const movesEl = () => document.getElementById('moves');
const remainEl = () => document.getElementById('steps-left');
const specialEl = () => document.getElementById('special-panel');
const statusEl = () => document.getElementById('status');
const logEl = () => document.getElementById('log');
const btnAi = () => document.getElementById('btn-ai');
const btnNew = () => document.getElementById('btn-new');

let state;
let busy = false;
let lastHighlight = null;
let rlReady = false;

const TIMINGS = {
  showPick: 700,
  showSwap: 850,
  showPowerup: 1000,
  showMerge: 900,
  showFall: 900,
  pauseBetweenChains: 250,
};

export async function initApp() {
  state = createGameState();
  rlReady = await checkRlServer();
  renderAll();
  bindEvents();
  if (rlReady) {
    log('RL 推理服务已连接');
    if (btnAi()) btnAi().textContent = 'RL 出手';
  } else {
    log('RL 推理服务未启动，请先运行 predict_server.py');
  }
}

function bindEvents() {
  btnAi()?.addEventListener('click', onAiMove);
  btnNew()?.addEventListener('click', () => {
    state = createGameState();
    resetFrameHistory();
    log('新一局开始');
    renderAll();
  });
}

function shapeClass(shape) {
  return `shape-${shape}`;
}

function powerupClass(powerupType) {
  return `power-${powerupType}`;
}

function renderGrid(
  board = state.board,
  {
    highlight = null,
    mergeCells = null,
    fallCells = null,
    powerCells = null,
    unfrozenCells = null,
    phaseClass = '',
  } = {}
) {
  const el = gridEl();
  if (!el) return;
  el.style.setProperty('--cols', String(COLS));
  el.className = `grid ${phaseClass}`.trim();
  el.innerHTML = '';

  const layout = state.layout || null;
  const mergedSet = new Set((mergeCells || []).map((p) => `${p.r},${p.c}`));
  const fallSet = new Set((fallCells || []).map((p) => `${p.r},${p.c}`));
  const powerSet = new Set((powerCells || []).map((p) => `${p.r},${p.c}`));
  const unfrozenSet = new Set((unfrozenCells || []).map((p) => `${p.r},${p.c}`));

  for (let r = 0; r < ROWS; r++) {
    for (let c = 0; c < COLS; c++) {
      const cell = board[r][c];
      const isVoid = layout && !layout[r][c];
      const div = document.createElement('div');
      div.className = 'cell';

      if (isVoid) {
        div.classList.add('void');
        el.appendChild(div);
        continue;
      }

      if (highlight || lastHighlight) {
        const { from, to } = highlight || lastHighlight;
        if ((r === from.r && c === from.c) || (r === to.r && c === to.c)) div.classList.add('highlight');
      }
      if (cell && state.targetShapes.includes(cell.shape)) div.classList.add('target-cell');
      if (cell && isFrozen(cell)) div.classList.add('frozen');
      if (!cell) div.classList.add('empty');

      if (cell) {
        const inner = document.createElement('div');
        inner.className = `piece ${shapeClass(cell.shape)} level-${cell.level}`;
        if (isPowerup(cell)) {
          inner.classList.add('powerup', powerupClass(cell.powerupType));
          inner.innerHTML = `<span class="power-tag">${powerupLabel(cell.powerupType)}</span>`;
        } else {
          inner.innerHTML = `<span class="level-num">${cell.level}</span>`;
        }
        div.appendChild(inner);
      }

      if (mergedSet.has(`${r},${c}`)) div.classList.add('merge-hit');
      if (fallSet.has(`${r},${c}`)) div.classList.add('fall-changed');
      if (powerSet.has(`${r},${c}`)) div.classList.add('power-hit');
      if (unfrozenSet.has(`${r},${c}`)) div.classList.add('unfreeze-hit');
      el.appendChild(div);
    }
  }
}

function powerupLabel(type) {
  if (type === 'column') return '列';
  if (type === 'bomb') return '炸';
  if (type === 'color') return '同';
  return '?';
}

function renderHud() {
  const left = Math.max(0, state.totalSteps - state.stepsUsed);
  if (scoreEl()) scoreEl().textContent = String(state.score);
  if (chainEl()) chainEl().textContent = String(state.chainScoreTotal);
  if (movesEl()) movesEl().textContent = `${state.stepsUsed}/${state.totalSteps}`;
  if (remainEl()) remainEl().textContent = String(left);

  const sp = specialEl();
  if (sp) {
    sp.innerHTML = SHAPES.map((shape) => {
      const task = state.taskScores[shape] || 0;
      const isTarget = state.targetShapes.includes(shape);
      const done = isTarget && task >= 4;
      return `
        <div class="special-row ${isTarget ? 'is-target' : ''} ${done ? 'done' : ''}">
          <span class="special-icon ${shapeClass(shape)}"></span>
          <span>${SHAPE_NAMES[shape]}</span>
          <span class="special-val">${task}${isTarget ? ' / 4' : ''}</span>
          ${isTarget ? '<span class="badge">目标</span>' : ''}
        </div>
      `;
    }).join('');
  }

  const st = statusEl();
  if (st) {
    if (state.won) {
      st.textContent = '🎉 通关！目标图形任务达成';
      st.className = 'status win';
    } else if (state.over) {
      st.textContent = '⏹ 步数用尽，本局结束';
      st.className = 'status lose';
    } else if (!rlReady) {
      st.textContent = 'RL 推理服务未连接 — 请先启动 predict_server.py';
      st.className = 'status lose';
    } else {
      const names = state.targetShapes.map((s) => SHAPE_NAMES[s]).join('、');
      st.textContent = `RL 已连接 · 目标：${names} 任务分各达 4 · 剩余 ${left} 步`;
      st.className = 'status';
    }
  }

  if (state.over) btnAi()?.setAttribute('disabled', 'true');
  else btnAi()?.removeAttribute('disabled');
}

function renderAll() {
  renderGrid();
  renderHud();
}

function log(msg) {
  const el = logEl();
  if (!el) return;
  const line = document.createElement('div');
  line.className = 'log-line';
  line.textContent = `[${new Date().toLocaleTimeString()}] ${msg}`;
  el.prepend(line);
  while (el.children.length > 80) el.removeChild(el.lastChild);
}

async function onAiMove() {
  if (busy || state.over) return;
  busy = true;
  btnAi()?.setAttribute('disabled', 'true');

  if (!rlReady) {
    rlReady = await checkRlServer();
  }
  if (!rlReady) {
    log('RL 推理服务未启动，无法走棋');
    busy = false;
    if (!state.over) btnAi()?.removeAttribute('disabled');
    return;
  }

  let move;
  try {
    move = await findRlMove(state);
  } catch (err) {
    log(`RL 推理失败: ${err.message}`);
    busy = false;
    if (!state.over) btnAi()?.removeAttribute('disabled');
    return;
  }
  if (!move?.from || !move?.to) {
    log('RL 未返回有效走法');
    busy = false;
    if (!state.over) btnAi()?.removeAttribute('disabled');
    return;
  }

  const preview = simulateSwap(state.board, move.from, move.to, {
    captureBoards: true,
    includeFinalBoard: true,
  }, state.layout || null);
  if (!preview?.finalBoard) {
    log('预演失败，本次跳过');
    busy = false;
    if (!state.over) btnAi()?.removeAttribute('disabled');
    return;
  }

  const fromLabel = `(${move.from.r + 1},${move.from.c + 1})`;
  const toLabel = `(${move.to.r + 1},${move.to.c + 1})`;
  lastHighlight = { from: move.from, to: move.to };

  renderGrid(state.board, { highlight: lastHighlight, phaseClass: 'phase-pick' });
  log(`准备交换 ${fromLabel} ↔ ${toLabel}`);
  await sleep(TIMINGS.showPick);

  if (preview.afterSwapBoard) {
    renderGrid(preview.afterSwapBoard, { highlight: lastHighlight, phaseClass: 'phase-swap' });
    await sleep(TIMINGS.showSwap);
  }

  let previousBoard = preview.afterSwapBoard ? cloneBoard(preview.afterSwapBoard) : cloneBoard(state.board);
  for (const step of preview.steps || []) {
    if (step.type === 'powerup' && step.boardAfterPowerup) {
      const powerCells = [];
      const unfrozenCells = [];
      for (const ev of step.events || []) powerCells.push(...(ev.targets || []));
      for (const ev of step.events || []) unfrozenCells.push(...(ev.unfrozen || []));
      renderGrid(step.boardAfterPowerup, { powerCells, unfrozenCells, phaseClass: 'phase-power' });
      await sleep(TIMINGS.showPowerup);
      previousBoard = cloneBoard(step.boardAfterPowerup);
      continue;
    }
    if (step.type === 'merge' && step.boardAfterMerge) {
      const mergeCells = [];
      const unfrozenCells = step.unfrozen || [];
      for (const ev of step.mergeEvents || []) mergeCells.push(...(ev.match?.cells || []));
      renderGrid(step.boardAfterMerge, { mergeCells, unfrozenCells, phaseClass: 'phase-merge' });
      await sleep(TIMINGS.showMerge);
      previousBoard = cloneBoard(step.boardAfterMerge);
      continue;
    }
    if (step.type === 'refill' && step.boardAfterRefill) {
      const changed = boardDiffChangedCells(previousBoard, step.boardAfterRefill);
      renderGrid(step.boardAfterRefill, { fallCells: changed, phaseClass: 'phase-fall' });
      await sleep(TIMINGS.showFall);
      previousBoard = cloneBoard(step.boardAfterRefill);
      await sleep(TIMINGS.pauseBetweenChains);
    }
  }

  const res = commitPreparedMove(state, move.from, move.to, preview, preview.finalBoard);
  if (res.ok) {
    const base = `交换 ${fromLabel}↔${toLabel} +${res.result.totalScore} 分`;
    const chain = res.result.chainScore > 0 ? `（连锁 +${res.result.chainScore}）` : '';
    const power = (res.result.steps || []).some((s) => s.type === 'powerup') ? ' · 触发道具' : '';
    log(`${base}${chain}${power} · ${move.reason}`);
  }

  lastHighlight = null;
  renderAll();
  if (state.won) log(`通关！总分 ${state.score}，步数 ${state.stepsUsed}/${state.totalSteps}`);
  else if (state.over) log(`步数用尽，最终总分 ${state.score}`);

  busy = false;
  if (!state.over) btnAi()?.removeAttribute('disabled');
}

function sleep(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

document.addEventListener('DOMContentLoaded', initApp);
