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
  getValidSwaps,
  simulateSwap,
  commitPreparedMove,
  reshuffleBoard,
} from './game.js';
import { findBestMove } from './ai.js';

const gridEl = () => document.getElementById('grid');
const scoreEl = () => document.getElementById('score');
const chainEl = () => document.getElementById('chain-score');
const movesEl = () => document.getElementById('moves');
const specialEl = () => document.getElementById('special-panel');
const statusEl = () => document.getElementById('status');
const logEl = () => document.getElementById('log');
const btnAi = () => document.getElementById('btn-ai');
const btnNew = () => document.getElementById('btn-new');

/** @type {ReturnType<typeof createGameState>} */
let state;
let busy = false;
let lastHighlight = null;

const TIMINGS = {
  showPick: 700,
  showSwap: 900,
  showMerge: 900,
  showFall: 1000,
  pauseBetweenChains: 300,
};

export function initApp() {
  state = createGameState();
  renderAll();
  bindEvents();
}

function bindEvents() {
  btnAi()?.addEventListener('click', onAiMove);
  btnNew()?.addEventListener('click', () => {
    state = createGameState();
    log('新一局开始');
    renderAll();
  });
}

function shapeClass(shape) {
  return `shape-${shape}`;
}

function renderGrid(
  board = state.board,
  {
    highlight = null,
    mergeCells = null,
    fallCells = null,
    phaseClass = '',
  } = {}
) {
  const el = gridEl();
  if (!el) return;
  el.style.setProperty('--cols', String(COLS));
  el.className = `grid ${phaseClass}`.trim();
  el.innerHTML = '';

  const mergedSet = new Set((mergeCells || []).map((p) => `${p.r},${p.c}`));
  const fallSet = new Set((fallCells || []).map((p) => `${p.r},${p.c}`));

  for (let r = 0; r < ROWS; r++) {
    for (let c = 0; c < COLS; c++) {
      const cell = board[r][c];
      const div = document.createElement('div');
      div.className = 'cell';
      div.dataset.r = String(r);
      div.dataset.c = String(c);

      if (highlight || lastHighlight) {
        const { from, to } = highlight || lastHighlight;
        if ((r === from.r && c === from.c) || (r === to.r && c === to.c)) {
          div.classList.add('highlight');
        }
      }

      if (state.targetShapes.includes(cell?.shape)) {
        div.classList.add('target-cell');
      }

      if (!cell) {
        div.classList.add('empty');
      } else {
        const inner = document.createElement('div');
        inner.className = `piece ${shapeClass(cell.shape)} level-${cell.level}`;
        inner.innerHTML = `<span class="level-num">${cell.level}</span>`;
        div.appendChild(inner);
      }

      if (mergedSet.has(`${r},${c}`)) div.classList.add('merge-hit');
      if (fallSet.has(`${r},${c}`)) div.classList.add('fall-changed');

      el.appendChild(div);
    }
  }
}

function renderHud() {
  scoreEl() && (scoreEl().textContent = String(state.score));
  chainEl() && (chainEl().textContent = String(state.chainScoreTotal));
  movesEl() && (movesEl().textContent = String(state.moves));

  const sp = specialEl();
  if (sp) {
    sp.innerHTML = SHAPES.map((shape) => {
      const n = state.specialScores[shape] || 0;
      const isTarget = state.targetShapes.includes(shape);
      const done = isTarget && n >= 4;
      return `
        <div class="special-row ${isTarget ? 'is-target' : ''} ${done ? 'done' : ''}">
          <span class="special-icon ${shapeClass(shape)}"></span>
          <span>${SHAPE_NAMES[shape]}</span>
          <span class="special-val">${n}${isTarget ? ' / 4' : ''}</span>
          ${isTarget ? '<span class="badge">目标</span>' : ''}
        </div>
      `;
    }).join('');
  }

  const st = statusEl();
  if (st) {
    if (state.won) {
      st.textContent = '🎉 通关！两种目标图形的特殊积分均已达到 4';
      st.className = 'status win';
      btnAi()?.setAttribute('disabled', 'true');
    } else {
      const names = state.targetShapes.map((s) => SHAPE_NAMES[s]).join('、');
      st.textContent = `本局目标：${names} 的特殊积分各达到 4 方可通关`;
      st.className = 'status';
      btnAi()?.removeAttribute('disabled');
    }
  }
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
  if (busy || state.won) return;
  busy = true;
  btnAi()?.setAttribute('disabled', 'true');

  let swaps = getValidSwaps(state.board);
  if (swaps.length === 0) {
    reshuffleBoard(state.board);
    log('无可行交换，已自动洗牌');
    renderAll();
    busy = false;
    btnAi()?.removeAttribute('disabled');
    return;
  }

  const move = findBestMove(state);
  if (!move) {
    log('未找到有效移动');
    busy = false;
    btnAi()?.removeAttribute('disabled');
    return;
  }

  const preview = simulateSwap(state.board, move.from, move.to, {
    captureBoards: true,
    includeFinalBoard: true,
  });
  if (!preview.success || !preview.finalBoard) {
    log('预演失败，本次跳过');
    busy = false;
    btnAi()?.removeAttribute('disabled');
    return;
  }

  const fromLabel = `(${move.from.r + 1},${move.from.c + 1})`;
  const toLabel = `(${move.to.r + 1},${move.to.c + 1})`;

  lastHighlight = { from: move.from, to: move.to };
  renderGrid(state.board, { highlight: lastHighlight, phaseClass: 'phase-pick' });
  log(`准备交换 ${fromLabel} ↔ ${toLabel}`);
  await sleep(TIMINGS.showPick);

  if (preview.afterSwapBoard) {
    renderGrid(preview.afterSwapBoard, {
      highlight: lastHighlight,
      phaseClass: 'phase-swap',
    });
    await sleep(TIMINGS.showSwap);
  }

  let previousBoard = preview.afterSwapBoard
    ? cloneBoard(preview.afterSwapBoard)
    : cloneBoard(state.board);
  for (let i = 0; i < preview.steps.length; i++) {
    const step = preview.steps[i];
    if (step.type === 'merge' && step.boardAfterMerge) {
      const mergeCells = collectMergeCells(step.mergeEvents || []);
      renderGrid(step.boardAfterMerge, {
        highlight: lastHighlight,
        mergeCells,
        phaseClass: 'phase-merge',
      });
      await sleep(TIMINGS.showMerge);
      previousBoard = cloneBoard(step.boardAfterMerge);
      continue;
    }

    if (step.type === 'refill' && step.boardAfterRefill) {
      const changed = collectChangedCells(previousBoard, step.boardAfterRefill);
      renderGrid(step.boardAfterRefill, {
        fallCells: changed,
        phaseClass: 'phase-fall',
      });
      await sleep(TIMINGS.showFall);
      previousBoard = cloneBoard(step.boardAfterRefill);
      await sleep(TIMINGS.pauseBetweenChains);
    }
  }

  const res = commitPreparedMove(
    state,
    move.from,
    move.to,
    preview,
    preview.finalBoard
  );
  if (res.ok) {
    const chain = res.result.chainScore;
    log(
      `交换 ${fromLabel}↔${toLabel} ` +
        `+${res.result.totalScore} 分` +
        (chain > 0 ? `（连锁 +${chain}）` : '') +
        ` · ${move.reason}`
    );
  }

  lastHighlight = null;
  renderAll();

  if (state.won) {
    log(`通关！总步数 ${state.moves}，总分 ${state.score}`);
  }

  busy = false;
  if (!state.won) btnAi()?.removeAttribute('disabled');
}

function sleep(ms) {
  return new Promise((r) => setTimeout(r, ms));
}

function collectMergeCells(mergeEvents) {
  const out = [];
  for (const ev of mergeEvents) {
    for (const p of ev.match?.cells || []) out.push(p);
  }
  return out;
}

function collectChangedCells(prevBoard, nextBoard) {
  const changed = [];
  for (let r = 0; r < ROWS; r++) {
    for (let c = 0; c < COLS; c++) {
      const a = prevBoard[r][c];
      const b = nextBoard[r][c];
      const same =
        (!a && !b) ||
        (a && b && a.shape === b.shape && a.level === b.level);
      if (!same) changed.push({ r, c });
    }
  }
  return changed;
}

document.addEventListener('DOMContentLoaded', initApp);
