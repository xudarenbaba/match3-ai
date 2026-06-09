/**
 * 消消乐游戏引擎（v2）
 * - 10x10 网格，4 种图形 × 3 个等级
 * - 固定步数（默认 100），允许无合并交换
 * - 随机道具：列消 / 九宫格 / 全图同形消除
 */

export const ROWS = 10;
export const COLS = 10;
export const SHAPES = ['circle', 'square', 'triangle', 'star'];
export const SHAPE_NAMES = {
  circle: '圆形',
  square: '方块',
  triangle: '三角形',
  star: '五角星',
};
export const POWERUP_TYPES = ['column', 'bomb', 'color'];
export const MAX_STEPS = 100;
export const INITIAL_FROZEN_RATIO = 0.12;
export const POWERUP_SPAWN_RATE = 0.025;

/** @typedef {{ kind: 'normal', shape: string, level: number, frozen: boolean }} NormalCell */
/** @typedef {{ kind: 'powerup', shape: string, powerupType: 'column'|'bomb'|'color', level: 3, frozen: boolean }} PowerCell */
/** @typedef {NormalCell | PowerCell} Cell */
/** @typedef {Cell | null} BoardCell */

export function createNormalCell(shape = null, level = null) {
  if (shape == null) {
    shape = SHAPES[Math.floor(Math.random() * SHAPES.length)];
  }
  if (level == null) {
    const r = Math.random();
    level = r < 0.6 ? 1 : r < 0.9 ? 2 : 3;
  }
  return { kind: 'normal', shape, level, frozen: false };
}

export function createPowerupCell(shape = null, powerupType = null) {
  if (shape == null) {
    shape = SHAPES[Math.floor(Math.random() * SHAPES.length)];
  }
  if (powerupType == null) {
    powerupType = POWERUP_TYPES[Math.floor(Math.random() * POWERUP_TYPES.length)];
  }
  return { kind: 'powerup', shape, powerupType, level: 3, frozen: false };
}

/**
 * 随机普通格子（小概率道具）
 */
export function createCell() {
  if (Math.random() < POWERUP_SPAWN_RATE) return createPowerupCell();
  return createNormalCell();
}

export function isPowerup(cell) {
  return Boolean(cell && cell.kind === 'powerup');
}

export function isFrozen(cell) {
  return Boolean(cell && cell.frozen);
}

export function createEmptyBoard() {
  return Array.from({ length: ROWS }, () => Array(COLS).fill(null));
}

export function cloneBoard(board) {
  return board.map((row) => row.map((c) => (c ? { ...c } : null)));
}

export function inBounds(r, c) {
  return r >= 0 && r < ROWS && c >= 0 && c < COLS;
}

function cellKey(r, c) {
  return `${r},${c}`;
}

function cellsEqual(a, b) {
  if (!a && !b) return true;
  if (!a || !b) return false;
  return (
    a.kind === b.kind &&
    a.shape === b.shape &&
    a.level === b.level &&
    Boolean(a.frozen) === Boolean(b.frozen) &&
    (a.powerupType || '') === (b.powerupType || '')
  );
}

/**
 * 仅普通格参与合并：横/竖 >= 3，且同形状同等级
 */
export function findMatches(board) {
  const matches = [];

  // 横向
  for (let r = 0; r < ROWS; r++) {
    let c = 0;
    while (c < COLS) {
      const cell = board[r][c];
      if (!cell || cell.kind !== 'normal' || cell.frozen) {
        c += 1;
        continue;
      }
      let end = c + 1;
      while (end < COLS) {
        const next = board[r][end];
        if (
          !next ||
          next.kind !== 'normal' ||
          next.frozen ||
          next.shape !== cell.shape ||
          next.level !== cell.level
        ) {
          break;
        }
        end += 1;
      }
      if (end - c >= 3) {
        const cells = [];
        for (let i = c; i < end; i++) cells.push({ r, c: i });
        matches.push({ cells, shape: cell.shape, level: cell.level });
      }
      c = end;
    }
  }

  // 纵向
  for (let c = 0; c < COLS; c++) {
    let r = 0;
    while (r < ROWS) {
      const cell = board[r][c];
      if (!cell || cell.kind !== 'normal' || cell.frozen) {
        r += 1;
        continue;
      }
      let end = r + 1;
      while (end < ROWS) {
        const next = board[end][c];
        if (
          !next ||
          next.kind !== 'normal' ||
          next.frozen ||
          next.shape !== cell.shape ||
          next.level !== cell.level
        ) {
          break;
        }
        end += 1;
      }
      if (end - r >= 3) {
        const cells = [];
        for (let i = r; i < end; i++) cells.push({ r: i, c });
        matches.push({ cells, shape: cell.shape, level: cell.level });
      }
      r = end;
    }
  }

  // 十字重叠合并
  if (matches.length <= 1) return matches;

  const parent = matches.map((_, i) => i);
  const find = (x) => (parent[x] === x ? x : (parent[x] = find(parent[x])));
  const unite = (a, b) => {
    parent[find(a)] = find(b);
  };
  const posToIndices = new Map();

  for (let i = 0; i < matches.length; i++) {
    for (const p of matches[i].cells) {
      const k = cellKey(p.r, p.c);
      if (!posToIndices.has(k)) posToIndices.set(k, []);
      posToIndices.get(k).push(i);
    }
  }
  for (const indices of posToIndices.values()) {
    for (let i = 1; i < indices.length; i++) unite(indices[0], indices[i]);
  }

  const groups = new Map();
  for (let i = 0; i < matches.length; i++) {
    const root = find(i);
    if (!groups.has(root)) {
      groups.set(root, {
        shape: matches[i].shape,
        level: matches[i].level,
        cells: [],
      });
    }
    const g = groups.get(root);
    for (const p of matches[i].cells) {
      if (!g.cells.some((x) => x.r === p.r && x.c === p.c)) g.cells.push(p);
    }
  }

  return Array.from(groups.values()).filter((g) => g.cells.length >= 3);
}

function freezeRandomCells(board, ratio = INITIAL_FROZEN_RATIO) {
  const positions = [];
  for (let r = 0; r < ROWS; r++) {
    for (let c = 0; c < COLS; c++) {
      positions.push({ r, c });
    }
  }
  const total = Math.floor(ROWS * COLS * ratio);
  for (let i = 0; i < positions.length; i++) {
    const j = i + Math.floor(Math.random() * (positions.length - i));
    const tmp = positions[i];
    positions[i] = positions[j];
    positions[j] = tmp;
  }
  for (let i = 0; i < total; i++) {
    const { r, c } = positions[i];
    if (board[r][c]) board[r][c].frozen = true;
  }
}

function buildMergeAdjacencySet(matches) {
  const set = new Set();
  const dirs = [
    [1, 0],
    [-1, 0],
    [0, 1],
    [0, -1],
  ];
  for (const m of matches) {
    for (const p of m.cells) {
      for (const [dr, dc] of dirs) {
        const nr = p.r + dr;
        const nc = p.c + dc;
        if (inBounds(nr, nc)) set.add(cellKey(nr, nc));
      }
    }
  }
  return set;
}

function unfreezeAdjacentByMatches(board, matches) {
  const adj = buildMergeAdjacencySet(matches);
  const unfrozen = [];
  adj.forEach((k) => {
    const [r, c] = k.split(',').map(Number);
    const cell = board[r][c];
    if (cell && cell.frozen) {
      cell.frozen = false;
      unfrozen.push({ r, c });
    }
  });
  return unfrozen;
}

function unfreezeTargets(board, targets) {
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

export function pickMergePositions(matches, swapFrom, swapTo) {
  return matches.map((m) => {
    const cellSet = new Set(m.cells.map((p) => cellKey(p.r, p.c)));
    for (const pos of [swapTo, swapFrom]) {
      if (pos && cellSet.has(cellKey(pos.r, pos.c))) return pos;
    }
    let best = m.cells[0];
    for (const p of m.cells) {
      if (p.r > best.r || (p.r === best.r && p.c > best.c)) best = p;
    }
    return best;
  });
}

export function applyMerges(board, matches, mergePositions) {
  let score = 0;
  const specialGained = {};
  const mergeEvents = [];
  const cleared = new Set();
  const results = new Map();
  const unfrozenByAdjacency = unfreezeAdjacentByMatches(board, matches);

  matches.forEach((m, i) => {
    const pos = mergePositions[i];
    const n = m.cells.length;
    const level = m.level;

    if (level === 1) {
      score += n;
      results.set(cellKey(pos.r, pos.c), createNormalCell(m.shape, 2));
    } else if (level === 2) {
      score += n * 2;
      results.set(cellKey(pos.r, pos.c), createNormalCell(m.shape, 3));
    } else {
      score += n * 3;
      specialGained[m.shape] = (specialGained[m.shape] || 0) + 1;
      results.set(cellKey(pos.r, pos.c), null);
    }

    m.cells.forEach((p) => {
      if (p.r === pos.r && p.c === pos.c) return;
      cleared.add(cellKey(p.r, p.c));
    });

    mergeEvents.push({ match: m, position: pos, level, count: n });
  });

  cleared.forEach((k) => {
    const [r, c] = k.split(',').map(Number);
    board[r][c] = null;
  });
  results.forEach((val, k) => {
    const [r, c] = k.split(',').map(Number);
    board[r][c] = val;
  });

  return { score, specialGained, mergeEvents, unfrozenByAdjacency };
}

export function applyGravityAndRefill(board) {
  for (let c = 0; c < COLS; c++) {
    const stack = [];
    for (let r = ROWS - 1; r >= 0; r--) {
      if (board[r][c]) stack.push(board[r][c]);
    }
    for (let r = ROWS - 1; r >= 0; r--) {
      const idx = ROWS - 1 - r;
      board[r][c] = idx < stack.length ? stack[idx] : createCell();
    }
  }
}

export function swapCells(board, from, to) {
  const tmp = board[from.r][from.c];
  board[from.r][from.c] = board[to.r][to.c];
  board[to.r][to.c] = tmp;
}

function powerupTargets(board, powerPos, partnerPos) {
  const powerCell = board[powerPos.r][powerPos.c];
  const partnerCell = board[partnerPos.r][partnerPos.c];
  const targets = new Set();

  if (!powerCell || powerCell.kind !== 'powerup') return [];

  if (powerCell.powerupType === 'column') {
    for (let r = 0; r < ROWS; r++) targets.add(cellKey(r, powerPos.c));
  } else if (powerCell.powerupType === 'bomb') {
    for (let dr = -1; dr <= 1; dr++) {
      for (let dc = -1; dc <= 1; dc++) {
        // 炸弹以“交换后炸弹所在位置”为中心触发 3x3
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

const DIRS = [
  [0, 1],
  [0, -1],
  [1, 0],
  [-1, 0],
];

export function getAdjacentSwaps(board) {
  const swaps = [];
  const seen = new Set();
  for (let r = 0; r < ROWS; r++) {
    for (let c = 0; c < COLS; c++) {
      if (!board[r][c]) continue;
      for (const [dr, dc] of DIRS) {
        const nr = r + dr;
        const nc = c + dc;
        if (!inBounds(nr, nc) || !board[nr][nc]) continue;
        const key =
          r < nr || (r === nr && c < nc)
            ? `${r},${c}-${nr},${nc}`
            : `${nr},${nc}-${r},${c}`;
        if (seen.has(key)) continue;
        seen.add(key);
        swaps.push({ from: { r, c }, to: { r: nr, c: nc } });
      }
    }
  }
  return swaps;
}

export function simulateSwap(board, from, to, options = {}) {
  const sim = cloneBoard(board);
  const result = trySwap(sim, from, to, options);
  if (options.includeFinalBoard) {
    return { ...result, finalBoard: cloneBoard(sim) };
  }
  return result;
}

export function reshuffleBoard(board, maxAttempts = 200) {
  for (let attempt = 0; attempt < maxAttempts; attempt++) {
    for (let r = 0; r < ROWS; r++) {
      for (let c = 0; c < COLS; c++) board[r][c] = createCell();
    }
    if (findMatches(board).length === 0) return true;
  }
  return false;
}

export function pickTwoShapes() {
  const copy = [...SHAPES];
  const a = copy.splice(Math.floor(Math.random() * copy.length), 1)[0];
  const b = copy[Math.floor(Math.random() * copy.length)];
  return [a, b];
}

export function createGameState() {
  const targetShapes = pickTwoShapes();
  const board = createEmptyBoard();
  reshuffleBoard(board);
  freezeRandomCells(board, INITIAL_FROZEN_RATIO);
  return {
    board,
    score: 0,
    chainScoreTotal: 0,
    taskScores: Object.fromEntries(SHAPES.map((s) => [s, 0])),
    targetShapes,
    totalSteps: MAX_STEPS,
    stepsUsed: 0,
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

export function checkVictory(state) {
  return state.targetShapes.every((s) => (state.taskScores[s] || 0) >= 4);
}

export function executeMove(state, from, to) {
  if (state.over) return { ok: false, reason: '本局已结束' };
  const result = trySwap(state.board, from, to);

  state.score += result.totalScore;
  state.chainScoreTotal += result.chainScore;
  applyTaskProgress(state, result);
  state.stepsUsed += 1;
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
  state.history.push({ from, to, ...preparedResult });

  if (checkVictory(state)) {
    state.won = true;
    state.over = true;
  } else if (state.stepsUsed >= state.totalSteps) {
    state.over = true;
  }

  return { ok: true, result: preparedResult };
}

export function boardDiffChangedCells(prevBoard, nextBoard) {
  const changed = [];
  for (let r = 0; r < ROWS; r++) {
    for (let c = 0; c < COLS; c++) {
      if (!cellsEqual(prevBoard[r][c], nextBoard[r][c])) changed.push({ r, c });
    }
  }
  return changed;
}
