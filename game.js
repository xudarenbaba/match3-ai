/**
 * 消消乐游戏引擎
 * 10x10 网格，4 种图形 × 3 个等级
 */

export const ROWS = 10;
export const COLS = 10;
export const SHAPES = ['circle', 'square', 'triangle', 'star'];
export const SHAPE_NAMES = { circle: '圆形', square: '方块', triangle: '三角形', star: '五角星' };

/** @typedef {{ shape: string, level: number }} Cell */
/** @typedef {Cell | null} BoardCell */

/**
 * @returns {import('./game.js').Cell}
 */
export function createCell(shape = null, level = null) {
  if (shape == null) {
    shape = SHAPES[Math.floor(Math.random() * SHAPES.length)];
  }
  if (level == null) {
    const r = Math.random();
    level = r < 0.6 ? 1 : r < 0.9 ? 2 : 3;
  }
  return { shape, level };
}

/**
 * @returns {BoardCell[][]}
 */
export function createEmptyBoard() {
  return Array.from({ length: ROWS }, () => Array(COLS).fill(null));
}

/**
 * @param {BoardCell[][]} board
 */
export function cloneBoard(board) {
  return board.map((row) => row.map((c) => (c ? { ...c } : null)));
}

/**
 * @param {BoardCell[][]} board
 * @param {number} r
 * @param {number} c
 */
export function inBounds(r, c) {
  return r >= 0 && r < ROWS && c >= 0 && c < COLS;
}

/**
 * @param {BoardCell[][]} board
 */
export function isBoardFull(board) {
  for (let r = 0; r < ROWS; r++) {
    for (let c = 0; c < COLS; c++) {
      if (!board[r][c]) return false;
    }
  }
  return true;
}

/**
 * 检测所有可合并的连续段（横/竖 ≥3，同形状同等级）
 * @returns {{ cells: {r:number,c:number}[], shape: string, level: number }[]}
 */
export function findMatches(board) {
  const matches = [];
  const used = new Set();

  const key = (r, c) => `${r},${c}`;

  // 横向
  for (let r = 0; r < ROWS; r++) {
    let c = 0;
    while (c < COLS) {
      const cell = board[r][c];
      if (!cell) {
        c++;
        continue;
      }
      let end = c + 1;
      while (
        end < COLS &&
        board[r][end] &&
        board[r][end].shape === cell.shape &&
        board[r][end].level === cell.level
      ) {
        end++;
      }
      const len = end - c;
      if (len >= 3) {
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
      if (!cell) {
        r++;
        continue;
      }
      let end = r + 1;
      while (
        end < ROWS &&
        board[end][c] &&
        board[end][c].shape === cell.shape &&
        board[end][c].level === cell.level
      ) {
        end++;
      }
      const len = end - r;
      if (len >= 3) {
        const cells = [];
        for (let i = r; i < end; i++) cells.push({ r: i, c });
        matches.push({ cells, shape: cell.shape, level: cell.level });
      }
      r = end;
    }
  }

  // 合并重叠的 match（十字交叉时按连通分量合并）
  if (matches.length <= 1) return matches;

  const cellToMatches = new Map();
  matches.forEach((m, idx) => {
    m.cells.forEach(({ r, c }) => {
      const k = key(r, c);
      if (!cellToMatches.has(k)) cellToMatches.set(k, []);
      cellToMatches.get(k).push(idx);
    });
  });

  const parent = matches.map((_, i) => i);
  const find = (x) => (parent[x] === x ? x : (parent[x] = find(parent[x])));
  const unite = (a, b) => {
    parent[find(a)] = find(b);
  };

  for (let r = 0; r < ROWS; r++) {
    for (let c = 0; c < COLS; c++) {
      const list = cellToMatches.get(key(r, c));
      if (list && list.length > 1) {
        for (let i = 1; i < list.length; i++) unite(list[0], list[i]);
      }
    }
  }

  const groups = new Map();
  matches.forEach((m, idx) => {
    const root = find(idx);
    if (!groups.has(root)) {
      groups.set(root, {
        cells: [],
        shape: m.shape,
        level: m.level,
      });
    }
    const g = groups.get(root);
    m.cells.forEach((cell) => {
      const k = key(cell.r, cell.c);
      if (!used.has(k)) {
        used.add(k);
        g.cells.push(cell);
      }
    });
  });

  return Array.from(groups.values()).filter((g) => g.cells.length >= 3);
}

/**
 * 为每个 match 选择合并落点
 * @param {{ cells: {r:number,c:number}[], shape: string, level: number }[]} matches
 * @param {{r:number,c:number}|null} swapFrom
 * @param {{r:number,c:number}|null} swapTo
 */
export function pickMergePositions(matches, swapFrom, swapTo) {
  const swapCells = [];
  if (swapFrom) swapCells.push(swapFrom);
  if (swapTo) swapCells.push(swapTo);

  return matches.map((m) => {
    const cellSet = new Set(m.cells.map(({ r, c }) => `${r},${c}`));

    // 优先落在 swapTo，其次 swapFrom
    for (const pos of [swapTo, swapFrom]) {
      if (pos && cellSet.has(`${pos.r},${pos.c}`)) return pos;
    }

    // 默认取 match 中最后一个格子（偏右/偏下）
    let best = m.cells[0];
    for (const cell of m.cells) {
      if (cell.r > best.r || (cell.r === best.r && cell.c > best.c)) best = cell;
    }
    return best;
  });
}

/**
 * 执行一轮合并
 * @returns {{ score: number, specialGained: Record<string, number>, mergeEvents: object[] }}
 */
export function applyMerges(board, matches, mergePositions) {
  let score = 0;
  /** @type {Record<string, number>} */
  const specialGained = {};
  const mergeEvents = [];

  // 同一格子可能属于多个 match 的并集，按 match 分别处理会冲突；
  // 已用 pickMergePositions 按连通分量合并过
  const cleared = new Set();
  const results = new Map(); // "r,c" -> { shape, level } | 'empty'

  matches.forEach((m, i) => {
    const pos = mergePositions[i];
    const n = m.cells.length;
    const level = m.level;

    if (level === 1) {
      score += n * 1;
      results.set(`${pos.r},${pos.c}`, { shape: m.shape, level: 2 });
    } else if (level === 2) {
      score += n * 2;
      results.set(`${pos.r},${pos.c}`, { shape: m.shape, level: 3 });
    } else {
      score += n * 3;
      specialGained[m.shape] = (specialGained[m.shape] || 0) + 1;
      results.set(`${pos.r},${pos.c}`, 'empty');
    }

    m.cells.forEach(({ r, c }) => {
      if (r === pos.r && c === pos.c) return;
      cleared.add(`${r},${c}`);
    });

    mergeEvents.push({ match: m, position: pos, level, count: n });
  });

  // 先清空参与合并的非落点格
  cleared.forEach((k) => {
    const [r, c] = k.split(',').map(Number);
    board[r][c] = null;
  });

  // 再设置落点
  results.forEach((val, k) => {
    const [r, c] = k.split(',').map(Number);
    if (val === 'empty') board[r][c] = null;
    else board[r][c] = val;
  });

  return { score, specialGained, mergeEvents };
}

/**
 * 重力下落 + 顶部随机补充
 */
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

/**
 * 交换两格（不验证）
 */
export function swapCells(board, from, to) {
  const tmp = board[from.r][from.c];
  board[from.r][from.c] = board[to.r][to.c];
  board[to.r][to.c] = tmp;
}

/**
 * 完整解析：合并 → 下落 → 再检测，直到稳定
 * @returns {{
 *   totalScore: number,
 *   chainScore: number,
 *   specialGained: Record<string, number>,
 *   steps: object[],
 *   hadMatch: boolean
 * }}
 */
export function resolveBoard(board, swapFrom = null, swapTo = null) {
  let options = {};
  if (
    typeof swapFrom === 'object' &&
    swapFrom &&
    !('r' in swapFrom) &&
    !('c' in swapFrom)
  ) {
    options = swapFrom;
    swapFrom = null;
    swapTo = null;
  } else if (
    typeof swapTo === 'object' &&
    swapTo &&
    !('r' in swapTo) &&
    !('c' in swapTo)
  ) {
    options = swapTo;
    swapTo = null;
  } else if (arguments.length >= 4 && typeof arguments[3] === 'object') {
    options = arguments[3] || {};
  }

  const captureBoards = Boolean(options.captureBoards);
  let totalScore = 0;
  let chainScore = 0;
  let isFirstMerge = true;
  /** @type {Record<string, number>} */
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

    const { score, specialGained: sg, mergeEvents } = applyMerges(
      board,
      matches,
      mergePositions
    );

    totalScore += score;
    if (!isFirstMerge) chainScore += score;

    Object.entries(sg).forEach(([shape, n]) => {
      specialGained[shape] = (specialGained[shape] || 0) + n;
    });

    steps.push({
      type: 'merge',
      score,
      mergeEvents,
      specialGained: { ...sg },
      boardAfterMerge: captureBoards ? cloneBoard(board) : null,
    });
    applyGravityAndRefill(board);
    steps.push({
      type: 'refill',
      boardAfterRefill: captureBoards ? cloneBoard(board) : null,
    });
    isFirstMerge = false;
  }

  return {
    totalScore,
    chainScore,
    specialGained,
    steps,
    hadMatch,
  };
}

/**
 * 尝试交换并解析；无效则还原
 */
export function trySwap(board, from, to) {
  let options = {};
  if (arguments.length >= 4 && typeof arguments[3] === 'object') {
    options = arguments[3] || {};
  }
  swapCells(board, from, to);
  const afterSwapBoard = options.captureBoards ? cloneBoard(board) : null;
  const result = resolveBoard(board, from, to, options);
  if (!result.hadMatch) {
    swapCells(board, from, to);
    return { success: false, afterSwapBoard, ...result };
  }
  return { success: true, afterSwapBoard, ...result };
}

/** 相邻四方向 */
const DIRS = [
  [0, 1],
  [0, -1],
  [1, 0],
  [-1, 0],
];

/**
 * 获取所有能通过交换触发合并的相邻对
 */
export function getValidSwaps(board) {
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

        const sim = cloneBoard(board);
        const from = { r, c };
        const to = { r: nr, c: nc };
        const res = trySwap(sim, from, to);
        if (res.success) swaps.push({ from, to, preview: res });
      }
    }
  }
  return swaps;
}

/**
 * 模拟交换（不修改原棋盘）
 */
export function simulateSwap(board, from, to) {
  let options = {};
  if (arguments.length >= 4 && typeof arguments[3] === 'object') {
    options = arguments[3] || {};
  }
  const sim = cloneBoard(board);
  const result = trySwap(sim, from, to, options);
  if (result.success && options.includeFinalBoard) {
    return { ...result, finalBoard: cloneBoard(sim) };
  }
  return result;
}

/**
 * 洗牌：重新填充，保证至少一个合法交换
 */
export function reshuffleBoard(board, maxAttempts = 200) {
  for (let attempt = 0; attempt < maxAttempts; attempt++) {
    for (let r = 0; r < ROWS; r++) {
      for (let c = 0; c < COLS; c++) {
        board[r][c] = createCell();
      }
    }
    // 消除初始就存在的三连（简单重抽该格）
    let matches = findMatches(board);
    let guard = 0;
    while (matches.length > 0 && guard++ < 500) {
      matches.forEach((m) => {
        m.cells.forEach(({ r, c }) => {
          board[r][c] = createCell();
        });
      });
      matches = findMatches(board);
    }
    if (getValidSwaps(board).length > 0) return true;
  }
  return false;
}

/**
 * 游戏状态
 */
export function createGameState() {
  const targetShapes = pickTwoShapes();
  const board = createEmptyBoard();
  reshuffleBoard(board);

  return {
    board,
    score: 0,
    chainScoreTotal: 0,
    specialScores: Object.fromEntries(SHAPES.map((s) => [s, 0])),
    targetShapes,
    moves: 0,
    won: false,
    lost: false,
    history: [],
  };
}

export function pickTwoShapes() {
  const copy = [...SHAPES];
  const a = copy.splice(Math.floor(Math.random() * copy.length), 1)[0];
  const b = copy[Math.floor(Math.random() * copy.length)];
  return [a, b];
}

/**
 * @param {ReturnType<typeof createGameState>} state
 */
export function checkVictory(state) {
  return state.targetShapes.every((s) => (state.specialScores[s] || 0) >= 4);
}

/**
 * 执行一步 AI/玩家交换
 */
export function executeMove(state, from, to) {
  const result = trySwap(state.board, from, to);
  if (!result.success) return { ok: false, reason: '无效交换' };

  state.score += result.totalScore;
  state.chainScoreTotal += result.chainScore;
  Object.entries(result.specialGained).forEach(([shape, n]) => {
    state.specialScores[shape] = (state.specialScores[shape] || 0) + n;
  });
  state.moves += 1;
  state.history.push({ from, to, ...result });

  if (checkVictory(state)) {
    state.won = true;
  } else if (getValidSwaps(state.board).length === 0) {
    reshuffleBoard(state.board);
  }

  return { ok: true, result };
}

/**
 * 使用“已预演结果”结算（用于动画后提交，保证展示与最终结果一致）
 */
export function commitPreparedMove(state, from, to, preparedResult, finalBoard) {
  if (!preparedResult?.success) return { ok: false, reason: '无效交换' };
  if (!finalBoard) return { ok: false, reason: '缺少最终棋盘' };

  state.board = cloneBoard(finalBoard);
  state.score += preparedResult.totalScore;
  state.chainScoreTotal += preparedResult.chainScore;
  Object.entries(preparedResult.specialGained).forEach(([shape, n]) => {
    state.specialScores[shape] = (state.specialScores[shape] || 0) + n;
  });
  state.moves += 1;
  state.history.push({ from, to, ...preparedResult });

  if (checkVictory(state)) {
    state.won = true;
  } else if (getValidSwaps(state.board).length === 0) {
    reshuffleBoard(state.board);
  }

  return { ok: true, result: preparedResult };
}
