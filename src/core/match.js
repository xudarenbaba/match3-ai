import { ROWS, COLS } from './constants.js';
import { cellKey, inBounds } from './board.js';
import { createNormalCell } from './cells.js';

export function findMatches(board) {
  const matches = [];

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
