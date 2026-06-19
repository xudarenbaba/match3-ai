// JS 引擎对拍输出脚本。与 test_parity.py 配对，对比 JS/Python 引擎确定性逻辑。
// 由 test_parity.py 以 cwd=tests/ 调用，故用绝对路径解析 src/core。
import { fileURLToPath } from 'url';
import { dirname, resolve } from 'path';

const __dirname = dirname(fileURLToPath(import.meta.url));
const coreDir = resolve(__dirname, '../../src/core');

const { findMatches, pickMergePositions, applyMerges } = await import(resolve(coreDir, 'match.js'));
const { applyPowerupEffect } = await import(resolve(coreDir, 'powerup.js'));

const N = (shape, level) => ({ kind: 'normal', shape, level, frozen: false });
const P = (shape, powerupType) => ({ kind: 'powerup', shape, powerupType, level: 3, frozen: false });
const empty = () => Array.from({ length: 10 }, () => Array.from({ length: 10 }, () => null));

const out = {};

// ── 场景A：横向4连 → direction=row ──
{
  const b = empty();
  for (let c = 0; c < 4; c++) b[0][c] = N('circle', 1);
  const m = findMatches(b);
  const pos = pickMergePositions(m, null, null);
  applyMerges(b, m, pos);
  const rc = b[pos[0].r][pos[0].c];
  out.A_row4 = { dir: m[0].direction, count: m[0].cells.length, powerup: rc?.powerupType ?? null };
}

// ── 场景B：纵向4连 → direction=col ──
{
  const b = empty();
  for (let r = 0; r < 4; r++) b[r][0] = N('circle', 1);
  const m = findMatches(b);
  const pos = pickMergePositions(m, null, null);
  applyMerges(b, m, pos);
  const rc = b[pos[0].r][pos[0].c];
  out.B_col4 = { dir: m[0].direction, count: m[0].cells.length, powerup: rc?.powerupType ?? null };
}

// ── 场景C：L2目标三连消 → special_gained ──
{
  const b = empty();
  for (let c = 0; c < 3; c++) b[0][c] = N('circle', 2);
  const m = findMatches(b);
  const pos = pickMergePositions(m, null, null);
  const merged = applyMerges(b, m, pos);
  out.C_L2merge = { score: merged.score, special: merged.specialGained, cleared: b[pos[0].r][pos[0].c] === null };
}

// ── 场景D：row道具升级（同行 L1*5 + L2目标格*1） ──
{
  const b = empty();
  for (let c = 0; c < 10; c++) b[5][c] = N('star', 1);
  b[5][2] = N('circle', 2); // 目标L2，升级后L3消失
  b[5][5] = P('circle', 'row');
  const res = applyPowerupEffect(b, { r: 5, c: 5 }, { r: 5, c: 6 }, null, ['circle', 'square']);
  // 统计升级后该行等级分布
  const levels = { 1: 0, 2: 0, 3: 0, null: 0 };
  for (let c = 0; c < 10; c++) {
    const cell = b[5][c];
    if (!cell) levels.null++;
    else levels[cell.level]++;
  }
  out.D_rowUpgrade = {
    special: res.specialGained,
    cleared: res.clearedByShape,
    levelsAfter: levels,
  };
}

// ── 场景E：color道具升级（全图同形 star） ──
{
  const b = empty();
  for (let r = 0; r < 10; r++) for (let c = 0; c < 10; c++) b[r][c] = N('star', 1);
  b[3][3] = N('star', 2); // L2 star，非目标
  b[5][5] = P('star', 'color');
  const res = applyPowerupEffect(b, { r: 5, c: 5 }, { r: 5, c: 6 }, null, ['circle', 'square']);
  let l2 = 0, l3gone = 0;
  for (let r = 0; r < 10; r++) for (let c = 0; c < 10; c++) {
    const cell = b[r][c];
    if (cell && cell.level === 2) l2++;
    if (!cell) l3gone++;
  }
  out.E_colorUpgrade = { special: res.specialGained, cleared: res.clearedByShape, l2count: l2, goneCount: l3gone };
}

console.log(JSON.stringify(out, null, 2));
