"""JS/Python 引擎对拍测试，确保两侧确定性逻辑完全一致。

通过 node 运行 tests/parity_js.mjs 得到 JS 引擎输出，与 Python 引擎对相同场景的
输出逐项对比。覆盖：match 方向检测、道具生成、L2 合并计分、道具升级逻辑。

若本机无 node，测试自动跳过。
"""

import json
import os
import random
import shutil
import subprocess
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from match3_engine.cells import NormalCell, PowerCell
from match3_engine.board import create_empty_board
from match3_engine.match import find_matches, pick_merge_positions, apply_merges
from match3_engine.powerup import powerup_upgrade_targets

JS_SCRIPT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "parity_js.mjs")


def build_py_results() -> dict:
    rng = random.Random(0)

    def N(shape, level):
        return NormalCell(shape=shape, level=level)

    def P(shape, powerup_type):
        return PowerCell(shape=shape, powerup_type=powerup_type)

    out = {}

    # 场景A：横向4连 → row 道具
    b = create_empty_board()
    for c in range(4):
        b[0][c] = N("circle", 1)
    m = find_matches(b)
    pos = pick_merge_positions(m, None, None)
    apply_merges(b, m, pos, rng)
    rc = b[pos[0]["r"]][pos[0]["c"]]
    out["A_row4"] = {"dir": m[0]["direction"], "count": len(m[0]["cells"]),
                     "powerup": getattr(rc, "powerup_type", None)}

    # 场景B：纵向4连 → column 道具
    b = create_empty_board()
    for r in range(4):
        b[r][0] = N("circle", 1)
    m = find_matches(b)
    pos = pick_merge_positions(m, None, None)
    apply_merges(b, m, pos, rng)
    rc = b[pos[0]["r"]][pos[0]["c"]]
    out["B_col4"] = {"dir": m[0]["direction"], "count": len(m[0]["cells"]),
                     "powerup": getattr(rc, "powerup_type", None)}

    # 场景C：L2目标三连消
    b = create_empty_board()
    for c in range(3):
        b[0][c] = N("circle", 2)
    m = find_matches(b)
    pos = pick_merge_positions(m, None, None)
    merged = apply_merges(b, m, pos, rng)
    out["C_L2merge"] = {"score": merged["score"], "special": merged["special_gained"],
                        "cleared": b[pos[0]["r"]][pos[0]["c"]] is None}

    # 场景D：row 道具升级
    b = create_empty_board()
    for c in range(10):
        b[5][c] = N("star", 1)
    b[5][2] = N("circle", 2)
    b[5][5] = P("circle", "row")
    _t, sp, cl = powerup_upgrade_targets(b, {"r": 5, "c": 5}, {"r": 5, "c": 6}, None, ["circle", "square"])
    levels = {"1": 0, "2": 0, "3": 0, "null": 0}
    for c in range(10):
        cell = b[5][c]
        levels["null" if cell is None else str(cell.level)] += 1
    out["D_rowUpgrade"] = {"special": sp, "cleared": cl, "levelsAfter": levels}

    # 场景E：color 道具升级
    b = create_empty_board()
    for r in range(10):
        for c in range(10):
            b[r][c] = N("star", 1)
    b[3][3] = N("star", 2)
    b[5][5] = P("star", "color")
    _t, sp, cl = powerup_upgrade_targets(b, {"r": 5, "c": 5}, {"r": 5, "c": 6}, None, ["circle", "square"])
    l2 = sum(1 for r in range(10) for c in range(10) if b[r][c] and b[r][c].level == 2)
    gone = sum(1 for r in range(10) for c in range(10) if b[r][c] is None)
    out["E_colorUpgrade"] = {"special": sp, "cleared": cl, "l2count": l2, "goneCount": gone}

    return out


def test_js_python_engine_parity():
    node = shutil.which("node")
    if node is None:
        pytest.skip("node 未安装，跳过 JS/Python 对拍")

    proc = subprocess.run(
        [node, JS_SCRIPT], capture_output=True, text=True, cwd=os.path.dirname(JS_SCRIPT)
    )
    assert proc.returncode == 0, f"node 运行失败: {proc.stderr}"
    js_results = json.loads(proc.stdout)
    py_results = build_py_results()

    assert js_results.keys() == py_results.keys()
    for key in py_results:
        assert js_results[key] == py_results[key], (
            f"场景 {key} JS/Python 不一致:\n  JS={js_results[key]}\n  PY={py_results[key]}"
        )


if __name__ == "__main__":
    test_js_python_engine_parity()
    print("JS/Python 引擎对拍通过")
