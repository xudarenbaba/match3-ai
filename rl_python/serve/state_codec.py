from __future__ import annotations

from match3_engine.cells import NormalCell, PowerCell
from match3_engine.constants import SHAPES, MAX_STEPS, TASK_TARGET
from match3_engine.game import GameState
from match3_engine.layouts import get_layout


def json_to_game_state(data: dict) -> GameState:
    board = []
    for row in data["board"]:
        brow = []
        for cell in row:
            if cell is None:
                brow.append(None)
            elif cell.get("kind") == "powerup":
                brow.append(
                    PowerCell(
                        shape=cell["shape"],
                        powerup_type=cell.get("powerupType") or cell.get("powerup_type"),
                        frozen=bool(cell.get("frozen", False)),
                    )
                )
            else:
                brow.append(
                    NormalCell(
                        shape=cell["shape"],
                        level=int(cell["level"]),
                        frozen=bool(cell.get("frozen", False)),
                    )
                )
        board.append(brow)

    task_scores = {s: 0 for s in SHAPES}
    for key, value in (data.get("taskScores") or {}).items():
        task_scores[key] = int(value)

    # 布局：从前端传来的 layoutName 还原，或使用 layoutMask（直接传 0/1 数组）
    layout_name = data.get("layoutName", "full")
    if data.get("layoutMask"):
        layout = data["layoutMask"]
    else:
        layout = get_layout(layout_name)

    return GameState(
        board=board,
        score=int(data.get("score", 0)),
        chain_score_total=int(data.get("chainScoreTotal", 0)),
        task_scores=task_scores,
        target_shapes=list(data.get("targetShapes", [])),
        task_target=int(data.get("taskTarget", TASK_TARGET)),
        total_steps=int(data.get("totalSteps", MAX_STEPS)),
        steps_used=int(data.get("stepsUsed", 0)),
        last_action=int(data.get("lastAction", -1)),
        won=bool(data.get("won", False)),
        over=bool(data.get("over", False)),
        layout=layout,
        layout_name=layout_name,
    )
