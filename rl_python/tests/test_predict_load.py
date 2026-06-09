import os
import random
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from sb3_contrib import MaskablePPO
from match3_engine.game import create_game_state
from env.observation import build_observation
from match3_engine.actions import build_action_mask, decode_action


def main():
    path = os.path.join(ROOT, "runs", "ppo_match3", "final_model.zip")
    assert os.path.isfile(path), path
    model = MaskablePPO.load(path)
    state = create_game_state(random.Random(0))
    obs = build_observation(state)
    mask = build_action_mask(state.board)
    action, _ = model.predict(obs, action_masks=mask.astype(bool), deterministic=True)
    swap = decode_action(int(action))
    assert swap is not None
    print("predict ok", int(action), swap)


if __name__ == "__main__":
    main()
