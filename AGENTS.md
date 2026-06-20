# AGENTS.md

## Repo structure

Two independent layers — neither depends on the other's toolchain:

- `src/` — vanilla ES-module JavaScript, served raw (no build step, no npm)
- `rl_python/` — Python RL training, environment, and inference server

No package.json, no Makefile, no CI. Python deps only in `rl_python/requirements.txt`.

## Python setup

All Python commands run from `rl_python/`, not the repo root.

```bash
# one-time install (conda env named "rlgame")
conda activate rlgame
pip install -r requirements.txt
```

Scripts add `rl_python/` to `sys.path` themselves — packages are never installed, just path-patched.

## Developer commands

```bash
# tests
python -m pytest tests/test_engine.py -v        # run from rl_python/
python -m pytest tests/test_parity.py -v        # JS/Python engine parity (needs node)
python tests/test_predict_load.py               # requires a saved model

# inference server (must be running for the frontend's RL button to work)
# 2-step lookahead: score = W*r_imm + gamma*V(s'), W=8 (reward weight), top-k=12, fixed-seed sim
python serve/predict_server.py --model runs/ppo_match3_v2/final_model
python serve/predict_server.py --model runs/ppo_match3_v2/final_model --stochastic
python serve/predict_server.py --model runs/ppo_match3_v2/final_model --top-k 16

# frontend (serve from repo root)
python -m http.server 8080

# training (CNN extractor + 2048 n_steps + vf_coef=1.0 + ent_coef=0.01; ~2.5M steps recommended)
python train/train_ppo.py --curriculum 3 --timesteps 2500000 --n-envs 8 --save-dir runs/ppo_match3_v2

# evaluation
python train/eval.py --model runs/ppo_match3_v2/final_model --curriculum 3 --episodes 100

# tensorboard
tensorboard --logdir runs/ppo_match3_v2/tb
```

## Critical invariant: JS/Python parity

The game engine and RL helpers are implemented **twice** in parallel. If you change one side, update the other:

| Python | JavaScript |
|---|---|
| `rl_python/match3_engine/` | `src/core/` |
| `rl_python/env/observation.py` | `src/rl/observation.js` |
| `rl_python/match3_engine/actions.py` | `src/actions/encoding.js` |
| `rl_python/env/reward.py` | `src/rl/reward.js` |

`tests/test_parity.py` runs the JS engine via node and diffs deterministic outputs (match direction, powerup generation, L2 merge scoring, powerup upgrade) against Python. Run it after touching either engine.

**Cell-generation invariant (a past parity bug):** `create_cell` = 97.5% L1 normal + 2.5% powerup; `create_initial_cell` = pure L1, no powerup. Both sides must match — if Python falls back to random levels, the training distribution diverges from the real game and the model learns to wait for L2 cells to "fall from the sky."

**Known intentional divergence:** Python reward includes `reverse_swap_penalty: -0.25` and `empty_swap: -0.2`; JS omits the first and uses `-0.05` for the second. Do not "fix" this — the JS reward is for display only.

## Inference server / frontend wiring

The browser's "RL 出手" button POSTs to `http://127.0.0.1:8765/predict`. This URL is hardcoded in `src/ai/rl-policy.js`. The inference server must be started separately before using the RL feature in the browser.

## Observation / network

Observation is `Dict{board: (90,10,10), global: (15,)}` — 30 channels/frame × 3 stacked frames. Channel 12-15 are powerup types `row/column/bomb/color` (must match `POWERUP_TYPES` order). Changing `BOARD_CHANNELS` requires retraining. Training uses a custom CNN extractor `train/features.py:Match3CnnExtractor` (two 3×3 convs over board, concatenated with global).

## Trained model in git

`rl_python/runs/ppo_match3_v1/final_model.zip` and `runs/ppo_match3_v1/best/` are **committed to git** (trained under an older 29-channel observation, before the powerup-upgrade rework and CNN extractor). It is **incompatible** with current code (30-channel board + CNN). A full retrain targeting `runs/ppo_match3_v2/` is required. `checkpoints/`, `tb/`, and `eval/` subdirs are gitignored.

## Action space

180-dimensional Discrete:
- 0–89: horizontal swaps — `action = row * 9 + col`
- 90–179: vertical swaps — `action = 90 + row * 10 + col`

## No linters, no type checkers, no formatters configured

There is no ruff, mypy, eslint, prettier, or tsconfig. Don't add config for these without discussion.
