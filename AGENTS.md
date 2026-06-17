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
python tests/test_predict_load.py               # requires a saved model

# inference server (must be running for the frontend's RL button to work)
# uses 2-step lookahead by default (top-k=8); add --top-k N to adjust
python serve/predict_server.py --model runs/ppo_match3_v2/final_model
python serve/predict_server.py --model runs/ppo_match3_v2/final_model --stochastic
python serve/predict_server.py --model runs/ppo_match3_v2/final_model --top-k 12

# frontend (serve from repo root)
python -m http.server 8080

# training
python train/train_ppo.py --curriculum 3 --timesteps 500000 --n-envs 8

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

**Known intentional divergence:** Python reward includes `reverse_swap_penalty: -0.25` and `empty_swap: -0.2`; JS omits the first and uses `-0.05` for the second. Do not "fix" this — the JS reward is for display only.

## Inference server / frontend wiring

The browser's "RL 出手" button POSTs to `http://127.0.0.1:8765/predict`. This URL is hardcoded in `src/ai/rl-policy.js`. The inference server must be started separately before using the RL feature in the browser.

## Trained model in git

`rl_python/runs/ppo_match3_v1/final_model.zip` and `runs/ppo_match3_v1/best/` are **committed to git** (trained under old merge rules — L2 merge did not count as task score). A retrain targeting `runs/ppo_match3_v2/` is recommended. `checkpoints/`, `tb/`, and `eval/` subdirs are gitignored.

## Action space

180-dimensional Discrete:
- 0–89: horizontal swaps — `action = row * 9 + col`
- 90–179: vertical swaps — `action = 90 + row * 10 + col`

## No linters, no type checkers, no formatters configured

There is no ruff, mypy, eslint, prettier, or tsconfig. Don't add config for these without discussion.
