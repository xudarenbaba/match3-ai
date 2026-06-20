# AI 消消乐（强化学习版）

10×10 网格消消乐。项目包含一套 Python 训练/推理链路和一套浏览器 JS 对局链路，二者通过统一的状态、观测与动作编码对齐。

![模拟游戏界面-1](images/imag1.png)

## 一、核心特性

- **MaskablePPO 强化学习**：动作空间 180 维（90 横向 + 90 纵向相邻交换）
- **CNN 特征提取**：`board` 经两层 3×3 卷积提取空间结构，再与 `global` 拼接送入 MLP，显著提升对「局部 3 连/4 连机会」的识别能力
- **多输入观测**：`board` 30 通道 × 3 帧堆叠 = 90 通道 × 10×10 棋盘 + `global` 15 维全局向量（当前帧）
- **本地推理服务**：浏览器每回合请求 Python 服务获取 RL 推荐走法
- **课程学习**：难度 1–3 逐步增加步数限制、任务目标与冻结格
- **3 帧堆叠时间视野**：将最近 3 帧棋盘观测沿通道轴拼接（90 通道），使模型感知棋盘变化趋势
- **2-step lookahead 推理**：推理时对 top-K 候选动作各模拟一步，按 `W·即时奖励 + γ·V(s')` 选择最优动作（即时奖励加权放大，避免被 value 估值量级淹没）；用真实 3 帧历史估值、固定种子模拟保证可复现
- **训练防抖**：
  - 仅允许「有效交换」（能触发消除或道具）进入动作 mask
  - 空转交换惩罚、每步成本、连续重复同一动作惩罚
  - 观测包含上一动作特征 `last_action`
- **推理模式**：支持 `--stochastic` 非确定性采样，`--top-k` 调节 lookahead 候选数

## 二、快速开始（训练 → 推理 → 游戏）

### 1) 安装依赖（一次）

```bash
conda activate rlgame
cd rl_python
pip install -r requirements.txt
```

### 2) 训练模型

默认输出到 `rl_python/runs/ppo_match3/`：

```bash
cd rl_python
python train/train_ppo.py --curriculum 3 --timesteps 2500000 --n-envs 8 --save-dir runs/ppo_match3_v2
```

> **注意**：v1 模型已多次迭代过时（旧合并规则、29 通道、旧奖励尺度），与当前代码不兼容，必须重新训练。

常用参数：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--curriculum` | `3` | 课程难度 1（简单）/ 2（中等）/ 3（完整） |
| `--timesteps` | `2500000` | 总训练步数（CNN 需更多步收敛，推荐 ≥2.5M） |
| `--n-envs` | `8` | 并行环境数 |
| `--save-dir` | `runs/ppo_match3` | 模型、checkpoint、TensorBoard 日志目录 |
| `--seed` | `42` | 随机种子 |

当前训练超参（`train_ppo.py`）：

| 超参 | 值 | 说明 |
|------|----|------|
| `n_steps` | `2048` | GAE 窗口，覆盖约 20 局 |
| `batch_size` | `512` | 配合 n_steps 增大 |
| `ent_coef` | `0.01` | 熵系数（从 0.03 下调，减少探索让策略收敛，此前 ep_rew 停滞） |
| `vf_coef` | `1.0` | value 损失权重（从 0.5 上调，加强 value 学习，提升 V(s') 估值精度） |
| `gamma` | `0.99` | 折扣因子 |
| `learning_rate` | `3e-4` | Adam 学习率 |
| `features_extractor` | `Match3CnnExtractor` | board 两层 3×3 卷积（C→32→64）+ Linear(256)，global 直接拼接 |
| `net_arch` | `pi=[128], vf=[128]` | CNN 后接的策略/价值 MLP（CNN 已提供主要特征） |

训练产物说明：

```text
runs/<实验名>/
├─ checkpoints/     # 每 1 万步中间模型（gitignored）
├─ best/            # 评估最优模型 best_model.zip（已提交 git）
├─ eval/            # 评估日志 evaluations.npz（gitignored）
├─ tb/              # TensorBoard 事件文件（gitignored）
└─ final_model.zip  # 训练结束最终模型（已提交 git）
```

查看训练曲线：

```bash
tensorboard --logdir runs/ppo_match3_v2/tb
```

### 3) 启动推理服务（必须先启动）

使用最终模型（路径可带或不带 `.zip`）：

```bash
conda activate rlgame
cd rl_python
python serve/predict_server.py --model runs/ppo_match3_v2/final_model
```

使用评估最优模型：

```bash
python serve/predict_server.py --model runs/ppo_match3_v2/best/best_model
```

非确定性模式（更不容易重复动作，推荐对局时使用）：

```bash
python serve/predict_server.py --model runs/ppo_match3_v2/final_model --stochastic
```

调节 2-step lookahead 候选数（默认 8，越大越精确但略慢）：

```bash
python serve/predict_server.py --model runs/ppo_match3_v2/final_model --top-k 12
```

自定义端口：

```bash
python serve/predict_server.py --model runs/ppo_match3_v2/final_model --host 127.0.0.1 --port 8765
```

看到 `推理服务: http://127.0.0.1:8765` 表示启动成功。

API：

- `GET  /health` — 健康检查，返回 `{"ok": true, "model_loaded": true}`
- `POST /predict` — 传入前端序列化的局面 JSON，返回 `{action, from, to, reason}`

### 4) 启动前端游戏

```bash
# 从仓库根目录运行
python -m http.server 8080
```

浏览器打开 [http://localhost:8080](http://localhost:8080)，点击「RL 出手」即可调用模型。

> 推理服务默认地址为 `http://127.0.0.1:8765`，定义在 `src/ai/rl-policy.js` 的 `RL_API`。

## 三、强化学习观测向量（obs）

策略网络输入为 Gymnasium `Dict` 观测，Python 与 JS 侧编码逻辑一致。

### 3.1 总体结构

| 字段 | 形状 | 取值范围 | 说明 |
|------|------|----------|------|
| `board` | `(90, 10, 10)` | `[0, 1]` | 3 帧堆叠棋盘特征：30 通道/帧 × 3 帧，最旧帧在前，最新帧在后 |
| `global` | `(15,)` | `[0, 1]` | 当前帧全局标量特征（步数、分数、任务进度等） |

**帧堆叠说明**：每次决策时将当前帧与前 2 帧的棋盘观测沿通道轴拼接，使模型能感知棋盘的变化趋势。历史帧不足时用零帧补齐。`global` 向量始终取最新帧，不堆叠。

对应常量：`rl_python/env/observation.py` 中 `FRAME_STACK=3`、`BOARD_CHANNELS=30`、`STACKED_BOARD_CHANNELS=90`。

对应代码：`rl_python/env/observation.py`、`src/rl/observation.js`。

### 3.2 单帧 `board` 通道（30 通道/帧，共 3 帧）

棋盘为 10 行 × 10 列，每格在同一 `(r, c)` 位置可激活多个通道。以下为单帧的通道定义，网络实际接收的是 3 帧按此顺序拼接的结果，共 90 通道。

| 通道索引 | 名称 | 含义 |
|----------|------|------|
| 0–2 | `circle` L1 / L2 / L3 | 圆形普通格，等级 1–3 |
| 3–5 | `square` L1 / L2 / L3 | 方形普通格，等级 1–3 |
| 6–8 | `triangle` L1 / L2 / L3 | 三角形普通格，等级 1–3 |
| 9–11 | `star` L1 / L2 / L3 | 星形普通格，等级 1–3 |
| 12 | `powerup_row` | 行升级道具格（横向4连生成） |
| 13 | `powerup_column` | 列升级道具格（纵向4连生成） |
| 14 | `powerup_bomb` | 九宫格炸弹道具格 |
| 15 | `powerup_color` | 同形升级道具格（5连+生成） |
| 16 | `frozen` | 该格被冻结（不可交换） |
| 17 | `target_circle_L1` | 目标圆形，等级 1 |
| 18 | `target_circle_L2` | 目标圆形，等级 2（当前最高级） |
| 19 | `target_circle_L3` | 目标圆形，等级 3（道具格占位） |
| 20 | `target_square_L1` | 目标方形，等级 1 |
| 21 | `target_square_L2` | 目标方形，等级 2（当前最高级） |
| 22 | `target_square_L3` | 目标方形，等级 3 |
| 23 | `target_triangle_L1` | 目标三角形，等级 1 |
| 24 | `target_triangle_L2` | 目标三角形，等级 2（当前最高级） |
| 25 | `target_triangle_L3` | 目标三角形，等级 3 |
| 26 | `target_star_L1` | 目标星形，等级 1 |
| 27 | `target_star_L2` | 目标星形，等级 2（当前最高级） |
| 28 | `target_star_L3` | 目标星形，等级 3 |
| 29 | `layout_mask` | 活跃格标志（1 = 有效格，0 = void 格） |

空位（`null`）不激活任何通道。

### 3.3 `global` 向量（15 维）

| 索引 | 名称 | 计算公式 | 说明 |
|------|------|----------|------|
| 0 | `steps_used_ratio` | `steps_used / total_steps` | 已用步数占比 |
| 1 | `steps_left_ratio` | `steps_left / total_steps` | 剩余步数占比 |
| 2 | `score_norm` | `min(1, score / 5000)` | 总分归一化 |
| 3 | `chain_score_norm` | `min(1, chain_score_total / 3000)` | 连锁分归一化 |
| 4–7 | `task_score_*` | `task_scores[shape] / 4` | 四种图形各自任务进度（目标为 4） |
| 8–11 | `is_target_*` | `1` 若该图形是本局目标，否则 `0` | 目标图形 one-hot |
| 12 | `min_target_progress` | `min(task_scores[target] / 4)` | 最慢目标图形的完成度 |
| 13 | `won` | `1` 若已通关，否则 `0` | 胜负标志 |
| 14 | `last_action_norm` | `(last_action + 1) / 180` | 上一步动作编号归一化（无历史时为 `-1` → `0`） |

### 3.4 动作空间与 mask

- **动作总数**：180（`MAX_ACTIONS`）
  - `0–89`：水平相邻交换，`action = r * 9 + c`（`(r,c)` 与 `(r,c+1)`）
  - `90–179`：垂直相邻交换，`action = 90 + r * 10 + c`（`(r,c)` 与 `(r+1,c)`）
- **动作 mask**：仅「有效交换」为 `1`（交换后能消除、触发道具或得分）；若无有效交换则退化为全部相邻交换，避免环境卡死。

对应代码：`rl_python/match3_engine/actions.py`、`src/actions/encoding.js`。

### 3.5 课程难度（`--curriculum`）

| 等级 | 步数上限 | 任务目标数 | 目标图形数 | 冻结格 |
|------|----------|------------|------------|--------|
| 1 | 150 | 2 分/图形 | 1 种 | 否 |
| 2 | 120 | 3 分/图形 | 2 种 | 是 |
| 3 | 100 | 4 分/图形 | 2 种 | 是 |

### 3.6 道具系统

道具由 L1 的多连消生成，效果为「升级」而非「整片消除」：

| 道具 | 生成条件 | 效果 |
|------|----------|------|
| **行（row）** | 横向 4 连消 | 该行其余普通格等级 +1；升到 L2 的目标格直接计任务分并消失 |
| **列（column）** | 纵向 4 连消 | 该列其余普通格等级 +1；升到 L2 的目标格直接计任务分并消失 |
| **同（color）** | 5 连+消 | 全图相同形状的普通格等级 +1；升到 L2 的目标格直接计任务分并消失 |
| **炸（bomb）** | 随机掉落 | 九宫格范围直接消除（保留原消除逻辑） |

> 升级说明：L1→L2 保留在原位；L2→升级即视为完成该格目标（计 1 个 `special_gained`）并消失。升级后触发重力补位与连锁。
>
> 道具生成的任务分统一只走 `special_gained`，不重复计入 `task_from_powerup`。

**关键不变量（格子生成分布）**：顶部掉落补充用 `create_cell`（97.5% L1 普通格 + 2.5% 道具），初始棋盘用 `create_initial_cell`（纯 L1、无道具）。JS 与 Python 两侧严格一致，确保训练分布 = 实际游戏分布。

## 四、项目目录与文件职责

```text
match3-ai/
├─ index.html                         # 前端入口：棋盘 UI、侧边栏统计、RL/新局按钮
├─ style.css                          # 页面样式（布局、棋盘格、按钮、日志面板）
├─ README.md                          # 项目说明文档
├─ .gitignore                         # Git 忽略规则（训练产物、__pycache__ 等）
├─ src/
│  ├─ core/
│  │  ├─ index.js                     # core 模块统一导出（供 UI 层引用）
│  │  ├─ constants.js                 # 棋盘尺寸、图形类型、动作空间、步数上限等常量
│  │  ├─ cells.js                     # 普通格/道具格数据结构与类型判定
│  │  ├─ board.js                     # 棋盘创建、克隆、边界判断
│  │  ├─ gravity.js                   # 消除后掉落补充、随机洗牌、冻结格生成
│  │  ├─ match.js                     # 三连匹配检测、合并升级、道具生成
│  │  ├─ powerup.js                   # 四种道具（行/列升级、炸弹消除、同形升级）效果计算
│  │  ├─ resolver.js                  # 交换后整步结算（消除链、道具连锁、得分）
│  │  └─ game-state.js                # 对局状态创建、提交移动、记录 lastAction
│  ├─ actions/
│  │  └─ encoding.js                  # 动作编解码（180 维）、相邻交换枚举
│  ├─ rl/
│  │  ├─ observation.js               # JS 侧观测编码（30 通道单帧 + 15 维 global；帧堆叠由 rl-policy.js 维护）
│  │  └─ reward.js                    # JS 侧奖励定义（与 Python 训练奖励对齐参考）
│  ├─ ai/
│  │  ├─ rl-policy.js                 # 调用推理服务 /health、/predict；序列化局面 JSON
│  │  └─ heuristic.js                 # 启发式策略（无模型时的备用走棋逻辑）
│  └─ ui/
│     └─ render.js                    # DOM 渲染、交换动画、RL 按钮、操作日志
└─ rl_python/
   ├─ requirements.txt                # Python 依赖（gymnasium、sb3、torch 等）
   ├─ match3_engine/                  # 游戏引擎（与 JS core 规则对齐）
   │  ├─ __init__.py                  # 包导出
   │  ├─ constants.py                 # ROWS/COLS/SHAPES/MAX_ACTIONS 等常量
   │  ├─ cells.py                     # NormalCell、PowerCell 数据结构
   │  ├─ board.py                     # 棋盘克隆、边界、空格判定
   │  ├─ gravity.py                   # 掉落、补充、洗牌、冻结
   │  ├─ match.py                     # 匹配检测与合并升级
   │  ├─ powerup.py                   # 道具效果与范围计算
   │  ├─ resolver.py                  # try_swap / 整步结算主逻辑
   │  ├─ actions.py                   # 动作编解码、有效交换 mask 构造
   │  └─ game.py                      # GameState、create_game_state、execute_move
   ├─ env/
   │  ├─ __init__.py                  # 包导出
   │  ├─ match3_env.py                # Gymnasium 环境（reset/step/action_masks/课程学习）
   │  ├─ observation.py               # 训练观测编码（单帧 30ch；堆叠后 90ch × 3帧 + global 15d）
   │  └─ reward.py                    # 训练奖励（消除分、任务分、空转/重复动作惩罚）
   ├─ train/
   │  ├─ train_ppo.py                 # MaskablePPO 训练入口（checkpoint + eval 回调）
   │  ├─ features.py                  # Match3CnnExtractor 自定义 CNN 特征提取器
   │  └─ eval.py                      # 模型评估（胜率 / 平均分 / 平均回报）
   ├─ serve/
   │  ├─ state_codec.py               # 前端 JSON ↔ Python GameState 转换
   │  └─ predict_server.py            # HTTP 推理 API（帧堆叠 + 2-step lookahead）
   ├─ tests/
   │  ├─ test_engine.py               # 引擎与环境冒烟测试（步进、观测形状）
   │  ├─ test_parity.py               # JS/Python 引擎对拍测试（需 node）
   │  ├─ parity_js.mjs                # 对拍 JS 侧输出脚本
   │  └─ test_predict_load.py         # 加载模型并执行一次 predict 的脚本
   └─ runs/                           # 训练产物（部分已提交 git，见下方说明）
       ├─ ppo_match3_v1/               # v1 模型（旧合并规则下训练，建议重训覆盖）
       └─ ppo_match3_v2/               # 推荐新训练目录（final_model.zip 和 best/ 提交 git）
```

## 五、训练与推理链路

**训练阶段（Python 内闭环）：**

1. `Match3Env.reset()` 按课程难度创建新局，初始化帧缓冲（`_frame_buffer`）
2. `build_observation()` 生成单帧 `{"board"(30ch), "global"}`，压入帧缓冲
3. `stack_observations()` 将最近 3 帧拼接为 `{"board"(90ch), "global"}`
4. 模型根据堆叠 `obs + action_mask` 选择动作
5. `step(action)` 调用引擎执行交换与结算，新帧压入缓冲
6. `compute_reward()` 计算奖励并进入下一步

**推理阶段（浏览器 + Python 服务）：**

1. 前端 `buildObservation(state)` 构建当前单帧，压入本地 `_frameHistory`（最多保留 3 帧）
2. `findRlMove()` 将当前局面 JSON + `frameHistory` 一起 POST 到 `/predict`
3. 服务端从 `frameHistory` 重建各帧 numpy 数组，调用 `stack_observations()` 堆叠
4. **2-step lookahead**：取 top-K（默认 12）候选动作，对每个候选用 Python 引擎模拟一步，用真实 3 帧历史 `[f_{t-1}, f_t, f_{t+1}]` 经 value head 估算下一状态价值，选择 `W·即时奖励 + 0.99·V(s')`（`W=8`，放大即时奖励权重）最高的动作。模拟用固定种子保证结果可复现
5. 返回 `from/to` 坐标给前端执行并播放动画

## 六、奖励函数设计

对应代码：`rl_python/env/reward.py`。引擎侧 `cleared_by_shape` 由 `match3_engine/match.py` 的 `apply_merges` 生成，经 `resolver.py` 汇总后传入。

### 6.1 密集信号（每步即时）

| 信号 | 公式 | 说明 |
|------|------|------|
| **目标色块消除** | `cleared_target_cells × +0.3` | 每消除 1 格目标色块即时奖励（L1 三连消 cleared=n-1；L2 三连消 cleared=n） |
| **L2 目标格合并额外奖励** | `+1.5` / 次 | L2 为最高级，直接计任务分，在 target_clear 基础上额外叠加 |
| **非目标色块消除** | `cleared_non_target_cells × -0.05` | 轻微惩罚，制造对目标的相对偏好 |
| **4连消额外奖励** | `+1.0` / 次 | 让 4连明显优于 3连（目标/非目标均计），解决模型忽视多连消机会的问题 |
| **5连+消额外奖励** | `+2.0` / 次 | 让 5连明显优于 4连（目标/非目标均计） |
| **生成 column 道具**（4连消） | `+0.4` | 鼓励多连消触发道具 |
| **生成 color 道具**（5连+消） | `+0.6` | 鼓励多连消触发道具 |
| 基础得分 | `total_score × 0.005` | 降权保留，避免与密集信号重叠 |
| 连锁得分 | `chain_score × 0.005` | 鼓励连锁消除 |

### 6.2 稀疏信号（L2 合并完成时）

| 信号 | 公式 | 说明 |
|------|------|------|
| **任务进度** | `task_delta × +3.0` | L2 三连消 → `special_gained` → `task_score +1` 时触发（L2 为当前最高级） |

消除目标 L2 时各信号叠加：`target_clear(+0.9)` + `L2_bonus(+1.5)` + `task_delta(+3.0)` + `score(+0.03)` - `step(-0.03)` = **+5.4**，远高于其他任何单步操作，强化"完成目标"这个最终动作。

### 6.3 步数惩罚

| 信号 | 值 | 条件 |
|------|-----|------|
| 每步成本 | `-0.03` | 每步无条件 |
| 空挥惩罚 | `-0.2` | 无消除且无道具 |
| 重复动作 | `-0.25` | 与上一步动作相同 |

### 6.4 终局信号

| 结果 | 奖励 |
|------|------|
| 胜利 | `+12.0 + 剩余步数 × 0.05` |
| 失败 | `-4.0 + 每目标缺口数 × -1.0` |

> 终局奖励尺度已下调（原 `win_bonus=50`）。原因：过大的终局奖励使 episode 回报量级达 ~116，value head 难以精确拟合，估值误差（±3.8）恰好淹没单步即时奖励差异（~4），导致 2-step lookahead 的决策被 value 噪声主导而非即时奖励。下调后回报量级降至 ~40，value 精度相对单步信号显著提升。过程奖励（任务分、多连消等）保持不变。

### 6.5 即时奖励优先级排序

下表为各典型操作的单步即时奖励（不含终局信号），反映模型被训练成优先选择的操作顺序：

| 优先级 | 操作 | 即时奖励 | 说明 |
|--------|------|---------|------|
| 1 | **L2 目标 3连消**（task+1） | **+5.39** | 唯一直接得任务分的操作，信号最强 |
| 2 | L1 目标 5连消 → color 道具 | +3.79 | 多连消奖励 + 道具奖励叠加 |
| 3 | L1 非目标 5连消 → color 道具 | +2.39 | 非目标但多连消信号强 |
| 4 | L1 目标 4连消 → column 道具 | +2.29 | 多连消奖励 + 道具奖励叠加 |
| 5 | L1 非目标 4连消 → column 道具 | +1.24 | 非目标但有多连消奖励 |
| 6 | L1 目标 3连消 | +0.58 | 普通目标消除，为凑 L2 铺垫 |
| 7 | L1 非目标 3连消 | -0.12 | 轻微负收益，避免浪费步数 |
| — | 空交换（无消除） | -0.23 | 步数惩罚 + 空挥惩罚 |

**关键设计取舍：**
- L2 目标消除（+5.4）>> L1 目标 5 连消（+3.8）：直接消 L2 仍优先于凑大连消
- 目标 4连（+2.3）vs 目标 3连（+0.6）：差距从 +0.7 扩大到 **+1.7**，模型现在有足够信号学会识别多连消机会
- 非目标 5连（+2.4）> 目标 3连（+0.6）：生成 color 道具的价值高于普通目标消除，避免模型死守目标格忽视大连消机会

## 七、评估模型

```bash
# 在 rl_python/ 目录下运行
python train/eval.py --model runs/ppo_match3_v2/final_model --curriculum 3 --episodes 100

# 随机策略基线（不传 --model）
python train/eval.py --curriculum 3 --episodes 100
```

输出指标：

| 指标 | 含义 |
|------|------|
| `win_rate` | 通关率 |
| `avg_score` | 平均游戏分 |
| `avg_return` | 平均 RL 累积回报 |

## 八、运行测试

```bash
# 在 rl_python/ 目录下运行
python -m pytest tests/test_engine.py -v
```

手动验证模型可加载（需本地已有对应模型文件）：

```bash
python tests/test_predict_load.py
```
