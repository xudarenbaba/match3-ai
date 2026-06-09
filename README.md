# AI 消消乐（强化学习版）

10x10 网格消消乐。项目包含一套 Python 训练/推理链路和一套浏览器 JS 对局链路，二者通过统一的状态与动作编码对齐。

## 一、核心特性

- MaskablePPO 强化学习：动作空间 180 维（90 横向 + 90 纵向交换）
- 本地推理服务：浏览器每回合请求 Python 服务获取 RL 推荐走法
- 训练防抖增强：
  - 仅允许“有效交换”（能触发消除或道具）进入动作 mask
  - 空转动作惩罚增强
  - 连续重复同一动作惩罚
  - 观测新增上一动作特征 `last_action`
- 推理防抖增强：
  - 若模型重复上一动作，优先切换到其他合法动作
  - 支持 `--stochastic` 非确定性推理模式

## 二、快速开始（训练 -> 推理 -> 游戏）

### 1) 安装依赖（一次）

```powershell
conda activate rlgame
cd D:\otherwise\AI_best\rl_python
pip install -r requirements.txt
```

### 2) 训练模型

```powershell
cd D:\otherwise\AI_best\rl_python
python train/train_ppo.py --curriculum 3 --timesteps 500000 --n-envs 8 --save-dir runs/ppo_match3
```

训练完成后输出：`rl_python/runs/ppo_match3/final_model.zip`

### 3) 启动推理服务（必须先启动）

确定性模式（可复现）：

```powershell
conda activate rlgame
cd D:\otherwise\AI_best\rl_python
python serve/predict_server.py --model runs/ppo_match3/final_model
```

非确定性模式（更不容易重复动作）：

```powershell
conda activate rlgame
cd D:\otherwise\AI_best\rl_python
python serve/predict_server.py --model runs/ppo_match3/final_model --stochastic
```

看到 `推理服务: http://127.0.0.1:8765` 表示启动成功。

### 4) 启动前端游戏

```powershell
cd D:\otherwise\AI_best
python -m http.server 8080
```

浏览器打开 [http://localhost:8080](http://localhost:8080)，点击“RL 出手”即可调用模型。

## 三、项目目录与文件职责

```text
AI_best/
├─ index.html                         # 前端入口页面
├─ README.md                          # 项目说明文档
├─ src/
│  ├─ core/
│  │  ├─ index.js                     # core 对外导出聚合
│  │  ├─ constants.js                 # 棋盘尺寸、图形类型、动作空间大小等常量
│  │  ├─ cells.js                     # 普通格/道具格的数据结构与判定
│  │  ├─ board.js                     # 棋盘基础工具（边界判断、克隆等）
│  │  ├─ gravity.js                   # 掉落补充、冻结相关逻辑
│  │  ├─ match.js                     # 连线匹配、合并升级、产出道具
│  │  ├─ powerup.js                   # 道具作用范围与解冻逻辑
│  │  ├─ resolver.js                  # 交换后整步结算（消除、连锁、道具、补充）
│  │  └─ game-state.js                # 对局状态创建与提交移动（含 lastAction 记录）
│  ├─ actions/
│  │  └─ encoding.js                  # 动作编码/解码、动作 mask 构造（JS 侧）
│  ├─ rl/
│  │  ├─ observation.js               # JS 侧观测编码（20 通道 + 15 维全局）
│  │  └─ reward.js                    # JS 侧奖励定义（可用于分析/对齐）
│  ├─ ai/
│  │  ├─ rl-policy.js                 # 调用 Python 推理服务 /health 与 /predict
│  │  └─ heuristic.js                 # 启发式策略（备用）
│  └─ ui/
│     └─ render.js                    # UI 渲染、动画、RL 按钮行为
└─ rl_python/
   ├─ requirements.txt                # Python 训练与推理依赖
   ├─ match3_engine/
   │  ├─ __init__.py                  # 包导出
   │  ├─ constants.py                 # Python 侧常量（与 JS 对齐）
   │  ├─ cells.py                     # Python 侧格子结构
   │  ├─ board.py                     # Python 侧棋盘工具
   │  ├─ gravity.py                   # Python 侧掉落补充逻辑
   │  ├─ match.py                     # Python 侧匹配与合并
   │  ├─ powerup.py                   # Python 侧道具效果
   │  ├─ resolver.py                  # Python 侧交换结算主逻辑
   │  ├─ actions.py                   # 动作编码/解码与“有效动作”mask
   │  └─ game.py                      # GameState、状态创建、执行一步
   ├─ env/
   │  ├─ __init__.py                  # 包导出
   │  ├─ match3_env.py                # Gym 环境（reset/step/action_masks）
   │  ├─ observation.py               # 训练观测编码（20 通道 + 15 维全局）
   │  └─ reward.py                    # 训练奖励函数（含反重复惩罚）
   ├─ train/
   │  ├─ train_ppo.py                 # 训练入口（MaskablePPO）
   │  └─ eval.py                      # 评估入口（胜率/平均分/平均回报）
   ├─ serve/
   │  ├─ state_codec.py               # 前端 JSON <-> Python GameState 转换
   │  └─ predict_server.py            # 推理 API 服务（含动作去抖）
   ├─ tests/
   │  ├─ test_engine.py               # 引擎逻辑测试
   │  └─ test_predict_load.py         # 模型与推理服务加载测试
   └─ runs/
      └─ ppo_match3/                  # 训练产物目录（checkpoints/best/final）
```

## 四、训练与推理链路

训练阶段（Python 内闭环）：

1. `Match3Env.reset()` 创建新局状态
2. `build_observation()` 生成 `{"board", "global"}`
3. 模型根据 `obs + action_mask` 选择动作
4. `step(action)` 调用引擎执行交换与结算
5. `compute_reward()` 计算奖励并进入下一步

推理阶段（浏览器 + Python 服务）：

1. 前端 `serializeState(state)` 上送当前局面
2. 服务端 `json_to_game_state()` 还原状态
3. 生成 `obs + mask`，模型 `predict()`
4. 返回 `from/to` 给前端
5. 前端执行交换并播放动画

## 五、评估模型

```powershell
cd D:\otherwise\AI_best\rl_python
python train/eval.py --model runs/ppo_match3/final_model --curriculum 3 --episodes 100
```

输出指标：

- `win_rate`：通关率
- `avg_score`：平均游戏分
- `avg_return`：平均 RL 回报
