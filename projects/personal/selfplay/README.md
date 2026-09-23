# 自我对弈策略项目（selfplay）

人工维护的策略项目，有两条互不共享检查点的训练路线，都用 PyTorch 训练：

- **规则教师指导**（[`guided/`](guided/)，新实验使用）：以规则研究的冻结冠军为教师，先纯模仿，再用真实桌面分做 PPO 并保留按师生差距调整的教师辅助损失；Actor 由独立 GPU 服务推理，正式内网比赛部署在 4070。用法见[训练手册](guided/README.md)，决定见 [ADR-0011](../../../docs/adr/0011-rule-guided-policy-learning.md)。
- **训练方案迭代**（`recipes/` + `guided/trial.py`）：自动研究里的 RL 方案分支。研究模型反复修改模仿之后的强化学习训练方案，每次从固定起点限时训练，训练出的模型交给研究框架，和规则分支一样做分段配对评估、参加共同大乱斗；进度在工作台 RL 分支的卡片上查看。决定见 [ADR-0013](../../../docs/adr/0013-unified-research-branches-and-common-brawl.md)、[ADR-0012](../../../docs/adr/0012-recipe-iteration-for-rl-training.md)，接口见 [recipes/API.md](recipes/API.md)。
- **自我对弈 PPO**（`train/`，下文）：联赛池自我对弈，ONNX Runtime 发布到平台。设计见 [ADR-0009](../../../docs/adr/0009-independent-selfplay-trainer-for-personal-project.md)。

术语（联赛池、桌面分、阶段总分、教师策略）见 [CONTEXT.md](../../../CONTEXT.md)。

## 目录

```text
selfplay/
├── project.toml        # id=personal-selfplay, family=selfplay, dependencies=[onnxruntime]
├── src/                # 原样进入发布包，只依赖 numpy + onnxruntime + mahjong_ai 运行时
│   ├── strategy.py     # build(model_dir) 入口
│   └── selfplay_core/  # actions（231 动作编号）、encoding（观察编码）、policy（ONNX 推理）
├── guided/             # 规则教师指导路线，依赖 torch，不进发布包；见 guided/README.md
├── recipes/            # 训练方案迭代：iteration.toml 固定条件、baseline/ 初始方案、API.md 接口
├── train/              # 自我对弈训练器，依赖 torch，不进发布包
│   ├── run.py          # 入口：fork 环境进程 → GPU 批量推理 + PPO + 联赛池
│   ├── env.py          # 直接驱动 HandState 的对局；critic 隐藏状态；hold-back 轨迹装配
│   ├── actors.py       # 环境 worker 进程；rollout.py 共享内存样本存储与 GAE
│   ├── server.py       # 按模型槽位分批的 GPU 推理
│   ├── learner.py      # PPO：裁剪目标、symlog 双热价值头、辅助头
│   ├── league.py       # 历史检查点 PFSP 采样 + 固定对手
│   ├── models/         # mlp / resnet / transformer，共用编码与头
│   ├── export.py       # torch → ONNX 与一致性校验
│   ├── evaluate.py     # 离线评估：固定对手、分块置信区间、多策略互打
│   ├── sweep_nodes.py  # 逐节点评估：导出联赛快照与 best，同种子配对初筛，前几名大乱斗复核
│   ├── eval_stability.py  # 评估规模测量：多种子长评估取前缀，给出各场数下的标准误与排名稳定性
│   └── configs/        # 三种架构与冒烟配置
├── scripts/            # 参赛流程：大乱斗选优、冻结、上线与自动续跑
├── tests/              # 项目内测试
└── models/ runs/ releases/   # 训练产物与发布，不提交
```

## 使用

训练与离线评估需要带 CUDA 的 torch。仓库 `.venv` 在 Mac 上可装可选依赖 `.[model]` 的 torch，只用于研究的模型座位推理（MPS），不用于本路线训练；本机用系统 conda 的 Python 3.11（自带 torch），并已为它就地编译 `_fast`，命令前加 `PYTHONPATH=src`。发布、沙箱与平台对战仍用 `.venv/bin/mahjong-ai`。

```bash
# 训练（配置见 train/configs/，主训练用 transformer-main.toml）
PYTHONPATH=src python3 projects/personal/selfplay/train/run.py --config projects/personal/selfplay/train/configs/transformer-main.toml --hours 72
PYTHONPATH=src python3 projects/personal/selfplay/train/run.py --config projects/personal/selfplay/train/configs/transformer-main.toml --resume
# 换固定对手：--opponents 指向对手名单 JSON（与 evaluate.py --entries 同格式），--fixed-prob-end 设退火后固定对手占非本人座位的比例
PYTHONPATH=src python3 projects/personal/selfplay/train/run.py --config projects/personal/selfplay/train/configs/transformer-main.toml --run-name <run> --resume --opponents <entries.json> --fixed-prob-end 0.5
# 离线评估导出的 actor；--brawl 让多方均衡混战（'auto' = 自动研究现役代表，'builtin' = 四个内置基线）
PYTHONPATH=src python3 projects/personal/selfplay/train/evaluate.py --policy projects/personal/selfplay/runs/transformer-main/export --games 1024
PYTHONPATH=src python3 projects/personal/selfplay/train/evaluate.py --policy <export...> --brawl auto builtin --games 2048
# 逐节点评估（--wait-pid 等训练进程退出再开始；--entries 冻结的对手名单写在 <run>/sweep/opponents.json）
PYTHONPATH=src python3 projects/personal/selfplay/train/sweep_nodes.py --run projects/personal/selfplay/runs/transformer-main --wait-pid <训练 PID>
# 评估规模测量（结果写到 runs/stability/summary.md；--analyze 只重算不重跑）
PYTHONPATH=src python3 projects/personal/selfplay/train/eval_stability.py --seeds 11 12 13 14 15
# 发布：把导出复制到 models/ 后走通用流程
cp projects/personal/selfplay/runs/resnet/export/{policy.onnx,meta.json} projects/personal/selfplay/models/
.venv/bin/mahjong-ai release-build --project projects/personal/selfplay --version v0.1.0
.venv/bin/mahjong-ai competition-register projects/personal/selfplay/releases/v0.1.0
# 项目测试
.venv/bin/python -m pytest -q projects/personal/selfplay/tests
```

`scripts/tonight.sh` 把「多方大乱斗 → 冻结胜者（RL 走 release-build，自动研究候选走 `mahjong-ai export`）→ 沙箱试跑 → `--go` 时用 `online-play --join` 上线」串成一条流程；`scripts/go-ertest.sh` 直接用已冻结的参赛包上线并在异常退出时用同一目录自动续跑。两者都读 `.local/credentials/free-play.json` 里的锦标赛令牌。

训练日志在 `runs/<run_name>/log.jsonl`，检查点在 `runs/<run_name>/ckpt/`，联赛快照在 `runs/<run_name>/league/`，结束时导出到 `runs/<run_name>/export/` 并做 torch 与 ONNX 的输出比对。`--resume` 从 `ckpt/latest.pt` 续跑，配置沿用 `runs/<run_name>/config.json`，只有命令行显式给出的参数会覆盖，所以换过的对手名单、比例等不会因漏带参数被换回默认；进度表按「累计时长 ÷ `--hours`」计算，续跑时改 `--hours` 会让学习率、熵系数与塑形权重跳到新位置，且 `best.pt` 在续跑第一轮会被覆盖。日志的 `eval` 与 `best.pt` 取最近 `eval_window`（默认 1024）场评估：对随机抽取的固定对手、不配对、跨若干次迭代的权重，约 ±1.6，只看趋势；`best.pt` 是这个窗口创新高时的快照，含挑选偏差，只作为 `sweep_nodes.py` 的一个节点，选版以逐节点配对评估和换种子复核为准。联赛快照每 30 分钟一份，移出联赛池后文件仍保留，`sweep_nodes.py` 以它们为节点，结果写到 `runs/<run_name>/sweep/summary.md`；同种子评估可复现，相邻节点配对约把差值标准误降三成。

## 关键约定

- 奖励是每局本人收支除以底分，整场之和等于桌面分；向听势能奖励只在训练前两成时间线性退火。
- 房间规则跟随线上锦标赛配置（2026-09-17 探针：二测海选 Rounds 16、BaseScore 1、有财必拷响关闭）：每场局数全部按 `rounds_choices`（默认 16 局，二测锦标赛四个阶段都是 Rounds=16）；`rounds_choices_prob` 小于 1 时其余对局才在 `rounds_min`..`rounds_max` 之间随机，默认不启用；正式赛开赛前用探针确认配置；有财必拷响默认关闭，`you_cai_bi_kao_prob` 可改；底分固定 1；策略看得到当前局与总局数。配置变化时用 `mahjong-ai online-probe` 重新读取。
- 固定对手默认是四个内置基线加本机自动研究的现役代表；设置 `opponent_entries` 后只用名单里的对手，训练、日志评估和 `sweep_nodes.py` 选版都用同一份。训练 `fixed_opponent_anneal_hours` 之后，非本人座位按 `fixed_opponent_prob_end` 的概率坐固定对手，其余坐联赛快照。
- critic 额外读取三家手牌与牌墙构成（`env.encode_oracle`），actor 与发布包不读取。
- `src/` 里不能出现 torch；训练器从 `src/` 导入编码，保证训练与发布一致。
