# 自我对弈策略项目（selfplay）

人工维护的策略项目：PyTorch 自我对弈 PPO 训练，ONNX Runtime 发布。设计决定见 [ADR-0009](../../../docs/adr/0009-independent-selfplay-trainer-for-personal-project.md)，术语（联赛池、桌面分、阶段总分）见 [CONTEXT.md](../../../CONTEXT.md)。

## 目录

```text
selfplay/
├── project.toml        # id=personal-selfplay, family=selfplay, dependencies=[onnxruntime]
├── src/                # 原样进入发布包，只依赖 numpy + onnxruntime + mahjong_ai 运行时
│   ├── strategy.py     # build(model_dir) 入口
│   └── selfplay_core/  # actions（231 动作编号）、encoding（观察编码）、policy（ONNX 推理）
├── train/              # 训练器，依赖 torch，不进发布包
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
│   └── configs/        # 三种架构与冒烟配置
├── scripts/            # 参赛流程：大乱斗选优、冻结、上线与自动续跑
├── tests/              # 项目内测试
└── models/ runs/ releases/   # 训练产物与发布，不提交
```

## 使用

训练与离线评估需要带 CUDA 的 torch。仓库 `.venv` 只装了 onnxruntime；本机用系统 conda 的 Python 3.11（自带 torch），并已为它就地编译 `_fast`，命令前加 `PYTHONPATH=src`。发布、沙箱与平台对战仍用 `.venv/bin/mahjong-ai`。

```bash
# 训练（配置见 train/configs/，主训练用 transformer-main.toml）
PYTHONPATH=src python3 projects/personal/selfplay/train/run.py --config projects/personal/selfplay/train/configs/transformer-main.toml --hours 72
PYTHONPATH=src python3 projects/personal/selfplay/train/run.py --config projects/personal/selfplay/train/configs/transformer-main.toml --resume
# 离线评估导出的 actor；--brawl 让多方均衡混战（'auto' = 自动研究现役代表，'builtin' = 四个内置基线）
PYTHONPATH=src python3 projects/personal/selfplay/train/evaluate.py --policy projects/personal/selfplay/runs/transformer-main/export --games 1024
PYTHONPATH=src python3 projects/personal/selfplay/train/evaluate.py --policy <export...> --brawl auto builtin --games 2048
# 逐节点评估（--wait-pid 等训练进程退出再开始；--entries 冻结的对手名单写在 <run>/sweep/opponents.json）
PYTHONPATH=src python3 projects/personal/selfplay/train/sweep_nodes.py --run projects/personal/selfplay/runs/transformer-main --wait-pid <训练 PID>
# 发布：把导出复制到 models/ 后走通用流程
cp projects/personal/selfplay/runs/resnet/export/{policy.onnx,meta.json} projects/personal/selfplay/models/
.venv/bin/mahjong-ai release-build --project projects/personal/selfplay --version v0.1.0
.venv/bin/mahjong-ai competition-register projects/personal/selfplay/releases/v0.1.0
# 项目测试
.venv/bin/python -m pytest -q projects/personal/selfplay/tests
```

`scripts/tonight.sh` 把「多方大乱斗 → 冻结胜者（RL 走 release-build，自动研究候选走 `mahjong-ai export`）→ 沙箱试跑 → `--go` 时用 `online-play --join` 上线」串成一条流程；`scripts/go-ertest.sh` 直接用已冻结的参赛包上线并在异常退出时用同一目录自动续跑。两者都读 `.local/credentials/free-play.json` 里的锦标赛令牌。

训练日志在 `runs/<run_name>/log.jsonl`，检查点在 `runs/<run_name>/ckpt/`，联赛快照在 `runs/<run_name>/league/`，结束时导出到 `runs/<run_name>/export/` 并做 torch 与 ONNX 的输出比对。`--resume` 从 `ckpt/latest.pt` 续跑；进度表按「累计时长 ÷ `--hours`」计算，续跑时改 `--hours` 会让学习率、熵系数与塑形权重跳到新位置，且 `best.pt` 在续跑第一轮会被覆盖。联赛快照每 30 分钟一份，移出联赛池后文件仍保留，`sweep_nodes.py` 以它们为节点，结果写到 `runs/<run_name>/sweep/summary.md`；同种子评估可复现，相邻节点配对约把差值标准误降三成。

## 关键约定

- 奖励是每局本人收支除以底分，整场之和等于桌面分；向听势能奖励只在训练前两成时间线性退火。
- 房间规则跟随线上锦标赛配置（2026-09-17 探针：Rounds 16、BaseScore 1、有财必拷响关闭）：八成对局 16 局，其余在 4 到 20 局之间均匀采样以保持通用；有财必拷响默认关闭，`you_cai_bi_kao_prob` 可改；底分固定 1；策略看得到当前局与总局数。配置变化时用 `mahjong-ai online-probe` 重新读取。
- critic 额外读取三家手牌与牌墙构成（`env.encode_oracle`），actor 与发布包不读取。
- `src/` 里不能出现 torch；训练器从 `src/` 导入编码，保证训练与发布一致。
