# 规则教师指导训练（guided）

`projects/personal/selfplay/` 的第二条训练路线：以规则研究的冻结冠军为教师，Actor 先纯模仿，再用真实桌面分做 PPO，并按师生差距保留教师辅助损失。设计取舍与实验得出的决定见 [ADR-0011](../../../../docs/adr/0011-rule-guided-policy-learning.md)，术语（教师策略、规则冠军、教师指导强度、评估代）见 [CONTEXT.md](../../../../CONTEXT.md)。

本路线不读取、不迁移 `train/` 的检查点，也不产出平台 ONNX 发布包；旧路线见[项目 README](../README.md)。

模仿之后的强化学习目前通过训练方案迭代改进：研究模型修改训练方案（`../recipes/`），本目录的 `trial.py` 按固定时限训练、配对评估并跑大乱斗，由工作台服务里的编排器调度。决定见 [ADR-0012](../../../../docs/adr/0012-recipe-iteration-for-rl-training.md)，方案接口见 [recipes/API.md](../recipes/API.md)。

## 机器分工

| 角色 | 主机 | 用途 |
| --- | --- | --- |
| Mac | 本机 | 控制面：冻结源码与教师、写运行计划、启动远端任务、汇总报告；不跑 GPU 任务 |
| 4090 | `gpu-4090-wsl` | 训练大模型（384/8/8），并承担规模试验、离线评估与大乱斗 |
| 4070 | `gpu-4070-wsl` | 训练小模型（256/6/8）；正式内网比赛只在这里推理，部署两个模型中同条件表现最好的那个。不训练大模型，不跑显存压力探针 |

- 连接由 [`remote.py`](remote.py) 的 `HOSTS`、`ROOT`、`PYTHON` 定义：经 Windows 宿主 SSH，再以 `wsl.exe -d Ubuntu-20.04 -u user` 进入 WSL；命令按 argv 传递，不拼接 shell。长任务用 `remote.command(..., background_log=...)` 启动，进程脱离 wsl.exe 会话。
- 两机项目路径都是 `~/code/leihuo_majiang`，解释器 `~/softwares/miniconda3/envs/py311_cu130/bin/python` 自带 CUDA torch，不要重装 CUDA 环境。统一版本的 numpy 放在 `.local/guided-deps/numpy-2.1.2`，`remote.command` 会自动加进 `PYTHONPATH`，不改共享 Conda 环境。
- 训练与推理测试串行执行，不在满负载训练时测部署时延。

## 实验如何冻结与运行

每个实验是 Mac 上的一个目录 `projects/personal/selfplay/.local/<种类>/<时间戳>/`，同类实验的现役入口是该种类下的 `current.json`：

- `source/source-manifest.json`：冻结源码的文件清单与 SHA-256，清单摘要即 `source_id`。源码上传到远端 `.local/guided-code/<source_id>/`，入口用 `bootstrap.verify_source()` 核对，与 `plan.json` 不符即拒绝运行。
- 引擎缺少编译的 `_fast` 扩展时会静默退回纯 Python，向听计算和观察编码慢一个数量级。训练与评估的 worker 绑定教师包运行时，其 `_fast` 由 `guided.sync` 在目标机编译；推理服务、测试和其他不绑定教师包的进程用快照自己的 `src/mahjong_ai/`，上传后要在目标机用快照里的 `_fast.c` 编译（setuptools `build_ext`）。`guided.encoding` 导入时检查扩展，缺失即报错。
- `plan.json`：该实验的主机、模型规模、教师包、初始权重、时长、评估场数和牌墙种子。`overnight`、`brawl`、`brawl_pipeline`、`bc_curve`、`tune_evaluation` 都以 `--root <实验目录>` 读取它。
- 准备、部署、启动与监控用的操作脚本（`prepare.py`、`deploy.py`、`start.py`、`monitor.py` 等）随实验目录冻结，报告写在同目录 `REPORT.md`。新实验复制上一份脚本后修改，不回写旧实验。
- 教师包由 Mac 从已完成的共同评测回执导出到 `.local/teachers/<id>/`，远端位于 `.local/guided-teachers/<id>/`。接收、运行校验与启用分开记录；未完整收到或不能运行的包不替换现役教师。

所有实验都是有限任务：结束后不会自动进入下一阶段，也没有常驻的训练截止时间。

## 模块

| 模块 | 职责 |
| --- | --- |
| `run` | 单机训练入口：`--bc-seconds` 纯模仿阶段、联合训练、`--snapshot-seconds` 定时快照、`--deadline` 绝对截止、`--resume` |
| `learning` / `networks` / `encoding` | PPO 与教师交叉熵、Actor 与带隐藏信息的 PopArt Critic、含本手完整公开历史的合法观察编码 |
| `environment` / `collection` | CPU worker 驱动完整对局，父进程集中做 GPU 推理 |
| `evaluation` / `statistics` / `inference` | 共同牌墙、四座轮换的配对评估，指导强度函数，评估专用的 Actor 批量推理 |
| `artifacts` / `sync` | 教师包的导出、校验与跨机发布 |
| `deployment` | Actor 导出、基准测试与独立 HTTP 推理服务 |
| `resources` / `tune_evaluation` | 按 CPU 亲和性、cgroup 配额与可用显存给出初始并发，再在固定赛程上实测校准 |
| `brawl` / `brawl_pipeline` | 5、8、10 或 11 个冻结策略的均衡大乱斗；流水线按校准、混合对局冒烟、计分比赛顺序执行，遇错停止 |
| `trial` / `recipe_jobs` | 训练方案迭代：远端框架在固定时限内调用 `recipe.train(ctx)`、做配对评估与五方大乱斗；Mac 侧任务负责部署、上传方案、脱离运行与轮询 |
| `overnight` | 同机启动方式对照（A 直接联合训练，B 先纯模仿再联合训练） |
| `bc_curve` | 纯模仿时长曲线：定时快照逐个与规则冠军配对评估 |
| `scale_probe` / `scale_batch` / `scale_study` / `scale_monitor` | 模型规模的资源探针、微批次测量、较长的监督学习对照与只读监控 |
| `experiment` / `node` | 双机一周协调器与远端单任务回执；保留实现，目前不使用 |
| `remote` / `bootstrap` / `config` | 控制面、源码校验与新实验默认参数 |

## 常用命令

在远端冻结源码根目录执行：

```bash
TRAIN_PY="$HOME/softwares/miniconda3/envs/py311_cu130/bin/python"
export PYTHONPATH="$HOME/code/leihuo_majiang/.local/guided-deps/numpy-2.1.2:src:projects/personal/selfplay"

# 单机训练：先纯模仿 N 秒，再进入联合训练
"$TRAIN_PY" -m guided.run --run <run-dir> \
  --bundle <teachers/<id>/manifest.json> --panel <teachers/<initial-id>/manifest.json> \
  --config <config.json> --deadline <unix-epoch-seconds> \
  --initial <initial.pt> --bc-seconds 2400 --snapshot-seconds 600

# 从已有的纯模仿快照直接联合训练，并把起点作为每代评估的固定参照
"$TRAIN_PY" -m guided.run --run <run-dir> --bundle ... --panel ... --config <config.json> \
  --deadline <unix-epoch-seconds> --initial <actor+critic.pt> --reference <start-actor.pt> \
  --updates <teachers/current.json>

# 实验目录驱动的对照与比赛
"$TRAIN_PY" -m guided.overnight preflight|suite|evaluate --root <dir> [--route A|B]
"$TRAIN_PY" -m guided.bc_curve suite|evaluate --root <dir>
"$TRAIN_PY" -m guided.brawl_pipeline --root <dir>

# Actor 导出、基准测试与推理服务；不含 Critic、优化器或轨迹
"$TRAIN_PY" -m guided.deployment export <run-dir>/best.pt <actor-dir>
"$TRAIN_PY" -m guided.deployment benchmark <actor-dir> --device cuda --count 200
"$TRAIN_PY" -m guided.deployment serve <actor-dir> --host 127.0.0.1 --port 8810

# 测试；Mac 的 .venv 没有 torch，guided 测试在 WSL 运行
"$TRAIN_PY" -m pytest -q projects/personal/selfplay/tests tests/test_guided_*.py
```

`--config` 覆盖 [`config.py`](config.py) 的 `DEFAULTS`，未知键直接报错；模型规模用 `dim`/`layers`/`heads`。

## 关键约定

- 教师只在学生实际访问的合法观察上标注动作，标签只进入监督交叉熵；PPO 比率只用学生自己的行为与概率，教师代打的样本不冒充 on-policy 数据。历史示范缓存只参与监督项。
- 纯模仿阶段只更新 Actor，Critic 与其优化器不动；进入联合训练前保存 `bc-final.pt`。定时快照在跨过时间节点后的完整更新边界保存，记录实际时间、更新次数与样本量，不覆盖中途版本。
- 联合训练中的指导强度 `w = clip((d + u) / D, 0, 1)`，`d` 为教师减学生的配对场均桌面分差，`u` 为 95% 区间半宽；首次评估前 `w = 1`。`--reference` 给定的固定 Actor 与学生、教师在同一批牌墙上评估，结果里的 `reference_minus_student` 衡量相对起点的进步；续跑沿用记录的参照。
- 奖励是每局本人收支除以底分，整场之和等于桌面分，不做塑形和裁剪。Critic 读取三家手牌与牌墙构成，只在训练进程内使用，不随 Actor 发布。
- 评估面板在一个实验内固定：同组牌墙四座轮换，按完整四座组计算配对均值差与区间。区间跨零只表示没分出高下，不等于持平或已到学习上限；多组比较未做多重校正。
- 大乱斗每组内各策略的出场与座位次数一致：5 方覆盖全部 5 种四人阵容，8 方覆盖全部 70 种，10 方覆盖全部 210 种，11 方按循环移位的 30 个轨道排程。跨项目比赛的赛程、身份、回执和报告放仓库根 `.local/competition/runs/`，与规则自迭代的研究榜单分开。
- `guided.deployment` 是独立 GPU 服务：`POST /act` 接收 `{"observation_payload": ...}` 完整公开观察并返回合法动作，`GET /health` 报告状态。它不是 ONNX 发布包，不改变平台 CPU/ONNX 发布契约，导出与测速也不会自动报名比赛。部署验收以 4070 实机的 10 桌并发为准，计时覆盖 HTTP、排队、编码、推理和回传。

运行状态、完整配置和回执放本项目 `.local/`；各次实验的结论与待办汇总在本机 `.local/reports/guided-status.md`，具体成绩不写入本手册。
