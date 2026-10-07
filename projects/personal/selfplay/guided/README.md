# 规则教师指导训练（guided）

`projects/personal/selfplay/` 的第二条训练路线：以规则研究的冻结冠军为教师，Actor 先纯模仿，再用真实桌面分做 PPO，并按师生差距保留教师辅助损失。设计取舍与实验得出的决定见 [ADR-0011](../../../../docs/adr/0011-rule-guided-policy-learning.md)，术语（教师策略、规则冠军、教师指导强度、评估代）见 [CONTEXT.md](../../../../CONTEXT.md)。

本路线不读取、不迁移 `train/` 的检查点，也不产出平台 ONNX 发布包；旧路线见[项目 README](../README.md)。

模仿阶段由 `imitate.py` 完成；之后的强化学习是自动研究里的 RL 方案分支：研究模型修改训练方案（`../recipes/`），本目录的 `trial.py` 从固定起点按时限训练，训练出的模型交回研究框架，做分段配对评估并参加共同大乱斗。决定见 [ADR-0013](../../../../docs/adr/0013-unified-research-branches-and-common-brawl.md)，方案接口见 [recipes/API.md](../recipes/API.md)。

## 机器分工

| 角色 | 主机 | 用途 |
| --- | --- | --- |
| Mac | 本机 | 控制面：冻结源码与教师、写运行计划、启动远端任务、汇总报告；研究评估中的模型座位用 MLX／MPS 推理，不训练 |
| 4090 | `gpu-4090-wsl` | 第一性原理 RL 线的大规格（384/8/8）尝试位；此前也做模仿阶段、RL 方案分支的训练与评估、规模试验与共同大乱斗（大乱斗派到这里时，本机训练进程被暂停，结束后恢复） |
| 4070 | `gpu-4070-wsl` | 用户的办公机，显卡同时负责桌面显示；正式内网比赛在这里推理。第一性原理 RL 线的小规格（128/4/4）尝试位（训练与评估都在本机），对局进程与显存留出余量给桌面（[ADR-0016](../../../../docs/adr/0016-maintainer-iteration-lines.md)）。不跑显存压力探针 |
| 5070 Ti | `gpu-5070ti-wsl` | 第一性原理 RL 线的大规格（384/8/8）尝试位（训练与评估都在本机）；用户借用的机器，开放时段由维护者用 `harness.line hosts` 设定；为桌面保留 CPU、显存和内存余量 |

- 连接由 [`remote.py`](remote.py) 的 `PROFILES` 定义：经 Windows 宿主 SSH，再按主机的分发名与用户进入 WSL；命令按 argv 传递，不拼接 shell。长任务用 `remote.command(..., background_log=...)` 启动，进程脱离 wsl.exe 会话。
- 4070 与 4090 的项目路径都是 `~/code/leihuo_majiang`，解释器 `~/softwares/miniconda3/envs/py311_cu130/bin/python` 自带 CUDA torch，不要重装 CUDA 环境。统一版本的 numpy 放在 `.local/guided-deps/numpy-2.1.2`，`remote.command` 会自动加进 `PYTHONPATH`，不改共享 Conda 环境。
- 5070 Ti 经 `ssh user@gpu-5070ti` 进入 `Ubuntu` 分发，用户为 `user`；项目在 `/home/user/code/leihuo_majiang`，独立解释器为项目内 `.venv/bin/python`（Python 3.12、torch 2.10.0/cu130、numpy 2.1.2）。调用方用 `remote.root(host)`、`remote.python(host)` 取路径。
- 训练与推理测试串行执行，不在满负载训练时测部署时延。

## 实验如何冻结与运行

每个实验是 Mac 上的一个目录 `projects/personal/selfplay/.local/<种类>/<时间戳>/`，同类实验的现役入口是该种类下的 `current.json`：

- `source/source-manifest.json`：冻结源码的文件清单与 SHA-256，清单摘要即 `source_id`。源码上传到远端 `.local/guided-code/<source_id>/`，入口用 `bootstrap.verify_source()` 核对，与 `plan.json` 不符即拒绝运行。
- 训练对局默认绑定教师包自带的运行时副本；方案线任务带 `runtime` 时（第一性原理迭代线传对手池的运行时）改绑那份，worker 里的对手包按 `isolation = "none"` 直接进程内加载。
- 引擎缺少编译的 `_fast` 扩展时会静默退回纯 Python，向听计算和观察编码慢一个数量级。训练与评估的 worker 绑定的运行时（教师包自带的那份，或任务指定的对手池运行时），其 `_fast` 由 `guided.sync` 在目标机编译；推理服务、测试和其他不绑定教师包的进程用快照自己的 `src/mahjong_ai/`，上传后要在目标机用快照里的 `_fast.c` 编译（setuptools `build_ext`）。`guided.encoding` 导入时检查扩展，缺失即报错。
- `plan.json`：该实验的主机、模型规模、教师包、初始权重、时长、评估场数和牌墙种子。`overnight`、`brawl`、`brawl_pipeline`、`bc_curve`、`tune_evaluation` 都以 `--root <实验目录>` 读取它。
- 准备、部署、启动与监控用的操作脚本（`prepare.py`、`deploy.py`、`start.py`、`monitor.py` 等）随实验目录冻结，报告写在同目录 `REPORT.md`。新实验复制上一份脚本后修改，不回写旧实验。
- 教师包由 Mac 从已完成的共同评测回执导出到 `.local/teachers/<id>/`，远端位于 `.local/guided-teachers/<id>/`。接收、运行校验与启用分开记录；未完整收到或不能运行的包不替换现役教师。

所有实验都是有限任务：结束后不会自动进入下一阶段，也没有常驻的训练截止时间。

## 模块

| 模块 | 职责 |
| --- | --- |
| `imitate` | 纯模仿：BC（老师出牌）→ DAgger（学生出牌、老师标注）→ 学习率线性衰减。新鲜样本上的老师交叉熵在 `window_seconds` 内改善不足 `tolerance` 即换阶段，衰减时长为已训练时间的 `decay_fraction`；给出 `hands`（学习座位的总局数，每场 16 局）时改为按局数固定比例切换（`bc_fraction`、`dagger_fraction`，其余衰减）并在打满后停止；每 `snapshot_seconds` 存快照，另存各阶段结束点与 `final`。进程被暂停（让位大乱斗）的时间不计入训练时间；`--resume` 续跑时可以换 worker 与桌数 |
| `run` | 单机训练入口：`--bc-seconds` 纯模仿阶段、联合训练、`--snapshot-seconds` 定时快照、`--deadline` 绝对截止、`--resume` |
| `learning` / `networks` / `encoding` | PPO 与教师交叉熵、Actor 与带隐藏信息的 PopArt Critic、含本手完整公开历史的合法观察编码 |
| `environment` / `collection` | CPU worker 驱动完整对局，父进程集中做 GPU 推理 |
| `evaluation` / `statistics` / `inference` | 共同牌墙、四座轮换的配对评估，指导强度函数，评估专用的 Actor 批量推理 |
| `artifacts` / `sync` | 教师包的导出、校验与跨机发布 |
| `deployment` | Actor 导出、基准测试与独立 HTTP 推理服务 |
| `expand` | 把导出的 Actor 扩成更宽的观察编码：新的平面和标量输入列置零，事件末字段的嵌入表每行取原第 0 行，其余参数照抄，决策逐位不变，产物按权重 SHA-256 写进模型库；`--belief-head` 另加信念头、`--share-suits` 改成花色共享嵌入（见下文 Actor 可选结构，后者会稍微改变决策） |
| `resources` / `tune_evaluation` | 按 CPU 亲和性、cgroup 配额与可用显存给出初始并发，再在固定赛程上实测校准 |
| `brawl` / `brawl_pipeline` | 5、8、10 或 11 个冻结策略的均衡大乱斗；流水线按校准、混合对局冒烟、计分比赛顺序执行，遇错停止 |
| `trial` / `recipe_jobs` | 训练方案迭代：远端框架在固定时限和显存、内存上限内调用 `recipe.train(ctx)`，对手取研究冻结的评估对手（规则版本按各自运行时隔离运行）；训练出的 Actor 拉回 Mac 交给研究框架评估。Mac 侧任务负责部署、上传方案、脱离运行与轮询，主机不可达时无限期等待，主机重启时记为主机故障、从头重跑 |
| `platform_replays` / `platform_dataset` | 平台牌谱转成模仿数据（[ADR-0015](../../../../docs/adr/0015-platform-replays-as-training-data.md)）：每手按牌谱的起手牌与摸牌顺序在本地引擎里重放，每个多选决策点取决策方的合法观察，标签是该玩家在平台上的实际动作；摸牌、合法性、每手结算与总分任一处与平台不一致，这一场整场丢弃。`scan` 增量入库并刷新外部玩家实力表 `players.json`，`freeze` 按文件哈希冻结训练清单，`rows` 在训练时现场还原并编码，默认去掉平台代打的超时和我方座位 |
| `collection` | 采集父进程：CPU worker 打整场，父进程跑所有网络。CUDA 上整个决策（前向、掩码、采样或 argmax、选中动作的对数概率）按模型和补齐形状捕获 CUDA Graph 原地重放（`inference.ActorGraphs`），线性层以 fp16 autocast 计算，输入直接写入固定内存缓冲，答案在同一条流上拷回固定内存后才读（各图共用一个内存池，别的图重放会覆盖设备上的输出）。父进程保持两轮在途：一轮发到设备后先收下一轮请求（`gather_seconds` 内到达的一起攒成更大的批），再读上一轮的答案回复；内置 Critic 或纯模仿时仍走完整批同步执行。教师是冻结模型时（方案用 `cfg["teacher"]` 选 `ctx.catalog` 里的模型），父进程在学生出牌前对同一批学习行跑教师 Actor（有 Graph 时用它自己的一组），取合法动作的 argmax 随答案发回 worker 写进 `teacher`；worker 不跑规则教师，对局仍可走 C 引擎。每次采集把各阶段耗时与批大小写入运行目录的 `collection.jsonl` |
| `model_server` / `mlx_actor` / `mlx_runner` | 每台主机一个批量模型服务，按权重 SHA-256 加载 Actor。CUDA 用 Graph，macOS 26+ 的 Apple Silicon 安装 `.[model]` 后自动用 MLX 0.32；较早系统或未安装受支持的 MLX 时用 MPS；可用 `--device mps` 显式选择 Torch 后端。线性层 fp16，embedding、残差和归一化 fp32；冻结权重只转换一次，学习 Actor 保持 fp32 参数。事件循环保留两批在途，Apple 后端按历史长度分组减少补齐，CUDA 轮询完成事件，Apple 等待线程通过管道唤醒。服务每分钟记录累计请求、批次与耗时。模型目录的 `meta.json` 带 `"sampling": {"temperature": T}` 时，该权重按温度采样出牌（CUDA 图内的 Gumbel-max；MLX 后端不支持），用于要保留随机性的对手，例如平台代理对手；采样版本另存一份权重，按自己的 SHA-256 寻址 |
| `overnight` | 同机启动方式对照（A 直接联合训练，B 先纯模仿再联合训练） |
| `bc_curve` | 纯模仿时长曲线：定时快照逐个与规则冠军配对评估 |
| `scale_probe` / `scale_batch` / `scale_study` / `scale_monitor` | 模型规模的资源探针、微批次测量、较长的监督学习对照与只读监控 |
| `experiment` / `node` | 双机一周协调器与远端单任务回执；保留实现，目前不使用 |
| `remote` / `bootstrap` / `config` | 控制面、源码校验与新实验默认参数 |

推理后端或网络计算布局改变时，浮点舍入可能改变接近同分的动作。新版本须使用新冻结运行时与对手池并重跑参照，不能将旧成绩直接并入；公式相同不代表跨设备、跨后端逐位一致。

## 常用命令

在远端冻结源码根目录执行：

```bash
TRAIN_PY="$HOME/softwares/miniconda3/envs/py311_cu130/bin/python"
export PYTHONPATH="$HOME/code/leihuo_majiang/.local/guided-deps/numpy-2.1.2:src:projects/personal/selfplay"

# 纯模仿：BC → DAgger → 衰减，阶段按数据自动切换
"$TRAIN_PY" -m guided.imitate --run <run-dir> --bundle <teachers/<id>/manifest.json> \
  --config <config.json> --deadline <unix-epoch-seconds> [--schedule '{"snapshot_seconds": 600}'] [--resume]

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

# 测试；Mac 的 .venv 装了 .[model] 后也能跑 guided 测试（CPU/MPS），GPU 相关项在 WSL 运行
"$TRAIN_PY" -m pytest -q projects/personal/selfplay/tests tests/test_guided_*.py
```

平台牌谱入库与实力表（Mac 上用仓库 `.venv`，数据在本项目 `.local/platform-data/`）：

```bash
cd projects/personal/selfplay
PYTHONPATH=../../../src:src:. ../../../.venv/bin/python -m guided.platform_dataset scan
PYTHONPATH=../../../src:src:. ../../../.venv/bin/python -m guided.platform_dataset freeze --users <user_id ...> --output <manifest.json>
```

`--config` 覆盖 [`config.py`](config.py) 的 `DEFAULTS`，未知键直接报错；模型规模用 `dim`/`layers`/`heads`。

## 关键约定

- 教师只在学生实际访问的合法观察上标注动作，标签只进入监督交叉熵；PPO 比率只用学生自己的行为与概率，教师代打的样本不冒充 on-policy 数据。历史示范缓存只参与监督项。
- 纯模仿阶段只更新 Actor，Critic 与其优化器不动；进入联合训练前保存 `bc-final.pt`。定时快照在跨过时间节点后的完整更新边界保存，记录实际时间、更新次数与样本量，不覆盖中途版本。
- 联合训练中的指导强度 `w = clip((d + u) / D, 0, 1)`，`d` 为教师减学生的配对场均桌面分差，`u` 为 95% 区间半宽；首次评估前 `w = 1`。`--reference` 给定的固定 Actor 与学生、教师在同一批牌墙上评估，结果里的 `reference_minus_student` 衡量相对起点的进步；续跑沿用记录的参照。
- 奖励是每局本人收支除以底分，整场之和等于桌面分，不做塑形和裁剪。Critic 读取三家手牌与牌墙构成，只在训练进程内使用，不随 Actor 发布。
- 评估面板在一个实验内固定：同组牌墙四座轮换，按完整四座组计算配对均值差与区间。区间跨零只表示没分出高下，不等于持平或已到学习上限；多组比较未做多重校正。
- 大乱斗每组内各策略的出场与座位次数一致：5 方覆盖全部 5 种四人阵容，8 方覆盖全部 70 种，10 方覆盖全部 210 种，11 方按循环移位的 30 个轨道排程。自动研究的共同大乱斗由研究框架执行；本模块的大乱斗用于研究框架之外的临时比赛，赛程、身份、回执和报告放仓库根 `.local/competition/runs/`。
- 观察编码有名字，导出的 Actor 在 `meta.json` 的 `encoding` 里记着自己的编码。`guided-current-hand-v2` 在 v1 的平面后追加 16 个鸣牌、杠前瞻平面（`encoding.call_planes`）：碰、吃、明杠、暗杠或补杠之后的手牌，取 v1 逐张出牌平面同样的向听、进张、胡牌张数和番值；不合法的动作为 0。新平面只由 v1 行算出，所以 worker 的 Python 与 C 编码、模型服务收到的二进制请求都用同一个 `encoding.augment` 扩成 v2，match worker 照旧只发 v1 行。v2 行的前 57 个平面就是 v1 行，每个 Actor 只读自己那部分前缀；训练时 trial 取在座模型里最宽的编码交给 worker（`cfg["encoding"]`）。
- `guided-current-hand-v3` 在 v2 后再追加 38 个平面、35 个标量，并给事件的末字段（v1、v2 恒为 0）赋值，同样只由 v1 行算出（`encoding.v3_inputs`，重活在 `native.c` 的 `v3_shape`、`v3_events`，经 `_fast.load_native` 在本机编译）。
  - 平面：逐张出牌的平胡与七对分开的向听、进张、胡牌张数；打后是否爆头听、几张摸牌能变成爆头听；单人多步前瞻（k = 1、2、3、5 与剩余自家摸牌数内听牌、爆头听、自摸的概率和自摸时的期望番，向听 1 的打法前三名逐张走一遍，其余按均值近似）；离四白差几张；被吃、被碰的风险；杠后补牌自摸率、杠后链数与再杠概率；四家衰减河。
  - 标量：参考手的前瞻、剩余摸牌数、可胡时的放弃与继续、留白与打白的向听差、打白是否成财飘及其后的链与飘、巡目、各家距上次鸣牌的巡数、刚吃碰后出牌。
  - 事件末字段：强制 × 8 + 距今巡数分档（0、1、2、3、4、5–6、7–9、10+）。强制 1 为抓打圈内非圈主的出牌（只能打刚摸的牌），2 为本家吃碰后紧接的出牌（必从手里打），其余为 0；平台不给摸切、手切，只有这两条规则推出的事实。
  - 定义逐项见 `v3_inputs` 的文档。每个 Actor 只读自己那部分平面、标量前缀；嵌入表只有 4 行的旧 Actor 把末字段读作 0，所以 v1、v2 Actor 坐在 v3 的桌上也逐位不变。
- `guided-current-hand-v4` 在 v3 的标量末尾追加 1 个风险系数 β/β_max（`stage_value.py`）：V(s) 为按独立正态估的晋级概率（半决赛取前二、决赛取第一），β = −V″/V′，单位为每基础分，中心差分步长 1 个基础分，剩余手数含当前这手，V′ < 1e-6 时取 0，截断在 ±0.05。β > 0 表示 V 凹（领先、求稳），β < 0 表示 V 凸（落后、要方差）。它是唯一不由 v1 行算出的输入：`encode` 在半决赛、决赛的观察上把它记在行的 `risk_beta`，自由对战、开发集等其余观察没有这个键，输入为 0；C 引擎的行由 `stage.rewrite` 连同其他阶段标量一起写入；match worker 的二进制请求只在发给读这个系数的模型（v4）时于掩码后多带一个 float16，客户端对每个模型向服务问一次（`POST /inputs`，没有答复就按不读处理）；发给 v1–v3 模型的请求与 v4 之前逐字节相同，旧服务照收（`selfplay_core.wire`）。`expand` 可把 v1、v2、v3 Actor 零初始化扩成 v4。
- Actor 的可选结构记在导出 `meta.json` 的 `config` 里（`networks.actor_options`），缺省时权重、输出和导出文件与没有这些选项时逐位相同；`model_server` 的 MLX 后端实现同一方程。
  - `"belief_head": true`（至少两层）：主干第 `layers // 2` 个块之后，从 34 个牌 token 预测三家对手（按出牌顺序）每种牌的暗手张数，输出 3×34 的 logits，sigmoid 为张数 / 4；sigmoid 停梯度后经 `encoder.belief_in` 线性投影加回牌 token，再进后面的块。`forward(..., return_belief=True)` 返回 `(logits, belief)`，训练用行里的 `oracle_planes` 监督（`networks.belief_target`，ADR-0009 允许训练标签读隐藏状态）；策略损失不经注入回传到头。头只读合法观察，推理仍是一次前向、只返回 logits。`expand --belief-head` 把现有 Actor 扩成带头的版本：头的输出层和注入层置零，隐层取固定种子，logits 逐位不变。
  - `"suit_embedding": "shared"`：Actor 里按牌的嵌入表（牌 token 的 `encoder.tile_embedding`、事件的牌与吃牌两张表）改为「点数嵌入 + 花色嵌入」（`networks.SuitEmbedding`）：万、筒、条共用 9 行点数，各加一个花色向量，7 张字牌和事件的「无牌」单独一行。Actor 其余部分本来就按牌共享（`tile_in`、出牌头、鸣牌头对每个牌 token 一样，平面里有花色独热和点数），所以这几张表是输入侧仅有的逐牌参数；`state_head` 的 63 个吃牌 logits 仍按花色各一行。参数形状变了，不能逐位扩展：`expand --share-suits` 从现有 Actor 起步，取原表的最小二乘加性拟合（点数取三个花色的平均，花色取该花色各点数的平均减总平均，字牌与「无牌」照抄），丢掉的只是原表里点数加花色之外的部分。MLX 后端加载时按同一公式算好整表。
- `guided.deployment` 是独立 GPU 服务：`POST /act` 接收 `{"observation_payload": ...}` 完整公开观察并返回合法动作，`GET /health` 报告状态。它不是 ONNX 发布包，不改变平台 CPU/ONNX 发布契约，导出与测速也不会自动报名比赛。部署验收以 4070 实机的 10 桌并发为准，计时覆盖 HTTP、排队、编码、推理和回传。

运行状态、完整配置和回执放本项目 `.local/`；各次实验的结论与待办汇总在本机 `.local/reports/guided-status.md`，具体成绩不写入本手册。
