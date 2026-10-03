# 训练方案接口

训练方案迭代改进的对象是一个名为 `recipe` 的 Python 包，它是自动研究里 RL 方案分支的候选（[ADR-0013](../../../../docs/adr/0013-unified-research-branches-and-common-brawl.md)，早期设计见 [ADR-0012](../../../../docs/adr/0012-recipe-iteration-for-rl-training.md)）。框架 `guided/trial.py` 在远端导入它，调用 `recipe.train(ctx)`，在时限内拿回训练后的 Actor，交给研究框架评估。[`baseline/recipe`](baseline/recipe) 是初始方案：多步小步策略梯度，每步用约 60 万个同策略决策算一次全量梯度，按 KL 0.03 定步长。设计理由写在 `loop.py` 和 `targets.py` 的文档字符串里，测量依据见 ADR-0012 的“基线方案与训练时长”。

## 固定不变

- Actor 的网络结构和观察编码。`ctx.actor` 从本方案线的固定起点加载；返回的权重必须能严格载入同一结构，否则本次尝试记为崩溃。
- 游戏环境、规则和奖励：每局结束时，学习座位得到该局桌面分（除以底分）。
- 教师：默认是冻结规则包里的规则冠军（`ctx.teacher`），方案可以用 `cfg["teacher"]` 换成 `ctx.teachers` 里的任一规则，或 `ctx.catalog` 里的模型。训练对手由方案通过 `cfg["opponents"]` 选择，默认是本次迭代冻结的评估对手（最近一次共同大乱斗前三名，可能包含 RL 模型）；harness 可以在 `ctx.catalog` 里另外提供冻结版本供方案选作对手或教师；没有评估对手时用规则包里的规则代表。
- 时间预算：从 `train` 被调用开始计时，到 `ctx.deadline` 截止；超过宽限期，进程被强制结束。只按局数限额的 harness（例如迭代线）把截止时刻设得很远，只用来防卡死。
- 局数预算（可选，方案线配置 `hand_budget`）：学习座位累计打过的训练局数（每整场 16 局）。用完后 `ctx.collect` 报错；接近用完时，`samples` 会被压到剩余局数按已见每局行数折算的行数；在途的整场照常打完，所以最后一次采集可能略超，超出部分如实记入训练回执。自检运行不设局数预算。
- 资源上限：显存按方案线配置的比例封顶，超出时在进程内抛出 CUDA out of memory，不会溢出到共享内存；训练进程组（含对局进程）的常驻内存超过上限时整组被强制结束。两种情况都记为训练失败并消耗额度，上限见 `ctx.resources`，训练回执里有实际峰值。
- 评估和晋级：由研究框架做分段配对评估。训练后的 Actor 和现役 Actor 面对同一组评估对手，在同样的牌墙和座位上对局，每边 4096 → 8192 → 16384 场，累计计数；前两段 99% 区间下界 > 0 提前晋级、上界 < 0 提前淘汰，最后一段 95% 区间下界 > 0 才晋级。晋级的模型参加下一次共同大乱斗。

## `train(ctx)`

必须在 `ctx.deadline` 之前返回训练后的 Actor 或它的 `state_dict`。返回 `None` 时，框架使用最近一次 `ctx.save_actor` 保存的权重。

`ctx` 提供：

| 成员 | 说明 |
| --- | --- |
| `actor` | 已加载起点权重、在 GPU 上的 Actor |
| `device`、`seed`、`deadline`、`time_left()` | 设备、随机种子、截止时刻、剩余秒数 |
| `hand_budget`、`hands`、`hands_left()` | 局数预算、已打局数、剩余局数；未设局数预算时为 `None` |
| `label_hand_budget`、`labeled_hands`、`labeled_hands_left()` | 带教师标签（`labels=True`）的局数上限、已打局数、剩余局数；它们也计入 `hands`。上限用完后再带标签采集会报错；未设上限时为 `None` |
| `smoke` | 自检运行时为真，此时时限只有约 90 秒 |
| `resources` | `cpu_capacity`、`cpu_count`、`gpu_total_bytes`、`gpu_limit_bytes`、`ram_limit_bytes`、建议的 `workers` 与 `tables` |
| `teacher`、`teachers`、`rules`、`bundle` | 默认教师 ID、可以当教师的 ID（规则包里的规则、harness 提供的规则版本和 `ctx.catalog` 里的模型）、规则包里的规则代表 ID、冻结包路径 |
| `opponents` | 本次迭代冻结的评估对手 ID，也是默认训练对手（规则版本按各自运行时运行，其余是模型） |
| `catalog` | harness 另外提供的冻结版本 ID，可以选作对手或教师（harness 标明只作对手的除外，例如平台代理）；没有时为空 |
| `new_actor()` | 同结构的新 Actor，例如用作自身历史快照对手 |
| `collect(actor, cfg, samples=, iteration=, labels=False, history=None, hooks=None, inference_session=None, starts=None)` | 采集训练对局，见下文 |
| `starts(name)` | 读取随方案冻结的残局集 `recipe/starts/<name>.json.gz`，见「残局开局」 |
| `log(**metrics)` | 写一行训练诊断，评估报告和下一轮交接都会引用 |
| `save_actor(actor)` | 保存 Actor 检查点，防止超时丢失成果 |

## 采集对局

`ctx.collect` 让 `actor` 坐在一个随机座位上学习，其余三座是对手，返回 `(rows, records, resources)`：

- `rows`：学习座位的每个决策。包含 Actor 输入（`planes`、`scalars`、`events`、`mask`）、`action`、`logprob`、`value`、`reward`（只在每局最后一个决策上，值为该局桌面分；学习座位整局没有决策时，该局分不记到任何行，只留在整场账本里）、`hand`（第几局，1～16）、`adv` 和 `ret`（由 `finish_game` 计算）、教师标签 `teacher`（`labels=True` 时是教师在同一局面的动作，否则为 −1；教师在某个决策上报错或给出非法动作时，这一行也是 −1，采集照常进行（纯模仿采集在这一步改由学生采样出牌，教师损失不计这一行），worker 报告里的 `teacher_errors` 计数，首次出错的堆栈写进训练日志。规则教师打标签时采集走 Python 引擎，比纯 PG 采集慢；教师是模型时，标签是它在合法动作上的 argmax，由采集父进程在学生出牌前与学生同批推理（精度与模型对手相同），不会是 −1，对局照常可走 C 引擎，`label_version` 为 `model:teacher:<ID>`），以及 `critic_features` 返回的数组。同一整场的决策按顺序连续排列。
- `records`：每整场一条，含四家得分和座位。
- `resources`：采集耗时和各 worker 的场数、样本数。

`cfg` 里由方案决定的键：

| 键 | 作用 |
| --- | --- |
| `workers`、`tables` | 对局进程数（不超过 CPU 核数）和每个进程同时打的桌数。达到 `samples` 后不再开新局，但在途的整场要打完，实际样本约多出 `workers × tables` 场。计划里带 `execution` 时（迭代线按主机固定），`workers`、`tables`、`gather_seconds` 以它为准，方案给的值被覆盖 |
| `gather_seconds` | 父进程有一轮推理在设备上时，再等这么久收齐其他 worker 的请求合成一批（默认 0，只收已到达的） |
| `opponents` | 对手座位均匀抽取的 ID 列表，取自 `ctx.opponents`、`ctx.catalog` 或 `ctx.rules`；缺省为全部 `ctx.opponents` |
| `teacher` | `labels=True` 时给学习座位打标签的规则或模型，取自 `ctx.teachers`；缺省为 `ctx.teacher` |
| `teacher_plays` | 为真时（须 `labels=True`）学习座位走教师的动作（行为克隆，采集教师自己的轨迹），行里的 `action` 是教师动作，`logprob` 不对应它；缺省为假，学生自己出牌、教师只打标签（DAgger） |
| `history_probability` | 对手座位从 `history`（`{"model:名字": Actor}`）抽取的概率，其余按 `opponents` 抽取 |
| `critic_features` | `"模块:函数"`，在对局进程里对每个学习决策调用 `f(state, seat)`，返回的数组并入该行；`state` 是完整的 HandState，可读对手手牌和按摸牌顺序排列的牌墙 |
| `finish_game` | `"模块:函数"`，每整场结束时在对局进程里调用 `f(rows, cfg)`，必须给每行写入 `adv` 和 `ret` |
| `playout_baseline` | 为真时，学习座位每次采样动作与概率最大动作（argmax）不同（行的 `deviated` 为真），框架复制这一局，在该处改走 argmax，之后学习座位一律 argmax、其余座位照常，打到这一局结束，把学习座位这一局的桌面分（底分为单位）写入该行的 `playout`，下一局由学习座位坐庄时 `playout_deals_next` 为 1.0。所有学习决策都带 `deviated`。牌墙与种子固定、对手按观察确定地决策时，下一次偏离之前实际路径就是 argmax 路径，所以每个决策的 argmax 价值都可得：本局下一次偏离的 `playout`，其后没有偏离则是实际结果。对手有随机性时这个基线仍不依赖本步动作，只是更不准；开启后模型对手也改取 argmax。`finish_game` 在该场所有复制局打完后才调用。复制局不产生训练行、不计入 `samples` 和局数预算，但要多打：采集变慢，多出的推理请求约等于偏离次数乘以偏离后本局剩余的学习决策数 |

训练计划的 `execution` 还可以固定按主机校准的推理调度：`inference_history_band` 按完整历史长度分组，`fine_graph_batches` 细化 CUDA Graph 的批次大小。它们覆盖同名 `cfg` 项，只减少 padding，不截断历史、缩减训练量或改变并发资源；`execution` 里出现其他键时训练直接报错。自定义采样／价值 hook 保持原有的批次组合。

## 残局开局

`ctx.collect(..., starts=局面列表)` 时，每场只打一局，从随机抽到的局面开始：学习座位坐在局面的 `focal`，其余座位照常按 `cfg["opponents"]` 抽取。教师标签（`labels=True`）、`playout_baseline` 和钩子都照常可用；这种对局走 Python 引擎，每个对局进程最多同时打 8 桌，每场计 1 局的局数预算（记录的 `hands` 为 1，`collection.jsonl` 的 `openings` 为真）。

局面（`guided.starts`）是一个字典，内容如下：
- 牌墙种子 `seed`、`dealer`、`hand_no`、`scores`；
- `prefix`：从发牌起的每一步动作，每步存成 1 字节的动作编号（十六进制串）。几万个局面在每个对局进程里也只占几十 MB。这些动作在引擎里重放到该局面，所以事件和观察与真实对局一致，但没有之前各局的历史；
- 学习座位 `focal`，它接下来要做选择；
- `kind` 等标签。

局面集由维护者用 `harness.starts` 从最好规则版本的四家自打中生成，放进 `recipe/starts/`（不进 Git），用 `ctx.starts(name)` 读取。普通采集和残局采集的比例由方案自己调用两次 `collect` 来混合。

## 关键窗口搜索（专家迭代）

`cfg["search"] = {"kinds": ["kong", "fish"], "worlds": 8, "max_windows": 1, "deal_value": 0.0}` 时，每场里学习座位的前 `max_windows` 个关键窗口要先做一次搜索，用于专家迭代。

**窗口**
- `kong`：学习座位可以杠；
- `fish`：持白且可以自摸胡（胡还是继续）。

窗口按学习座位自己的合法观察判定。

**候选动作**
- 第一个候选是 Actor 的 argmax；
- 另一类动作里，杠或胡数量少，逐个列为候选；
- 其余动作由 Actor 在屏蔽了 argmax 那一类后再选一次，这次推理照常走采集父进程。

**假想世界与续打**
- 每个候选在同一批 `worlds` 个假想世界里各打一遍。假想世界由 `mahjong_ai.sdk.simulation.sample_world` 从合法观察和独立种子生成，未见牌均匀分配。
- 之后学习座位走 argmax，其余座位照常，打到本局结束。
- 价值是本局得分（底分为单位），学习座位坐下一局庄时再加 `deal_value`。

**写回该行的字段**
- `search_kind`、`search_actions`（argmax 在首位）；
- `search_values`（世界 × 候选）、`search_q`（各候选均值）；
- `search_worlds`（成功生成的世界数）。

**执行**
- 带搜索的采集走 Python 引擎；
- 一场的搜索全部打完后才调用 `finish_game`；
- 搜索只在训练采集里做，评估和上场时 Actor 照常直接出牌。

**建议**
- 和残局开局（`starts`）一起用：每场一开局就是窗口，算力集中在这些局面上。
- 损失由方案决定，例如：只在最好候选与 argmax 的配对差显著时，把 Actor 拉向最好候选，其余行用 KL 锚住冻结副本。

`hooks` 在主进程里对学习座位生效：

- `hooks["values"](rows) -> 数组`：给出每行的价值估计，写入 `value`。采集时 Critic 为空，所以使用 PPO 的方案必须提供它。
- `hooks["sample"](logits, mask) -> (actions, logprobs)`：替换默认的按策略概率采样，例如调温度来增加探索。返回的 `logprobs` 必须是实际采样分布下的对数概率。

评估时不调用任何钩子，Actor 取概率最大的动作。

在同一学习步里多次采集且 Actor、对手权重均不变时，可用 `with guided.collection_session.InferenceSession(actor) as session`，将 `inference_session=session` 传给各次 `ctx.collect`。会话只复用设备端推理快照与 CUDA Graph；每次采集仍创建独立游戏和规则策略，保持各自的 seed、完整收尾与预算统计。参数、buffer、模型对象或推理配置在会话内改变会报错。必须在训练更新前退出会话；不能用 `.data` 等绕过版本计数的方式修改会话内权重。推理快照与 CUDA Graph 会跨学习步保留（带不带会话都一样），每次采集开始时把 Actor 当前的全部参数与 buffer 原位复制进快照再复用，对手、推理配置或结构变了才重建；这部分显存在训练期间一直占用。

## 约定

- Actor 只能读取合法观察。隐藏信息只能进入 Critic 或训练目标，这是 ADR-0009 的训练期例外。
- 钩子模块在对局进程里加载时，引擎运行时已经绑定，模块顶层可以导入 `mahjong_ai`。
- 方案可以导入 `guided` 里的构件（例如 `guided.networks.HistoryEncoder`、`PopArt`），但不能修改框架文件。
- 只能使用训练环境已有的依赖：PyTorch、NumPy 和仓库代码。
- 训练主机不可达时，远端训练照常进行，框架无限期等待。主机重启导致训练丢失时，记为主机故障：不消耗尝试次数，主机恢复后同一次训练从头重跑。
