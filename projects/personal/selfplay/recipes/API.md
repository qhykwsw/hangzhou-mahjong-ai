# 训练方案接口

训练方案迭代改进的对象是一个名为 `recipe` 的 Python 包，它是自动研究里 RL 方案分支的候选（[ADR-0013](../../../../docs/adr/0013-unified-research-branches-and-common-brawl.md)，早期设计见 [ADR-0012](../../../../docs/adr/0012-recipe-iteration-for-rl-training.md)）。框架 `guided/trial.py` 在远端导入它，调用 `recipe.train(ctx)`，在时限内拿回训练后的 Actor，交给研究框架评估。[`baseline/recipe`](baseline/recipe) 是初始方案：多步小步策略梯度，每步用约 60 万个同策略决策算一次全量梯度，按 KL 0.03 定步长。设计理由写在 `loop.py` 和 `targets.py` 的文档字符串里，测量依据见 ADR-0012 的“基线方案与训练时长”。

## 固定不变

- Actor 的网络结构和观察编码。`ctx.actor` 从本方案线的固定起点加载；返回的权重必须能严格载入同一结构，否则本次尝试记为崩溃。起点可以带可选结构（起点 `meta.json` 的 `config`，如信念头、花色共享嵌入，见 [guided README](../guided/README.md) 的「Actor 的可选结构」），`ctx.new_actor()` 与导出沿用它；带信念头（`belief_head`）时 `actor(..., return_belief=True)` 另返回三家对手手牌的 3×34 logits，方案可用行里的 `oracle_planes` 经 `guided.networks.belief_target` 监督；`"belief_head": "v2"` 时另返回的是 `BeliefV2(tiles [n, 7, 34], state [n, 6])`，目标用 `guided.networks.belief_target_v2(oracle_planes, oracle_scalars, opp_waits)`，`opp_waits` 由 `critic_features` 调用 `guided.encoding.opponent_waits(state, seat)` 加进行里（或直接用钩子 `guided.encoding:belief_features`）。起点带 oracle 输入（`oracle_inputs`）时见「Oracle 输入」；它只在训练里存在，导出时剥离，所以返回的权重带不带这部分参数都能导出。
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

- `rows`：学习座位的每个决策。包含 Actor 输入（`planes`、`scalars`、`events`、`mask`）、`action`、`logprob`、`value`、`reward`（只在每局最后一个决策上，值为该局桌面分；学习座位整局没有决策时，该局分不记到任何行，只留在整场账本里）、`hand`（第几局，1～16）、`adv` 和 `ret`（由 `finish_game` 计算）、教师标签 `teacher`（`labels=True` 时是教师在同一局面的动作，否则为 −1；教师在某个决策上报错或给出非法动作时，这一行也是 −1，采集照常进行（纯模仿采集在这一步改由学生采样出牌，教师损失不计这一行），worker 报告里的 `teacher_errors` 计数，首次出错的堆栈写进训练日志。规则教师打标签时采集走 Python 引擎，比纯 PG 采集慢；教师是模型时，标签是它在合法动作上的 argmax，由采集父进程在学生出牌前与学生同批推理（精度与模型对手相同），不会是 −1，对局照常可走 C 引擎，`label_version` 为 `model:teacher:<ID>`），以及 `critic_features` 返回的数组；开 `cfg["oracle_inputs"]` 时另有该决策实际喂给 Actor 的 `oracle_input_planes`、`oracle_input_scalars`（见「Oracle 输入」）。同一整场的决策按顺序连续排列。
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
| `vine` | 与 `playout_baseline` 同用。为真时，学习座位在对局里每个决策都走 argmax，采样动作与 argmax 不同处（行的 `deviated` 为真）由复制局走采样动作、之后学习座位一律 argmax、打到本局结束。训练状态因此始终是 argmax 路径上的状态（与上场分布一致），采样多偏离也不改变它；偏离行的 `playout` 是「这步走采样动作」的本局分，对局本身的本局结果就是这处的 argmax 价值。行的 `action`、`logprob` 仍是采样动作的 |
| `stable_wall` | 为真时训练对局用稳定牌墙（`guided/walls.py`）：每家的普通摸牌来自自己那一串交错位置（按转换时从牌墙尾部往前数，下一位摸牌者取最后一张，下一家取前一张，依次相隔四张），杠后补牌仍取牌墙前端。没有吃碰时与普通牌墙摸到的牌完全相同；有吃碰时各家仍摸到自己那一串，所以偏离分支与 argmax 分支的配对不因吃碰错位。牌墙是均匀洗乱的，每次摸牌在未见牌里仍然均匀，对局分布不变。整场在发牌后转换；残局在重放前缀之后转换。复制局（playout）沿用同一副稳定牌墙 |
| `luck` | 为真时记录学习座位的摸牌运气（AIVAT 式控制变量）：每次摸牌（庄家第一张除外）若摸牌者已听牌，f(y) 为「摸到 y 立即自摸」时学习座位的收益（底分为单位），d(y) 为此时学习座位是否坐下一局庄；累计 f(摸到的牌) − E f 与 d(摸到的牌) − E d，期望按剩余牌墙精确计算。两者在给定历史下均值为零，所以一个价值减去从某决策起累计的运气，期望不变、方差变小。行带 `luck_pay`、`luck_deal`（决策时的累计值）；每局学习座位最后一行带 `end_luck_pay`、`end_luck_deal`（本局结束时）；偏离行带 `playout_luck_pay`、`playout_luck_deal`（续打结束时）。坐庄溢价为 D 时，校正量是 `pay + D × deal` 的差 |
| `luck_kind` | 摸牌运气的种类，与 `luck` 同用：`"immediate"`（默认，即上一行）或 `"race"`。`"race"` 时上一行的字段照记，另记竞速运气（公式与近似见 `guided/walls.py`）：四家的每次摸牌（含杠后补牌）都累计 v(摸到的牌) − E v，期望按剩余牌墙精确计算。摸牌者摸到 y 立即自摸时 v(y) 就是上一行的 f(y)；否则 v(y) 是摸走 y 之后的竞速值：各听牌者按剩余牌墙里和张的张数与剩余摸牌次数，算本局先于别家自摸的概率（各家自己的摸牌按无放回抽样，各家之间视为独立，听牌、番数按当时手牌不变，不计吃碰改变的摸牌顺序），乘以它自摸时学习座位的平均收益再求和；坐庄部分是学习座位先自摸的概率，学习座位是庄时再加上无人自摸的概率。v 只用摸牌前的状态和摸到的牌定义，给定历史均值为零，近似只影响降方差的多少。行另带 `luck_race_pay`、`luck_race_deal`，每局最后一行带 `end_luck_race_pay`、`end_luck_race_deal`，偏离行带 `playout_luck_race_pay`、`playout_luck_race_deal`，用法与上一行相同 |
| `opponent_jitter` | `{"sd": 0.1}`：训练对局里每个 race 成员座位（固定对手池里整步在 `race_act` 决策的成员）这一场用自己的一份 race profile，浮点常数（q、saved、one_step、far_step、q_early、q_late、hunt、draw_risk、tie）各乘以 1 + sd × N(0,1)，正态数取自本场种子的独立随机流；再截到合理范围：概率类 q、q_late、draw_risk 在 [0, 0.95]，q_early 在 [0, q_late]，折扣 far_step 在 [0, 1]，其余不小于 0；整数与开关字段不变。fp-rule-131 不抖动：它按角色和局次从九份 profile 里选一份，另有 profile 之外的常数和表，单抖 profile 不等于抖这个策略；其他在 Python 里决策的规则座位没有 profile，也不抖。两种引擎都在 `native.c` 里用这份副本决策，结果逐位一致。只作用于训练对局，评估对局不受影响；sd 为 0 或不给这个键时对局逐位不变。说明见 `guided/opponents.py` |
| `opponent_timeout` | `{"mean": 0.15, "shape": 2.0}`：训练对局里每个规则对手座位按本场种子抽一个超时率 ε ~ Beta(mean × shape, (1 − mean) × shape)，shape 是集中度（默认 2，均值 0.15 时中位数约 0.05，约九分之一的座位超过 0.5）。该座位每个有选择的决策以概率 ε 改走平台的超时默认动作，规则本身不被调用：吃碰窗口过；轮到出牌时能胡就胡，否则打出手牌按摸入顺序的最后一张（摸牌后即摸切，吃碰后是手里最后一张），与 `HandState.timeout_action` 和平台指南一致（`docs/reference/platform.md`）。逐决策的随机数来自按「本场种子、局号、座位」定的流，C 与 Python 引擎相同。学习座位与模型座位不受影响；mean 为 0 或不给这个键时对局逐位不变 |
| `starts_reshuffle` | 为真时残局在重放前缀之后按本场种子重洗剩余牌墙：位置之后的牌序本是隐藏的，重洗让每次重放都是新的隐藏样本，而不是同一副录制牌墙 |

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

`cfg["search"] = {"kinds": ["kong", "fish"], "worlds": 8, "max_windows": 1, "continuation": "recipe.value:after_hand"}` 时，每场里学习座位的前 `max_windows` 个关键窗口要先做一次搜索，用于专家迭代。

**窗口**
- `kong`：学习座位可以杠；
- `fish`：持白且可以自摸胡（胡还是继续）。

窗口按学习座位自己的合法观察判定。

**候选动作**
- 第一个候选是 Actor 的 argmax；
- 另一类动作里，杠或胡数量少，逐个列为候选；
- 其余动作由 Actor 在屏蔽了 argmax 那一类后再选一次，这次推理照常走采集父进程。

**假想世界与续打**
- 每个候选在同一批 `worlds` 个世界里各打一遍。每个世界是当前真实牌局的副本，只按种子重洗牌墙，对手保留真实手牌。
  - 这和 playout 基线续打真实牌局一样，只在训练进程里用隐藏状态给学习座位的选择估值（ADR-0009），Actor 仍只看合法观察。
  - 实测：把对手手牌均匀重发（sample_world）会严重偏向「继续」。在 300 个持白可胡窗口上，续打里对手胡的比例从 34% 变成 8%，「继续」被高估 9.6 分。
  - 因为价值带着真实隐藏手牌，单行的最好候选含有「知道对手牌」的成分。应跨行平均使用，例如把价值当作优势，不要逐行挑最好候选去做 CE。
- 之后学习座位走 argmax，其余座位照常，打到本局结束。
- 价值是本局得分（底分为单位），再加下一局开局状态的价值。
  - 价值由 `continuation`（`"模块:函数"`，在对局进程里解析）给出。
  - 它收到一个字典：`seat`；`dealer`（谁坐下一局庄：胜者，流局时原庄）；下一局的 `hand_no` 和 `rounds`；`scores`（本局后的四家分数，底分为单位）；`last`（本局是否是本场最后一局）。
  - 没有 `continuation` 时，学习座位坐下一局庄就加 `deal_value`。
- playout 基线的续打行同样写入 `playout_dealer`（续打后谁坐下一局庄）和 `playout_scores`（续打后的四家分数），方案可以用同一个价值函数计算那一侧。

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

## 分支候选（branches）

`cfg["branches"] = K`（正整数，须同时开 `playout_baseline`）时，每个学习决策除了对局本身和 playout 基线的单个偏离副本，还可以由方案点名最多 K 个候选动作，各开一个副本打到本局结束。实现与字段定义见 `guided/branches.py`。

**候选**
- 父进程在学习座位选定动作之后调用 `hooks["branches"](logits, mask, actions)`，返回 (行数, K) 的整数张量，每行是要额外评估的动作编号，空位填 −1；少于 K 列自动补 −1，多于 K 列报错。`actions` 是这一步实际选中的动作（采样钩子或默认采样的结果）。
- 走 CUDA 图时，带分支的学习决策只重放 logits 的图，采样钩子（标了 `capturable` 的也一样）和分支钩子都在图外、同一条流上对这一批的行运行，结果与 eager 路径的算法相同。
- 默认钩子由方案提供，框架只负责调用和执行；`cfg["branches"]` 开着却没给钩子时报错。

**副本**
- 每个候选从同一决策点各 fork 一份，走该候选；之后学习座位走当前 Actor 的 argmax，其余座位照常（规则照常决策，模型对手取 argmax），牌墙、稳定牌墙与运气记录沿用对局本身的。
- 对局本身和单个偏离副本的行为不变；两种引擎（Python 的 `Game`、C 的 `FastGame`）逐位一致。

**续打策略与控制副本**
- `cfg["branch_continuation"] = {"learner": "greedy" | ID, "opponents": {ID: ID} | None}`：副本走完候选之后怎么打。
  - `learner` 是学习座位：`"greedy"`（默认，当前 Actor 的 argmax）、一个规则版本 ID（如 `"fp-rule-131"`，race 成员和 fp-rule-131 在 C 里决策，其他规则在 Python 里决策），或 `ctx.catalog`／`ctx.opponents` 里的模型（父进程取它的 argmax），也可以是 `history` 里的 `"model:名字"`。
  - `opponents` 把对局里某个对手（键按 `cfg["opponents"]` 的 ID 写）换成另一个规则或模型，用于抽象对手、便宜对手续打；没列出的座位照对局本身打。
  - ID 取自 `ctx.opponents`、`ctx.catalog` 或 `ctx.rules`，未知 ID 报错；不能用正在采样的学习者本身。续打里的规则每个座位各有一份策略，按本场种子重置。
- `cfg["branch_control"] = True`：有候选的行另开一个副本，走 argmax、用同一续打策略，写进 `branch_control`（字段同 `branches` 的一项）。它和候选副本从同一个 fork 出发、同一副牌墙、同样的座位，候选减控制就是配对差。

**截断**
- `cfg["branch_truncate"] = k`（正整数）：每个副本（含控制副本）在候选之后学习座位的第 k 个决策处停下，不打完本局；本局先结束的副本照常是完整结果。
- 「决策」指学习座位有不止一个合法动作的回合，不论续打的是 Actor 还是规则。
- 停下的副本写 `{"action", "truncated": True, "playout_scores", "leaf_row"}`，开 `luck` 时另有到此为止的 `playout_luck_pay`、`playout_luck_deal`：
  - `playout_scores` 是到此为止的四家分数（底分为单位）。一局只在结束时结算，所以就是本局开始前的分数。
  - `leaf_row` 是停下时学习座位的编码行，只含合法观察的 `planes`、`scalars`、`events`、`mask`，编码与正常行相同。
- 方案在父进程里用自己的价值网给 `leaf_row` 估值，框架不估值。

**世界重采样**
- `cfg["branch_resample"] = {"weights": None | "模块:函数"}`：每行的所有副本（候选与控制）不从真实牌局出发，而从按学习座位视角重发的同一个世界出发。
- 重发的做法：
  - 把三家的暗手牌和剩余牌墙合成一池，先排序，再用该行自己的随机流（本场种子、`"branch-world"`、行号）洗乱；按各家原来的暗手张数发回，余下的按原长度做牌墙。
  - 先排序，所以世界只取决于学习座位看不到的那组牌，与真实谁拿了什么无关。
- 保持不变的：
  - 学习座位的手牌，以及所有公开信息：副露、牌河、各家张数、财神状态（抓打圈、连杠与财飘计数）、分数、本局事件。
  - 别家的暗杠作为副露整副保留，所以牌池是学习座位的未见牌减去这些暗杠的牌。
  - 别家过去的私有事件（发牌、摸牌）仍是原牌。只读当前手牌和公开信息的规则（race 成员、fp-rule-131）不受影响。
- 稳定牌墙：重发的牌按牌墙顺序填进剩余位置，位置结构不变。此后各家仍从自己的位置摸牌，所以同一行的副本之间仍然配对；只是位置上的牌是重发世界的，不是真实牌墙的。运气照常累计，期望按重发后的牌墙算。
- `weights`（在对局进程里解析）对每个重发世界调用一次 `f(row, world)`：
  - `row` 只含该行的 `planes`、`scalars`、`events`、`mask`；
  - `world = {"seat", "hands"（按绝对座位的四家手牌，含学习座位自己）, "wall"（按摸牌顺序）, "opened"（各家副露数）}`；
  - 返回该世界的权重，例如信念模型给出的对手手牌似然。
- 该行每个结果都带 `world_weight`，没有 `weights` 时为 1.0。两种引擎用同一个随机流重发，结果逐位一致。

**写回该行的字段**
- `branches`：按候选顺序的列表，每项 `{"action", "truncated": False, "playout", "playout_scores", "playout_dealer", "playout_deals_next"}`，含义同偏离行的同名字段；开 `luck` 时另有 `playout_luck_pay`、`playout_luck_deal`；重采样时另有 `world_weight`。截断的项见上。
- `branch_control`：开 `branch_control` 时，有候选的行才带。
- 没有候选的行不带 `branches`。`finish_game` 在该场所有副本打完后才调用。
- 副本不产生训练行、不计入 `samples` 和局数预算，但要多打：多出的推理请求约等于候选数乘以本局剩余的学习决策数。

## Oracle 输入（oracle guiding）

Suphx 式 oracle guiding（[ADR-0018](../../../../docs/adr/0018-oracle-guided-policy-training.md)）：Actor 在训练采集里额外读到按保留率 γ 随机遮盖的隐藏状态，方案把 γ 从 1 逐步降到 0，让策略学会不依赖它。框架部分见 `guided/oracle.py`；γ 的日程、学习率与样本筛选由方案决定。

**Actor**
- 结构选项 `"oracle_inputs": true`（`guided.networks.Actor`）。`actor(planes, scalars, events, oracle=(oracle_planes, oracle_scalars))`：`oracle_planes` 为 [n, 16, 34]，`oracle_scalars` 为 [n, 4]，在输入端经无偏置、零初始化的 `encoder.oracle_tile_in`（16→dim）加到每个牌 token，经 `encoder.oracle_state_in`（4→dim）加到状态 token。`oracle=None` 跳过这一路；无偏置，所以不给、给全零、剥离后三者 logits 逐位相同。
- 两种得到方式：起点就带（维护者用 `python -m guided.expand <起点> <模型库> --oracle-inputs` 做，logits 与原起点逐位相同，`ctx.actor`、`ctx.new_actor()` 都带它）；或方案在训练开始时自己加：`actor = guided.networks.with_oracle_inputs(ctx.actor)`（零初始化，同设备，原 Actor 不变；之后采集、优化器都用这个新对象）。
- 导出（`ctx.save_actor` 的快照、最终产物）一律剥离选项和参数，写出前核对剥离后的 logits 与不给 oracle、给全零 oracle 时逐位一致，不一致即报错。导出的是不读 oracle 的策略：γ 没降到 0 就导出，导出物仍合法，只是训练时它一直有 oracle 可依赖。
- oracle 参数不随导出跨尝试保留：从某个版本 `--from` 续训时，起点没有 oracle 输入，要么用带它的固定起点，要么方案重新加（重新零初始化）。所以 γ 的整个衰减要在一次尝试里走完。
- 评估、自由对战、比赛、模型服务、对手与教师都通过 `guided.deployment.load_actor` 或模型服务加载，二者拒绝 `meta.json` 或权重里带 oracle 通道的模型。

**采集**
- `cfg["oracle_inputs"] = {"keep": γ}`（0 ≤ γ ≤ 1），可加 `"continuations": False`。不给、`None` 或 γ = 0 时关闭：不计算、行和请求里都没有 oracle 字段，学习者按 `oracle=None` 决策，与剥离后的 Actor 完全相同（采集结果与不带这个键时逐位一致）。开启时 `actor` 必须有 oracle 输入，否则报错。
- 训练对局里学习座位的每个决策：worker 算出该状态的 oracle 特征（`guided.encoding.oracle`，与 Critic 的 `oracle_planes`/`oracle_scalars`、配方 `critic_features` 里的同名数组同一定义；C 引擎 `Hand.oracle` 逐位相同）：16 个平面依次是下家、对家、上家暗手里各种牌张数 ≥1..4，再是牌墙的；4 个标量是三家的 (8 − 向听) / 9 与牌墙张数 / 136。548 个值各以概率 γ 保留、否则置 0（逐元素伯努利，不放大），随机流为 `seed_for(本场种子, "oracle", 局号, k)`，k 是本局里该路径上学习座位此前的决策数；两种引擎逐位一致。
- 遮盖后的值以 float16 随推理请求发到父进程，父进程只把它喂给学习者自己的 Actor（`model:current`；CUDA 图路径有单独的图和缓冲），并留在行上：`oracle_input_planes`（float16 [16, 34]）、`oracle_input_scalars`（float16 [4]）。方案的梯度用的正是行为策略看到的输入。
- 「学习座位」包括 playout 基线的偏离副本、分支副本里续打为 `"greedy"` 的学习座位（`GREEDY`，同一个策略），它们各自按同样的公式取 k：副本沿对局本身路径走时，抽到的遮盖与对局后续各行相同（vine 配对的公共随机数）。`"continuations": False` 时这些续打不给 oracle。对手（模型或规则）、给同一批行打标签的模型教师、规则教师和评估都拿不到 oracle。

**方案的梯度**
- 用框架的 `guided.learning.make_batch`：有行带 oracle 字段时批里多出 `oracle_input_planes`、`oracle_input_scalars`（没有的行为 0，等于不给），`guided.oracle.argument(batch)` 给出 Actor 的 `oracle` 参数，没有时为 `None`：`actor(planes, scalars, events, oracle=guided.oracle.argument(batch))`。
- 方案自己的批（例如 `recipe/batch.py` 的 `make_batch`、`ResidentRows`）：`guided.oracle.stack(rows)` 返回 (planes [n,16,34], scalars [n,4]) 的 float16 数组（没有的行为 0），所有行都没有时返回 `None`；常驻列可按 `FIELDS = ("oracle_input_planes", float16), ("oracle_input_scalars", float16)` 加进去，前向时 `oracle=(fields["oracle_input_planes"], fields["oracle_input_scalars"])`。CUDA 图捕获的梯度回放要把这两列也作为静态输入。
- 计算旧策略概率、KL、重要性比时同样带上行里的 oracle；γ = 0 之后的行没有这两个字段，`argument`/`stack` 给 `None`，前向即剥离后的策略。

`hooks` 在主进程里对学习座位生效：

- `hooks["values"](rows) -> 数组`：给出每行的价值估计，写入 `value`。采集时 Critic 为空，所以使用 PPO 的方案必须提供它。
- `hooks["sample"](logits, mask) -> (actions, logprobs)`：替换默认的按策略概率采样，例如调温度来增加探索。返回的 `logprobs` 必须是实际采样分布下的对数概率。
- `hooks["branches"](logits, mask, actions) -> (行数, K) 动作编号`：`cfg["branches"]` 开启时点名每行的候选，见「分支候选」。

评估时不调用任何钩子，Actor 取概率最大的动作。

在同一学习步里多次采集且 Actor、对手权重均不变时，可用 `with guided.collection_session.InferenceSession(actor) as session`，将 `inference_session=session` 传给各次 `ctx.collect`。会话只复用设备端推理快照与 CUDA Graph；每次采集仍创建独立游戏和规则策略，保持各自的 seed、完整收尾与预算统计。参数、buffer、模型对象或推理配置在会话内改变会报错。必须在训练更新前退出会话；不能用 `.data` 等绕过版本计数的方式修改会话内权重。推理快照与 CUDA Graph 会跨学习步保留（带不带会话都一样），每次采集开始时把 Actor 当前的全部参数与 buffer 原位复制进快照再复用，对手、推理配置或结构变了才重建；这部分显存在训练期间一直占用。

## 约定

- Actor 在评估和上场时只读取合法观察。隐藏信息只能进入 Critic、训练目标（ADR-0009 的训练期例外）或训练采集里学习者的 oracle 输入（ADR-0018，见「Oracle 输入」，导出时剥离）。
- 钩子模块在对局进程里加载时，引擎运行时已经绑定，模块顶层可以导入 `mahjong_ai`。
- 方案可以导入 `guided` 里的构件（例如 `guided.networks.HistoryEncoder`、`PopArt`），但不能修改框架文件。
- 只能使用训练环境已有的依赖：PyTorch、NumPy 和仓库代码。
- 训练主机不可达时，远端训练照常进行，框架无限期等待。主机重启导致训练丢失时，记为主机故障：不消耗尝试次数，主机恢复后同一次训练从头重跑。
