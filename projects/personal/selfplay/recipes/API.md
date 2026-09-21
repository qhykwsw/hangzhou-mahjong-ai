# 训练方案接口

训练方案迭代（见 [ADR-0012](../../../../docs/adr/0012-recipe-iteration-for-rl-training.md)）改进的对象是一个名为 `recipe` 的 Python 包。框架 `guided/trial.py` 在远端导入它，调用 `recipe.train(ctx)`，在时限内拿回训练后的 Actor，再做配对评估。[`baseline/recipe`](baseline/recipe) 是初始方案。

## 固定不变

- Actor 的网络结构和观察编码。`ctx.actor` 从本方案线的固定起点加载；返回的权重必须能严格载入同一结构，否则本次尝试记为崩溃。
- 游戏环境、规则和奖励：每局结束时，学习座位得到该局桌面分（除以底分）。
- 对手和教师：冻结的四个规则代表，教师是其中的规则冠军。
- 时间预算：从 `train` 被调用开始计时，到 `ctx.deadline` 截止；超过宽限期，进程被强制结束。
- 评估和晋级：训练后的 Actor 和现役 Actor 在同样的牌墙、座位和对手上各打 1024 场，配对差的 95% 区间下界大于 0 才晋级。

## `train(ctx)`

必须在 `ctx.deadline` 之前返回训练后的 Actor 或它的 `state_dict`。返回 `None` 时，框架使用最近一次 `ctx.save_actor` 保存的权重。

`ctx` 提供：

| 成员 | 说明 |
| --- | --- |
| `actor` | 已加载起点权重、在 GPU 上的 Actor |
| `device`、`seed`、`deadline`、`time_left()` | 设备、随机种子、截止时刻、剩余秒数 |
| `smoke` | 自检运行时为真，此时时限只有约 90 秒 |
| `resources` | `cpu_capacity`、`cpu_count`、`gpu_total_bytes`、建议的 `workers` 与 `tables` |
| `teacher`、`rules`、`bundle` | 教师 ID、四个规则对手 ID、冻结包路径 |
| `new_actor()` | 同结构的新 Actor，例如用作自身历史快照对手 |
| `collect(actor, cfg, samples=, iteration=, labels=False, history=None, hooks=None)` | 采集训练对局，见下文 |
| `log(**metrics)` | 写一行训练诊断，评估报告和下一轮交接都会引用 |
| `save_actor(actor)` | 保存 Actor 检查点，防止超时丢失成果 |

## 采集对局

`ctx.collect` 让 `actor` 坐在一个随机座位上学习，其余三座是对手，返回 `(rows, records, resources)`：

- `rows`：学习座位的每个决策。包含 Actor 输入（`planes`、`scalars`、`events`、`mask`）、`action`、`logprob`、`value`、`reward`（只在每局最后一个决策上，值为该局桌面分）、`hand`（第几局，1～16）、`adv` 和 `ret`（由 `finish_game` 计算）、教师标签 `teacher`（`labels=True` 时是冠军在同一局面的动作，否则为 −1），以及 `critic_features` 返回的数组。同一整场的决策按顺序连续排列。
- `records`：每整场一条，含四家得分和座位。
- `resources`：采集耗时和各 worker 的场数、样本数。

`cfg` 里由方案决定的键：

| 键 | 作用 |
| --- | --- |
| `workers`、`tables` | 对局进程数和每个进程同时打的桌数 |
| `history_probability` | 对手座位从 `history`（`{"model:名字": Actor}`）抽取的概率，其余抽规则代表 |
| `critic_features` | `"模块:函数"`，在对局进程里对每个学习决策调用 `f(state, seat)`，返回的数组并入该行；`state` 是完整的 HandState，可读对手手牌和按摸牌顺序排列的牌墙 |
| `finish_game` | `"模块:函数"`，每整场结束时在对局进程里调用 `f(rows, cfg)`，必须给每行写入 `adv` 和 `ret` |

`hooks` 在主进程里对学习座位生效：

- `hooks["values"](rows) -> 数组`：给出每行的价值估计，写入 `value`。采集时 Critic 为空，所以使用 PPO 的方案必须提供它。
- `hooks["sample"](logits, mask) -> (actions, logprobs)`：替换默认的按策略概率采样，例如调温度来增加探索。返回的 `logprobs` 必须是实际采样分布下的对数概率。

评估时不调用任何钩子，Actor 取概率最大的动作。

## 约定

- Actor 只能读取合法观察。隐藏信息只能进入 Critic 或训练目标，这是 ADR-0009 的训练期例外。
- 钩子模块在对局进程里加载时，引擎运行时已经绑定，模块顶层可以导入 `mahjong_ai`。
- 方案可以导入 `guided` 里的构件（例如 `guided.networks.HistoryEncoder`、`PopArt`），但不能修改框架文件。
- 只能使用训练环境已有的依赖：PyTorch、NumPy 和仓库代码。
