# 第一性原理迭代线（firstprinciples）

两条由迭代 Agent 推进的策略改进序列，设计与取舍见 [ADR-0016](../../../docs/adr/0016-maintainer-iteration-lines.md)，术语（迭代线、尝试、迭代 Agent、维护者、固定对手池）见 [CONTEXT.md](../../../CONTEXT.md)。

- **规则线**：`rule/` 是唯一被修改的目录。v0 是最朴素的牌效策略，之后按杭州麻将规则的第一性原理逐次改进，大方向是速度优先、必胜时追求翻倍。规则线也可以从主线的分叉口接着走：`checkout rule <版本>` 把任意评估过的规则版本放回 `rule/`，`start --from <版本>` 沿这条分支自己的版本接着改（见[规则线手册](handbooks/rule.md)）。
- **RL 线**：沿用训练方案框架（`../selfplay/recipes`、`guided.trial`）。模型规格按主机固定：4070 与 5070 Ti 是小规格 128/4/4（约 0.84M 参数），起点是它对 piao-call-channel 做 10 万局（每场 16 局）的模仿；4090 在 2026-10-04 至 10-05 跑过中规格 256/6/8（约 4.8M 参数，起点是随机初始化的 random-medium-256x6），10-05 起也跑小规格（用户决定 RL 线只留 4090 一条路线）；中规格与此前的大规格 384/8/8（约 14.3M 参数，起点 imitation-piao-1h）版本仍可比较，但不再训练。各种规格共享一个最好版本，续训只在同规格之间。每次尝试的训练局数由迭代 Agent 选（默认 100 万；小规格与中规格最多 500 万，大规格最多 1000 万、其中带教师标签的最多 500 万，大规格靠一段段续训累积训练量），可以从固定起点、当前最好版本、任意一个评估过的 RL 版本出发，或在方案里从随机初始化开始（例如换教师重做模仿），在 4070、4090 或 5070 Ti 上进行。任意两个版本可以用 `harness.evaluate compare <A> <B>` 在开发集上配对比较。
- **规模训练**（维护者）：大规格从随机初始化开始，按开跑前固定的方案与计划（`scale/recipe`）模仿 500 万局、RL 1 亿局，分 12 段，每段是 RL 线的一次尝试，照常评估与确认。训练对手是第三代训练对手（fp-rule-068 派生的 7 种风格，`harness.rl training-rules` 登记），评估仍对固定对手池。`harness.scale begin` 冻结方案并开跑第一段，之后的段在上一段记录后自动开跑；进行期间它的主机只给它用。`harness.scale status` 看各段进度。2026-10-03 用户决定提前结束（PG 段到平台，见 ADR-0016），4090 交回 RL 线做大规格难例蒸馏 fp-rule-127。

## 评估

`harness/evaluate.py` 让一次尝试面对固定对手池（7 个版本，现役成员见 `.local/pool.json`，组成的取舍见 ADR-0016）：每场被评估版本坐一个座位，其余三座取对手池的三个，35 种阵容循环、四座轮换，牌墙与各方种子只由局组编号决定，所以开发集对每次尝试都是同一批牌局。规则线在 Mac 本机评估（GPU 主机借给规则线时也可在那台主机上评估，`hosts lend <主机> --line rule`），RL 线在训练它的 GPU 主机上就地评估，每台机器同一时间只跑一个评估；复用研究的对局执行器与冻结运行时，时限为出牌 2.0 秒、应答 0.6 秒，超时按平台默认动作代打。

对局不用沙箱：对手池的代码包是维护者自己的，对局进程直接在进程内运行它们（`InlinePolicy`），每个进程打一桌或几桌（一桌一个线程），模型座位共用主机的推理服务。对手池全是规则时一个进程只打一桌：规则座位吃满一个核，多一桌只会让座位互相等解释器锁、超时翻倍。规则包的缓存要小，一个进程里所有座位共用垃圾回收，大缓存让全量回收停顿一秒以上。并发数属于按主机设置的执行参数（`EVAL_WORKERS`、`EVAL_TABLES`），不进对手池 id；运行时用 `harness.evaluate runtime` 从 `src/mahjong_ai` 冻结，换运行时就要重建对手池并重跑参照。对局任务目录按池 id 隔离（`.local/jobs/<池 id>/<尝试>/`）：中断后续跑会复用已完成的任务，换池后绝不会把旧池的结果当成新结果。模型座位的线性层以 fp16 推理，归一化与残差保留 fp32。接近同分的动作可能受设备、后端或批次形状的浮点舍入影响，不能保证跨配置逐场得分完全一致；更换推理实现也要冻结新版本并重跑参照。

- 开发集：每个版本（尝试、参照、RL 起点）都打满同一批 16384 场，一个版本只有一个成绩；同一版本有多条记录时，图、排名和 `status` 只显示场数最多、最新的那条。
- 确认集：开发集超过最好成绩后，换全新牌墙与最好版本各打 8192 场，配对差的单侧 90% 下界大于 0 才算成功。两条线做的是半决赛与决赛用的锦标赛版本，只按对手池判定。10-05 曾计划让开发集过线的版本加打「最近一周最强 15 人」的模仿代理，检验发现代理只还原风格、不还原强度（131 对它们每场 +15.3，与对人群代理的 +15.2 相同，而线上这 15 人与我们大致持平），用户决定不用；现有的强手替身仍是固定对手池。确认期间最好版本换了（RL 线三台主机并行时会发生），先用新最好版本存档的开发集成绩比一次，不领先就直接判失败，领先才让它重打这批新牌墙。
- 新的最好版本还会在开发集上与自由对战的上场版本配对，下界大于 0 时汇报「高于参照线，可以上场」。2026-09-30 起用户授权维护者直接换上（`harness.line promote <完整编号>`）、事后告知；比赛最终上场版本仍由用户决定。
- 代理桌（2026-10-05 起）：海选人群的替身。三家是平台玩家的代理，即用他们自己的牌谱模仿出的 Actor（ADR-0015），只学该玩家最近一周的对局，模仿他们现在的 AI；它们还原风格（验证一致率中位数 0.91）和人群的整体水平，但不还原强手的强度。面板 v3 是最近一周遇到最多的 63 个玩家，加上其余 78 个少见玩家按强弱分两档合成的长尾代理；按约 150 人参赛规划，63 个常客按人平均共占约 42% 的座位，长尾占其余。同名代理一桌最多一个，长尾代理可以占几个座位。启用的（`.local/proxy/panel.json`）只是留出一半（偶数号常客加每档的一个长尾），用于判定海选线；另一半（`.local/proxy/train-panel.json`）供海选候选训练和调参，候选不得对着留出的一半训练或调参。两条线成功的新最好版本在同一台主机上补打一批（默认 6000 场），与被它取代的最好版本、参照版本（fp-rule-131）逐组配对，只作记录，供网页「代理桌」和海选线比较。参照与各版本在每台主机各打一次，之后复用，因为不同推理后端在接近同分时的选择可能不同，配对只在同一台主机上进行。等价替换沿用最好版本的代理桌成绩。
- 等价替换：规则线只提速、不改决策的尝试（`start --equivalent`）不跑开发集。`harness/equivalence.py` 在新牌局里让它与最好版本面对同一批局面（最好版本坐一座，其余三座是对手池里的规则），逐个决策比较；全部一致才沿用最好版本的成绩成为新的最好版本，否则记为不等价。每个规则结果都附每决策耗时，与 RL 教师清单同一口径（Mac 单进程实测）。

## 海选线

2026-10-05 起（用户决定）比赛准备两个版本：两条迭代线做半决赛、决赛用的**锦标赛版本**，对手是固定对手池（强手）；**海选版本**面向海选里大多较弱的人群，允许对强手的期望略降，但要明显多赢人群。海选线有两条，海选 1 由 Codex 主导、海选 2 由 Claude 帮手主导，规则与 RL 都可以用；维护者统筹，不亲自做。工作方式见[海选手册](handbooks/qualifier.md)。

`harness/qualifier.py` 判定海选候选，对手主要是代理桌，与本线最好的海选版本比（第一个之前与参照 fp-rule-131 比）：

1. 固定的海选代理桌（留出一半，6000 场），配对差大于 0，否则止于此；
2. 开发集（16384 场）：比两条线最好的锦标赛版本低不超过 2 分/场（`POOL_FLOOR`），否则止于此；
3. 换新牌墙、新抽阵容的代理桌，候选与本线最好的海选版本各打 6000 场，配对差的单侧 90% 下界大于 0 才成功。

记录在 `.local/qualifier.jsonl`，代理桌成绩也进网页的代理桌图。命令：`python -m harness.qualifier submit <规则包目录 | Actor 目录 | 版本> --line q1|q2 --host <主机> --name "..."`，中断后 `submit --resume <编号>`，`list [--line]` 看结果。

## 迭代方式

用户在 Claude Code 或 Codex 里说「开始/继续执行规则迭代」或「开始/继续执行 RL 迭代」，那个会话就成为该线的迭代 Agent，按[公共手册](handbooks/common.md)循环：`harness.line wait` 带回汇报和下一步，`harness.line start` 开跑一次尝试。尝试在脱离会话的驱动进程里运行，状态全部在 `.local/running/<编号>.json`；驱动进程死了（会话关闭、Mac 重启），下一次 wait 会把它重新拉起，从原来的阶段接着做。同一条线同一时间只有一个会话持有推进权。

RL 线最初的最好版本就是模仿起点本身，三台主机从一开始就能各朝一个方向改；三个尝试位共享一个最好版本。主机离线时尝试等它回来；主机重启丢了训练，尝试排队由同规格的空闲主机重做。用户可以随时关闭或开启某台主机，也可以预约关闭时刻（主机只借用到某个时间）：到点后任一条线的 wait 关闭它，没跑完的尝试排队到其他主机重做，研究方向不中断。

## 维护者

迭代 Agent 只改 `rule/` 或 `rl/recipe/`。下面这些是维护者的工作，每次改完都要重新封存（`seal`），否则迭代会拒绝运行：

训练采集的执行参数按主机校准，设置在 `harness/rl.py` 的 `EXECUTION`（进程数、桌数、攒批时间与推理调度），随训练计划冻结并封存。推理调度里，历史长度分组只减少 padding，细化批次大小只减少空行，都保留完整观察与训练量；CPU 瓶颈与 GPU 瓶颈的主机使用各自实测的配置。

```bash
cd projects/personal/firstprinciples
../../../.venv/bin/python -m harness.evaluate runtime                                       # 把 src/mahjong_ai 冻结为运行时
../../../.venv/bin/python -m harness.evaluate pool [--model <weights-sha> --name <模型名>] [--rules <研究候选 ID 或代码包目录，逗号分隔>] [--runtime <目录>] [--replace]  # 冻结对手池
../../../.venv/bin/python -m harness.evaluate reference <研究候选 ID | 池内模型名> [--weights <sha>] [--host local|4070|4090|5070ti]  # 参照线
../../../.venv/bin/python -m harness.starts make --kinds kong-window,kept-fourth,fish-window --count <N> --name <集名> [--workers 8]  # RL 线的残局集（.local/rl/starts/）
../../../.venv/bin/python -m harness.rl export <远端 checkpoint> --name <模型名> --host <主机>  # 导出别处训练的模型（如对手池成员）
../../../.venv/bin/python -m harness.rl setup <主机上的 checkpoint | 权重 sha> --name <起点名> --host 4070|4090|5070ti  # RL 起点（每台主机一次）
../../../.venv/bin/python -m harness.rl baseline [--host 4070]                                   # 起点的完整开发集成绩，作为 RL 线最初的最好版本（每个对手池一次）
../../../.venv/bin/python -m harness.rl proxies <panel.json>                                     # 把代理面板的成员登记为可选训练对手（替换上一次登记）
../../../.venv/bin/python -m harness.rl training-rules <代码包目录>... [--scores <json>]            # 把规则代码包登记为可选训练对手（不参与评估，替换上一次登记）
../../../.venv/bin/python -m harness.evaluate implement <对手池规则成员> <代码包目录>          # 登记成员的等价实现（更快、决策一致）
../../../.venv/bin/python -m harness.proxy register <actors 目录> --job <job.json> --name "代理面板 vN"  # 代理 Actor 入模型库，写出面板描述
../../../.venv/bin/python -m harness.proxy activate <panel.json> [--reference fp-rule-131] [--name ...]  # 启用代理桌（之后 seal）
../../../.venv/bin/python -m harness.proxy run <版本> --host <主机>                               # 给启用前的版本补测代理桌
../../../.venv/bin/python -m harness.evaluate seal                                          # 接受当前的评估条件
```

- 对手池成员的等价实现：成员的冻结代码包太慢时，写一份决策完全相同的快版，用 `implement` 登记。它在 16384 局新牌局里逐个决策与冻结版本比较，全部一致才登记到 `pool.json` 的 `implementations`，之后评估、等价检查与 RL 训练对手都运行快版；对手池 id 与已有成绩不变。登记改变了封存的条件，随后 `seal`。

- 封存覆盖 `harness/` 全部代码（评估、训练预算与按主机的执行参数、循环）、对手池、启用的代理桌（`.local/proxy/panel.json`），以及每台 RL 主机的训练框架快照与起点。
- 条件变了还没重新封存时，harness 不开始新尝试；正在循环的迭代 Agent 的 wait 继续等，一次 wait 结束（最多 9 分钟）仍不一致才让它停下。所以改完随即 `seal`，迭代不会中断。
- 换对手池（新成员或新运行时）按这个顺序：
  1. `harness.evaluate drain on "<原因>"`：两条线不再开新尝试，在途尝试照常跑完，迭代 Agent 继续等待；
  2. 在途尝试全部结束后，`pool ... --replace` 冻结新池并 `seal`；
  3. 在新池下补齐成绩：每个成员的参照（`reference`）、`harness.rl baseline`（RL 起点），以及两条线每个成为过最好版本的版本与自由对战上场版本（`rebase <编号> ... --host <主机>`，可分到几台主机并行，中断后重跑会跳过已完成的）；
  4. `rebase-finish rule` 与 `rebase-finish rl`（在 baseline 之后）：重算的版本沿用原编号记入新池，分数从低到高写入，所以每条线重算分最高的成为它在新池下的最好版本；这些记录不作为新结果汇报；
  5. `drain off`，告诉两条线新旧分数不可比。一条线的重算先完成时，可以先记入它，用 `drain on "<原因>" --only <另一条线>` 只让另一条线继续等；记入之后不再给这条线补记重算，否则低分记录会排在后面成为「最好版本」。
- 评估条件在同一个池里变了（成员登记了等价快版、主机执行参数改了），或最好版本的开发集存档带着超时：`drain on "<原因>" --only <线>`，等这条线在途尝试结束后 `refresh <线>`，在当前条件下重打它最好版本的开发集并追加同编号记录（时间位置不变），再 `drain off`。开发集配对差读的是最好版本的存档，存档条件与候选不同时会系统性偏高或偏低；确认集每次重打冠军，不受影响。
- 规则线候选可以自带 `native.c`，由运行时的 `_fast.load_native` 按源码在每台机器上编译一次、缓存在包外；start 开跑前先编译并检查 C 源码（[ADR-0017](../../../docs/adr/0017-rule-candidates-carry-their-own-c.md)，用法见[规则线手册](handbooks/rule.md)）。
- 给引擎加速器新增或改进函数（例如对手池成员的 C 实现）：先改 `src/mahjong_ai/_fast.c` 并通过全量测试，再 `drain on "<原因>"`，等在途评估和 `implement` 结束后 `upgrade-runtime`：以现役运行时为底换上仓库的 `_fast.c`。竞速部分（C 成员的函数，从 race valuation 横幅到方法表）以外的原有代码逐行保留、只许新增；竞速部分可以改，这时每个已登记的成员实现都要在新运行时下按 `implement` 的手数重新比对，逐决策与冻结成员相同。规则线最好版本在新旧运行时下重放同一批开发集、逐组成绩相同才切换；池编号、开发集和成绩都不变，远端主机在下次同步时自动编译。之后 `seal`，用 `implement` 登记用到新函数的成员实现，`drain off`。改动竞速部分以外的原有 C 代码时不能原池升级，要换新的对手池。
- 修改 `projects/personal/selfplay/guided/` 后：把 `.local/rl/source` 按其 `source-manifest.json` 的 id 归档到 `.local/rl/sources/<id>`，再对每台主机重新 `setup <起点 sha>`，最后 `seal`。`setup` 从当前工作区冻结框架（`guided/` 与 `src/`）：工作区里有其他会话未提交的改动时，先用 `git worktree` 取一份干净的 HEAD，把 `recipe_jobs.REPO` 指向它冻结到 `.local/rl/source`，再 `setup`。
- 往 4070 上传教师包时，清单里登记的文件要全部带上，包括 Mac 的 `.so`，否则 `read_bundle` 会拒绝；每个运行时目录都要在主机上单独编译 `_fast` 扩展。对手池里有模型成员时，评估由对局执行器从 Mac 的模型库自动同步，不要手动往主机拷贝模型。
- 借主机给规则线：`harness.line hosts lend <主机> --line rule`（还回 `--line rl`），这台主机跑完手上的尝试后只接规则线的尝试；规则线自己的研究脚本（续打采数、联合调参）用 `python -m harness.hostrun push|run|status|fetch|stop --host <主机>` 在那台主机上跑（用法见 `harness/hostrun.py` 文件头）。
- 流程检查：把 `.local/pool.json`、`.local/seal.json` 和 `.local/rl/setup-*.json` 复制到仓库内一个单独的目录，设 `FP_LOCAL` 指向它，再用 `FP_DEV`（开发集局组数，每组 4 场）、`FP_CONFIRM`、`FP_EQUIVALENCE_HANDS`、`FP_RL_HANDS`、`FP_RL_BUDGET` 缩短场数与训练预算，跑一遍 wait/start。这些缩短的设置只在 `FP_LOCAL` 下被接受，流程检查不提交 Git、不改动自由对战，跑完删掉这个目录。

## 运行数据

都在本项目 `.local/`：`pool.json`（现役对手池）、`pools/<池 id>.json`（每个对手池的存档，换池后网页按它画各代的参照线）、`seal.json`、`attempts.jsonl`（每次尝试一条记录）、`rebase/<池 id>-unrecorded/`（只供网页展示、不记入记录的换池重算）、`running/`（在途尝试的状态）、`reported.json`（已汇报给用户的记录，按「编号@完成时间」登记）、`hosts.json`（主机开关）、`notes/<线>.md`（迭代笔记）、冻结的尝试快照与任务目录。研究网页的「迭代线」标签页只读展示。
