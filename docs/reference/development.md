# 开发索引

## 模块与职责

| 模块 | 入口与职责 |
| --- | --- |
| `domain/` | 牌、规则、观察、动作、真实对局与完整结算 |
| `sdk/` | 精确牌形、公开轨迹、假想状态生成与模拟 |
| `strategies/` | 可运行启发式、收益预测、PPO 与搜索基线 |
| `research/` | 长期注册表、实验、分支迭代、共同大乱斗与主机可用性的状态转换，以及网页只读进度投影 |
| `evaluation/` | 有放回对手抽样、配对评估、四座赛程与成绩 |
| `execution/` | 模型配置、会话、依赖沙箱、资源与进程调度；`remote.py` 远程主机的同步、批量派发、探测与大乱斗抢占训练，`model_seat.py`/`model_service.py` 模型座位（可带一个按阶段比分切换的第二个模型，目前只在 `harness.stage` 的模拟阶段里用）与本机批量推理服务 |
| `persistence/` | SQLite 事务、幂等回执和冻结工件 |
| `platform/` | 导出、官方协议与牌谱审计 |
| `interfaces/` | CLI、研究 MCP、HTTP 与网页 |

操作系统适配集中在 `interfaces/service_host.py`（launchd/systemd 与浏览器）、`execution/candidate_runtime.py` 的沙箱入口、`execution/linux_sandbox.py`（Bubblewrap/seccomp）和 `execution/process_usage.py`（RSS）。规则、研究状态机与评估口径不因操作系统改变。Linux/WSL2 的候选限制与 macOS 相同，不提供无沙箱降级。隔离只有一种显式例外：对局任务的 `resources.isolation = "none"` 让任务拥有者信任的代码包（例如维护者自己冻结的固定对手池）直接在对局进程里运行（`candidate_runtime.InlinePolicy`），省去每次决策的序列化与进程往返；时限改为事后判定，超时仍按代打计。这不是沙箱不可用时的回退，研究执行器不设置这个字段。同一任务的 `resources.tables` 让一个对局进程同时打几桌（一桌一个线程），模型座位等推理服务时其他桌继续。

研究入口为 `mahjong`，独立引擎、训练、吞吐测试、导出与平台命令为 `mahjong-ai`。以 `--help` 为准。

人工自我对弈项目 `projects/personal/selfplay/` 的训练器直接驱动 `HandState`，不经过候选运行时；它只复用引擎、规则、SDK 与发布流程，见该项目 README 与 [ADR-0009](../adr/0009-independent-selfplay-trainer-for-personal-project.md)。同项目的规则教师指导路线 `guided/` 同样直接驱动 `HandState`，教师以冻结规则包的运行时参与对局，见[训练手册](../../projects/personal/selfplay/guided/README.md)与 [ADR-0011](../adr/0011-rule-guided-policy-learning.md)。训练方案迭代的编排器 `src/mahjong_ai/recipes/` 与规则研究同在工作台服务进程，状态存于该项目 `.local/recipes/`；它为方案分支设计并训练候选（远端由冻结的 `guided/trial.py` 执行），训练好的模型交回研究框架做评估与大乱斗，见 [ADR-0013](../adr/0013-unified-research-branches-and-common-brawl.md)。

维护者的迭代线（规则线与 RL 线）不走研究框架：`projects/personal/firstprinciples/harness/` 负责固定对手池评估、封存、尝试循环与主机分配，见[项目说明](../../projects/personal/firstprinciples/README.md)与 [ADR-0016](../adr/0016-maintainer-iteration-lines.md)。策略项目的组织、冻结发布包和跨项目注册见[策略项目](projects.md)。默认自动研究根目录是 `projects/agent-research/`。

独立 CLI 的 `rule`、`rl` 是基础策略入口；多方向研究的组合基线由 `strategies/catalog.py` 提供。CLI 的基础吞吐测试不替代各研究组合在冻结对手池上的实际训练和评估测量。

研究状态由 `Research` 的接口修改，`Store` 在事务中提交。调度器先登记任务再创建进程；任务结果使用 实验/job 唯一标识，比赛逐场落盘，恢复时复用完整回执。暂停是执行状态，不是候选质量结论。

研究证据的职责分开：`research/conclusions.py` 投影累计结论与修订，`research/selection.py` 判定完整配对证据是否达到晋级门槛，`research/agent_context.py` 生成交接、状态增量与按需读取结果。MCP 调用这些研究接口，网页使用完整快照；查询去重不能改写证据或影响选优。会话边界和读取协议见[研究手册](strategy-agent.md#会话累计结论与按需读取)。

策略只操作公开 SDK 和自身产物。`sdk/simulation.py` 创建的是假想世界，真实 `HandState` 保留在宿主进程；修改两者需同时核对观察过滤、合法动作和牌张守恒。规则、结算不得为了某类算法而改变。

## 修改与验证

```bash
.venv/bin/python -m pytest -q
.venv/bin/python setup.py build_ext --inplace   # 可选：构建牌形 C 加速器
ruff check src tests scripts --select F401,F821,F822,F823
ruff format --check src tests scripts
```

测试默认用 pytest-xdist 并行（`pyproject.toml` 的 `addopts`），进程数取本机当前空闲的核数、至少 4 个（`tests/conftest.py`），规则线评估占着 Mac 时也不会饿到依赖真实时钟的测试；全量约半分钟以内。调试单个用例时加 `-n 0` 在一个进程里跑。脚本化的线上对局测试有 30 秒看门狗，错过窗口会失败而不是卡住。

`src/mahjong_ai/_fast.c` 是可选的牌形加速器：`sdk/shape.py` 的向听/进张与
`domain/rules.py` 的胡牌形状在它存在时走 C，负数（和牌边界）用同样经过参考回归的原生胡牌谓词裁决；
没有构建扩展时自动回退到同一套 Python 实现。
`sdk/lookahead.py` 提供摸牌后弃牌的爆头进张与听牌路线价值；C 路径整段枚举并跳过不可能听牌的目的地，Python 路径保留相同定义。固定规则包采用这些函数时派生新包，旧冻结源码不改写。`tests/test_shape_fast.py` 用参考实现核对两者一致，
修改任一侧都要重跑该测试。冻结运行时按整棵包树复制，指纹覆盖同一集合。

编译扩展、人工项目的 `models/`、`releases/` 与 `runs/` 都是本机生成物，不提交 Git。人工项目测试在临时目录真实训练、构建并校验发布包，不依赖仓库内预置的模型或旧发布版本。

操作系统适配修改先运行 `tests/test_sandbox.py`、`tests/test_service_host.py` 和 `tests/test_candidate_runtime.py`：验证只读/私有文件、断网、禁止派生进程但允许线程、沙箱子进程 RSS 限制，以及服务真实启停和重启。systemd 用户管理器不可用时真实服务测试跳过；不能把跳过当成后台服务验收通过。macOS 分支有命令与配置回归，发布双平台能力仍需在 Mac 实机运行测试。

引擎与平台事实以官方实战快照、完整牌谱及对应回归为准。`tests/fixtures/official-v25-*` 保留基础协议样本，较新的规则反例与控制见同目录 `official-v34-*`、`official-v35-*`。抓打圈的豁免、吃碰后续飘与断链另有合成契约回归。四白板的 `docs/sources/fan-calc-four-whites.json` 仅为闭手试算控制；实战计番兼容分支及正向控制见[平台规则](platform.md)。理论规则、闭手试算和合成用例均不能覆盖或替代真实对局回执。

`tests/fixtures/official-v34-parity.json` 保存官方完成牌谱的整局片段，覆盖碰后补杠和圈外吃后财飘；玩家身份与房间标识已移除，局内记录保持原貌。`official-v34-post-peng-kong-snapshot.json` 保存当前服务器接受补杠时的玩家快照。`tests/test_engine_parity.py` 将这些记录送入真实引擎，并覆盖底分传递、暗杠隐私、副露来源与事件特征一致性。

`platform/replay_audit.py` 直接用 `HandState.step` 重放官方动作和摸牌顺序，检查合法性、番数、逐家分数、连庄、牌张守恒和官方出牌超时的代打选择；不维护第二份吃碰杠状态机。缺少结算分数不能报告通过。补齐本地顺序 pass 只用于推进响应窗口，网络时序与跨窗请求仍由完整平台日志审计。

改引擎或规则后运行 `scripts/audit_archive_parity.py`：它把 `.local/competition/` 下全部存档房间的官方牌谱用当前引擎重放审计，并在我方每个决策点比对平台快照与本地状态（手牌、弃牌、副露、摸牌、抓打圈、飘链），再按下一个事件给被拒动作归因；有任何不一致时退出码为 1。它只覆盖存档对局实际走过的路径和出现过的房间设置，没走过的动作和其他设置仍靠官方样本回归。

`official-v35-catch-owner-draw.json` 保留一局官方牌谱及打财神者再次摸牌后的真实决策快照。此时线上仍保留抓打圈及豁免座位，直到本人出非财神牌才解除；`test_engine_parity.py` 验证本地观察一致。这类观察字段差异可能不改变已执行动作和最终分数，验收须同时比较线上决策快照，不能仅检查整局重放。

已执行动作的重放只验证已走过的路径，不能证明两端合法动作集合相同。升级验收还应保存被拒动作的快照和请求时间，并在官方超时代打发生前比较本地 `timeout_action()`；确认服务器拒绝本地合法动作时，要排除截止时间、已被消费的响应窗口及其他玩家抢先声明等因素。`tests/test_youcai_parity.py` 保留“有财必拷响”杠后拒胡快照和完整超时局，核对 Python/C、平台投影与假想模拟，并验证审计器能检出本地额外允许胡牌的情况。指南冲突与证据边界见[平台规则](platform.md)。假想模拟会将已知刚摸牌恢复到手牌末尾，用于一致的超时代打；没有刚摸牌时，排序观察不能恢复完整历史插入顺序。

`HandState` 按座位增量维护策略可见事件视图（`public_views()`），结算循环把视图随历史一起传给下一局，观察构造不再逐次重投影整场历史；`tests/test_engine_views.py` 用固定种子对局逐步比对增量视图与旧的两遍脱敏投影完全一致，改动事件投影必须重跑。`domain/observations.py` 统一策略可见事件与吃后财飘判断；本地与平台共用该语义。公开观察协议或特征语义改变时使用新实验与新冻结运行时，旧候选继续携带自身实现。`public-state-action-rank-v3` 对应统一后的事件特征，不把旧版事件编码的模型直接用于新语义。

`tests/test_kong_decisions.py` 单独覆盖 v32/v33：暗杠/补杠后听牌态从非爆头变为爆头、明杠保持等价判定、普通杠开不误加爆头倍率，以及三种杠补牌后的主动胡、弃胡续飘/续杠和超时胡。用例检查牌张守恒与 `OfficialAdapter` 的合法动作、胡牌倍率投影；它们是按指南构造的合成场景，不是官方抓取样本。无状态番型接口目前只校准闭手牌，不能替代这些状态转换回归。

研究测试覆盖分支隔离、完整配对、幂等入账、最好版本保留、跨实验历史、部分分支迭代、全体对比、评估分片切分与合并、前 K 自动更新与轮次终点、迭代后与立即停止安排，以及训练到限冻结检查点、迭代开始必须先写研究判断、候选创建幂等、每次迭代换会话并交接、提交前运行自检、启发式与搜索家族不提供训练槽。策略 SDK 测试用独立牌形枚举核对结构距离，检查假想世界与合法观察一致，并验证 PPO 真实训练更新；模型启动测试核对上下文与自动压缩阈值确实透传到隔离 worker。

修改晋级口径时核对 `tests/test_selection.py` 与研究集成测试，覆盖置信下界、完整同量、四座配对对齐和执行错误阻断。修改 Agent 读接口时运行 `tests/test_agent_context.py`，覆盖累计结论无损交接、状态回执丢失与重连、历史查询隔离和分页、显式详情重读及长源码完整遍历；`tests/test_research_conclusions.py` 核对证据与修订，`tests/test_research.py` 核对同次会话续接和原迭代额度。

模型启动必须同时处理 model、provider、reasoning effort 和 catalog。只更换模型名称不足以切换提供方。自定义模型采用直接暴露研究工具的目录副本，原用户配置不改写。新增模型或修改启动方式需真实调用验证，命令构造测试不能证明网关可用。

端到端验收使用隔离的策略注册表和实验，记录在 `projects/agent-research/.local/research/reports/`。按实际算法测量并发和训练资源，再调整固定评估场数；不得将简单规则的吞吐当作搜索或 PPO 的吞吐。网页只读，需验证主机状态与排空提示、分支三态与归档卡片、阶段条与卡片动态、卡片上的现役做法与研究判断、详情里的判断历史、方案分支卡片与尝试详情、置顶的排行榜与历史排名，以及页面上没有修改入口。

完整线上阶段身份映射、最高并发与响应窗口仍需专项平台实战验证。本地通过不等于正式赛事已验证。

## 领域与决策

[术语表](../../CONTEXT.md) 定义研究方向、组合、分支、候选、派生父代与现役代表。架构取舍见 [ADR](../adr/)，现行操作见[研究手册](strategy-agent.md)。新的通用说明更新现有手册；具体成绩、耗时和核实日期保存在研究报告中。
