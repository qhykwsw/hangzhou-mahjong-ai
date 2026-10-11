# 开发索引

## 模块与职责

| 模块 | 入口与职责 |
| --- | --- |
| `domain/` | 牌、规则、观察、动作、真实对局与完整结算 |
| `sdk/` | 精确牌形、公开轨迹、假想状态生成与模拟 |
| `strategies/` | 策略规格（`PolicySpec`）与可运行的规则、收益预测、搜索基线 |
| `evaluation/` | 四座赛程、分组不确定度与成绩 |
| `execution/` | 对局任务（`jobs.py`）、候选进程与沙箱、资源与进程调度；`remote.py` 远程主机的同步、批量派发、探测与大乱斗抢占训练，`model_seat.py`/`model_service.py` 模型座位（可带一个按阶段比分切换的第二个模型，目前只在 `harness.stage` 的模拟阶段里用；平台不下发阶段累计比分，切换在正式赛不会触发。`stage_view` 决定模型看到的阶段：`platform` 只留平台客户端给的阶段类型与本场比分、局数，`blind` 再把阶段类型当作海选）与本机批量推理服务 |
| `persistence/` | 原子写入、文件锁、运行时指纹和冻结工件 |
| `platform/` | 官方协议客户端、自由对战与牌谱审计 |
| `interfaces/` | CLI、迭代线只读网页与服务托管 |

操作系统适配集中在 `interfaces/service_host.py`（launchd/systemd 与浏览器）、`execution/candidate_runtime.py` 的沙箱入口、`execution/linux_sandbox.py`（Bubblewrap/seccomp）和 `execution/process_usage.py`（RSS）。规则与评估口径不因操作系统改变。Linux/WSL2 的候选限制与 macOS 相同，不提供无沙箱降级。隔离只有一种显式例外：对局任务的 `resources.isolation = "none"` 让任务拥有者信任的代码包（例如维护者自己冻结的固定对手池）直接在对局进程里运行（`candidate_runtime.InlinePolicy`），省去每次决策的序列化与进程往返；时限改为事后判定，超时仍按代打计。这不是沙箱不可用时的回退。同一任务的 `resources.tables` 让一个对局进程同时打几桌（一桌一个线程），模型座位等推理服务时其他桌继续。

网页入口为 `mahjong`（`start`、`status`、`stop`），独立引擎、训练、吞吐测试与平台命令为 `mahjong-ai`。以 `--help` 为准。

RL 线的训练框架 `projects/personal/selfplay/guided/` 直接驱动 `HandState`，不经过候选运行时；教师以冻结规则包的运行时参与对局，远端训练由冻结的 `guided/trial.py` 执行，见[训练手册](../../projects/personal/selfplay/guided/README.md)与 [ADR-0011](../adr/0011-rule-guided-policy-learning.md)。

迭代线（规则线与 RL 线）由 `projects/personal/firstprinciples/harness/` 负责固定对手池评估、封存、尝试循环与主机分配，见[项目说明](../../projects/personal/firstprinciples/README.md)与 [ADR-0016](../adr/0016-maintainer-iteration-lines.md)。模型库与运行时库沿用已退役自动研究的路径 `projects/agent-research/.local/research/{models,runtimes}`，见 [ADR-0019](../adr/0019-retire-automated-research.md)。

独立 CLI 的 `rule`、`rl` 是基础策略入口；CLI 的基础吞吐测试不替代在冻结对手池上的实际评估测量。

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

编译扩展、模型与训练运行目录都是本机生成物，不提交 Git。

操作系统适配修改先运行 `tests/test_sandbox.py`、`tests/test_service_host.py` 和 `tests/test_candidate_runtime.py`：验证只读/私有文件、断网、禁止派生进程但允许线程、沙箱子进程 RSS 限制，以及服务真实启停和重启。systemd 用户管理器不可用时真实服务测试跳过；不能把跳过当成后台服务验收通过。macOS 分支有命令与配置回归，发布双平台能力仍需在 Mac 实机运行测试。

引擎与平台事实以官方实战快照、完整牌谱及对应回归为准。`tests/fixtures/official-v25-*` 保留基础协议样本，较新的规则反例与控制见同目录 `official-v34-*`、`official-v35-*`。抓打圈的豁免、吃碰后续飘与断链另有合成契约回归。四白板的 `docs/sources/fan-calc-four-whites.json` 仅为闭手试算控制；实战计番兼容分支及正向控制见[平台规则](platform.md)。理论规则、闭手试算和合成用例均不能覆盖或替代真实对局回执。

`tests/fixtures/official-v34-parity.json` 保存官方完成牌谱的整局片段，覆盖碰后补杠和圈外吃后财飘；玩家身份与房间标识已移除，局内记录保持原貌。`official-v34-post-peng-kong-snapshot.json` 保存当前服务器接受补杠时的玩家快照。`tests/test_engine_parity.py` 将这些记录送入真实引擎，并覆盖底分传递、暗杠隐私、副露来源与事件特征一致性。

`platform/replay_audit.py` 直接用 `HandState.step` 重放官方动作和摸牌顺序，检查合法性、番数、逐家分数、连庄、牌张守恒和官方出牌超时的代打选择；不维护第二份吃碰杠状态机。缺少结算分数不能报告通过。补齐本地顺序 pass 只用于推进响应窗口，网络时序与跨窗请求仍由完整平台日志审计。

改引擎或规则后运行 `scripts/audit_archive_parity.py`：它把 `.local/competition/` 下全部存档房间的官方牌谱用当前引擎重放审计，并在我方每个决策点比对平台快照与本地状态（手牌、弃牌、副露、摸牌、抓打圈、飘链），再按下一个事件给被拒动作归因；有任何不一致时退出码为 1。它只覆盖存档对局实际走过的路径和出现过的房间设置，没走过的动作和其他设置仍靠官方样本回归。

`official-v35-catch-owner-draw.json` 保留一局官方牌谱及打财神者再次摸牌后的真实决策快照。此时线上仍保留抓打圈及豁免座位，直到本人出非财神牌才解除；`test_engine_parity.py` 验证本地观察一致。这类观察字段差异可能不改变已执行动作和最终分数，验收须同时比较线上决策快照，不能仅检查整局重放。

已执行动作的重放只验证已走过的路径，不能证明两端合法动作集合相同。升级验收还应保存被拒动作的快照和请求时间，并在官方超时代打发生前比较本地 `timeout_action()`；确认服务器拒绝本地合法动作时，要排除截止时间、已被消费的响应窗口及其他玩家抢先声明等因素。`tests/test_youcai_parity.py` 保留“有财必拷响”杠后拒胡快照和完整超时局，核对 Python/C、平台投影与假想模拟，并验证审计器能检出本地额外允许胡牌的情况。指南冲突与证据边界见[平台规则](platform.md)。假想模拟会将已知刚摸牌恢复到手牌末尾，用于一致的超时代打；没有刚摸牌时，排序观察不能恢复完整历史插入顺序。

`HandState` 按座位增量维护策略可见事件视图（`public_views()`），结算循环把视图随历史一起传给下一局，观察构造不再逐次重投影整场历史；`tests/test_engine_views.py` 用固定种子对局逐步比对增量视图与旧的两遍脱敏投影完全一致，改动事件投影必须重跑。`domain/observations.py` 统一策略可见事件与吃后财飘判断；本地与平台共用该语义。公开观察协议或特征语义改变时使用新的冻结运行时，已冻结的对手池成员继续携带自身实现。`public-state-action-rank-v3` 对应统一后的事件特征，不把旧版事件编码的模型直接用于新语义。

`tests/test_kong_decisions.py` 单独覆盖 v32/v33：暗杠/补杠后听牌态从非爆头变为爆头、明杠保持等价判定、普通杠开不误加爆头倍率，以及三种杠补牌后的主动胡、弃胡续飘/续杠和超时胡。用例检查牌张守恒与 `OfficialAdapter` 的合法动作、胡牌倍率投影；它们是按指南构造的合成场景，不是官方抓取样本。无状态番型接口目前只校准闭手牌，不能替代这些状态转换回归。

## 网页与服务

`interfaces/app.py` 只读展示迭代线（`/api/lines`：各对手池的尝试、代理桌与线上自由对战），页面上没有修改入口；修改后在浏览器核对图表、对手池切换与表格。服务由 macOS 用户域 launchd 或 Linux/WSL2 的 systemd 用户服务管理（`interfaces/service_host.py`），不注册登录启动项；状态与日志在仓库的 `.local/web/`。优先端口 8765，冲突时选择可用端口；Tailscale 已连接时同时监听回环与 Tailscale IPv4 网卡，连接状态改变后重启服务。WSL 需先在 `/etc/wsl.conf` 的 `[boot]` 节启用 `systemd=true`，`systemctl --user show-environment` 应能正常执行。

完整线上阶段身份映射、最高并发与响应窗口仍需专项平台实战验证。本地通过不等于正式赛事已验证。

## 领域与决策

[术语表](../../GLOSSARY.md) 定义迭代线、尝试、固定对手池与评测口径。架构取舍见 [ADR](../adr/)，迭代线的现行操作见[项目说明](../../projects/personal/firstprinciples/README.md)。新的通用说明更新现有手册；具体成绩、耗时和核实日期保存在迭代线的记录里。
