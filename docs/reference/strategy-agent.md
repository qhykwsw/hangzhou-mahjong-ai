# 策略研究与 Agent 接口

## 长期策略与实验

策略按“研究方向 → 策略组合 → 独立分支 → 候选版本”组织。分支持续积累历史、最好版本、研究会话与迭代次数；实验记录本次对象、参数、执行任务和成绩，不拥有策略身份。新增实验不会清空已有分支。术语见 [CONTEXT.md](../../CONTEXT.md)，架构取舍见 [ADR-0006](../adr/0006-decouple-strategy-history-from-experiments.md)。

研究分支分两种：策略分支的候选是规则代码；方案分支的候选是训练方案，由它从固定起点训练出的模型参赛。两种分支共用现役代表、晋级、评估对手、共同大乱斗与排行榜，只在候选如何产出、在哪台主机执行和评估场数上不同，见 [ADR-0013](../adr/0013-unified-research-branches-and-common-brawl.md)。分支阵容以项目 [research.toml](../../projects/agent-research/research.toml) 为准；框架在未提供项目配置时采用的备用阵容见 [config.py](../../src/mahjong_ai/research/config.py)。

研究模型是分支属性，可以随时在配置里切换，从该分支的下一次迭代生效并保留变更历史。分支源码、模型和会话彼此独立；其他分支只共享公开成绩。

## 控制方式与连续迭代

```bash
.venv/bin/mahjong start
.venv/bin/mahjong status
.venv/bin/mahjong create [--candidate-runtime PATH] [--opponents ID ...]
.venv/bin/mahjong control pause|resume|pause_after|stop_now|clear_stop|close
.venv/bin/mahjong stop
```

网页是只读的单页：从上到下依次是实验状态、主机状态（在线、离线、排空中、大乱斗中）、排行榜和分支卡片；方案分支的卡片与规则分支同样大小，显示训练主机、模型、当前方案判断、训练阶段和最近尝试，点开可看全部尝试与方案累计结论。页面不提供任何修改入口。分支与实验只有两个控制面，见 [ADR-0014](../adr/0014-control-plane-resilience-and-live-branches.md)：

- `research.toml` 是分支阵容、分支状态、研究模型、固定参赛者与主机分工的唯一来源。后台检测到文件变化就应用到运行中的实验。
- `mahjong create` 与 `mahjong control` 负责开实验、暂停、继续、停止安排与结束实验。

分支有三种状态，由 `[modes]` 按分支 ID 指定。已配置但未列出的分支默认迭代中；不再配置的已注册分支自动归档：

| 状态 | 产生候选 | 现役代表参加大乱斗、可成为评估对手 |
| --- | --- | --- |
| `iterating`（迭代中） | 是 | 是 |
| `competing`（仅参赛） | 否 | 是 |
| `archived`（归档） | 否 | 否 |

新组合或新分支写进配置后立即初始化并开始迭代，初始版本从下一次大乱斗起参赛。分支离开迭代中时，在途的那次迭代照常跑完。参赛的总是分支当前的现役代表。新进入大乱斗的参赛者（新分支、恢复参赛的分支、新增的 `guests`/`anchors`）各计一次待定晋级。运行中改动评估条件或计分相关的键（评估场数与段数、置信水平、对局设置、动作时限、资源、训练额度、已有组合的参数或场数）不会生效，页面显示被拒绝的键；这些键只能随新实验改变。

每条迭代中的分支完成一次迭代就开始下一次，不等其他分支，直到暂停：

1. 迭代开始时冻结评估对手：最近一次共同大乱斗排名前三的非锚点版本，可以包含本分支自己的版本，也可以是方案分支的模型。实验不带对手创建时，以开场大乱斗产生第一批评估对手。策略分支的每次迭代先整理尚未复盘的累计研究结论，再由 Agent 写下研究判断（对现役做法与轨迹的判断、本次方向、原因），然后派生草稿并实现；没有判断就不能创建或提交候选。方案分支的候选由训练方案服务设计并训练，交接回来的模型进入同样的评估。
2. 每次分支迭代默认提交一个候选，上限可配置为一至四个。Agent 提示词和提交校验都使用该迭代冻结的 `candidate_limit`；超限或重复的候选在自检前就被拒绝，未提交的草稿不参与评估。历史草稿须作为父代派生出本次的新候选后再提交，以保留原迭代的证据归属。
3. 全部准备成功的候选与原代表面对相同的对手安排、牌局和座位，做分段配对评估，每段每边场数加倍，累计计数。前几段用更严格的 `early_confidence`：区间下界 > 0 提前晋级，上界 < 0 提前淘汰。最后一段用 `final_confidence`，下界 > 0 才晋级。晋级还要求双方完整同量，且没有报错、超时或代打；多个候选达标时按场均桌面分选最高者，其余情况保留原代表。
4. 选优主口径是完整结算的场均桌面分，与线上非决赛第一排序键（总分）一致。场均分相同时依次按场均名次分（每场 +3 / +1 / −1 / −3，同分取平均）、白板获取数和稳定身份排序；决赛按总分并依规则加赛。名次效用（4−名次）与逐局读数只作对照，不参与选优。
5. 单次分支迭代到达 `iteration_seconds` 上限时，执行器把最近一次已提交的训练检查点冻结成候选版本，照常参加配对选优；只有完全没有检查点时才记为执行失败。

累计 `brawl_promotions` 次晋级后发起共同大乱斗；有晋级在等、且距上次大乱斗超过 `brawl_fallback_seconds` 时也会发起。参赛的是所有未归档分支的现役代表，加上 `anchors`、`guests` 与常驻对手，每个参赛者出场 `brawl_appearances` 次。赛程出场均衡、四座轮换，尽量减少重复的对手组合。大乱斗完整结束后固定排行榜，之后开始的迭代以新的前三名为评估对手。参照锚点只参与排名展示，不成为评估对手。实际参赛者不足四个时，由三种启动启发式补位。

新实验冻结的运行时发生变化时，生成对应的新启动策略版本。旧启动版本的源码、路径和运行时都保留，不迁移或改写已冻结的分支候选；`--candidate-runtime` 可以让延续过来的旧版本统一运行在新冻结的运行时上。

## 主机分工与离线

Mac 是唯一的控制面，运行研究模型会话、策略分支的评估和调度。`hosts` 中配置的远程主机按角色承担其余工作：`brawl_host` 执行共同大乱斗；方案分支的 `evaluation_host` 负责它的训练与评估；`overflow_hosts` 空闲时（没有训练进程）承接本机排不下的规则评估。大乱斗派到某台主机时，先暂停这台主机上的训练进程（进程组 `SIGSTOP`，显存保留），这台主机上的大乱斗全部结束或被取消后再恢复；被暂停的时长不计入训练时间。

任何远程主机掉线都不影响本机的规则迭代。某台主机连续 10 分钟探测失败即视为离线：

- **方案分支所在主机离线**：该分支不开新迭代，也不预先设计方案；现役模型照常参赛、照常当评估对手，这不依赖那台主机。在途的训练或评估等主机回来接着做。
- **大乱斗主机离线**：溢出到它的规则评估迁回本机。大乱斗到期时进入大乱斗排空：本机不再开新迭代，等本机在跑的迭代（含评估）全部完成后，在本机跑大乱斗，跑完恢复迭代，场数不打折。排空期间主机恢复、且大乱斗还没开跑，就改回原主机。大乱斗跑到一半主机离线时，已取回的分片保留，缺的分片按本机的分片大小拆开补跑。
- **主机故障**：远程主机重启导致工作丢失，或主机不可达导致启动失败，都记为主机故障。它不消耗迭代次数、不计入失败，工作在主机恢复后重做；方案分支的训练遇到主机故障，同一个训练任务重新开始。

模型座位在 Mac 上用 MPS fp16 推理，在远程主机上用 CUDA fp32。接近平局的动作选择可能不同，视为同一评测条件。一类新模型（例如 RL 训练后不再像模仿模型那样笃定的模型）首次参评前，取几千个真实对局决策点，比较两种推理的选中动作；不一致率明显时，Mac 改用 fp32。

## 成绩、历史与恢复

实验与进度、全体排行榜是独立页面。配对选优成绩附带实验与对手条件，不能直接混成全局实力排名。排行榜绑定实际参赛版本、研究模型与推理等级，支持查看历史对比记录；大乱斗尚未结束时标为临时排名。排名表达有限样本的表现，不计算 Elo。

配对选优记录和排行榜同时提供标准误差与评估组数。主口径按四场轮换组（同副牌、同对手、四座轮换）先聚合出场均桌面分，再以组均值的样本标准差除以组数平方根估计标准误差，避免把同一组内的四场当作独立样本；不足两组时不估计。场均名次分、名次效用与逐局读数作为辅助一并展示，不参与排序。候选与基线共用同一份赛程，页面额外给出场口径的逐块配对差与标准误，用来判断本次选优是否落在噪声以内。晋级使用配对差的双侧 Student-t 区间：`delta ± t(1−(1−c)/2, groups−1) × standard_error`，`c` 取该段的置信水平。区间写入 `paired.confidence_lower` / `confidence_upper`，`promotion` 保存达标结果和原因；网页同步显示。少于两组或配对不完整时不晋级。单版本标准误及逐局读数仍仅供参考，大乱斗排名和前三名仍按均值。区间针对本次冻结评估对手的比较，不代表跨迭代的同时置信保证。公式依据 [NIST 配对均值差区间](https://itl.nist.gov/div898/handbook/prc/section3/prc312.htm)，t 分位数由 SciPy 计算，版本随运行时环境记录。

一次只运行一个实验，内部并发多个分支。暂停只停止开新工作：本机在途的任务跑完才算暂停完成；远程主机上的评估、大乱斗和方案训练不打断，继续后去主机上认领结果。服务重启后实验保持暂停，须用 `mahjong control resume` 继续。重启时本机被打断的任务从头重跑，远程任务按启动记录重新认领。停止安排有两种：`pause_after` 在本次迭代结束后停，各分支跑完当前这一次就收工；`stop_now` 立即结束当前迭代，取消在途任务，删除未完成的迭代记录及其草稿候选，分支回到就绪。正在进行的大乱斗都会先跑完。安排不会被暂停或服务重启清掉，`clear_stop` 才撤销，触发一次后自动清空。改框架协议时先暂停并结束本次实验，再开新实验；已完成的候选与分支历史保留。

候选可以从本分支任意历史版本派生，但选优始终与本次迭代开始时的代表比较。有效评估、实际提升和执行失败分别统计。失败尝试消耗一次分支迭代额度；暂停不消耗额外额度，关闭未完成实验会结束当前尝试。

## 配置与资源

配置合并顺序为框架默认值、项目 [research.toml](../../projects/agent-research/research.toml)、本机 `projects/agent-research/.local/machine.toml`。`machine.toml` 不进 Git，只接受 `agent_concurrency`、`training_workers`、`match_workers` 三个并发键，用于不同核数的机器各自设定并发；评估条件不能按机器改变，写入其他键会拒绝启动。权威默认值和校验见 [config.py](../../src/mahjong_ai/research/config.py)。已有实验按自身冻结的配置运行，不能用项目文件的现值推断历史实验的参数。

| 设置 | 生效方式 |
| --- | --- |
| 并发、分支阵容与 `modes`、研究模型、`anchors`/`guests`、`hosts`/`brawl_host`/`overflow_hosts`、方案分支的 `evaluation_host` | 写入 `research.toml` 后自动应用到运行中的实验 |
| 评估场数与段数、置信水平、大乱斗场数与触发条件、对局设置、动作时限、资源、训练额度、已有组合的参数与场数 | 随新实验冻结；运行中修改会被拒绝并提示 |
| 评估对手具体版本 | 每次迭代开始时取最近一次大乱斗的前三名 |
| 框架、规则与动作窗口 | 新实验；旧版本仍使用其冻结运行时，或经 `--candidate-runtime` 统一到新运行时 |

`training_caps` 按组合家族限制训练额度：启发式与搜索家族为 0，`strategy.train` 会被运行检查拒绝，`research_submit_batch` 也不接受 `training_games`；规则策略的实测比较由配对选优完成。value/PPO 保留训练槽，上限为 256 场；需要更大预算时改实验级配置，超限申请在运行检查之前拒绝。评估场数与段数在实验开始时固定；分段加场按冻结规则进行，不依据候选表现另行追加。

配对选优按完整四座轮换组分片，每个参评版本分别产生任务。`evaluation_shards` 是期望分片数（默认 4），实际值还会按本批参评版本数（含原代表）与比赛 worker 数适配，并且不超过轮换组数；比赛 worker 数取实验的实时值，热改后新建的评测立即按新值分片，单个分支评测时也能用满 worker；权威算法为 `research/service.py` 的 `evaluation_shard_count`。对手抽样、牌局与策略种子都由 `block` 决定，分片只改变任务切分，不增加样本、不改变任何一局；读数在分片到齐后合成，标准误仍按全部轮换组计算。实际并行度由可用 worker 和已就绪分片共同限制，不等于候选个数。

Agent 在下一次迭代提示中直接收到本分支上次的实际执行反馈，`research_history(kind="iterations")` 的各次迭代也提供 `execution_feedback`：申请训练场数、实际完成场数、参数更新数、研究 Agent 耗时、训练与准备耗时和单次执行上限。时间单位为秒，训练与准备耗时包括依赖准备、采集对局、训练和校验，排除排队与选优比赛；缺少记录的数值为 `null`。训练被时限打断而按检查点冻结的候选，`status` 为 `partial`、`interrupted` 说明原因、`completed_games` 为冻结时的实际场数。Agent 根据实际耗时、成绩及代码和对手变化自主调整后续训练场数，框架不自动放大或锁定场数。

单次分支迭代的执行预算由 `iteration_seconds` 决定（当前项目 600 秒，框架默认 3600 秒），累计模型思考、依赖准备与训练的实际经过时间，不计排队、暂停和框架评估。提示开头写明本次总预算与剩余秒数，并要求先写计划、尽早提交最小可行改动、有余时再派生改进；研究 MCP 的每个工具返回都附带 `budget`（`remaining_seconds`、`limit_seconds`、`advice`），剩余 60%、35%、15% 或 90 秒时建议依次收紧到立即提交，续跑调用同样带剩余时间。到限时把最近一次已提交的训练检查点冻结成候选版本继续评估，该候选记为 `partial`；没有检查点的候选标为超时，全部失败则保留原代表。Agent 提交后立即结束回复、释放槽位，不等待评估；重复提交同一批次返回已接收结果。会话与交接见[下文](#会话累计结论与按需读取)。

执行器面向本机 CPU，macOS 使用 sandbox-exec，Linux/WSL2 使用 Bubblewrap 与 libseccomp。两者都限制文件访问、网络与候选派生进程，同时允许算法计算线程；沙箱不可用时拒绝执行，不回退到无隔离运行。依赖下载单独允许网络，创建依赖环境单独允许子进程，不把这些权限交给候选。每个候选受 RSS 内存上限和计算线程配置约束，Linux 统计沙箱启动器及其子进程的合计 RSS，超限明确报错，不替换设备或静默缩小算法。默认弃牌三秒、响应一秒，超时或非法动作由环境执行默认动作，完整结果照常计分并保留错误与回退统计。

模型配置从 `CODEX_HOME` 读取，未设置时使用 `~/.codex`。需要 `config.toml`；研究注册表里的 `deepseek` 与 `qwen` 分别叠加 `netease-deepseek-flash.config.toml`、`netease-qwen-flash.config.toml`。本机提供方配置和凭据不随仓库分发，变更后须实际调用验证。研究工具直接暴露，不依赖第三方模型工具发现能力。

模型网关会按账号限流，因此 Agent 任务每隔 5 秒放出一个，不把并发上限一次性压上去；网关返回 429、5xx 等瞬时错误时执行器按 30、90、180 秒退避重试，重试属于同一次任务，不消耗分支迭代配额。真正的失败（非瞬时错误或重试用尽）才把该次迭代记为失败。

研究 Agent 以 `--ignore-user-config` 启动，因此只继承显式转发的模型设置：`model`、`model_provider`、`model_reasoning_effort`、`model_catalog_json`，以及全局 `model_context_window` 与 `model_auto_compact_token_limit`（同时写进该次任务副本的 model catalog）。超过自动压缩阈值后由客户端摘要历史，作为长上下文的兜底。

## 服务与运行时

服务由 macOS 用户域 launchd 或 Linux/WSL2 的 systemd 用户服务管理，不注册登录启动项。macOS 保持 `ProcessType=Interactive`，避免候选进程被后台 QoS 降级；Linux 使用临时服务单元，停止时先通知主进程收尾，再清理剩余进程。改动服务安装方式后要重新核对比赛吞吐。优先端口 8765，冲突时选择可用端口；日志位于 `projects/agent-research/.local/service/service.log`，服务环境文件权限为 `0600`。`--no-open` 或 SSH 会话仅输出地址；WSL 优先通过 `wslview` 打开 Windows 默认浏览器，打开失败不影响服务启动。启动时若 WSL/Linux 内的 Tailscale 已连接，则同时监听回环与该 Tailscale IPv4 网卡；不把 Windows 宿主的 Tailscale 地址当成本机网卡。连接状态改变后重启服务。地址以 `status` 的 `url`、`tailscale_url` 为准。

更新框架时，先让在途研究到达可停止的边界，再从经过验证的目标代码启动后台；新协议通过新实验生效。工作区源码、后台进程、实验运行时和候选运行时是不同对象：仅修改源码不会更新已运行的后台，仅重启后台也不会升级旧实验或旧候选。核验时分别确认服务实际加载的路径、实验冻结指纹和候选引用。

服务启动不自动继续研究。未完成的实验须用 `mahjong control resume` 继续；已结束的实验保留结果，只有新建实验才会再安排研究。

### WSL2 环境

在 Ubuntu 内安装 `bubblewrap`、`libseccomp2`、`libgomp1` 和 `build-essential`，使用 Python 3.12+ 创建项目自己的 `.venv`。若默认 Python 版本较低，可使用 `uv venv --python 3.12 --seed .venv`。不要复制 Mac 的 `.venv`、wheel 缓存或 C 扩展；安装项目时在本机重新构建。

`systemctl --user show-environment` 必须能正常执行。若 WSL 尚未启用 systemd，在 `/etc/wsl.conf` 的 `[boot]` 节设置 `systemd=true`，再从 Windows 执行 `wsl --shutdown` 后重新进入发行版，见[微软说明](https://learn.microsoft.com/en-us/windows/wsl/systemd)。服务不依赖启动它的终端，但 systemd 服务本身不保证 WSL 实例持续存活；长时间研究需要保持实例运行。Windows 重启或关闭 WSL 后，重新运行 `mahjong start` 并用 `mahjong control resume` 继续，不自动恢复实验执行。

## 会话、累计结论与按需读取

### 会话边界

| 场景 | 会话与上下文 |
| --- | --- |
| 开始一次新分支迭代 | 开新会话，默认携带本分支全部累计结论、现役代表、排行榜和上次判断与执行反馈 |
| 暂停后恢复同一次迭代 | 复用原会话，只发送恢复指引，查询实际状态，不重发完整交接 |
| 模型正常结束回复但未提交候选 | 在原会话中最多续接两次，沿用判断和草稿，已耗时计入原迭代上限 |

续接要求注册表确认仍未提交、存在会话标识、无执行错误或中断，且至少剩余 30 秒；用尽后仍未提交则记为执行失败。已提交的任务不因停止标记延迟而重跑。会话标识 `thread_id` 留在迭代记录中用于审计。

### 开场交接与复盘

累计研究结论已经是完整迭代记录的压缩展示：保留全部假设、结论、适用条件和失败原因，跨实验累积，不限于最近几次；开场不带完整旧聊天或逐场回执。重复的玩法、对手池和资源条件提取到 `research_conditions`，每条证据用 `condition_id` 引用，可无损还原，结论正文没有再摘要或缩短。交接体积仍会随记录增长。

[conclusions.py](../../src/mahjong_ai/research/conclusions.py) 根据已完成尝试投影记录。Agent 解释假设与证据，框架附加候选、原代表、派生父代、对手池、玩法、样本量、配对差和执行证据。结论分为已验证（`verified`）、已否定（`refuted`）、证据不足（`inconclusive`）、未有效验证（`untested`）；只有双方完整评估且无错误、超时或代打，才允许标记已验证或已否定。未提升不自动否定方向，仍需结合配对误差与适用条件判断；候选是否晋级由框架独立判定。

失败原因区分实现、资源限制、基础设施和原因未明。自检错误在修复成功后仍留档，没有提交候选的失败尝试也进入记录，不能把实现出错解释成方向无效。历史证据保留其原冻结运行时与评估条件。

每次迭代先整理交接中 `assessment` 为空的记录，再声明计划。`pending_conclusion_ids` 用于核对尚未处理的记录；`research_record_conclusion` 可修订判断，旧修订保留，同内容重复提交幂等。最后一次迭代的证据立即保留为待复盘，后续新研究再整理，不额外启动总结 Agent。

### 状态与详情

Agent 的读接口由 [agent_context.py](../../src/mahjong_ai/research/agent_context.py) 提供，不直接返回网页的全量快照。Agent 复用开场交接，只针对当前假设的知识缺口读取源码或历史。

| 需要什么 | 接口与返回范围 |
| --- | --- |
| 当前任务、代表、草稿、额度、待复盘 ID | `research_status(known_revision)`；不重复结论正文、已写计划或历史成绩 |
| 检索历史候选、迭代或结论 | `research_history(kind, query, offset, limit)`；只返回命中页，以 `next_offset` 继续 |
| 一次迭代的判断、聚合成绩和执行反馈 | `research_history(kind="iterations", query="完整迭代 ID")`；保留配对统计，省略逐组数组 |
| 单条结论的完整证据与全部修订 | `research_read_conclusion(record_id)`；显式重复读取仍返回详情 |
| 候选文件与公开 SDK 源码 | `research_read_file` / `research_read_source`；按字符分页，以 `next_offset` 继续 |

状态首次查询省略 `known_revision`，返回 `full` 和 `revision`。后续传入**实际收到的版本号**，未变化返回 `unchanged`；有变化返回 `delta`，用 `changes` 替换对应顶层字段并删除 `removed` 中的字段。回执丢失时仍用旧版本查询，不会漏掉变化；重连或缓存淘汰导致版本不可用时自动返回 `full`。上下文缺失时也可省略版本号，重新读取完整当前状态。

历史页按完整记录控制体积，不截断单条结论。源码的 `offset`、`limit` 按字符计数，每页还受 JSON 转义后的 UTF-8 字节预算限制；沿 `next_offset` 读取到 `null` 才是全文，跨页用 `revision` 核对文件是否变化，长行与嵌入源码也可完整遍历。完整原始比赛回执和网页历史保持不变。

## 候选契约与训练

权威契约为 [CONTRACT](../../src/mahjong_ai/research/contract.py) 与 [MCP schema](../../src/mahjong_ai/interfaces/mcp.py)。公开源码名称由 `research_read_source.name` 枚举给出，`strategies/common.py`（`RulePolicy`、`risk_preference` 等）使用 `policy` 读取，不接受 `common`、文件路径或组合 ID。`strategy.py` 声明 `COMBINATION`，提供 `build(model_dir)`，返回实现 `reset(seed)`、`act(observation)`、`finish_round(result, seat)` 的对象。动作必须来自 `observation.legal_actions`。

研究契约直接提供杭州麻将的关键规则：仅自摸、没有点炮或抢杠胡，白板与吃牌限制，以及杠后爆头重算和补牌弃胡窗口。不能把其他麻将玩法的“安全牌/现物”当作本玩法的规则保证；弃牌仍可能帮助对手吃碰杠。`research_read_source("rules")` 提供判胡与番型源码，当前合法动作仍由观察决定。规则提示随新实验的运行时冻结，旧实验与已冻结候选不自动改写。

`research_read_source("observations")` 提供策略事件规范化和财飘判断。`Observation.base_score` 保留房间底分；`PublicEvent.concealed` 标记暗杠，非本人视角隐藏其牌面。策略事件不包含配牌、庄家初摸和网络 pass/timeout，局号按当前场从 1 开始，避免训练与线上使用不同的事件特征。吃碰后仍可能合法杠牌，吃前爆头、吃后弃白保持爆头也可财飘，均以合法动作与 SDK 判断为准。

value/PPO 组合的 `train(ctx)` 可生成模型或其他文件。`ctx.play(learner)` 请求一次真实完整结算，`ctx.play_many(learner, games)` 顺序采集轨迹。`ctx.episodes` 限制训练对局数，`ctx.output` 保存产物，`ctx.initial` 是派生父代模型；`ctx.checkpoint(metadata)` 原子提交恢复点，`ctx.completed_episodes` 和 `ctx.checkpoint_metadata` 提供恢复信息。启发式与搜索组合没有训练槽：`ctx.play` 只在训练上下文可用，把规则调参塞进训练槽会占用本支迭代时间，并且与随后的配对选优重复。

训练与策略只接收合法观察和清理后的结算。收益预测对实际动作记录真实回报，搜索估计标签另记来源。PPO 基线记录旧动作概率和价值，执行裁剪目标、多轮小批次更新与价值学习。假想模拟器从合法观察与独立随机流生成世界，不接收宿主真实隐藏状态。

候选须保持组合算法与指定决策侧重点或风格。牌效优先（`heuristic-efficiency` / `style=efficiency`）以向听、剩余有效进张和胡牌速度为主要依据；番型经营（`heuristic-fan` / `style=fan`）权衡速度与白板、爆头、财飘、杠的合法倍率收益；局势自适应（`heuristic-adaptive` / `style=adaptive`）根据公开分差、剩余局数与庄位动态调整速度和收益偏好。三者都不使用训练槽，仍按共同的终局得分目标评估。启发式对象提供对应 `style` 和 `utilities(observation)`；自由规则组合 `heuristic-unrestricted` 不限定攻守风格、不要求 `style` 字段，只要求规则／启发式决策和 `utilities(observation)`，允许按局面动态调整偏好；学习、搜索对象提供对应 `algorithm`，包装类也必须直接暴露。学习算法的 `train(ctx)` 与 `model/training.json` 必须证明实际更新。完整校验器通过 `research_read_source("capabilities")` 读取。依赖使用精确 `name==version`，下载 wheel 并安装到隔离环境；相同精确依赖环境可以只读复用。

每次迭代先整理待复盘结论，再写研究判断与代码：`research_declare_plan` 要求三个字段 `assessment`（现役策略现在具体做什么、这条分支的实测轨迹说明什么）、`direction`（本次要尝试的方向）、`reason`（为什么这个方向最值得做）。未声明判断时 `research_create_candidate` 与 `research_submit_batch` 都拒绝执行，Agent 必须在同一次回复里补上。判断按迭代保存，展示在网页卡片和详情里；上一次判断随执行反馈一起进入下一次迭代的提示，供 Agent 对照结果修正自己的判断。

工具回执可能丢失而调用已经生效，因此契约要求把报错当作状态问题处理：先重新读 `research_status`，按实际落到注册表的结果继续。`research_create_candidate` 对同一分支重复同名调用返回已有候选（`existing` 为 true）而不是报冲突，`research_declare_plan` 可安全重写；只有 `research_submit_batch` 需要确认结果，不能盲目重复。

草稿必须先跑通再提交：`research_check_candidate` 在冻结运行时里导入包、用完整校验器执行 `build`。无 `train` 的候选还要实际打一局，覆盖 `reset`、动作选择和结算回调，运行异常、非法动作或超时均拒绝，单局自检最多 30 秒；有 `train` 时再打一场训练。`research_submit_batch` 会重跑同一套自检，失败就整批拒绝并返回原始错误，草稿保持可改、迭代仍为 researching，Agent 必须在同一次回复内修好重交；不能靠"写完就停"把跑不起来的代码送进准备阶段。通过自检的结果记在候选的 `preflight` 字段里；自检成败与原始错误另保存在 `checks` 中供累计复盘使用。

| 目的 | 工具 |
| --- | --- |
| 当前状态 / 历史与成绩 | `research_status` / `research_history` |
| 单条结论的证据与修订 | `research_read_conclusion` |
| 契约与公开 SDK | `research_read_source` |
| 整理累计结论、适用条件和失败原因 | `research_record_conclusion` |
| 迭代开始时声明研究判断 | `research_declare_plan` |
| 本支派生、读写 | `research_create_candidate`、`research_read_file`、`research_write_file` |
| 声明依赖 | `research_set_dependencies` |
| 提交前自检 | `research_check_candidate` |
| 提交后结束本轮会话 | `research_submit_batch` |

工具会话绑定当前实验和本分支；旧实验的会话不能修改新实验。研究 Agent 不控制评估条件、资源并发、用户轮数、选池规则或平台对战。

## 目录、备份与导出

- `projects/agent-research/.local/strategies/<combination>/<branch>/`：分支初版、草稿、冻结提交和准备完成的候选版本。组合下还保存共享初版与启动策略。
- `projects/agent-research/.local/research/`：实验配置与大乱斗回执、任务、按内容标识冻结的运行时、依赖环境、随机材料和 `reports/`。
- `projects/agent-research/.local/service/research.sqlite`：长期策略注册表、实验状态、事件和幂等回执。

三者一并备份才能完整恢复。框架代码在实验开始时冻结，候选引用自身的运行时，重启不会升级旧版本。按日期 run 划分的旧研究记录与旧研究协议都不迁移，也不建立兼容路径。

全新检出不包含历史实验、已训练模型或冻结发布包；首次打开控制台只初始化配置中的研究分支，历史和排行榜为空。运行数据均在本机生成且不纳入 Git。不同机器分别从新实验开始，不复制跨操作系统的运行时或继续旧机器的在途任务。

验收、吞吐与诊断记录保存到 `projects/agent-research/.local/research/reports/`，具体实验配置和大乱斗回执在 `projects/agent-research/.local/research/experiments/<id>/`。是否完成以完整任务和比赛回执为准，不能仅凭服务正常推断。

```bash
.venv/bin/mahjong-ai export --project projects/agent-research \
  --candidate <full-candidate-id> --output /path/to/bundle
```

导出候选及其对应冻结运行时。测试房、自由对战与正式比赛各自保存记录，外部成绩不回流研究训练，操作见[平台接入](platform.md)。
