# 策略研究与 Agent 接口

专用 Codex 在一个持久线程中通过研究 MCP 工作；shell、通用执行和其他 MCP 关闭。运行器管理基础设施与预算，Agent 负责策略代码、实验方案和证据判断。

## 目标与运行单位

优化完整四人结算的平均相对名次，第一至第四对应收益 3、2、1、0。比分用于确定名次，同名次下不额外奖励高分。策略可根据当前排名、分差与剩余进程调整风险，局数从观察读取。

规则和 RL 各维护一个长期 run，分别积累历史和种群，不交换 run 成果。初次创建见[README](../../README.md)。`--rounds` 是每场局数，`--games-per-round` 是每个完整结算的场数；评估中的“一轮”指一个完整结算。

## 目录与命名

长期迭代实验统一放在 `.local/runs/YYYYMMDD-{family}-NN/`：日期取创建当天的本地日期，`family` 为 `rule` 或 `rl`，`NN` 为同日同路线从 `01` 起的首个可用序号。`init --family rule` 自动分配目录；显式指定 `--root` 时，新建项目实验仍遵守此规范，已有 run 则使用原路径。

续跑和加预算沿用原目录，不因新的一批实验另建 run。候选、模型、批次、验收与冻结 runtime 留在 run 内，报告及会话辅助记录放其 `reports/`。平台测试、正式对战与凭据分别管理，不混入研究 run。

run 初始化时冻结源码和环境指纹，后续仓库更新或目录改名不会升级它。旧结果可能保存绝对路径，搬迁时须校验原文件内容一致，并保留必要的旧路径链接；不要批量改写冻结结果。

## 管理与续跑

在项目根目录执行，`<run-name>` 替换为 `list` 返回的名称。`list` 只读列出已有 run 的路径、路线、基线和冻结引擎版本，不启动实验。

```bash
.venv/bin/mahjong-research list
.venv/bin/mahjong-research status --root ".local/runs/<run-name>"
.venv/bin/mahjong-research history --root ".local/runs/<run-name>" --query 听牌
.venv/bin/mahjong-research budget --root ".local/runs/<run-name>" --seconds 3600 --unlimited-acceptance
.venv/bin/mahjong-research run --root ".local/runs/<run-name>"
```

`init` 只创建一次，`run` 恢复原线程和进度，`budget` 显式增加时间。默认预算 60 分钟且不限验收次数；新 run 可用 `--acceptance-slots N` 限制次数，续跑可用 `budget --slots N` 增加次数。`acceptance_remaining=null` 表示不限。

预算覆盖思考、工具和实验等待，暂停期间不计时。时间耗尽、中断或提供商错误后保存状态供续跑。

## 每批操作

1. 查看目标、预算、基线和历史；检查同类想法的既有结果，选择父代与参考候选。
2. 提出一个或多个有明确假设的方案，编辑候选代码。新 run 先提交 `initial` 批次完成初始化。
3. 一次提交本批候选；环境冻结代码，执行训练、开发对照、公开行为诊断及候选联赛。
4. 等本批所有任务成功、失败或取消后，再读取完整结果。批次执行期间可以查进度、读历史和取消任务。
5. 根据整批证据统一保留或归档候选，记录理由；失败方向也留档。归档版本仍可查询和派生。
6. 将有价值的候选提交独立验收，再继续下一批，直到预算用尽。

开发实验可并行，种群决定必须在批末统一完成。对照版本应置于相同实验条件；跨配置、对手或赛程的分数不能直接排序。保留多样性服务于后续探索，不改变最终名次目标。

## 验收与记忆

正式验收使用固定三个对手，旧、新使用相同种子与座位，各跑 1,000 个完整结算，共 1,000 对。名次收益差区间整体为正且通过运行质量检查才晋升；失败或证据不足保留基线。Elo 仅描述开发联赛表现，不代替晋升判据。

相同候选、基线及验收条件复用已有结果；基线改变后需要重新比较。取消的验收计入提交次数，恢复同一验收不重复计数。连续验收持续分配统计误判额度，不限次数不代表每次都独立重置置信标准；算法以[评估器](../../src/mahjong_ai/evaluator.py)和[研究状态实现](../../src/mahjong_ai/research.py)为准。

历史接口支持摘要搜索与分页详情。诊断提供最多 128 个固定公开观察，用于解释动作差异；它不是全部开发对局的完整牌谱。正式验收私有明细与随机种子不会返回研究 Agent。

测试房和本地实验可作为研究反馈。自由对战与正式比赛使用冻结策略，结果只作比赛记录，不回流迭代。

## 研究工具

开始工作先调用 `research_read_source("contract")` 读取该 run 的契约；观察字段、初始算法与规则分别读 `types`、`policy`、`rules`。下面是导航，完整参数以工具 schema 为准。

| 工作 | 工具 |
| --- | --- |
| 目标、预算与状态 | `research_status` |
| 搜索记忆、展开证据 | `research_history`、`research_details` |
| 选择父代与读 SDK | `research_sample_parents`、`research_read_source` |
| 派生、读写候选源码 | `research_create_candidate`、`research_read_file`、`research_write_file` |
| 声明精确依赖 | `research_set_dependencies`，使用 `name==version` |
| 冻结并执行一批 | `research_submit_batch` |
| 等待或取消 | `research_wait`、`research_cancel` |
| 批末统一决定 | `research_decide_batch` |
| 独立验收 | `research_submit_acceptance` |

`research_submit_batch.pairs` 是每候选的旧新开发结算对数，`training_episodes` 是 RL 训练完整结算数；它们与固定 1,000 对的正式验收分开。`research_details` 可读候选、批次、验收公开摘要和诊断观察，支持分页。

## 候选代码

候选包含 `strategy.py` 和任意辅助 Python 模块。`build(model_dir)` 返回实现 `reset(seed)`、`act(observation)`、`finish_round(result, seat)` 的对象；动作必须来自 `observation.legal_actions`。已冻结版本通过派生后代修改。

- **规则路线**：可重写牌形计算、规则、搜索或蒙特卡洛模拟，不加载训练模型。
- **RL 路线**：可改变算法、特征、网络、优化器和训练调度；必须实际训练，并在推理中使用得到的模型。
- **观察边界**：可使用全部合法可见信息。隐藏状态搜索只能从公开信息采样假想世界，不能访问真实对手手牌和未来牌墙。

`catch_play` 表示全桌处于抓打圈，`god_discarder_seat` 为公开的豁免座位（无圈为 -1），`catch_restricted` 才表示本人受限。豁免者可吃碰杠及任意出牌，具体动作以 `legal_actions` 为准；已有 run 的字段与规则以其冻结契约为准。

RL 额外实现 `train(ctx)`。`ctx.play(learner)` 执行真实完整结算并返回结果与参与者位置；模型写入 `ctx.output`，父代模型从 `ctx.initial` 读取。

保存模型、优化器、缓冲区和 RNG 状态后调用 `ctx.checkpoint(metadata)`，将文件和实际训练进度原子提交。恢复从 `ctx.completed_episodes`、`ctx.checkpoint_metadata` 和已提交文件继续；未实现检查点的训练中断后会从函数起点重跑。

候选在隔离进程执行，默认单动作上限 50 ms，可在 run 初始化配置。超时终止 worker，按第一个合法动作回退，直到下个结算重新初始化；运行错误和非法动作单独记录并阻止晋升。这一限时不代表通过线上并发时效验证。

源码权威：[工具 schema](../../src/mahjong_ai/research_mcp.py) · [候选契约与模板](../../src/mahjong_ai/research_templates.py) · [训练执行接口](../../src/mahjong_ai/candidate_worker.py)。
