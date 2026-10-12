# 项目开发约定

本项目提供杭州麻将引擎、由迭代 Agent 推进的规则线与 RL 线（迭代线），以及平台对战。安装和启动见 [README](README.md)，领域术语见 [GLOSSARY.md](GLOSSARY.md)。

## 沟通

- 面向用户的回复、说明与报告一律使用简体中文；代码、命令和技术标识保持英文。长会话中工具输出和系统消息多为英文，每条回复发出前核对语言，不要被带偏。

## 按修改范围读取

- 引擎、规则、观察或假想模拟：先读[平台规则](docs/reference/platform.md)和[开发索引](docs/reference/development.md)，核对官方样本与回归。
- 执行器、候选运行时、主机调度或网页：先读[开发索引](docs/reference/development.md)。
- 平台适配、导出或比赛：先读[平台接入](docs/reference/platform.md)。自由对战与测试房的赛后牌谱可作为训练数据，见 [ADR-0015](docs/adr/0015-platform-replays-as-training-data.md)。
- RL 训练框架（`projects/personal/selfplay/guided/`）、编码或训练方案接口：先读[训练手册](projects/personal/selfplay/guided/README.md)、[方案接口](projects/personal/selfplay/recipes/API.md)与 [ADR-0011](docs/adr/0011-rule-guided-policy-learning.md)；改 `guided/` 后按迭代线 README 给每台 RL 主机重新 `setup` 并封存。`src/` 不得引入 torch，模型推理在独立的推理服务进程里。
- 迭代线的 harness、对手池或主机（`projects/personal/firstprinciples/`，维护者的工作）：先读其 [README](projects/personal/firstprinciples/README.md) 与 [ADR-0016](docs/adr/0016-maintainer-iteration-lines.md)，改完重新封存；固定对手池、开发集与确认集口径不为某次尝试改变。

## 迭代线入口

- 用户说「开始/继续执行规则迭代」或「开始/继续执行 RL 迭代」：你是该线的迭代 Agent，读[公共手册](projects/personal/firstprinciples/handbooks/common.md)并照它循环，直到用户说「停」；你的范围与收尾以手册为准。

## 开发约束

- 迭代 Agent 只改本线的 `rule/` 或 `rl/recipe/`；harness、对手池与主机由维护者修改。评估条件与计分不能为了候选表现而改变。
- 每次尝试的代码快照、模型与评估记录冻结在迭代线的 `.local/`，收线时的脱敏快照在 `projects/personal/firstprinciples/records/`（只读存档，不随 `.local/` 更新）；对手池成员在冻结的运行时上运行，仓库源码的改动不影响已有成绩。本项目不迁移旧协议，不保留旧路径兼容层。
- 策略训练只接收合法观察和清理后的结算。假想隐藏状态可由公开信息与独立随机流生成；宿主真实隐藏牌、环境随机材料、凭据与外部反馈不进入候选。RL 训练器（`guided/`）是显式例外：价值网络与信念头的训练标签在训练进程内读取隐藏状态，见 [ADR-0009](docs/adr/0009-independent-selfplay-trainer-for-personal-project.md)；策略网络可在训练进程内经 oracle guiding 读取隐藏状态，读取强度在训练中衰减到零，导出时剥离，发布包与所有推理不读取，见 [ADR-0018](docs/adr/0018-oracle-guided-policy-training.md)；平台牌谱在引擎里重放成决策方的合法观察后可用于模仿，见 [ADR-0015](docs/adr/0015-platform-replays-as-training-data.md)。
- 模型库与运行时库在 `projects/agent-research/.local/research/models` 与 `runtimes`：这是已退役的自动研究留下的路径，对手池与各项记录按它引用，不移动；同目录下旧研究的数据只作归档（[ADR-0019](docs/adr/0019-retire-automated-research.md)）。模型、运行数据和本机编译产物不提交 Git（`records/` 快照除外）；本机相关的连接信息（如训练主机的 `.local/training-hosts.json`）也不提交。凭据目录权限 `0700`、文件 `0600`，日志和报告不写令牌。
- 规则与策略 SDK 不依赖网页或调度状态。

## 验证与文档

- 按范围运行测试；影响引擎、执行器、harness 或平台适配时运行 `.venv/bin/python -m pytest -q`，改 `guided/` 另跑 `.venv/bin/python -m pytest -q projects/personal/selfplay/tests`。网页修改做浏览器验证。
- 通用说明维护在 `docs/reference/`，官方原文与响应保留在 `docs/sources/`。机制以源码、schema 和测试为准。
- 迭代线的笔记、诊断与评估记录放 `projects/personal/firstprinciples/.local/`；比赛与线上对战结果放 `.local/competition/`。具体成绩、耗时、测试计数与核实日期不进入通用手册。
- 清理本次产生且不再需要的临时文件；冻结记录不因代码清理而改写。
