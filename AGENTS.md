# 项目开发约定

本项目提供杭州麻将引擎、多方向独立策略研究与平台对战。安装和启动见 [README](README.md)，领域术语见 [CONTEXT.md](CONTEXT.md)。

## 按修改范围读取

- 引擎、规则、观察或假想模拟：先读[平台规则](docs/reference/platform.md)和[开发索引](docs/reference/development.md)，核对官方样本与回归。
- 研究、训练、执行、候选或网页：先读[研究手册](docs/reference/strategy-agent.md)。候选接口修改同步 `research/contract.py` 的 CONTRACT 与 `interfaces/mcp.py` schema。
- 平台适配、导出或比赛：先读[平台接入](docs/reference/platform.md)，外部对战与研究训练隔离。

## 开发约束

- 研究 Agent 仅通过研究 MCP 修改本分支候选；框架修改在仓库源码进行。评估条件与计分不能为了候选表现而改变。
- 策略按组合和分支长期保存；新实验的运行时代码、提交源码、模型与完整比赛回执冻结。修改框架协议使用新实验，候选保留自身运行时，重启后台不会升级旧版本。本项目不迁移旧研究协议，不保留旧路径兼容层。
- 策略训练只接收合法观察和清理后的结算。假想隐藏状态可由公开信息与独立随机流生成；宿主真实隐藏牌、环境随机材料、凭据与外部反馈不进入候选。
- 自动研究项目的 `projects/agent-research/.local/service/research.sqlite`、`projects/agent-research/.local/strategies/` 与 `projects/agent-research/.local/research/` 一并保存研究；根目录旧 `.local/` 仅作迁移回滚备份。人工项目的训练和发布数据归各自项目。凭据目录权限 `0700`、文件 `0600`，日志和报告不写令牌。
- 研究状态转换由 `research/` 拥有，执行器提交任务结果，用户接口调用研究接口；规则与策略 SDK 不依赖网页或后台调度状态。

## 验证与文档

- 按范围运行测试；影响引擎、研究协议或平台适配时运行 `.venv/bin/python -m pytest -q`。模型配置变更做实际调用检查，网页修改做浏览器验证。
- 通用说明维护在 `docs/reference/`，官方原文与响应保留在 `docs/sources/`。机制以源码、schema 和测试为准。
- 自动研究的训练、吞吐、诊断和验收报告放 `projects/agent-research/.local/research/reports/`；跨项目比赛结果放 `.local/competition/runs/`。具体成绩、耗时、测试计数与核实日期不进入通用手册。
- 清理本次产生且不再需要的临时文件；新研究的冻结记录不因代码清理而改写。
