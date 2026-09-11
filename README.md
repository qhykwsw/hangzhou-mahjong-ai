# 杭州麻将策略研究

项目同时支持自动研究和人工维护的策略项目。自动研究的分支由 Codex 编写候选并经过完整配对评估；人工项目独立训练和发布具体版本。共享比赛可以让两类冻结版本在相同条件下对战。

## 启动

需要 macOS、Python 3.12+、已登录的 Codex CLI。LightGBM 需要本机 OpenMP 库，可通过 `brew install libomp` 安装。

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -e '.[dev]'
.venv/bin/mahjong start
```

启动打开研究控制台。对比对象由上一轮大乱斗前 K 名自动生成；在卡片列表勾选本轮要迭代的分支，设置研究总轮数、每轮分支迭代次数与并发后开始。当前自动研究项目的配置在 [projects/agent-research/research.toml](projects/agent-research/research.toml)，注册七种组合各一条 DeepSeek 分支（共七条），共 20 轮、每轮每支 5 次迭代，训练默认 128 场、上限 256 场。每轮结束后统一对比并更新前七名对手池，完整操作见[研究手册](docs/reference/strategy-agent.md)。

```bash
.venv/bin/mahjong status
.venv/bin/mahjong stop
.venv/bin/mahjong-ai play --rounds 8 --check-invariants
```

关闭终端后后台继续运行；停止服务或重启电脑后，重新启动并在网页继续。当前自动研究项目位于 `projects/agent-research/`，`mahjong start` 会在仓库根目录自动选择它；也可以显式传入 `--project projects/agent-research`。SSH 启动输出本机与已连接的 Tailscale 地址，不打开远端电脑浏览器。

## 入口

| 需要做什么 | 文档 |
| --- | --- |
| 配置、运行、暂停和理解排名 | [研究手册](docs/reference/strategy-agent.md) |
| 修改框架或策略 SDK | [开发索引](docs/reference/development.md) |
| 核对规则、导出或平台对战 | [平台规则与接入](docs/reference/platform.md) |
| 理解领域对象和架构取舍 | [术语表](CONTEXT.md) · [决策记录](docs/adr/) |
| 查阅官方材料 | [原始资料](docs/sources/README.md) |

自动研究项目的 `projects/agent-research/.local/strategies/`、`projects/agent-research/.local/research/` 和 `projects/agent-research/.local/service/research.sqlite` 需要一并备份。

自动研究迁移后的数据保存在 `projects/agent-research/.local/`；人工维护的 PPO、规则或搜索项目位于 `projects/personal/`，通过 `project.toml` 和冻结发布包接入共享比赛。跨项目注册表位于根目录 `.local/competition/competition.sqlite`。
