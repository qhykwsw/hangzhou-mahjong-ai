# 杭州麻将 AI 自迭代框架

通过本地真实对局持续改进规则与强化学习策略，再将冻结策略接入竞赛平台。优化目标是完整四人结算后的相对名次；支持多场累计、换座和可变局数。

## 开始使用

需要 Python 3.12+；长期策略研究还需要可用的 Codex CLI。候选隔离执行目前支持 macOS。

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -e '.[dev]'
.venv/bin/mahjong-ai play --rounds 8 --check-invariants
```

创建规则研究 run 并启动。默认运行 60 分钟，不限制正式验收次数。

```bash
.venv/bin/mahjong-research init --root .local/research/rule --family rule
.venv/bin/mahjong-research run --root .local/research/rule
```

RL 使用独立目录和 `--family rl`。已有 run 使用原目录续跑，管理命令见[研究流程](docs/reference/strategy-agent.md)。

## 文档导航

| 需要做什么 | 入口 |
| --- | --- |
| 玩法、计分、赛事与平台对战 | [平台规则与接入](docs/reference/platform.md) |
| 策略研究、候选编写与实验管理 | [策略研究与 Agent 接口](docs/reference/strategy-agent.md) |
| 修改框架、定位代码和验证 | [开发索引](docs/reference/development.md) · [AGENTS.md](AGENTS.md) |
| 查看平台原文与响应样本 | [原始资料](docs/sources/README.md) |

运行报告随各自 run 保存在 `<run>/reports/`。本地历史设计位于 `.local/documentation-archive/`，不作为现行规范。
