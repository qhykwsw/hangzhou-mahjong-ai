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
.venv/bin/mahjong-research init --family rule
.venv/bin/mahjong-research list
.venv/bin/mahjong-research run --root ".local/runs/<run-name>"
```

`init` 在 `.local/runs/` 下自动创建 run 并返回路径；将 `<run-name>` 替换为 `list` 返回的目录名。RL 使用 `--family rl` 创建独立 run。命名、续跑和预算管理见[研究手册](docs/reference/strategy-agent.md#目录与命名)。

## 文档导航

| 需要做什么 | 入口 |
| --- | --- |
| 玩法、计分、赛事与平台对战 | [平台规则与接入](docs/reference/platform.md) |
| 策略研究、候选编写与实验管理 | [策略研究与 Agent 接口](docs/reference/strategy-agent.md) |
| 修改框架、定位代码和验证 | [开发索引](docs/reference/development.md) · [AGENTS.md](AGENTS.md) |
| 查看平台原文与响应样本 | [原始资料](docs/sources/README.md) |

`.local/` 保存本地运行数据和凭据，被 Git 忽略；运行报告随各自 run 保存在 `<run>/reports/`。
