# 杭州麻将策略研究

多种算法各自维护独立研究分支，由 Codex 编写候选，经过完整配对评估选出分支代表，每轮再由全体分支代表进行大乱斗，前 K 名成为下一轮对比对象。策略历史按组合长期保留。

## 启动

需要 macOS、Python 3.12+、已登录的 Codex CLI。LightGBM 需要本机 OpenMP 库，可通过 `brew install libomp` 安装。

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -e '.[dev]'
.venv/bin/mahjong start
```

启动打开研究控制台。对比对象由上一轮大乱斗前 K 名自动生成、不由用户挑选；在卡片列表勾选本轮要迭代的分支，设置研究总轮数、每轮分支迭代次数与并发后开始。本项目 [research.toml](research.toml) 当前注册七种组合各一条 DeepSeek 分支（共七条），跑 2 轮研究、每轮每支 5 次迭代，训练默认 128 场、上限 256 场；代码默认是每组合 DeepSeek、Qwen 各一条（共十四条）、十轮 × 每支十次。每轮结束后统一对比并更新前七名对手池，完整操作见[研究手册](docs/reference/strategy-agent.md)。

```bash
.venv/bin/mahjong status
.venv/bin/mahjong stop
.venv/bin/mahjong-ai play --rounds 8 --check-invariants
```

关闭终端后后台继续运行；停止服务或重启电脑后，重新启动并在网页继续。SSH 启动输出本机与已连接的 Tailscale 地址，不打开远端电脑浏览器。

## 入口

| 需要做什么 | 文档 |
| --- | --- |
| 配置、运行、暂停和理解排名 | [研究手册](docs/reference/strategy-agent.md) |
| 修改框架或策略 SDK | [开发索引](docs/reference/development.md) |
| 核对规则、导出或平台对战 | [平台规则与接入](docs/reference/platform.md) |
| 理解领域对象和架构取舍 | [术语表](CONTEXT.md) · [决策记录](docs/adr/) |
| 查阅官方材料 | [原始资料](docs/sources/README.md) |

`.local/strategies/` 保存长期策略历史，`.local/research/` 保存冻结运行时、实验与任务，`.local/service/research.sqlite` 保存研究状态，三者需要一并备份。旧运行记录不兼容。
