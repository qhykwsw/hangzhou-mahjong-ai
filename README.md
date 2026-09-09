# 杭州麻将 AI 自迭代框架

多个独立研究方向持续编写候选，共同进入四人混合对战池。框架按每期完整比赛的平均名次效用排名、归档末位选手，再把结果交回各方向的同一个 Codex 会话。

## 开始使用

需要 macOS、Python 3.12+ 和已登录的 Codex CLI。模型与提供方沿用本机 Codex 配置，候选通过系统沙箱隔离执行。

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -e '.[dev]'
.venv/bin/mahjong start
```

命令启动项目专属的后台服务。首次启动创建默认研究并开始 60 分钟预算；关闭终端或当前对话后继续运行。重复启动打开已有记录，不新增服务或预算；服务不会在登录或重启电脑后自动运行。

在 gpu-4070 上通过 SSH 进入 MacBook 后，同样执行上述命令。终端会显示本机地址和可在 gpu-4070 打开的 Tailscale 地址；SSH 会话中不会打开 MacBook 浏览器。端口和远程访问方法见[启动与远程访问](docs/reference/strategy-agent.md#启动与远程访问)。

网页可选择研究记录、创建新研究、选择已配置的方向、暂停、继续、增加预算，查看实时对战、积分和历史事件。预算耗尽时计算暂停，网页保留。

```bash
.venv/bin/mahjong status
.venv/bin/mahjong stop
.venv/bin/mahjong-ai play --rounds 8 --check-invariants
```

默认配置在 [research.toml](research.toml)，方向模板在 [directions](directions)。规则与强化学习只是初始方向示例，新增方向通过配置和模板完成。

## 文档导航

| 需要做什么 | 入口 |
| --- | --- |
| 玩法、计分、赛事与平台对战 | [平台规则与接入](docs/reference/platform.md) |
| 策略研究、候选接口、方向配置 | [策略研究与 Agent 接口](docs/reference/strategy-agent.md) |
| 修改框架、定位代码和验证 | [开发索引](docs/reference/development.md) · [AGENTS.md](AGENTS.md) |
| 查看平台原文与响应样本 | [原始资料](docs/sources/README.md) |

`.local/` 保存运行状态与凭据，被 Git 忽略。新 run 位于 `.local/runs/YYYYMMDD-arena-NN/`，报告在各 run 的 `reports/`。旧 v1/v2 run 的冻结文件与证据保留为只读历史档案，新入口不续跑或自动迁移旧协议。
