# 杭州麻将策略研究

多种算法各自维护独立研究分支，由 Codex 编写候选，经过完整配对评估选出分支代表，再通过周期共同竞技排名和保护淘汰持续探索。

## 启动

需要 macOS、Python 3.12+、已登录的 Codex CLI。LightGBM 需要本机 OpenMP 库，可通过 `brew install libomp` 安装。

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -e '.[dev]'
.venv/bin/mahjong start
```

启动只打开研究控制台。在网页「新建研究」中设置组合、分支、模型和轮数，再开始运行。默认八种组合、每组合三条独立分支（Astra、DeepSeek、Qwen 各一条），共二十四条，每条默认研究一百轮；首次二十轮淘汰，其后每十轮淘汰。每条分支单轮可提交一至四个候选。配置入口为 [research.toml](research.toml)，模型准备和配置生效规则见[研究手册](docs/reference/strategy-agent.md#配置)。

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

`.local/service/research.sqlite` 保存研究状态，`.local/runs/` 保存冻结代码、候选、任务和报告，两者需要一并备份。新框架不读取旧研究协议。
