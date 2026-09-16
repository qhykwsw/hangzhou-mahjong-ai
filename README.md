# 杭州麻将策略研究

项目同时支持自动研究和人工维护的策略项目。自动研究的分支由 Codex 编写候选并经过完整配对评估；人工项目独立训练和发布具体版本。共享比赛可以让两类冻结版本在相同条件下对战。

## 启动

支持 macOS 和启用 systemd 的 Linux/WSL2，需要 Python 3.12+、已登录的 Codex CLI。

- macOS 使用系统自带的 launchd 和 sandbox-exec；LightGBM 的 OpenMP 库可通过 `brew install libomp` 安装。
- Ubuntu/WSL2 安装运行依赖：`sudo apt-get install bubblewrap libseccomp2 libgomp1 build-essential`。`systemctl --user show-environment` 应能正常执行；WSL 的 systemd 设置见[研究手册](docs/reference/strategy-agent.md)。
- 不复制其他机器的虚拟环境或编译产物；在当前机器安装依赖并构建 C 加速器。

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -e '.[dev]'
.venv/bin/mahjong start
```

如果系统的 `python3` 低于 3.12，可用 `uv venv --python 3.12 --seed .venv` 创建环境，再执行上述安装命令。

启动打开研究控制台，不自动运行实验。仓库只保存源码和配置，不附带历史实验、训练模型或发布版本。当前配置使用 DeepSeek，开始研究前需在 `CODEX_HOME`（默认 `~/.codex`）配置 `netease-deepseek-flash.config.toml`；仅启动控制台不需要这个配置。

当前项目注册四种启发式组合各两条 DeepSeek 分支。在卡片列表勾选要迭代的分支后开始；每轮先配对选优，再由全体代表大乱斗产生下一轮对手池。具体轮数、场数和并发以 [research.toml](projects/agent-research/research.toml) 与网页创建实验时的设置为准，完整操作见[研究手册](docs/reference/strategy-agent.md)。

```bash
.venv/bin/mahjong status
.venv/bin/mahjong stop
.venv/bin/mahjong-ai play --rounds 8 --check-invariants
```

服务独立于启动终端运行；WSL 实例本身必须保持运行，`wsl --shutdown`、Windows 重启或实例退出都会停止服务。停止服务或重启电脑后，重新启动并在网页继续。当前自动研究项目位于 `projects/agent-research/`，`mahjong start` 会在仓库根目录自动选择它；也可以显式传入 `--project projects/agent-research`。SSH 启动输出本机与已连接的 Tailscale 地址，不打开远端电脑浏览器；WSL 优先用 `wslview` 打开 Windows 默认浏览器，没有浏览器时仍输出访问地址。

## 入口

| 需要做什么 | 文档 |
| --- | --- |
| 配置、运行、暂停和理解排名 | [研究手册](docs/reference/strategy-agent.md) |
| 修改框架或策略 SDK | [开发索引](docs/reference/development.md) |
| 核对规则、导出或平台对战 | [平台规则与接入](docs/reference/platform.md) |
| 理解领域对象和架构取舍 | [术语表](CONTEXT.md) · [决策记录](docs/adr/) |
| 查阅官方材料 | [原始资料](docs/sources/README.md) |

自动研究的数据与备份边界见[研究手册](docs/reference/strategy-agent.md#目录备份与导出)；人工项目与跨项目比赛见[策略项目](docs/reference/projects.md)。运行数据不进入 Git，提交源码不能代替备份。
