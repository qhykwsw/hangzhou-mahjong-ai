# 杭州麻将策略迭代

项目包含杭州麻将引擎、两条由迭代 Agent 推进的策略线（规则线与 RL 线，合称迭代线）和官方平台的对战客户端。迭代线面对固定对手池评测，每次尝试改一处，统计上确认更强才成为新的最好版本；RL 线在 GPU 主机上用规则教师指导的训练框架训练。

## 安装

支持 macOS（Apple Silicon）和 Linux/WSL2，需要 Python 3.12+。

- macOS 先执行 `brew install libomp`；候选沙箱使用系统自带的 sandbox-exec。
- Ubuntu/WSL2 安装运行依赖：`sudo apt-get install bubblewrap libseccomp2 libgomp1 build-essential`。
- 不复制其他机器的虚拟环境或编译产物；在当前机器安装依赖并构建 C 加速器。

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -e '.[dev]'               # 模型推理另加 '.[model]'
.venv/bin/python setup.py build_ext --inplace            # 可选：C 加速，失败时自动回退纯 Python
.venv/bin/mahjong-ai play --rounds 8 --check-invariants  # 自检：本地打一场
```

如果系统的 `python3` 低于 3.12，可用 `uv venv --python 3.12 --seed .venv` 创建环境，再执行上述安装命令。

## 网页

```bash
.venv/bin/mahjong start    # 启动只读网页：迭代线、代理桌与线上自由对战
.venv/bin/mahjong status
.venv/bin/mahjong stop
```

网页由 macOS 用户域 launchd 或 Linux/WSL2 的 systemd 用户服务托管，独立于启动终端，优先使用端口 8765。Tailscale 已连接时同时监听回环与 Tailscale 地址；SSH 会话里启动只输出地址，不打开浏览器。

## 入口

| 需要做什么 | 文档 |
| --- | --- |
| 推进或维护规则线与 RL 线 | [第一性原理迭代线](projects/personal/firstprinciples/README.md) |
| 修改 RL 训练框架或训练方案接口 | [训练手册](projects/personal/selfplay/guided/README.md) · [方案接口](projects/personal/selfplay/recipes/API.md) |
| 修改引擎、执行器或策略 SDK | [开发索引](docs/reference/development.md) |
| 核对规则或平台对战 | [平台规则与接入](docs/reference/platform.md) |
| 理解领域对象和架构取舍 | [术语表](GLOSSARY.md) · [决策记录](docs/adr/) |
| 查阅官方材料 | [原始资料](docs/sources/README.md) |

运行数据不进入 Git，提交源码不能代替备份。
