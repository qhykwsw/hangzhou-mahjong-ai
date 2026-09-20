# ChatGPT 只读代码接入

`mahjong_ai.project_mcp` 是本仓库的独立只读 MCP，用于让 ChatGPT 阅读代码、文档、配置与 Git 历史。它和自动研究使用的 `mahjong_research` MCP 分开，启动时不导入研究服务，不创建实验或修改候选。

## 本地运行与验证

沿用根 `.venv` 与 `pyproject.toml` 的 MCP 依赖，无需额外安装一套环境。

```bash
bash scripts/run_project_mcp.sh
.venv/bin/python -m pytest -q tests/test_project_mcp.py tests/test_project_mcp_tunnel.py
```

第一条命令使用 stdio，供 MCP 客户端或隧道启动；不是交互式终端程序。服务只提供 `project_overview`、`list_files`、`read_file`、`search` 和 `git_history`。

开始时调用 `project_overview`，再按以下例子查询。

```json
{"tool": "list_files", "arguments": {"prefix": "src/mahjong_ai/", "limit": 50}}
{"tool": "read_file", "arguments": {"path": "README.md", "start_line": 1, "lines": 100}}
{"tool": "search", "arguments": {"query": "paired", "prefix": "src/"}}
{"tool": "git_history", "arguments": {"path": "src/mahjong_ai/research/selection.py", "limit": 5}}
```

这些 JSON 是工具调用示意。按返回的 `next_offset`、`next_line` 继续翻页；搜索每个文件只返回首个匹配，需读取上下文。`read_file` 默认读取已跟踪文件的工作区版本，含未提交修改；`list_files` 和 `read_file` 的 `revision` 可指定完整 commit hash，读取固定历史版本。新建但未跟踪的文件不可见。当前工作区不是原子快照。

白名单覆盖 `src/`、`scripts/`、`projects/`、`docs/`、`tests/` 的文本源码和选定根文件，包含 C 加速器源码。隐藏目录、凭据、模型、发布包、训练与比赛运行数据、原始日志、二进制和符号链接不开放。读取范围与容量上限以 `reader.py` 为准；输出脱敏，超限返回失败，不假装读完。没有实时训练指标时回答未知。

## 私有隧道

采用 [OpenAI Secure MCP Tunnel](https://developers.openai.com/api/docs/guides/secure-mcp-tunnels)。本机客户端主动连接 OpenAI，不需要公网监听。创建麻将项目自己的 tunnel，关联实际使用的个人 ChatGPT workspace，保留已有项目的 tunnel。

在 [Platform Tunnels](https://platform.openai.com/settings/organization/tunnels) 创建 `Leihuo Mahjong Read-only`，描述可用 `Read-only access to Hangzhou Mahjong source code, documentation and Git history.`。运行 key 需要该 organization 的 Tunnels Read + Use，管理隧道另需 Read + Manage。key 只放本机权限为 `600` 的文件，不写入命令行正文、聊天或仓库。

从[官方发布](https://github.com/openai/tunnel-client/releases/latest)安装并核验 `tunnel-client`，或复用本机已有官方二进制，然后配置。

```bash
.venv/bin/python scripts/project_mcp_tunnel.py configure \
  --tunnel-id tunnel_REPLACE_ME \
  --client /absolute/path/to/tunnel-client \
  --key-file /absolute/path/to/tunnel-runtime.key
.venv/bin/python scripts/project_mcp_tunnel.py doctor
.venv/bin/python scripts/project_mcp_tunnel.py start
.venv/bin/python scripts/project_mcp_tunnel.py status
```

配置存入不进 Git 的 `.local/project_mcp/`，只引用 key 文件；客户端二进制复制到该目录。入口清除可能覆盖专用 key 的环境变量。服务不需要其他项目的 Python 环境或业务代码。

`start` 在 macOS 安装独立的 `com.netease.leihuo-majiang.project-mcp` LaunchAgent，登录启动、退出重启。它运行本工作树中的服务；修改 MCP 实现后执行 `stop`、`start` 使新代码生效。已跟踪业务文件的修改直接可读。`stop` 同时移除登录启动项。Linux 使用 `run` 前台入口交给 systemd 等进程管理器托管。

健康地址为 `http://127.0.0.1:8789/readyz`，与 Runtime 项目的端口分开；`/ui` 是客户端管理页面。电脑关机或休眠时接入不可用。不要用 Agent 的临时 exec 会话托管长期服务。

## ChatGPT 连接与验收

按官方页面开启开发者模式，在 [ChatGPT Plugins](https://chatgpt.com/plugins) 新建连接，名称填 `Leihuo Mahjong Read-only`，选择 Tunnel 并填入麻将专用 `tunnel_id`。此 stdio 服务选择 No authentication，访问边界由隧道关联的组织和 workspace 提供；不要在表单里填 runtime key。

保持 `status` 显示 ready 后发现工具，并在新对话选中该连接，发出：

> 先调用 project_overview，再读取 README 和开发索引，解释自动研究与人工自我对弈项目如何共享引擎。引用源码路径和行号；找不到的运行结果回答未知，不执行修改。

本地协议通过、隧道 ready、ChatGPT 真实工具调用成功分别验证不同层次，不能互相替代。固定提交的十道检索题在 `tests/project_mcp_evaluation.xml`，用于后续模型检索验收。

审计保存在 `.local/project_mcp/audit.jsonl`，只记录时间、工具名、状态和耗时，按大小轮转一份；不记录参数或正文。隧道日志在 `.local/project_mcp/logs/`，排障时不要将原始日志或配置粘贴到聊天。停止访问可以停掉该 LaunchAgent，或在 ChatGPT 移除本连接。
