---
status: accepted
---

# 退役自动研究，只保留迭代线架构

2026-10-11 决定（用户 10-08 提出，为不影响 10-10 正式赛推迟到赛后）：删除早期由 Codex 与 DeepSeek 编写候选的自动研究系统，仓库只保留引擎、迭代线（ADR-0016）、RL 训练框架和平台对战。

自动研究在 2026-09 底之后就没有再推进，规则线与 RL 线早已取代它。它的代码仍和迭代线、网页、自由对战连在一起：harness 冻结对手池时读研究状态，网页服务同时是研究的控制循环，自由对战脚本向研究服务查询最强版本，后台的 launchd 配置里还带着模型提供方的密钥。留着它会继续拖慢测试、误导阅读，也让每次改执行器都要顾及一套不再使用的协议。

## 决定

- 删除的代码：研究状态机与配置（`src/mahjong_ai/research/`）、训练方案迭代编排（`src/mahjong_ai/recipes/`）、研究 MCP 与只读项目 MCP（`interfaces/mcp.py`、`project_mcp/`）、Agent 会话与模型配置（`execution/agent.py`、`controller.py`、`models.py`）、候选准备链路（`jobs.py` 的 prepare、`candidate_runtime.train_package`、`execution/dependencies.py`）、组合能力检查（`strategies/catalog.py`）、研究候选导出（`platform/export.py`）、研究存储（`persistence/store.py`）。
- 一并删除同属早期架构的部分：策略项目清单、冻结发布包与比赛注册（ADR-0008：`competition/`、`project.py`、`projects/personal/rules/`、`mahjong-ai release-build` 与 `competition-*` 命令），自我对弈 PPO 训练器与 ONNX 推理（ADR-0009 的 `train/`、`selfplay_core/policy.py`），以及自动研究里方案分支的配置（`recipes/iteration.toml`、`recipes/baseline/`）。
- 网页改成只读展示迭代线的小服务（`mahjong start|status|stop`），只有 `/api/lines`；服务状态放仓库的 `.local/web/`，环境里不再带 Codex 配置或任何密钥。
- harness 冻结对手池时只接受代码包目录，默认沿用现役对手池的成员、对局配置与资源；`reference` 只接受池内成员或按权重哈希指定的模型。
- 保留不动：
  - 模型库与运行时库留在 `projects/agent-research/.local/research/{models,runtimes}`。对手池、代理面板与尝试记录按这个路径引用运行时，移动会改变对手池身份。
  - 旧研究的数据（`research.sqlite`、`strategies/`、报告）留在本机作归档，不删除。
  - `guided/` 的代码，包括 `trial.py`、`recipe_jobs.py`。它是 RL 线冻结在各主机上的训练框架，改动要给每台主机重新 `setup`；其中只服务于旧研究的部分以后随框架升级一并清理。
  - `recipes/API.md`：它是 RL 线训练方案的接口说明。
- ADR-0001～0008、0012～0014 改为已被本决定取代；ADR-0009 部分取代：PPO 训练器已删除，训练进程读取隐藏状态的例外仍适用于 `guided/`。

## 考虑过的替代方案

- **只停用不删除**：代码留着，耦合点也留着；网页、harness 与自由对战仍依赖研究服务，每次修改执行器都要顾及旧协议。
- **连同数据一起删除**：研究数据量大（约 0.7 GB）且不可再生，删除不可逆；它不妨碍任何现行流程，只作归档。
- **移动模型库与运行时库到新路径**：路径参与对手池身份与历次记录，移动后需要整体重算，收益只是名字更整齐。

## 后果

- 已冻结的运行时是 `src/mahjong_ai` 的整棵副本，不受本次删除影响：对手池成员、代理面板与已有成绩照旧。
- `mahjong-ai export`、`release-build`、`competition-*` 不再可用；参赛包由 `harness.line promote` 导出。
- 自由对战常驻脚本只按 `entrant.pin`（或 `entrant.rotation`）出场，不再向研究服务查询排名。
- 运行时指纹因源码变化而改变，之后冻结的运行时与旧运行时不同；这只影响新冻结的运行时，不影响已有记录。
