# 开发索引

## 术语与索引

| 概念 | 含义与实现入口 |
| --- | --- |
| 手（Hand） | 一次发牌到胡牌或流局；[engine.py](../../src/mahjong_ai/engine.py)、[rules.py](../../src/mahjong_ai/rules.py) |
| 场（Game） | 多手累计分数；`RoundConfig.rounds` 是每场手数 |
| 完整结算（Round） | 四名固定参与者最终结算一次名次与奖励；[settlement.py](../../src/mahjong_ai/settlement.py) |
| 观察与结果 | [types.py](../../src/mahjong_ai/types.py)；`Observation.stage.seat` 是物理座位，`RoundResult.rank_by_seat` 按稳定参与者索引 |
| 策略与训练 | [policy.py](../../src/mahjong_ai/policy.py)、[training.py](../../src/mahjong_ai/training.py) |
| 研究方向、混合池、比赛期 | [arena.py](../../src/mahjong_ai/arena.py)；事务状态、调度、排名和方向隔离 |
| 进程、预算和回执 | [arena_controller.py](../../src/mahjong_ai/arena_controller.py)、[arena_jobs.py](../../src/mahjong_ai/arena_jobs.py) |
| 后台服务与网页 | [app.py](../../src/mahjong_ai/app.py)、[web](../../src/mahjong_ai/web) |
| 研究会话与契约 | [research_agent.py](../../src/mahjong_ai/research_agent.py)、[research_mcp.py](../../src/mahjong_ai/research_mcp.py)、[research_templates.py](../../src/mahjong_ai/research_templates.py) |
| 候选与依赖隔离 | [candidate_runtime.py](../../src/mahjong_ai/candidate_runtime.py)、[candidate_worker.py](../../src/mahjong_ai/candidate_worker.py)、[dependencies.py](../../src/mahjong_ai/dependencies.py) |
| 平台适配与审计 | [official.py](../../src/mahjong_ai/official.py)、[online.py](../../src/mahjong_ai/online.py)、[replay_audit.py](../../src/mahjong_ai/replay_audit.py) |

研究入口是 `mahjong start`，候选接口见[研究手册](strategy-agent.md)。独立引擎、训练、吞吐测试、导出和平台命令保留在 `mahjong-ai`，以 `--help` 为准。旧研究工作台、配对晋升、Elo、独立验收和 campaign 运行器已退出活动源码。

## 验证与边界

```bash
.venv/bin/python -m pytest -q
```

规则差异优先用 `tests/fixtures/official-v25-*` 的真实快照与牌谱定位；抓打圈豁免使用 [test_catch_play.py](../../tests/test_catch_play.py) 的合成契约场景，四白板回归读取 [fan-calc 原始响应](../sources/fan-calc-four-whites.json)。合成场景与无状态番型核验不能证明整个对局引擎等价。

本地引擎与平台适配支持抓打圈打财神者豁免、圈内响应与吃碰后的续飘/断链；牌谱审计采用同一规则。`Observation.god_discarder_seat` 是公开物理座位，`catch_restricted` 表示本人是否受限。本地按动作推进响应窗口，不人为等待整个平台窗口结束。

研究候选在 macOS `sandbox-exec` 中运行，禁止网络和私有文件访问。Linux/GPU 机器需要先适配强制隔离执行器。超时采用平台默认动作，迟到动作不能串入后续观察；回退次数、主进程耗时和错误诊断随真实结算保留。

服务与 run 的代码版本边界、地址和日志位置见[研究手册](strategy-agent.md#目录与命名)。现役协议以源码及 MCP schema 为准，旧 run 中的文档和源码属于冻结证据。

研究状态变更由 SQLite 事务提交。控制器先登记任务再创建进程，比赛按 run、期数与场次唯一入账，整期淘汰只执行一次。暂停可重跑当前未完成结算，不能改变已完成回执或已冻结工件。

完整平台阶段的身份映射、阶段上下文与最高并发窗口仍需专项实战验证。框架端到端验收记录留在对应 run 的 `reports/`，本地通过不等于线上阶段已验证。
