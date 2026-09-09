# 开发索引

## 术语与索引

| 概念 | 含义与实现入口 |
| --- | --- |
| 局（Hand） | 一次发牌到胡牌或流局；[engine.py](../../src/mahjong_ai/engine.py)、[rules.py](../../src/mahjong_ai/rules.py) |
| 场（Game） | 若干局，分数在局间累计；`RoundConfig.rounds` 是每场局数 |
| 完整结算（Round） | 一个评估样本，可含多场；场间累计，最终结算一次名次与 RL 奖励；[settlement.py](../../src/mahjong_ai/settlement.py) |
| 观察与结果 | [types.py](../../src/mahjong_ai/types.py)；`Observation.stage.seat` 是物理座位，`RoundResult.rank_by_seat` 按稳定参与者索引 |
| 策略与训练 | [policy.py](../../src/mahjong_ai/policy.py)、[training.py](../../src/mahjong_ai/training.py) |
| 评估与 Elo | [evaluator.py](../../src/mahjong_ai/evaluator.py)、[ratings.py](../../src/mahjong_ai/ratings.py)、[league.py](../../src/mahjong_ai/league.py) |
| 持久研究 | [research.py](../../src/mahjong_ai/research.py) 管状态与记忆；[research_jobs.py](../../src/mahjong_ai/research_jobs.py) 执行实验 |
| 研究入口与契约 | [research_cli.py](../../src/mahjong_ai/research_cli.py)、[research_mcp.py](../../src/mahjong_ai/research_mcp.py)、[research_templates.py](../../src/mahjong_ai/research_templates.py) |
| 候选隔离 | [candidate_runtime.py](../../src/mahjong_ai/candidate_runtime.py)、[candidate_worker.py](../../src/mahjong_ai/candidate_worker.py) |
| 平台适配与审计 | [official.py](../../src/mahjong_ai/official.py)、[online.py](../../src/mahjong_ai/online.py)、[replay_audit.py](../../src/mahjong_ai/replay_audit.py) |

长期开放代码研究使用 `mahjong-research`。`mahjong-ai iterate` 与 `scripts/run_campaign.py` 保留为旧受限提案入口，参数以各自 `--help` 为准。

## 验证与边界

```bash
.venv/bin/python -m pytest -q
```

规则差异优先用 `tests/fixtures/official-v25-*` 的真实快照与牌谱定位；抓打圈豁免使用 [test_catch_play.py](../../tests/test_catch_play.py) 的合成契约场景，四白板回归读取 [fan-calc 原始响应](../sources/fan-calc-four-whites.json)。合成场景与无状态番型核验不能证明整个对局引擎等价。

本地引擎与平台适配支持抓打圈打财神者豁免、圈内响应与吃碰后的续飘/断链；牌谱审计采用同一规则。`Observation.god_discarder_seat` 是公开物理座位，`catch_restricted` 表示本人是否受限。本地按动作推进响应窗口，不模拟平台固定等待时长；豁免路径仍需新增真实平台牌谱验证。更新源码不会升级已有 run 的冻结 runtime，旧 run 继续使用原协议。

研究候选在 macOS `sandbox-exec` 中运行，文件访问受限且禁用网络。Linux/GPU 机器需要先适配强制隔离执行器，当前会拒绝无隔离运行。动作超时、非法动作和运行错误必须保留诊断，不能计为正常成功。

本地多场按顺序执行；线上可并发且随机换座。完整阶段的身份映射、阶段上下文与最高并发窗口仍需专项实战验证，牌谱 timeout 的语义也有覆盖缺口。性能结论必须注明引擎、执行方式和样本配置。
