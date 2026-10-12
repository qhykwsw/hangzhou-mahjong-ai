# 迭代记录快照

规则线与 RL 线在 2026-10-11 收线时的记录，从迭代线的 `.local/` 复制而来。这是只读存档：harness 不读它，`.local/` 之后的变化也不会同步过来。

## 内容

| 路径 | 内容 |
| --- | --- |
| `notes/rule.md`、`notes/rl.md` | 两条线的迭代笔记：每次尝试的想法、诊断与结论 |
| `attempts.jsonl` | 每次尝试与参照各一条：所属线、对手池、描述、开发集与确认集成绩、是否成为最好版本 |
| `attempts/<编号>/package/` | 规则线每次尝试的策略代码 |
| `rl/attempts/<编号>/` | RL 线每次尝试的训练方案（`recipe/`）、运行计划（`plan/plan.json`）、模型目录（`catalog.json`）、训练指标（`metrics.jsonl`）与训练摘要（`training.json`）；在 GPU 主机上训练的尝试另有非空的 `train.log` |
| `rl/*.json`、`rl/starts/*.json` | 各主机的 RL 起点、训练对手登记与开局集清单 |
| `pool.json`、`pools/`、`pool-members/`、`training-rules/` | 固定对手池的定义与成员代码、RL 训练对手的代码 |
| `pdl-results.jsonl`、`stage-results.jsonl`、`proxy-results.json`、`proxy/`、`qualifier.jsonl`、`qualifier/`、`reported.json`、`scale/` | 其余评估（配对对局、阶段赛、代理桌、海选线）与规模训练的记录 |

## 没有收录

- 模型权重：按项目约定不进 Git。正式比赛出场的 fp-rl-270 参赛包在仓库的 Release 里，挂在标签 `competition-2026-10-10-r2`（正式比赛运行的代码）上。
- 评估的逐局记录（各尝试的 `jobs/`）：体量过大，成绩已汇总在 `attempts.jsonl`。
- 开局数据集（`*.json.gz`）、各版本训练框架的源码快照（与 Git 历史重复）、运行状态与锁文件。

## 脱敏

本机与训练主机的路径、主机名、用户名和地址已替换为占位：`/Users/user`、`/home/user`、`gpu-4090`、`gpu-4070`、`gpu-5070ti`、`user@gpu-4070`、`100.64.0.x` 等。其余内容与原记录逐字一致。
