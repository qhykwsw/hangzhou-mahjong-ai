# 第一性原理迭代线（firstprinciples）

维护者亲自推进的两条迭代线，设计与取舍见 [ADR-0016](../../../docs/adr/0016-maintainer-iteration-lines.md)，术语（迭代线、尝试、固定对手池）见 [CONTEXT.md](../../../CONTEXT.md)。

- **规则线**：`rule/strategy.py` 是唯一被修改的文件。v0 是最朴素的牌效策略，之后按杭州麻将规则的第一性原理逐次改进，大方向是速度优先、必胜时追求翻倍。
- **RL 线**：沿用训练方案框架（`../selfplay/recipes`、`guided.trial`），用最小的模型（128/4/4，约 0.84M 参数）。起点是它对 piao-call-channel 做 10 万局（每场 16 局）的模仿，每次尝试按局数预算训练（默认 40 万局），在 4070 或 4090 上进行。

## 评估

`harness/evaluate.py` 让一次尝试面对固定对手池（6 个规则版本加 1 个模仿模型）：每场被评估版本坐一个座位，其余三座取对手池的三个，35 种阵容循环、四座轮换，牌墙与各方种子只由局组编号决定，所以开发集对每次尝试都是同一批牌局。规则线在 Mac 本机评估，RL 线在训练它的 GPU 主机（4070 或 4090）上就地评估，每台机器同一时间只跑一个评估；复用研究的对局执行器与冻结运行时，时限为出牌 2.0 秒、应答 0.6 秒，超时按平台默认动作代打。

对局不用沙箱：对手池的代码包是维护者自己的，对局进程直接在进程内运行它们（`InlinePolicy`），每个进程同时打几桌（一桌一个线程，模型座位等推理服务时其他桌继续），模型座位共用主机的推理服务。这些都不改变对局结果，所以是按主机的执行参数（`EVAL_WORKERS`、`EVAL_TABLES`），不进对手池 id；运行时用 `harness.evaluate runtime` 从 `src/mahjong_ai` 冻结，换运行时就要重建对手池并重跑参照。

- 开发集阶梯：4096 → 8192 → 16384 场，低于当前最好成绩两倍误差以上即停。
- 确认集：开发集超过最好成绩后，换全新牌墙与最好版本各打 8192 场，配对差单侧 90% 下界大于 0 才算成功。

```bash
cd projects/personal/firstprinciples
../../../.venv/bin/python -m harness.evaluate runtime                                       # 把 src/mahjong_ai 冻结为运行时
../../../.venv/bin/python -m harness.evaluate pool --model <weights-sha> --name <模型名> [--runtime <目录>] [--replace]  # 冻结对手池
../../../.venv/bin/python -m harness.evaluate reference <研究候选 ID | 池内模型名> [--weights <sha>] [--host local|4070|4090]  # 参照线
../../../.venv/bin/python -m harness.evaluate history rule|rl                               # 全部尝试、最好版本与参照
../../../.venv/bin/python -m harness.evaluate restore rule|rl                               # 把源码目录恢复成最好版本
../../../.venv/bin/python -m harness.evaluate seal                                          # 接受改过的评估代码（须用户同意）
../../../.venv/bin/python -m harness.evaluate rule "<改动>" --design-start <epoch>
../../../.venv/bin/python -m harness.rl export <远端 checkpoint> --name <模型名> --host <主机>  # 导出别处训练的模型（如对手池成员）
../../../.venv/bin/python -m harness.rl setup <主机上的 checkpoint | 权重 sha> --name <起点名> --host 4070|4090  # RL 线起点（每台主机一次）
../../../.venv/bin/python -m harness.rl attempt "<改动>" --design-start <epoch> --host 4070|4090  # 训练并评估
```

冻结对手池时记下 `harness/evaluate.py` 的哈希，之后评估代码一改就拒绝评估，直到用户同意 `seal`。`FP_LOCAL`、`FP_STAGES`、`FP_CONFIRM`、`FP_RL_HANDS`、`FP_RL_BUDGET` 用单独的目录、缩短的场数与训练预算做流程检查，不进正式记录。

## 迭代方式

两条线各由一个独立的 Claude Code 会话推进，用户以 `/loop` 启动，每次唤醒完成一轮：读历史、改一处、评估、汇报、成功就提交、失败就恢复。RL 线的会话同时管两台 GPU 主机，每台各保持一个在途尝试；尝试编号由评估入口加锁预留，配对时重新读取当前最好版本。步骤与约束见[公共手册](handbooks/common.md)，各线的差异见[规则线手册](handbooks/rule.md)和 [RL 线手册](handbooks/rl.md)。

运行数据在本项目 `.local/`（`pool.json`、`attempts.jsonl`、冻结的尝试包与任务目录），研究网页的「迭代线」标签页只读展示。
