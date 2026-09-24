# 第一性原理迭代线（firstprinciples）

维护者亲自推进的两条迭代线，设计与取舍见 [ADR-0016](../../../docs/adr/0016-maintainer-iteration-lines.md)，术语（迭代线、尝试、固定对手池）见 [CONTEXT.md](../../../CONTEXT.md)。

- **规则线**：`rule/strategy.py` 是唯一被修改的文件。v0 是最朴素的牌效策略，之后按杭州麻将规则的第一性原理逐次改进，大方向是速度优先、必胜时追求翻倍。
- **RL 线**：沿用训练方案框架（`../selfplay/recipes`、`guided.trial`），起点是 piao-call-channel 的模仿模型；每次尝试在 4090 上训练 1 小时。

## 评估

`harness/evaluate.py` 让一次尝试面对固定对手池（6 个规则版本加 1 个模仿模型）：每局被评估版本坐一个座位，其余三座取对手池的三个，35 种阵容循环、四座轮换，牌墙与各方种子只由局组编号决定，所以开发集对每次尝试都是同一批牌局。规则线在 Mac 本机评估，RL 线在训练它的 4090 上就地评估（模型不经中继传输），每台机器同一时间只跑一个评估；复用研究的对局执行器与冻结运行时，时限为出牌 2.0 秒、应答 0.6 秒，超时按平台默认动作代打。

- 开发集阶梯：4096 → 8192 → 16384 场，低于当前最好成绩两倍误差以上即停。
- 确认集：开发集超过最好成绩后，换全新牌墙与最好版本各打 8192 场，配对差单侧 90% 下界大于 0 才算成功。

```bash
cd projects/personal/firstprinciples
../../../.venv/bin/python -m harness.evaluate pool --model <weights-sha> --name <模型名>   # 冻结对手池（一次）
../../../.venv/bin/python -m harness.evaluate reference <研究候选 ID>                       # 图上的参照线
../../../.venv/bin/python -m harness.evaluate rule "<改动>" --design-start <epoch>
../../../.venv/bin/python -m harness.evaluate model <actor 目录> "<改动>" --line rl
```

`FP_LOCAL`、`FP_STAGES`、`FP_CONFIRM` 用单独的目录与缩短的局数做流程检查，不进正式记录。

## 迭代纪律

- 串行：等上一次尝试的评估结果出来，再根据反馈设计下一次改动；每次设计加实现不超过 10 分钟，耗时记入日志。
- 每次只改一处，每次尝试都记录，失败也记；只继承成功的尝试。
- 除了沿当前方向改进，也有意尝试从未用过的观察和动作空间。
- 每次尝试结束向用户汇报：改了什么、结果如何、必要的数据与排名。

运行数据在本项目 `.local/`（`pool.json`、`attempts.jsonl`、冻结的尝试包与任务目录），研究网页的「迭代线」标签页只读展示。
