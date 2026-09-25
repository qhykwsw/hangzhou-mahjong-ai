# 第一性原理迭代线（firstprinciples）

两条由迭代 Agent 推进的策略改进序列，设计与取舍见 [ADR-0016](../../../docs/adr/0016-maintainer-iteration-lines.md)，术语（迭代线、尝试、迭代 Agent、维护者、固定对手池）见 [CONTEXT.md](../../../CONTEXT.md)。

- **规则线**：`rule/` 是唯一被修改的目录。v0 是最朴素的牌效策略，之后按杭州麻将规则的第一性原理逐次改进，大方向是速度优先、必胜时追求翻倍。
- **RL 线**：沿用训练方案框架（`../selfplay/recipes`、`guided.trial`），用最小的模型（128/4/4，约 0.84M 参数）。起点是它对 piao-call-channel 做 10 万局（每场 16 局）的模仿，每次尝试训练 100 万局，在 4070、4090 或 5070 Ti 上进行。

## 评估

`harness/evaluate.py` 让一次尝试面对固定对手池（6 个规则版本加 1 个模仿模型）：每场被评估版本坐一个座位，其余三座取对手池的三个，35 种阵容循环、四座轮换，牌墙与各方种子只由局组编号决定，所以开发集对每次尝试都是同一批牌局。规则线在 Mac 本机评估，RL 线在训练它的 GPU 主机上就地评估，每台机器同一时间只跑一个评估；复用研究的对局执行器与冻结运行时，时限为出牌 2.0 秒、应答 0.6 秒，超时按平台默认动作代打。

对局不用沙箱：对手池的代码包是维护者自己的，对局进程直接在进程内运行它们（`InlinePolicy`），每个进程同时打几桌（一桌一个线程，模型座位等推理服务时其他桌继续），模型座位共用主机的推理服务。并发数属于按主机设置的执行参数（`EVAL_WORKERS`、`EVAL_TABLES`），不进对手池 id；运行时用 `harness.evaluate runtime` 从 `src/mahjong_ai` 冻结，换运行时就要重建对手池并重跑参照。对局任务目录按池 id 隔离（`.local/jobs/<池 id>/<尝试>/`）：中断后续跑会复用已完成的任务，换池后绝不会把旧池的结果当成新结果。模型座位的线性层以 fp16 推理，归一化与残差保留 fp32。接近同分的动作可能受设备、后端或批次形状的浮点舍入影响，不能保证跨配置逐场得分完全一致；更换推理实现也要冻结新版本并重跑参照。

- 开发集：每个版本（尝试、参照、RL 起点）都打满同一批 16384 场，一个版本只有一个成绩；同一版本有多条记录时，图、排名和 `status` 只显示场数最多、最新的那条。
- 确认集：开发集超过最好成绩后，换全新牌墙与最好版本各打 8192 场，配对差单侧 90% 下界大于 0 才算成功。
- 新的最好版本还会在开发集上与自由对战的上场版本配对，下界大于 0 时汇报「高于参照线，可以上场」，由用户决定是否换上。

## 迭代方式

用户在 Claude Code 或 Codex 里说「开始/继续执行规则迭代」或「开始/继续执行 RL 迭代」，那个会话就成为该线的迭代 Agent，按[公共手册](handbooks/common.md)循环：`harness.line wait` 带回汇报和下一步，`harness.line start` 开跑一次尝试。尝试在脱离会话的驱动进程里运行，状态全部在 `.local/running/<编号>.json`；驱动进程死了（会话关闭、Mac 重启），下一次 wait 会把它重新拉起，从原来的阶段接着做。同一条线同一时间只有一个会话持有推进权。

RL 线最初的最好版本就是模仿起点本身，三台主机从一开始就能各朝一个方向改；三个尝试位共享一个最好版本。主机离线时尝试等它回来；主机重启丢了训练，尝试排队由任一空闲主机重做。用户可以随时关闭或开启某台主机。

## 维护者

迭代 Agent 只改 `rule/` 或 `rl/recipe/`。下面这些是维护者的工作，每次改完都要重新封存（`seal`），否则迭代会拒绝运行：

```bash
cd projects/personal/firstprinciples
../../../.venv/bin/python -m harness.evaluate runtime                                       # 把 src/mahjong_ai 冻结为运行时
../../../.venv/bin/python -m harness.evaluate pool --model <weights-sha> --name <模型名> [--runtime <目录>] [--replace]  # 冻结对手池
../../../.venv/bin/python -m harness.evaluate reference <研究候选 ID | 池内模型名> [--weights <sha>] [--host local|4070|4090|5070ti]  # 参照线
../../../.venv/bin/python -m harness.rl export <远端 checkpoint> --name <模型名> --host <主机>  # 导出别处训练的模型（如对手池成员）
../../../.venv/bin/python -m harness.rl setup <主机上的 checkpoint | 权重 sha> --name <起点名> --host 4070|4090|5070ti  # RL 起点（每台主机一次）
../../../.venv/bin/python -m harness.rl baseline [--host 4070]                                   # 起点的完整开发集成绩，作为 RL 线最初的最好版本（每个对手池一次）
../../../.venv/bin/python -m harness.evaluate seal                                          # 接受当前的评估条件
```

- 封存覆盖 `harness/` 全部代码（评估、训练预算与按主机的执行参数、循环）、对手池，以及每台 RL 主机的训练框架快照与起点。
- 换了对手池就要在新池下重跑参照（上场版本、对手池里的模型）和 `harness.rl baseline`（RL 起点，它同时就是模仿小模型的参照）；没有起点成绩时 RL 线的迭代会停下。
- 修改 `projects/personal/selfplay/guided/` 后：把 `.local/rl/source` 按其 `source-manifest.json` 的 id 归档到 `.local/rl/sources/<id>`，再对每台主机重新 `setup <起点 sha>`，最后 `seal`。
- 往 4070 上传教师包时，清单里登记的文件要全部带上，包括 Mac 的 `.so`，否则 `read_bundle` 会拒绝；每个运行时目录都要在主机上单独编译 `_fast` 扩展。对手池里的大模型是 384/8/8，评估时由对局执行器从 Mac 的模型库自动同步，不要手动往主机拷贝模型。
- 流程检查：把 `.local/pool.json`、`.local/seal.json` 和 `.local/rl/setup-*.json` 复制到仓库内一个单独的目录，设 `FP_LOCAL` 指向它，再用 `FP_DEV`（开发集局组数，每组 4 场）、`FP_CONFIRM`、`FP_RL_HANDS`、`FP_RL_BUDGET` 缩短场数与训练预算，跑一遍 wait/start。这些缩短的设置只在 `FP_LOCAL` 下被接受，流程检查不提交 Git、不改动自由对战，跑完删掉这个目录。

## 运行数据

都在本项目 `.local/`：`pool.json`、`seal.json`、`attempts.jsonl`（每次尝试一条记录）、`running/`（在途尝试的状态）、`hosts.json`（主机开关）、`notes/<线>.md`（迭代笔记）、冻结的尝试快照与任务目录。研究网页的「迭代线」标签页只读展示。
