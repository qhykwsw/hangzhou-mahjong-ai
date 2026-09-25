# RL 线手册

先读[公共手册](common.md)。本手册只写 RL 线的差异，训练方案接口见 [recipes/API.md](../../selfplay/recipes/API.md)。

## 本线参数

| 项目 | 内容 |
|---|---|
| 可以修改 | 只有 `rl/recipe/`（训练方案包，起点是 selfplay 基线方案的副本） |
| 尝试命令 | `python -m harness.rl attempt "<改动>" --design-start <epoch> --host 4070|4090|5070ti`：在该主机训练，再在该主机评估，最后打印汇报块 |
| 模型 | 最小规格 128/4/4（Actor 约 0.84M 参数）。起点是它对 piao-call-channel 模仿 10 万局（每场 16 局）的结果 |
| 训练预算 | 按局数：每次尝试训练 40 万局（`harness/rl.py` 的 `HANDS`，环境变量 `FP_RL_HANDS` 覆盖，0 表示改按时间），三台主机训练量相同，只看改动的趋势，放大规模留到最后。时间上限 `FP_RL_BUDGET`（默认 4 小时）只防止卡死 |
| 主机 | 4070、4090 与 5070 Ti 各自训练并就地评估，本线会话每台保持一个在途尝试（三个尝试并行）。4070 是办公机，显卡同时负责桌面，`harness/rl.py` 的 `LIMITS["4070"]` 为桌面留了 CPU、显存和内存余量，不要放宽；4090 与 5070 Ti 的上限分别在 `LIMITS["4090"]`、`LIMITS["5070ti"]`，5070 Ti 同样为桌面留出余量。Mac 不训练。训练对局用对手池的冻结运行时（不是教师包自带的那份），对手包在采集进程内运行 |
| 回退 | `restore rl` 用当前最好尝试保存的方案副本（`.local/rl/attempts/<编号>/recipe`）覆盖 `rl/recipe/` |

## 起步（每台主机的 `.local/rl/setup-<主机>.json` 存在就说明已完成；4070 的旧文件名是 `setup.json`）

1. 确认小模型模仿已完成：启动脚本和运行目录在 `projects/personal/selfplay/.local/imitation/20260924-piao-small/`，运行 `control.py status 4070` 查看状态。
2. 第一台主机用模仿的 `final.pt` 执行 `python -m harness.rl setup <远端路径> --name <起点名> --host <主机>`，它会把 RL 起点放到该主机，同时在 Mac 冻结一份；其他主机用同一个起点的权重 sha 执行 `setup <sha> --name <起点名> --host <主机>`，起点从该主机镜像的模型库复制过去。
3. 冒烟：
   - 把 `.local/pool.json` 和 `.local/rl/setup-<主机>.json` 复制到一个单独的 `FP_LOCAL` 目录（必须在仓库目录之内，远端任务目录按仓库相对路径镜像）。
   - 设 `FP_RL_BUDGET` 为几分钟，再缩短 `FP_STAGES` 和 `FP_CONFIRM`，然后运行一次尝试命令。
   - 确认三件事：训练回执里有局数；4070 能同步到对手池里的大模型；评估锁生效。
   - 跑完删掉这个目录。
4. 冒烟通过后，提交工作区里尚未提交的框架与文档改动，逐个写路径，不用 `-A`。
5. 第一次正式尝试用基线方案原样（v0）。

## 注意

- 冻结的训练框架源码在 `.local/rl/source`。修改 `projects/personal/selfplay/guided/` 后，先按 `source-manifest.json` 的 id 把这个目录归档到 `.local/rl/sources/<id>`，再对每台主机重新 `setup`，让框架重新冻结上传。
- 预算按局数，三台主机的尝试训练量相同，可以直接比较；各台按实际吞吐完成相同局数。
- 往 4070 上传教师包时，清单里登记的文件要全部带上，包括 Mac 的 `.so`，否则 `read_bundle` 会拒绝。每个运行时目录都要在 4070 上单独编译 `_fast` 扩展。
- 对手池里的大模型是 384/8/8。4070 评估时，对局执行器会从 Mac 的模型库自动同步它，不要手动往其他主机拷贝模型。
- 采集的执行参数在方案配置里：`tables`（每个 worker 同时打的桌数）与 `gather_seconds`（父进程有一轮推理在设备上时再等这么久攒批）。它们只改吞吐不改训练内容；默认每个 worker 开 32 桌；只增加桌数，不增加进程数，按实测吞吐与桌面余量选择。
- 学习步的 `resident_rows` 将本轮输入按列驻留 GPU，供两折 Critic、Actor 梯度与 KL 检查复用；样本顺序、历史补齐、计算精度与更新次数保持原方案。Actor 批次不包含 oracle 字段。驻留遵守原 GPU 上限，并预留反向传播空间；空间不足时沿用逐批搬运。该配置可关闭以做同工作量对照。
- 除了 `rl/recipe/`，本线还负责 `harness/rl.py` 和 RL 需要的 `projects/personal/selfplay/guided/` 框架。改动这些不算一次尝试，只在确有必要时进行，并告诉用户。
