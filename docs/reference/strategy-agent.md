# 策略研究与 Agent 接口

每个研究方向拥有一个独立、持久的 Codex 会话。Agent 阅读本方向历史，修改候选，一次提交 1 至 4 个，然后结束当前回复。控制器在整批完成当期比赛后继续同一个会话。候选源码、模型和父代只在本方向内使用，比赛成绩跨方向可见。

## 启动与远程访问

在 MacBook 的项目根目录执行；从 gpu-4070 通过 SSH 进入 MacBook 后使用相同命令。

```bash
.venv/bin/mahjong start
.venv/bin/mahjong status
.venv/bin/mahjong stop
```

也可用 `mahjong --project /path/to/project start` 指定项目。`start --no-open` 只输出链接；检测到 SSH 会话时也会跳过本机浏览器。

服务优先使用 8765 端口，冲突时自动选择可用端口，以终端输出为准。启动时已连接 Tailscale，会同时监听本机回环与 Tailscale IPv4 网卡，输出类似下面的地址。

```text
http://127.0.0.1:8765
Tailscale: http://<MacBook-Tailscale-IP>:8765
```

在 MacBook 本机打开第一条，在同一 tailnet 的 gpu-4070 打开第二条，无需 SSH 端口转发。`status` 返回实际监听的 `url` 与 `tailscale_url`；后者为空时尚未启用远程监听。若启动服务后才连接 Tailscale，先停止再启动服务以增加远程监听。

首次启动建立默认研究并开始计时；后续启动打开已有记录，不重新发放预算。每项目最多一个运行中的研究。网页中先暂停当前研究，再创建或继续另一条记录；查看历史记录不改变运行状态。

## 目录与命名

新 run 使用 `.local/runs/YYYYMMDD-arena-NN/`，日期为创建日，同日序号自动分配，续跑保持名称。目录保存冻结 runtime、方向模板、候选、提交工件、作业和报告；状态、事件与比赛回执保存在 `.local/service/arena.sqlite`。保留或备份研究时，两者都需要保存，单独复制 run 目录不能恢复网页中的记录。

服务由 macOS 用户域 launchd 持有，服务配置、地址和日志保存在 `.local/service/`。其中 `service.log` 用于排查启动失败；研究过程和候选准备错误从网页的会话日志、全部候选记录中查看。服务不会注册为登录启动项，停止服务会暂停计算并保留已有进度，再次启动后须在网页继续。

后台服务与网页使用当前仓库代码；研究作业和 MCP 使用创建 run 时冻结的代码。更新仓库后，服务端改动需重启服务，新候选执行器或研究协议需新建 run 才会生效。重启服务不会升级旧 run。

旧 v1/v2 run 仅作为只读历史档案，新服务不续跑或迁移它们。其源码、模型、报告和路径兼容链接保留，历史文档只描述当时的协议。

## 预算与资源

默认预算 60 分钟，包含模型思考、准备、比赛和等待。暂停期间不计时。预算耗尽时计算暂停，网页保持可访问；增加预算后仍需点击「继续」。Agent 无预算控制工具。

默认同时运行一个方向会话，每次最多五分钟；候选准备与比赛共享两个计算 worker。每批候选准备也受单次时间上限约束，涵盖依赖安装、训练和加载检查，配置见 [research.toml](../../research.toml)。暂停会取消当前作业，续跑重做未完成的完整结算，已落盘回执不重复计分；训练从最近已提交检查点恢复。

## 混合对战

以下为默认配置。四个普通种子选手启动，和新候选一样可以被淘汰。池容量最多 16 位，超过 8 位时才淘汰，淘汰后至少保留 8 位。每期冻结阵容并完成 128 局四人比赛，每局一场八手。调度先选择出场较少的选手，再尽量减少重复对手，随机打破平局；每组四人轮换四个座位。

最终名次对应效用 3、2、1、0，只按当期平均效用排序。同分时出场次数多者优先，仍相同则按稳定 ID 升序排列。每期归档末尾 `min(floor(N/4), max(0, N-8))` 位，历史及源码继续可读。排行榜表达相对当期阵容的成绩，不跨期累计排名。

每期最多接纳四个新候选，方向轮流优先，整批入场。冻结内容哈希相同的候选标记为重复，记录对应的既有选手，不占新的入场位置。准备与等待入场的候选默认最多 16 个；没有新候选时等待研究，不反复运行同一阵容。

候选加载失败会在入场前标记失败。批次全部失败或重复时直接返回研究方向修正；批次仍有新候选时等待整期比赛结果。比赛中的运行错误、非法动作和超时保留诊断并采用环境默认动作，实际结果照常计分，不附加淘汰条件。默认弃牌窗口三秒、响应窗口一秒。响应超时过牌；自己回合能胡则胡，否则从真实手牌插入顺序末尾选择合法弃牌。迟到回复会丢弃，正常完成的 worker 可继续响应新观察。

## 配置研究方向

在 `directions/<id>/direction.toml` 中配置名称、研究建议、模板和训练局数，例如

```toml
name = "搜索策略"
brief = "研究基于合法公开观察的浅层搜索与风险估计。"
template = "template"
training_games = 64
```

候选初始源码放在该方向的 `template/strategy.py`；模板目录不存在时使用普通规则选手。把方向 ID 加入 `research.toml` 的 `directions`，或者在网页新建研究时勾选。配置只影响新 run。新增或修改方向由框架开发 Agent 操作文件，网页没有方向编辑器。

方向建议不构成算法类型限制。任何方向都可以编写规则、搜索、神经网络或其他算法；是否训练由候选的 `train(ctx)` 能力决定。

## 候选契约

完整契约以 [research_templates.py](../../src/mahjong_ai/research_templates.py) 的 `CONTRACT` 和该 run 的 MCP schema 为准。`strategy.py` 提供 `build(model_dir)`，返回实现 `reset(seed)`、`act(observation)`、`finish_round(result, seat)` 的对象。动作必须来自 `observation.legal_actions`。

可选 `train(ctx)` 可生成模型、查表等任意文件。`ctx.play(learner)` 请求真实完整结算，`ctx.episodes` 限制可用训练局数。`ctx.output` 存训练产物，`ctx.initial` 指向父代模型；`ctx.checkpoint(metadata)` 原子提交恢复点，`ctx.completed_episodes` 和 `ctx.checkpoint_metadata` 提供续训上下文。没有真实对局的离线产物生成也允许。

训练对手从当时的冻结混合池抽样。候选只接收合法观察、独立策略随机种子和清理后的结算，不接收环境随机种子、未来牌墙、真实隐藏手牌或平台信息。

依赖按精确 `name==version` 声明，只下载 wheel 并安装到候选独立环境。候选执行禁用网络和宿主私有文件访问。训练产物和依赖指纹随冻结候选保留，提交后要修改必须派生新草稿。

| 用途 | 工具 |
| --- | --- |
| 状态与完整比赛历史 | `research_status`、`research_history` |
| 契约与公开 SDK 源码 | `research_read_source` |
| 本方向候选派生和读写 | `research_create_candidate`、`research_read_file`、`research_write_file` |
| 精确 wheel 依赖 | `research_set_dependencies` |
| 冻结一批候选并结束本轮 | `research_submit_batch` |

Agent 没有原生终端、其他 MCP、子 Agent、任意工作区执行或独立验收工具。研究理由保存在候选 hypothesis、自己的会话和历史比赛结果中。

## 导出与平台隔离

```bash
.venv/bin/mahjong-ai export --root .local/runs/<run-name> \
  --candidate <direction--candidate> --output /path/to/bundle
```

导出包含候选、运行时、策略描述与依赖清单。迁移机器时按清单准备依赖，使用导出 runtime 运行平台适配器。外部比赛单独建立记录，不回流给研究 MCP，也不改写本地积分。平台操作方法见[平台规则与接入](platform.md)。
