# 平台规则、导出与对战

[赛事公告](https://e.netease.com/community/post/165392) · [竞赛平台](https://10.240.169.190:18080/portal/) · [官方指南](https://10.240.169.190:18080/portal/api/guide) · [指南版本](https://10.240.169.190:18080/portal/api/guide/version)

赛事报名、平台 AI 身份与项目申报分别办理；赛程和实际房间配置以官方入口为准。[报名入口](https://survey.hz.netease.com/htmls/tyd1os/paper_internal.html) · [项目申报](https://e.netease.com/honor/award/150297/new)。

## 玩法与结算

**一致性验收以线上实际对局为准。** 本地训练引擎必须复现平台实际接受的动作、状态和结算；即使平台行为疑似有缺陷，也不能用理论规则、指南或无状态试算结果覆盖真实对局回执。出现冲突时，本地与线上不一致属于待修复的兼容性缺陷，相关验收保持未通过。用对照样本确定影响范围后在本地复现线上行为；不能把一个反例擅自扩大为所有相似牌型的规则。平台以后改变行为时，再按新的实测证据更新并冻结运行时。

- 136 张牌，白板为财神；仅自摸，无点炮或抢杠胡。白板不能吃、碰、杠，可以主动弃出。
- 最多两摊吃牌；七对要求无副露，包括暗杠。
- 吃碰后尚未摸牌时仍可按手牌补杠或暗杠，但不能胡牌；杠后补牌再进入胡牌决策窗口。
- 爆头通常按任意摸牌均可胡判断，四白板可与爆头叠加；已采到的线上例外按下文“四白兼容规则”处理。财飘与杠的连续动作链叠加倍率，不满足续链条件的弃牌会断链。
- 吃前已是爆头听牌态，吃后弃白仍保持爆头时可计财飘，不要求此前已经在抓打圈内；本地从本人手牌与公开吃碰记录恢复吃前状态。
- 杠爆按杠完成后、补牌前的听牌手牌重新判断爆头（v32），不能沿用上一次出牌时的状态。一次杠与爆头叠加为 ×4，再与其他番型叠加。
- 普通摸牌与暗杠、补杠、明杠后的补牌都允许弃胡：补牌可胡仍停在本人 `phase="draw"` 的决策窗口，可主动胡、续杠或按合法动作弃胡出牌；可胡但出牌超时则自动胡（v33）。
- 弃财神后进入抓打圈，其余三家只能弃刚摸牌，禁止吃、碰、明杠，仍可暗杠或自摸。打财神者本人豁免，可吃、碰、明杠、补杠并任意出牌；圈内吃碰后再弃财神则财飘链 +1 并重启圈，弃其他牌则断链并解除圈。
- 抓打圈标记保留到打财神者的下一次摸牌与决策窗口，本人始终豁免；不能在本人摸牌时提前清除 `catch_play` / `god_discarder_seat`，否则本地训练的策略观察与线上快照不同。本人出非财神牌才解除圈，再出财神则重启圈。
- 胡牌者成为下一局庄家，流局庄家不变；保留牌墙 20 张，此时不能杠。
- 设底分为 B、综合倍率为 F：庄家胡，其余各付 8BF；闲家胡，庄家付 8BF，其余各付 BF。

番型、倍率和状态转换的精确条件参见官方指南及本地 [rules.py](../../src/mahjong_ai/domain/rules.py)、[engine.py](../../src/mahjong_ai/domain/engine.py)。原文冲突排查与番型响应样本见[原始资料](../sources/README.md)。

本地引擎与平台适配器已实现抓打圈豁免及圈内续飘；验证范围与尚待实战验证的路径见[开发索引](development.md)。

策略观察使用统一事件序列：只保留摸、打、吃、碰、杠和局结算，排除配牌、庄家初摸与网络 pass/timeout。完整网络事件仍保存在外部对战日志中。暗杠事件带隐蔽标识，非本人视角隐藏牌面；副露来源从公开吃碰记录恢复，历史缺失时保留未知。房间底分通过 `Observation.base_score` 传入假想模拟。

## 一致性验证与兼容边界

本地 `calculate_fan` 从当前手牌扣除刚摸牌后计算爆头，不保存上次出牌时的爆头缓存；`HandState._draw` 在杠后补牌时同样保留决策窗口。这些实现覆盖指南描述的 v32/v33 核心流程，开关组合仍须实战核对。三种杠、普通杠开与杠爆、弃胡续飘/续杠、超时胡及平台快照投影的合成回归见 `tests/test_kong_decisions.py`；这些用例不是官方线上采样。

**指南与实测冲突：** 指南将杠开列为“有财必拷响”的豁免，但线上在开启该开关、手持白板、非爆头的明杠或补杠后会拒胡，超时也弃牌；已有截止前拒胡回执及完整牌谱佐证。本地按实际行为统一为“持白必须爆头”，杠后补牌不豁免，Python 与 C 计算保持一致；关闭开关、无白板或爆头时仍按原规则胡牌。暗杠采用同一门槛，组合条件由合成回归覆盖，尚无对应线上拒胡样本。官方反例见 `tests/fixtures/official-v34-youcai-rejected-hu.json`，回归见 `tests/test_youcai_parity.py`。实际已执行动作和结算能够重放，不能单独证明合法动作集合完全一致。

**四白兼容规则：** `tests/fixtures/official-v35-four-white-melds-mismatch.json` 保存一局关闭“有财必拷响”的真实牌谱：两组副露，摸前为三张 3 筒加四白，摸入 1 筒；线上只计四白、2 番。本地 Python 与 C 按已采样的窄牌形处理：摸前手牌为“四白＋一副自然刻子”、有两组副露时不计爆头，仍保留四白倍率。对应实战回归要求重放到官方 2 番及各家分数。

该兼容分支不适用于所有“四白加副露”：闭手、其他手牌结构，以及刚摸入第四张白的原判定保持不变。`tests/fixtures/official-v35-four-white-controls.json` 保留线上正向控制，包括持四白的闭手/单副露结算，以及双副露摸到第四张白时的爆头快照和随后财飘结算。此分支是基于实战反例的兼容处理，并非已查明服务端内部根因；反例赢家的实时 `god.baotou` 未留存，完整状态边界仍需同类线上快照补充。指南和闭手试算不能替代这些实战验证。

## 排名与策略目标

海选、半决赛和决赛共享麻将玩法。局数、并发与底分等房间配置在锦标赛级别只有一份（`tournament.config`），各阶段共用；二测四个阶段均为 `Rounds: 16`、`M: 10`，每人每阶段 10 场。非决赛排序为总分、名次分、白板数、稳定 ID；场名次分为 +3、+1、−1、−3，同分平均。决赛按总分，同分加赛并累计。自动匹配房由服务端按默认 M=10、Rounds=8 建房，`POST /api/match` 请求体里的 `Rounds` 只是可承受上限，不能要求 16 局；自建测试房可以自定局数。

本地训练与评估的主口径是完整结算的场均桌面分，与非决赛第一排序键一致，场均名次分只在同分时比较（[ADR-0010](../adr/0010-table-score-as-primary-evaluation-metric.md)）。评估每场局数跟随锦标赛配置，当前为 16 局。研究分支连续迭代，每次面对最近一次共同大乱斗的前三名做分段配对评估，累计晋级后全体现役代表再共同大乱斗，详见[研究手册](strategy-agent.md)。

## 冻结策略与外部记录

从研究导出完整候选 ID，使用导出的 `policy.json` 和对应 runtime 运行。外部测试房、自由对战与正式比赛各自建立新的输出目录；自由对战与测试房的赛后牌谱可以作为人工项目的训练数据（[ADR-0015](../adr/0015-platform-replays-as-training-data.md)），自动研究候选不读取平台记录；正式锦标赛没有赛后牌谱。命令入口位于 `platform/` 模块，脚本递归冻结全部模块。

源码中的引擎修复不自动更新旧候选、已导出的策略或远端冻结运行时。采用新规则需创建新实验或新发布包，核对其 `ENGINE_VERSION` 与运行时指纹；旧实验及原始对局证据保持不变。

## 对战操作

从门户取得令牌，保存在 `.local/credentials/`。测试房使用绑定房间的 scoped 令牌，自由对战使用门户 AI 身份的 global 令牌；匿名自注册不可用。请求使用 Bearer 认证。测试房凭据结构为 `{"tokens":["TOKEN_1","TOKEN_2","TOKEN_3","TOKEN_4"]}`，自由对战为 `{"token":"TOKEN"}`，占位符替换为实际令牌。

测试房四个策略依次传给 `--policies`，每个策略可用内置名称或导出的 `policy.json`：

```bash
.venv/bin/python scripts/run_platform.py \
  --credentials .local/credentials/test-room.json --insecure \
  --policies rule rule rule rule \
  --seconds 1200 --output .local/platform/test-01
```

自由对战使用选定的冻结策略：

```bash
.venv/bin/python scripts/run_free_play.py \
  --credentials .local/credentials/free-play.json \
  --policy rule --insecure \
  --seconds 1800 --output .local/competition/free-01
```

正式赛与测试赛使用门户报名后下发的参赛令牌（scoped，凭据 `{"token":"TOKEN"}`），开赛前启动：

```bash
.venv/bin/python scripts/run_tournament.py \
  --credentials .local/credentials/tournament.json \
  --policy .local/platform/entrant-<候选 ID>/policy.json \
  --insecure --output .local/competition/runs/<赛事>
```

参赛进程报名并确认到位，之后持续轮询以保持「在线」，打完分配的全部对局，赛事结束后退出；中途崩溃用同一输出目录重启即可续打。参赛令牌与自由对战的全局令牌属于同一平台用户、共享限流，开赛前须停下自由对战。

脚本冻结 runtime 并保存对局与审计结果。相同输出目录用于恢复，已完成对局不重复；新比赛使用新目录。`--insecure` 适用于平台自签证书。

规则审计与客户端执行审计分别验收。`response_window_expired` 与我方座位的 `timeout(response)` 表示错过响应窗口；错过通常按过牌处理，即使最终结算重放通过，也不能据此判定执行通过。单独统计我方窗口错过率（我方 `timeout(response)` ÷ 我方全部窗口）、出牌超时代打、被拒动作及尚未解释的拒绝。按并发数、开窗方式和开局阶段拆分：吃碰后紧接出牌的窗口与常规出牌窗口需要分别比较。具体测量结果随运行证据保存，不作为其他房间的性能保证。

改动客户端后，在目标并发及相同响应时限下验证；用请求时间、快照和回执区分客户端延迟、窗口已失效、其他玩家抢先声明与真实规则差异。对局记录保存在运行目录的 `game-<id>.json`、`.jsonl` 与 `.fetches` 中。

```bash
.venv/bin/python scripts/inspect_platform_run.py .local/platform/test-01
.venv/bin/python scripts/audit_platform_run.py .local/platform/test-01 \
  --output .local/platform/test-01/audit.json
.venv/bin/python scripts/audit_free_play_run.py .local/competition/free-01 --insecure
```

凭据文件结构、策略参数和导出选项以 [live_workflow.py](../../src/mahjong_ai/platform/live_workflow.py)、[free_play.py](../../src/mahjong_ai/platform/free_play.py) 及相应 CLI `--help` 为准。开放研究候选接入平台前须核对加载/导出兼容性；研究运行完成不代表候选已部署。

## 协议要点

| 场景 | 处理方式 |
| --- | --- |
| 确认身份与配置 | 读取 `/api/me` 与房间 rules；响应窗口、并发 M、每场局数 Rounds 从配置读取 |
| 正式阶段 | register、ready 后参与；新的 `stage_open` 需要重新 ready。报名只是意向，不占对局额度，不能在客户端按报名数量扣减可用场次。海选开赛那一刻冻结名单：已报名、已到位，且开赛前 90 秒内有过已认证请求，才算参赛；不在线就算未到位、不占桌，平台不等待。参赛进程必须在开赛前启动并保持轮询。海选分 M 批同时开打，每批随机同桌，余数进补桌池，由系统代打补满，每人恒打满 M 场 |
| 自由匹配 | 仅 `POST /api/match`，不 register/ready；服务端优先按近期战力相近匹配，无需客户端选房。结果未知先查 me，避免重复匹配。409 `MATCH_BUSY` 有两种可重试的触发：自动房已满，或入席结果未确认（此时席位可能已落定）；客户端延时后重调 `/api/match`，已入席时幂等返回原房，不会另开第二间 |
| 并发 | 每个 game 独立状态和游标，同一用户共享限流；自由匹配准入按正在进行的对局数 + 新房 M ≤ 16 判断，`MATCH_LIMIT_REACHED` 只来自 `/api/match`，报名不占额度。当前 state 请求按 14/s 控制（官方上限 16/s，实测 15/s 已会触发 429） |
| 窗口时序 | 碰窗期间记录应答；有人碰可提前结束窗口，碰者随即出牌，下一帧可能已是新窗口。无人碰时，到期为未应答者记录 `timeout`，再开启下家的吃窗；吃窗开启本身可能没有事件，不能只等 SSE。客户端本方为下家时，须按响应截止时间主动重拉；具体窗口时长从房间配置读取 |
| 唤醒与拉取 | 每桌订阅 `GET /api/games/{id}/notify` SSE 流（只带 seq 水位，不占 state 额度）作为唤醒信号，仍以 `state?seq=0` 全量快照为准。本方动作被接受后由 `ActionTimeline` 按上表时序预测每一帧：截止前的单个事件视为他人应答（整窗最多跳过一个，先等 80 ms 看是否有碰后紧随的出牌），两个及以上一起到达的事件立即拉取；截止时刻等 0.25 s 收齐超时爆发，爆发帧数超过尚未应答人数才拉取；之后下家吃窗应答跳过一次、按定时到达的下家摸牌跳过一次，再来的帧就是开窗出牌。凡是可能为认领的帧（截止前被当作应答的单帧、吃窗时段被跳过的单帧）都记下标记，其后真正到达的下一帧一律拉取：认领之后紧跟的出牌会开我方窗口，而普通应答之后要等到定时摸牌才有帧，因此这条不会凭空增加拉取。截止后到达的帧数与尚未应答人数不一致且不为零时也拉取，覆盖碰抢先结束窗口、超时帧少于预期的情形。本方为下家时截止后每帧都拉，并在截止后 0.12 s 与 0.4 s 定时检查静默开启的吃窗。与服务器的时钟偏差用本方出牌开出的窗口截止时间校准。限流 14/s 分高中低三级，中、低优先级分别等待 0.25 s、0.5 s 后自动晋级，避免被其他桌的本方回合请求饿死；局间 5 秒暂停只做低优先级定时轮询。策略未按时给出动作时，窗口剩余 0.4 s 以上则强制刷新后重试最多 2 次。每桌 `game-<id>.json` 的 `client` 字段记录帧数、各优先级拉取数、截止检查、跳过、等待超时、策略超时与重试、时钟偏差等计数，`game-<id>.fetches` 逐次记录拉取的触发原因、帧跳变、优先级、时延与结果，`.jsonl` 每条动作带 `ts`、`priority`、`fetch_ms`、`limiter_wait_ms`、`wake_to_post_ms`、`window_left_ms`。传输按每线程一条长连接复用（`http.client`），GET 在连接被对端回收时重连重试一次，提交不重放。`/state` 以外的请求不进入本地拉取预算；429 由每桌 `client.throttled` 与 `.fetches` 的 `throttled` 字段记录。只在房间未开打时为线程池预热连接，已开打则直接拉取对局。保留逐次时延记录诊断偶发停顿，延迟敏感的对战应与训练分机 |
| 事件恢复 | `seq=0` 重建；游标断档或 409 后重拉状态。补拉历史获得新快照时，同步采用该响应的 `seq`、局号和快照，并推进时间线水位，避免新快照记在旧序号下。回归见 `test_history_refresh_keeps_snapshot_sequence_and_round_together`。碰转吃可能无新事件，不能仅依赖事件长轮询 |
| 瞬时错误 | 赛事端点的 404 按响应体的 code 判型（v35）：`TOURNAMENT_GONE` 是房间暂时不可达（开赛、结算瞬间最常见），平台客户端对所有请求自动退避重试；`TOURNAMENT_NOT_FOUND` 或缺 code 表示房间不存在，放弃。参赛循环对 `TOURNAMENT_GONE`、429 与 5xx 同样视为瞬时错误，2 秒后重试，连续失败超过 5 分钟才退出；报名与出席确认每 30 秒重发一次，409 视为已完成。适配器抛快照错误先强制刷新历史与快照重试 3 次再判失败 |
| 合法动作 | 平台无 `allowed_actions`，适配器结合 stage、turn、responding_seats、drawn_tile 生成；state pending 不代表局势未变化 |
| 抓打圈豁免 | 适配器将 `god.god_discarder_seat` 传入观察，无圈时为 -1；`catch_restricted` 等价于 `catch_play && god_discarder_seat != seat`。圈内非财神出牌仍对豁免方开启固定时长响应窗口，不能因圈中或没有可吃碰的牌而跳过轮询 |
| 分数 | `round_ended.data.scores` 为本局增量，`game_ended.final_scores` 为累计分 |
| 终局留存 | 自动房间完成后约 60 秒清理，及时保存终局快照与牌谱 |
| 局间停顿 | 每局结算后固定停 5 秒才发下一局（v31，时长服务端写死）。窗口内任意 seq 的快照是上一局终态：`phase="settled"`、`round_no` 仍是刚结束那一局、`my_hand`/`hand_counts` 是上一局残牌，提交动作一律 409 `INVALID_ACTION`。按 phase 判定可操作性、等下一局全量快照，不用该手牌计算；轮询循环无需改动，每场 8 局因此多约 40 秒墙钟，`--seconds` 预算留余量 |
| 功能开关 | 管理员可在正式比赛期间关闭自由匹配与自建测试房（v29）。`POST /api/match` 新匹配、`POST /portal/api/test-rooms` 建房、finished 测试房的跨轮重开返回 403 `FEATURE_DISABLED`——永久条件，不重试（跨轮重开此前为 409，按 code 判型而非状态码）；在途对局、房内幂等重调与已建房管理不受影响；门户 `GET /portal/api/features` 可预检 |
| 姓名语义 | 正式锦标赛之外的一切姓名字段只下发 AI 昵称，昵称为空给空串（v30）。name 仅作展示，稳定身份一律用 `user_id` |
| 门户今日榜 | v34 的 `/portal/api/leaderboard` 新增 `last`：仅今日榜且上榜人数大于 32 时为 `{rooms, firsts, score, is_me}`，否则为 `null`；不包含身份。本项目不消费此门户榜单，无需接入，也不改变本地研究排名 |

测试反馈用于框架校准与回归；平台记录不进入研究 MCP。自由对战与测试房的牌谱由人工项目的 `guided.platform_dataset` 入库、还原与冻结，用法见[训练手册](../../projects/personal/selfplay/guided/README.md)。完整阶段身份映射、最高并发和响应窗口仍需专项线上验证，当前能力边界见[开发索引](development.md)。

自定义快照映射必须包含 `god_discarder_seat`。圈中缺少或包含无效豁免座位时拒绝生成动作；旧版无圈快照允许省略该字段，由官方适配器补为 -1。
