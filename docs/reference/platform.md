# 平台规则、导出与对战

[赛事公告](https://e.netease.com/community/post/165392) · [竞赛平台](https://10.240.169.190:18080/portal/) · [官方指南](https://10.240.169.190:18080/portal/api/guide) · [指南版本](https://10.240.169.190:18080/portal/api/guide/version)

赛事报名、平台 AI 身份与项目申报分别办理；赛程和实际房间配置以官方入口为准。[报名入口](https://survey.hz.netease.com/htmls/tyd1os/paper_internal.html) · [项目申报](https://e.netease.com/honor/award/150297/new)。

## 玩法与结算

- 136 张牌，白板为财神；仅自摸，无点炮或抢杠胡。白板不能吃、碰、杠，可以主动弃出。
- 最多两摊吃牌；七对要求无副露，包括暗杠。
- 爆头按任意摸牌均可胡判断，四白板不例外。财飘与杠的连续动作链叠加倍率，不满足续链条件的弃牌会断链。
- 杠爆按杠完成后、补牌前的听牌手牌重新判断爆头（v32），不能沿用上一次出牌时的状态。一次杠与爆头叠加为 ×4，再与其他番型叠加。
- 普通摸牌与暗杠、补杠、明杠后的补牌都允许弃胡：补牌可胡仍停在本人 `phase="draw"` 的决策窗口，可主动胡、续杠或按合法动作弃胡出牌；可胡但出牌超时则自动胡（v33）。
- 弃财神后进入抓打圈，其余三家只能弃刚摸牌，禁止吃、碰、明杠，仍可暗杠或自摸。打财神者本人豁免，可吃、碰、明杠、补杠并任意出牌；圈内吃碰后再弃财神则财飘链 +1 并重启圈，弃其他牌则断链并解除圈。
- 胡牌者成为下一局庄家，流局庄家不变；保留牌墙 20 张，此时不能杠。
- 设底分为 B、综合倍率为 F：庄家胡，其余各付 8BF；闲家胡，庄家付 8BF，其余各付 BF。

番型、倍率和状态转换的精确条件参见官方指南及本地 [rules.py](../../src/mahjong_ai/domain/rules.py)、[engine.py](../../src/mahjong_ai/domain/engine.py)。原文冲突排查与番型响应样本见[原始资料](../sources/README.md)。

本地引擎与平台适配器已实现抓打圈豁免及圈内续飘；验证范围与尚待实战验证的路径见[开发索引](development.md)。

本地 `calculate_fan` 从当前手牌扣除刚摸牌后计算爆头，不保存上次出牌时的爆头缓存；`HandState._draw` 在杠后补牌时同样保留决策窗口。因此 v32/v33 不需要改写本地计分和摸牌流程。三种杠、普通杠开与杠爆、弃胡续飘/续杠、超时胡及平台快照投影的合成回归见 `tests/test_kong_decisions.py`；这些用例不是官方线上采样。

## 排名与策略目标

海选、半决赛和决赛共享麻将玩法，场数、并发和晋级安排由阶段配置决定。非决赛排序为总分、名次分、白板数、稳定 ID；场名次分为 +3、+1、−1、−3，同分平均。决赛按总分，同分加赛并累计。

本地将完整结算的终局排名用于训练与评估，研究使用独立分支、固定对手池的完整配对评估与每轮全体共同竞技，详见[研究手册](strategy-agent.md)。策略根据实际进程和排名调整风险，不将固定局数或某一赛段写死。

## 冻结策略与外部记录

从研究导出完整候选 ID，使用导出的 `policy.json` 和对应 runtime 运行。外部测试房、自由对战与正式比赛各自建立新的输出目录；这些记录只用于平台运行与框架审计，不作为研究训练数据。命令入口位于 `platform/` 模块，脚本递归冻结全部模块。

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

脚本冻结 runtime 并保存对局与审计结果。相同输出目录用于恢复，已完成对局不重复；新比赛使用新目录。`--insecure` 适用于平台自签证书。

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
| 正式阶段 | register、ready 后参与；新的 `stage_open` 需要重新 ready。报名只是意向，不占对局额度，不能在客户端按报名数量扣减可用场次 |
| 自由匹配 | 仅 `POST /api/match`，不 register/ready；服务端优先按近期战力相近匹配，无需客户端选房。结果未知先查 me，避免重复匹配 |
| 并发 | 每个 game 独立状态和游标，同一用户共享限流；自由匹配准入按正在进行的对局数 + 新房 M ≤ 16 判断，`MATCH_LIMIT_REACHED` 只来自 `/api/match`，报名不占额度。当前 state 请求按 14/s 控制 |
| 事件恢复 | `seq=0` 重建；游标断档或 409 后重拉状态。碰转吃可能无新事件，不能仅依赖事件长轮询 |
| 合法动作 | 平台无 `allowed_actions`，适配器结合 stage、turn、responding_seats、drawn_tile 生成；state pending 不代表局势未变化 |
| 抓打圈豁免 | 适配器将 `god.god_discarder_seat` 传入观察，无圈时为 -1；`catch_restricted` 等价于 `catch_play && god_discarder_seat != seat`。圈内非财神出牌仍对豁免方开启固定时长响应窗口，不能因圈中或没有可吃碰的牌而跳过轮询 |
| 分数 | `round_ended.data.scores` 为本局增量，`game_ended.final_scores` 为累计分 |
| 终局留存 | 自动房间完成后约 60 秒清理，及时保存终局快照与牌谱 |
| 局间停顿 | 每局结算后固定停 5 秒才发下一局（v31，时长服务端写死）。窗口内任意 seq 的快照是上一局终态：`phase="settled"`、`round_no` 仍是刚结束那一局、`my_hand`/`hand_counts` 是上一局残牌，提交动作一律 409 `INVALID_ACTION`。按 phase 判定可操作性、等下一局全量快照，不用该手牌计算；轮询循环无需改动，每场 8 局因此多约 40 秒墙钟，`--seconds` 预算留余量 |
| 功能开关 | 管理员可在正式比赛期间关闭自由匹配与自建测试房（v29）。`POST /api/match` 新匹配、`POST /portal/api/test-rooms` 建房、finished 测试房的跨轮重开返回 403 `FEATURE_DISABLED`——永久条件，不重试（跨轮重开此前为 409，按 code 判型而非状态码）；在途对局、房内幂等重调与已建房管理不受影响；门户 `GET /portal/api/features` 可预检 |
| 姓名语义 | 正式锦标赛之外的一切姓名字段只下发 AI 昵称，昵称为空给空串（v30）。name 仅作展示，稳定身份一律用 `user_id` |
| 门户今日榜 | v34 的 `/portal/api/leaderboard` 新增 `last`：仅今日榜且上榜人数大于 32 时为 `{rooms, firsts, score, is_me}`，否则为 `null`；不包含身份。本项目不消费此门户榜单，无需接入，也不改变本地研究排名 |

测试反馈用于框架校准与回归；测试房、自由对战和正式比赛的记录均不回流研究 MCP。完整阶段身份映射、最高并发和响应窗口仍需专项线上验证，当前能力边界见[开发索引](development.md)。

自定义快照映射必须包含 `god_discarder_seat`。圈中缺少或包含无效豁免座位时拒绝生成动作；旧版无圈快照允许省略该字段，由官方适配器补为 -1。
