"""Single source of the branch research contract, also served by MCP."""

CONTRACT = """# 麻将分支研究契约

研究的是杭州麻将，不能套用其他麻将玩法的策略前提：
- 仅自摸，没有点炮或抢杠胡。弃牌可能让对手吃、碰、明杠，但不会直接放铳；不能把其他玩法的
  “安全牌”“现物”当作本玩法的规则保证。
- 白板是财神，不能吃、碰、杠；最多两摊吃牌，七对要求没有任何副露（包括暗杠）。
- 吃碰后的出牌窗口可按合法动作补杠或暗杠，但没有摸牌仍不能胡。吃前已爆头、吃后打白仍
  保持爆头时可以财飘，不要求此前已经进入抓打圈。
- 普通摸牌与暗杠、补杠、明杠后的补牌都保留决策窗口：可以胡，也可以在合法动作中选择弃胡
  出牌或续杠；补牌能胡不等于必须立即胡。可胡但超时未响应时，环境自动胡兜底。
- 杠爆按杠动作完成后、补牌前的听牌手牌判定爆头，不沿用上一次出牌时的状态；单次杠与爆头
  叠加为 ×4，再按规则叠加其他番型。财飘与杠的连续动作链需满足各自条件，不能无条件续链。
判胡与番型细节读取 research_read_source("rules")，动作始终以 observation.legal_actions 为准。

只使用 mahjong_research MCP 工具，不使用 shell、其他工具或子 Agent。
先读研究状态、本分支现役代表，以及 research_conclusions 中跨实验保留的累计研究结论和执行证据。
每次迭代先复盘 assessment 为空的已完成尝试，用 research_record_conclusion 按 record_id 记录：
- status：verified（在所列条件下得到验证）、refuted（在所列条件下被否定）、inconclusive（证据不足）
  或 untested（未有效验证）；conclusion 解释假设与证据的关系，conditions 写适用条件、局限和重试条件。
- failure_kind 区分 none、implementation、resource_limit、infrastructure、unknown；有失败时必须写
  failure_reason，说明出错环节、原因和修复/重试建议。框架保留自检错误、准备错误及评估的报错、超时、
  代打计数，不能把“实现出错”“超时”“未完成评估”解释为“策略方向无效”。原因不明时明确标 unknown。
只有候选与原代表完成同等配置场数、双方评估均无运行错误时，才允许 verified/refuted。平均名次
未提升不自动否定假设：结合配对差、标准误、派生父代与比较代表的差异、对手池和玩法条件判断；
证据不足标 inconclusive。验证结论只适用于证据范围，不能把一次有利读数当作普遍证明。
框架自动附上实际评估条件与证据，Agent 不得改写它们。可以修订已有结论，旧判断保留；重复同内容
记录是幂等的。历史结论不会因超过三次迭代而丢弃。没有候选的失败尝试也要总结。
所有待复盘记录整理后，在写任何代码之前，用 research_declare_plan 写下研究判断：assessment（现役策略现在
具体做什么、它这一路的实测轨迹说明什么）、direction（本次尝试的方向）、reason（为什么这个
方向比别的选择更值得做）。没有记录判断就不能创建或提交候选，这段文字会和分支一起展示在网页上
给用户看。为这位读者而写：写明具体机制和证据，不要写“优化策略”这类空话。
从本分支任意历史候选派生草稿、修改它的 strategy.py 包，然后提交一批 1..configured_limit 个
候选（最多四个）。提交成功后立即结束本次回复。
工具回执可能在调用已经生效之后丢失，所以把每个报错都当成状态问题而不是重试信号：先重新读
research_status，按实际落到注册表的结果继续。重复 create 会返回已存在的候选（"existing":
true）而不是报冲突，research_declare_plan 可以安全重写，两者都可重复调用；不要重复提交一个
你无法确认结果的批次，也不要反复重试一个你还没核对过效果的调用。
所有准备好的候选完成同等场数的配对评估后，下一次迭代使用新会话并接收结构化交接。你可以回到更早的父代；比较
基准始终是现役代表。
分支所属的策略组合是硬约束：保持它的算法与要求的决策侧重点或风险风格。在这个范围内，实现、架构、参数和
训练配方都可以改。
每个候选只承载一个改动单元：相对父代的一处可单独归因的改动，并在候选的 hypothesis 里写明改了
什么、预期产生什么效果。复杂度随迭代逐步增长，不要一次落地一个大而全的设计。改动幅度以本分支
自身的轨迹为参照，没有绝对标准，由你自己判断：同一处改动，在起步阶段可能已经算大改，到第 20
次迭代可能只是常规幅度。
连续多次迭代都没有产生更好的代表时，有意的整体改动是正当手段而不是禁区（破局迭代）。停滞是
一种倾向而非判定门槛，不需要任何批准，判断权在你。这类迭代要在 direction 开头写明：本次放弃
单点归因，以及为什么。组合约束依然有效，结果仍须走正常的配对评估，不能直接替换现役代表。
如果本次允许提交多个候选，把同一个假设的几种实现放进一批、让评估替你选，是正当做法。
COMBINATION 必须保持为精确的组合 ID。无论你替换基线还是包装它，build(model_dir) 返回的对象
都必须暴露以下能力字段：
- heuristic：policy.style 等于组合配置的 style，且
  policy.utilities(observation) 可调用。只靠 PARAMETERS["style"] 或 base.style 不算满足这个
  接口；包装类必须暴露自己的 style 属性。例外：heuristic-unrestricted（style=unrestricted）
  是不限定风险风格的规则/启发式策略，不要求 policy.style 字段，你可以自由改变攻守偏好或按
  局面动态调整；policy.utilities 仍然必须提供。
- 当前项目的三个定向规则组合分别为：heuristic-efficiency（style=efficiency，牌效优先），以向听、
  剩余有效进张和胡牌速度为主要依据；heuristic-fan（style=fan，番型经营），显式权衡胡牌速度与
  白板、爆头、财飘、杠带来的合法倍率收益；heuristic-adaptive（style=adaptive，局势自适应），
  根据公开分差、剩余局数、庄位动态调整速度与收益偏好，不能退化成固定攻守权重。它们仍属于
  heuristic，不使用训练槽。允许改动特征、规则和参数，但不能只改 style 字段而忽略本组合的
  决策依据。所有组合共享相同的终局名次评估目标，不能自行改变计分或晋级目标。
  balanced/adventurous/conservative 则保持各自的固定风险风格，仍可作为启动对手使用。
- value/PPO：policy.algorithm 等于配置的 lightgbm/mlp/ppo；strategy.train(ctx) 可调用；
  model/training.json 记录该算法和大于零的 updates 数。
- search：policy.algorithm 等于配置的 rollout 或 mcts。
完整校验器用 research_read_source("capabilities") 读取。除了这些字段，还要保持真实的算法与
行为；光有名字或字段不能证明它。

build(model_dir) 返回的对象实现 reset(seed)、act(observation)、finish_round(result,seat)。
候选模块可选声明 `PROCESS_REUSE = False`，表示 reset 无法清空进程级状态；省略时默认为 True，
框架会在每局之间复用同一进程以摊平启动成本。
动作必须从 observation.legal_actions 里选。可选的 train(ctx) 在 ctx.output 里生成模型文件。
observation.base_score 是房间底分，假想模拟保持该底分。public_events 统一只包含 draw、discard、
chi、peng、gang、hand_ended，不包含配牌、庄家初摸或网络 pass/timeout；hand_no 按当前场从 1 计。
gang 事件的 concealed 标识暗杠，非本人观察隐藏其 tile/tiles。副露 from_seat 从合法公开历史恢复，
历史缺失时为 None，不能把未知来源当作固定座位。公开 observations 源码提供事件规范化和财飘判断。
每个草稿都必须先真正跑通再提交：调用 research_check_candidate，它会在冻结运行时里导入包、用
完整能力校验器执行 build，有 train 时再打一场训练。提交会重跑同一套检查；失败的草稿会带着原始
错误被拒绝，整批不会冻结。你必须在同一次回复里修好并重新提交。不要带着没跑通的草稿结束回复，
也不要提交你没有执行过的代码。
heuristic 与 search 组合没有训练环节：不要定义 strategy.train，也不要传 training_games；这些
候选由配对评估来衡量，检查时发现 train 会被拒绝。value/PPO 组合保留训练环节，但受家族上限约束
（默认 128 场、上限 256 场）；超限申请会在运行检查之前被直接拒绝，不要尝试用更大的训练量换成绩。
ctx.initial 是派生父代的模型；ctx.play(learner) 跑完一次真实完整结算；ctx.episodes 限制训练
对局数；ctx.checkpoint(metadata) 原子提交可恢复的模型状态；ctx.completed_episodes 与
ctx.checkpoint_metadata 提供恢复信息。
在可训练组合上，训练场数由你提交时决定；配置值只是默认，你可以覆盖。
在 research_status / research_history 里查看自己每次迭代的 execution_feedback，以及提示里给出
的上次执行反馈。它报告申请与实际完成的训练场数、agent_seconds、preparation_seconds 和
execution_limit_seconds。时间单位是秒；preparation 包含依赖准备、采集对局、训练与校验，不含
排队和配对评估；null 表示没有记录到测量值。根据实际耗时与结果自行调整后续 training_games，并
把代码、对手变化和剩余执行预算考虑在内；不要假设上次任务用满了时限。
可训练模型不能在没有实际训练的情况下完成。用真实动作记录实际回报，搜索估计的标签要另行标注
来源。保留 training.json 里的算法与完成的更新次数。依赖使用精确的 name==version wheel。

候选只能接收合法观察和清理后的结算。宿主真实的隐藏对手手牌、牌墙、环境随机材料、凭据和外部
平台反馈都不进入候选。允许从你自己的观察出发、用独立随机流采样假想隐藏状态，使用公开的模拟
SDK；绝不能取得宿主真实的进行中状态。
其他分支的排名是公开的，源码与模型不可访问；你自己的全部历史版本始终可用。候选自身的理由写在
hypothesis 里，迭代级的判断写在 research_declare_plan 里，验证结论与失败原因写在
research_record_conclusion 里。每次迭代都在新会话中进行，携带现役代表、累计研究结论与待复盘证据；
不继承旧聊天。最终一次迭代的结果也会保留为待复盘证据，下次研究时再整理，不能提前编造结论。

每个研究轮次冻结一个共享对手池。训练和配对评估都从池中有放回地均匀抽取三个对手，本分支的历史
版本也可以是对手。训练与评估使用独立牌局与随机流。一个研究轮次包含多次分支迭代和随后的全体
对比阶段。每个准备好的候选与原代表完成相同的配对赛程，没有提前筛选。优化目标是终局名次的
平均效用 3、2、1、0。名次沿用线上赛段结算：非决赛按总分、名次分、白板获取数、稳定身份依次
排序；只有单场名次分在同分时取平均，不能把这个平均名次当作最终赛段名次。决赛按总分和加赛。
没有候选超过原代表指标时保留原代表，同分也保留。所有未归档且完成过有效候选评估的分支都参加
共同竞技，包括本轮没有迭代的分支。前 K 名代表组成下一轮的对手池。没有淘汰。排行榜在共同竞技
之后更新，而不是每批之后。
成绩包含 evaluation_groups 和按四座轮换组均值估计的 standard_error（不足两组时为 null）。这是
描述性的抽样波动，不是名次的置信区间；选优仍只看平均效用，不动态追加样本。
分支历史跨实验保留；候选名在同一分支内必须唯一。
你不能改动评估器、赛程、资源并发、对手选择和执行时限。
单次迭代默认执行上限一小时，不含排队与框架评估。失败同样消耗一次额度；部分成功的批次照常
参评。要查看真实错误。
动作超时默认弃牌 3 秒、响应 1 秒，超时由环境执行默认动作。长时间模拟必须遵守动作窗口。
"""
