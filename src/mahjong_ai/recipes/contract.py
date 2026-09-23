"""What the recipe Agent is told: the fixed contract, the handoff and the pacing."""

import json
from pathlib import Path
import time

ADR = (
    Path(__file__).resolve().parents[3]
    / "docs/adr/0012-recipe-iteration-for-rl-training.md"
)

CONTRACT = """你是训练方案迭代的研究模型。改进对象是一条方案线上的训练方案：一个名为 recipe 的 Python 包，框架调用 recipe.train(ctx)，在固定时限内从固定起点训练出麻将对局策略（Actor）。
目标：在训练时限内，从固定起点训练出一个在配对评估里稳定胜过现役模型的 Actor。本分支和规则策略分支同属一个研究框架：每次尝试开始时冻结评估对手（最近一次共同大乱斗的前三名，可能是规则策略，也可能是本分支自己的模型），ctx.opponents 就是这三个版本，训练时默认从中抽对手，方案可以改用 cfg["opponents"] 选择其中一部分或教师规则包里的规则。训练结束后，框架让你的模型和现役模型在同样的牌墙、座位和评估对手上分段对打（每边 4096、8192、16384 场逐段累计；前两段用 99% 区间可提前晋级或结束，最后一段 95% 区间下界大于 0 才晋级）。晋级若干次后，全部分支的代表参加共同大乱斗，排名决定之后的评估对手。
固定不变、你无法修改的：Actor 结构与观察编码、游戏环境与桌面分奖励、评估对手与教师、训练时限（最长 4 小时，可以提前结束，只比较最终强度）、评估与晋级规则、每次尝试都从同一个固定起点开始。接口细节只以 recipe_read_reference("api") 为准。
你能改的：recipe 包里的全部代码和配置，包括采集方式、优势与回报计算、Critic 结构与输入（可读完整隐藏状态，含按摸牌顺序排列的牌墙）、损失函数、优化器、教师约束、对手构成、探索方式和时间分配。Actor 只能读合法观察。
工作方式：每次尝试是一个改动单元，对应一个可证伪的假设；也可以在判断进入局部最优时整体重写方案，但仍要写清假设。先看证据再动手：交接里有现役方案、累计结论和上次尝试的训练诊断与评估结果。
流程（缺一步就无法提交）：
1. 用 recipe_record_conclusion 复盘交接里 unreviewed 列出的每个已完成尝试：区分假设本身的效果和实现、超时、资源问题；状态用 已验证 / 已否定 / 证据不足 / 未有效验证。
2. 用 recipe_declare_plan 写下 assessment（现役方案现在具体怎么训练、实测轨迹说明什么）、direction（本次改什么）、reason（为什么最值得做）。这段文字会展示给用户，用中文写具体机制和证据，不要空话。
3. 用 recipe_read_file / recipe_edit_file / recipe_write_file 修改工作区；需要换起点时用 recipe_reset。
4. 用 recipe_smoke 在目标主机上以很短时限实跑一遍，确认能训练并导出 Actor；它不评估棋力。冒烟通过的内容一旦再改，就要重新冒烟。
5. 用 recipe_submit 提交一句话摘要。提交后立即简短结束回复，不等待训练结果；下一次尝试会开新会话并携带交接。
注意：训练时限很短，方案要在时限内完成采集与更新并及时 ctx.save_actor；超时进程会被强制结束，只保留最后保存的权重。不要向用户提问。工具报错时先调用 recipe_status 对齐实际状态再决定下一步。"""


def trial_brief(trial):
    evaluation = trial.get("evaluation") or {}
    training = trial.get("training") or {}
    plan = trial.get("plan") or {}
    return {
        "trial": trial["number"],
        "outcome": trial.get("outcome"),
        "parent": trial.get("parent"),
        "direction": plan.get("direction"),
        "summary": trial.get("summary"),
        "difference": evaluation.get("difference"),
        "challenger": evaluation.get("challenger"),
        "incumbent": evaluation.get("incumbent"),
        "training": {
            k: training.get(k)
            for k in (
                "state",
                "reason",
                "seconds",
                "updates",
                "last",
                "gpu_peak_gb",
                "gpu_limit_gb",
                "ram_peak_gb",
                "ram_limit_gb",
            )
            if training.get(k) is not None
        }
        if training
        else None,
        "error": (trial.get("error") or "")[:300]
        or (training.get("error") or "")[-800:]
        or None,
    }


def handoff(s, line_id):
    line = s["lines"][line_id]
    trials = [
        t
        for t in sorted(s["trials"].values(), key=lambda t: t["number"])
        if t["line"] == line_id and t["status"] == "completed"
    ]
    incumbent = line["versions"][-1]
    prepared = s["prepared"] or {}
    return {
        "line": {
            "id": line["id"],
            "name": line["name"],
            "host": line["host"],
            "actor": line["model"],
            "trials_used": line["used"],
            "trials_limit": line["limit"],
        },
        "fixed": {
            "budget_seconds": line.get("budget_seconds")
            or s["config"]["trial"]["budget_seconds"],
            "smoke_seconds": s["config"]["trial"]["smoke_seconds"],
            "evaluation": "框架分段配对评估，见说明",
            "teacher": prepared.get("teacher"),
            "rules": prepared.get("rules"),
        },
        "incumbent": {
            "version": incumbent["version"],
            "trial": incumbent["trial"],
            "difference_when_promoted": incumbent.get("difference"),
            "note": (
                "第 0 版的模型是基线方案从固定起点训练一个时限的产物"
                if incumbent.get("training")
                else "第 0 版的模型就是固定起点本身，方案是未经验证的基线方案"
            )
            if incumbent["version"] == 0
            else None,
            "training": incumbent.get("training"),
        },
        "curve": [
            {
                "version": p["version"],
                "incumbent": (p.get("results") or {}).get("model:incumbent"),
                "rank": (p.get("ranking") or []).index("model:incumbent") + 1
                if p.get("ranking")
                else None,
                "error": p.get("error"),
            }
            for p in line["curve"]
        ],
        "conclusions": line["conclusions"],
        "unreviewed": [trial_brief(t) for t in trials if not t.get("reviewed")],
        "previous": trial_brief(trials[-1]) | {"plan": trials[-1].get("plan")}
        if trials
        else None,
        "recent": [
            {
                "trial": t["number"],
                "outcome": t.get("outcome"),
                "direction": (t.get("plan") or {}).get("direction"),
                "difference": (t.get("evaluation") or {})
                .get("difference", {})
                .get("mean"),
            }
            for t in trials[-12:]
        ],
    }


def diagnosis():
    """The start-point diagnosis section of ADR-0012, the single written source."""
    text = ADR.read_text() if ADR.exists() else ""
    start = text.find("## 起点诊断")
    if start < 0:
        return ""
    end = text.find("\n## ", start + 1)
    return text[start : end if end > 0 else None].strip()


def pacing(remaining, limit):
    return (
        f"\n\n时间：本次思考时限 {limit} 秒，现在剩余约 {int(remaining)} 秒；冒烟检查在远端运行的时间另行补偿。"
        "到点会强制结束，未提交的改动作废，本次尝试记为未提交并消耗一次额度。"
        f"大约在剩余 {int(remaining * 0.8)} 秒前写完复盘和计划，剩余 {int(remaining * 0.45)} 秒前开始冒烟。"
        "每次工具返回都带 budget，按它调整节奏；宁可提交一个小而确定的改动，也不要因为改动太大而交不出东西。"
    )


def prompt(s, trial, job, remaining):
    limit = s["config"]["agent"]["seconds"]
    if trial.get("thread_id"):
        reason = (
            "上一条回复已经结束，但还没有提交方案；这不算完成。"
            if job["continuations"]
            else "会话曾被暂停，现在恢复。"
        )
        return (
            f"继续同一次方案尝试（第 {trial['number']} 次）。{reason}"
            "先调用 recipe_status 对齐状态，再完成复盘、计划、修改、冒烟，recipe_submit 成功后立即结束。"
            + pacing(remaining, limit)
        )
    return (
        CONTRACT
        + "\n\n"
        + diagnosis()
        + f"\n\n本次是方案线 {trial['line']} 的第 {trial['number']} 次尝试。工作区已复制现役方案（第 {trial['incumbent_version']} 版）。"
        + "\n方案线交接（结论、现役、曲线、上次尝试的训练诊断与评估）：\n"
        + json.dumps(
            handoff(s, trial["line"]), ensure_ascii=False, separators=(",", ":")
        )
        + pacing(remaining, limit)
    )


def budget_note(task, extension):
    remaining = (
        task["deadline"]
        + max(0.0, extension - task.get("extension_base", 0.0))
        - time.time()
    )
    limit = task.get("execution_limit_seconds") or 300
    share = remaining / limit
    if remaining <= 0:
        advice = "时间已到，即将被终止。"
    elif remaining <= 45 or share <= 0.15:
        advice = "最后窗口：冒烟已通过就立刻 recipe_submit，不要再改代码。"
    elif share <= 0.45:
        advice = "现在就冒烟并提交，不要再扩大改动。"
    elif share <= 0.8:
        advice = "应当已经写完计划并在修改工作区了。"
    else:
        advice = "复盘、写计划，然后动手修改。"
    return {"remaining_seconds": int(max(0, remaining)), "advice": advice}
