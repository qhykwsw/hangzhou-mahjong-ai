"""One bounded Codex turn with its own watchdog and persisted thread receipt."""

import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from mahjong_ai.persistence.artifacts import atomic_json, read_json
from mahjong_ai.research.contract import CONTRACT
from mahjong_ai.research.agent_context import handoff_view

# 模型网关在并发启动时会对瞬时过载返回 429/5xx；Codex 自己的重试用尽后整轮
# 迭代会被记成失败并消耗配额，所以在执行器里再补一层带退避的重试。
RETRY_DELAYS = (30.0, 90.0, 180.0)
TRANSIENT_MARKERS = (
    "429",
    "too many requests",
    "rate limit",
    "overloaded",
    "502",
    "503",
    "504",
)



def _kill_group(pid, sig=signal.SIGKILL):
    """Signal a process group that may have just exited (macOS answers a group
    of unreaped zombies with EPERM, not ESRCH)."""
    try:
        os.killpg(pid, sig)
    except (ProcessLookupError, PermissionError):
        pass

def transient_failure(window):
    """判断一次失败的输出里是否只有网关瞬时错误。"""
    for line in window.splitlines():
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if event.get("type") not in ("error", "turn.failed"):
            continue
        message = json.dumps(event, ensure_ascii=False).lower()
        if any(marker in message for marker in TRANSIENT_MARKERS):
            return True
    return False


def toml_value(value):
    if isinstance(value, dict):
        return (
            "{"
            + ",".join(json.dumps(k) + "=" + toml_value(v) for k, v in value.items())
            + "}"
        )
    if isinstance(value, list):
        return "[" + ",".join(toml_value(v) for v in value) + "]"
    return json.dumps(value)


def codex_command(
    project,
    experiment_id,
    branch,
    thread=None,
    model_config=None,
    catalog_dir=None,
    deadline=None,
    budget=None,
    server=None,
    tool_timeout=45,
):
    """server replaces the research MCP: {"name", "module", "args"}."""
    from mahjong_ai.execution.models import resolve_model
    from mahjong_ai.research.config import MODELS

    argv = ["codex", "exec", "--ignore-user-config", "--skip-git-repo-check", "--json"]
    settings = {
        "sandbox_mode": "read-only",
        "approval_policy": "never",
        "features.shell_tool": False,
        "features.unified_exec": False,
        "features.multi_agent": False,
        "features.apps": False,
        "features.plugins": False,
        "features.browser_use": False,
        "features.computer_use": False,
        "features.skill_search": False,
        "features.skip_host_skill_discovery": True,
        "web_search": "disabled",
        "mcp_servers." + (server or {}).get("name", "mahjong_research"): {
            "command": sys.executable,
            "args": [
                "-B",
                "-m",
                *(
                    [server["module"], *server["args"]]
                    if server
                    else [
                        "mahjong_ai.interfaces.mcp",
                        str(project),
                        experiment_id,
                        branch,
                    ]
                ),
            ],
            # 每次工具返回都带上剩余时间：非交互会话里这是唯一可靠的提醒通道。
            "env": {
                "PYTHONPATH": str(Path(__file__).parents[2]),
                **(
                    {"MAHJONG_ITERATION_DEADLINE": f"{deadline:.0f}"}
                    if deadline
                    else {}
                ),
                **({"MAHJONG_ITERATION_BUDGET": f"{budget:.0f}"} if budget else {}),
            },
            "required": True,
            "tool_timeout_sec": tool_timeout,
            "startup_timeout_sec": 20,
        },
    }
    settings.update(resolve_model(model_config or MODELS["astra"], catalog_dir))
    for k, v in settings.items():
        argv += ["-c", k + "=" + toml_value(v)]
    if thread:
        argv += ["resume", thread]
    argv += ["-"]
    return argv


def budget_brief(task):
    """开场就把时间预算说清楚，并给出「先交最小可行改动，再用剩余时间改进」的节奏。"""
    deadline, total = task.get("deadline"), task.get("execution_limit_seconds")
    if not deadline:
        return ""
    remaining = max(0, int(deadline - time.time()))
    limit = (
        f"本次迭代总预算 {int(total)} 秒（{int(total) // 60} 分钟）；" if total else ""
    )
    return (
        f"\n\n时间预算：{limit}现在剩余约 {remaining} 秒，到点会强制终止，未提交的草稿全部作废。"
        "研究 MCP 每次返回都带 budget.remaining_seconds 与 advice，按它调整节奏。"
        f"要求按这个顺序推进：先在约 {max(60, remaining // 6)} 秒内写完结论与计划，"
        f"再在剩余约 {max(90, remaining // 3)} 秒时提交第一版最小可行改动，"
        "提交成功后如果还有时间，可以派生新草稿继续改进并再次提交。"
        "任何时刻都要保证手上有一个能提交的版本：宁可先交一个小而确定的改动，"
        "也不要为了更大的改动把时间用光而交不出东西。不要为了省时间跳过自检。"
    )


def agent_prompt(task):
    if task.get("continuation"):
        return (
            "继续当前同一次研究迭代，沿用本会话已有判断、草稿和累计结论。"
            "上一条回复正常结束，但注册表尚未收到候选提交；这不算完成任务。"
            "先读取 research_status 对齐实际状态，再完成待复盘结论、计划、"
            "候选实现和自检，调用 research_submit_batch 提交成功后才能结束。"
            "不要重复创建已有草稿或重复提交已接收的批次。原迭代时间上限不变。"
            + budget_brief(task)
        )
    if task.get("thread_id"):
        return (
            "继续当前同一次研究迭代，沿用会话中的累计结论、判断和草稿，不重新投喂交接。"
            "先调用 research_status 核对当前状态；已知 revision 时传 known_revision，"
            "上下文压缩后需要恢复当前状态时省略版本号，缺少某条证据时主动读取详情。"
            "完成本次候选实现、自检和 research_submit_batch，提交成功后立即结束。"
            + budget_brief(task)
        )
    prompt = (
        CONTRACT
        + "\n当前策略组合："
        + task["brief"]
        + (
            "\n交接已包含全部累计结论；先核对当前状态与现役源码，按本次假设的具体缺口读取资料，不要遍历 SDK 或重新获取已给出的历史、契约和未变化源码。必须派生新草稿后修改和提交。"
            "先用 research_record_conclusion 整理累计研究结论中 assessment 为空的记录，"
            "区分假设效果和实现、资源、基础设施失败。再用 research_declare_plan 写下三件事：assessment（现役策略现在具体"
            "在做什么、它这一路的实测轨迹说明什么）、direction（本次要尝试的方向）、reason（为什么"
            "这个方向最值得做）。没有计划就创建不了候选；这段文字会展示在网页卡片上给用户看，要写具体"
            "机制和证据，用中文写，不要写“优化策略”这类空话。"
            "工具报错不等于操作没生效：先重新读 research_status 对齐实际状态，再决定下一步；"
            "重复调用 create 会直接返回已存在的那个候选（existing 为 true），"
            "research_declare_plan 可以安全重写；不要在没有确认结果前反复重试同一个调用，"
            "也不要重复提交一个你没有看到结果的批次。"
            f"本次最多提交一批 1 至 {task['candidate_limit']} 个候选；"
            "未提交草稿不参与评估，不要把它们当作已有验证结果。提交成功后立即用简短结论结束本次回复，"
            "不等待比赛。提交前先用 research_check_candidate 跑通每个草稿，框架在提交时会再跑一次；"
            "被拒绝必须在本次回复内修好并重新提交，不能带着跑不通的代码结束回复。"
            "比赛完成后的下一次迭代会开启新会话并携带结构化交接。不要向用户提问。"
        )
        + budget_brief(task)
    )
    if task.get("previous_plan"):
        plan = task["previous_plan"]
        prompt += (
            "\n上次迭代你写下的研究判断（用于对照实际结果）：\n"
            + json.dumps(plan, ensure_ascii=False)
            + "\n如果实际结果与当时的方向或原因不符，在本次 assessment 里明确说明。"
        )
    if task.get("handoff"):
        prompt += (
            "\n本分支的结构化交接（每次迭代开新会话，携带累计研究结论与待复盘证据，"
            "结论不按最近三次截断；每条 evidence.condition_id 指向 research_conditions 中的共享条件；不要重新通读完整历史，缺少细节时按关键词检索 research_history 或用 research_read_conclusion 读取一条完整证据）：\n"
            + json.dumps(
                handoff_view(task["handoff"]), ensure_ascii=False, separators=(",", ":")
            )
        )
    if task.get("previous_iteration_feedback"):
        prompt += (
            "\n上次分支迭代的实际执行反馈（时间单位为秒）：\n"
            + json.dumps(task["previous_iteration_feedback"], ensure_ascii=False)
            + "\n结合实际训练场数、准备耗时和执行上限，自行调整本次 training_games。"
            "preparation_seconds 包含依赖准备、采集对局、训练与校验，"
            "不含排队和选优比赛；不要把总等待时间当成训练耗时。"
            "上次耗时很短时可以增加训练场数，但应结合收益和本次代码、对手变化判断。"
        )
    return prompt


def main():
    root = Path(sys.argv[1])
    task = read_json(root / "task.json")
    cancelled = False
    process = None

    def stop(signum, frame):
        nonlocal cancelled
        cancelled = True

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    started = time.monotonic()
    prompt = task.get("prompt") or agent_prompt(task)

    def deadline():
        """Remote checks recorded in extension.json do not count against the limit."""
        extra = 0.0
        path = root / "extension.json"
        if path.exists():
            try:
                extra = float(read_json(path)["seconds"]) - task.get(
                    "extension_base", 0.0
                )
            except (ValueError, KeyError):
                extra = 0.0
        return task["deadline"] + max(0.0, extra)

    log = root / "codex.jsonl"
    workspace = root / "workspace"
    workspace.mkdir(exist_ok=True)
    error = None
    interrupted = False

    def run_once():
        """跑一次 codex exec，原地更新本次尝试的结果。"""
        nonlocal process, error, interrupted
        process = None
        error = None
        interrupted = False
        try:
            with log.open("ab") as out, (root / "codex.stderr").open("ab") as err:
                process = subprocess.Popen(
                    codex_command(
                        task["project"],
                        task["experiment"],
                        task["branch"],
                        task["thread_id"],
                        task["model_config"],
                        root,
                        task.get("deadline"),
                        task.get("execution_limit_seconds"),
                        task.get("mcp_server"),
                        task.get("tool_timeout_seconds", 45),
                    ),
                    stdin=subprocess.PIPE,
                    stdout=out,
                    stderr=err,
                    cwd=workspace,
                    start_new_session=True,
                )
                atomic_json(root / "codex-process.json", {"pid": process.pid})
                process.stdin.write(prompt.encode())
                process.stdin.close()
                while process.poll() is None:
                    if (
                        cancelled
                        or time.time() >= deadline()
                        or os.getppid() != task["owner_pid"]
                        or (root / "cancel.flag").exists()
                        or (root / "submitted.flag").exists()
                    ):
                        interrupted = True
                        _kill_group(process.pid, signal.SIGTERM)
                        break
                    time.sleep(0.2)
                try:
                    process.wait(timeout=4)
                except subprocess.TimeoutExpired:
                    _kill_group(process.pid)
                    process.wait()
                if process.returncode and not interrupted:
                    error = "Codex 未正常完成；请检查登录、模型配置及会话日志。"
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"[:2000]
        finally:
            if process is not None and process.poll() is None:
                _kill_group(process.pid)
                process.wait()

    def pause_before_retry(delay):
        end = time.monotonic() + delay
        while time.monotonic() < end:
            if (
                cancelled
                or time.time() >= deadline() - 30
                or os.getppid() != task["owner_pid"]
                or (root / "cancel.flag").exists()
                or (root / "submitted.flag").exists()
            ):
                return False
            time.sleep(0.5)
        return True

    for attempt in range(len(RETRY_DELAYS) + 1):
        offset = log.stat().st_size if log.exists() else 0
        run_once()
        if not error or interrupted or attempt >= len(RETRY_DELAYS):
            break
        with log.open(errors="replace") as handle:
            handle.seek(offset)
            window = handle.read()
        if not transient_failure(window):
            break
        delay = RETRY_DELAYS[attempt]
        if time.time() + delay >= deadline() - 30:
            break
        with log.open("a") as handle:
            handle.write(
                json.dumps(
                    {"type": "retry.scheduled", "attempt": attempt + 1, "delay": delay}
                )
                + "\n"
            )
        if not pause_before_retry(delay):
            break

    thread, usage = task["thread_id"], []
    if log.exists():
        for line in log.read_text(errors="replace").splitlines():
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            if event.get("type") == "thread.started":
                thread = event.get("thread_id")
            if event.get("type") == "turn.completed" and event.get("usage"):
                usage.append(event["usage"])
    atomic_json(
        root / "result.json",
        dict(
            status="paused"
            if cancelled or (root / "cancel.flag").exists()
            else "completed"
            if not error
            else "failed",
            thread_id=thread,
            usage={"wall_seconds": time.monotonic() - started, "model": usage},
            usage_complete=bool(usage) and not interrupted and not error,
            interrupted=interrupted,
            error=error,
        ),
    )


if __name__ == "__main__":
    main()
