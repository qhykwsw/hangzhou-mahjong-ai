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
    project, experiment_id, branch, thread=None, model_config=None, catalog_dir=None
):
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
        "mcp_servers.mahjong_research": {
            "command": sys.executable,
            "args": [
                "-B",
                "-m",
                "mahjong_ai.interfaces.mcp",
                str(project),
                experiment_id,
                branch,
            ],
            "env": {"PYTHONPATH": str(Path(__file__).parents[2])},
            "required": True,
            "tool_timeout_sec": 45,
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


def agent_prompt(task):
    prompt = (
        CONTRACT
        + "\n当前策略组合："
        + task["brief"]
        + (
            "\n先读取状态、本分支现役代表和全部相关历史。必须派生新草稿后修改和提交。"
            "动手写代码之前，先用 research_declare_plan 写下三件事：assessment（现役策略现在具体"
            "在做什么、它这一路的实测轨迹说明什么）、direction（本次要尝试的方向）、reason（为什么"
            "这个方向最值得做）。没有计划就创建不了候选；这段文字会展示在网页卡片上给用户看，要写具体"
            "机制和证据，用中文写，不要写“优化策略”这类空话。"
            "工具报错不等于操作没生效：先重新读 research_status 对齐实际状态，再决定下一步；"
            "重复调用 create 会直接返回已存在的那个候选（existing 为 true），"
            "research_declare_plan 可以安全重写；不要在没有确认结果前反复重试同一个调用，"
            "也不要重复提交一个你没有看到结果的批次。"
            "本次最多提交一批 1 至 4 个候选；提交成功后立即用简短结论结束本次回复，"
            "不等待比赛。提交前先用 research_check_candidate 跑通每个草稿，框架在提交时会再跑一次；"
            "被拒绝必须在本次回复内修好并重新提交，不能带着跑不通的代码结束回复。"
            "框架会在比赛完成后继续同一个会话。不要向用户提问。"
        )
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
            "\n本分支的结构化交接（每次迭代开新会话，上一轮对话已归档，"
            "不要引用你记不得的细节，需要就去查 research_status / research_history）：\n"
            + json.dumps(task["handoff"], ensure_ascii=False)
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
    prompt = agent_prompt(task)
    log = root / "codex.jsonl"
    error = None
    interrupted = False
    try:
        workspace = root / "workspace"
        workspace.mkdir(exist_ok=True)
        with log.open("ab") as out, (root / "codex.stderr").open("ab") as err:
            process = subprocess.Popen(
                codex_command(
                    task["project"],
                    task["experiment"],
                    task["branch"],
                    task["thread_id"],
                    task["model_config"],
                    root,
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
                    or time.time() >= task["deadline"]
                    or os.getppid() != task["owner_pid"]
                    or (root / "cancel.flag").exists()
                    or (root / "submitted.flag").exists()
                ):
                    interrupted = True
                    os.killpg(process.pid, signal.SIGTERM)
                    break
                time.sleep(0.2)
            try:
                process.wait(timeout=4)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
            if process.returncode and not interrupted:
                error = "Codex 未正常完成；请检查登录、模型配置及会话日志。"
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"[:2000]
    finally:
        if process is not None and process.poll() is None:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait()
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
