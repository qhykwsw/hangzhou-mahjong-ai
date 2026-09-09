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
    project, run_id, branch, thread=None, model_config=None, catalog_dir=None
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
                run_id,
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
    prompt = (
        CONTRACT
        + "\n当前策略组合："
        + task["brief"]
        + (
            "\n先读取状态、本分支现役代表和全部相关历史。必须派生新草稿后修改和提交。"
            "本次最多提交一批 1 至 4 个候选；提交成功后立即用简短结论结束本次回复，"
            "不等待比赛。框架会在比赛完成后继续同一个会话。不要向用户提问。"
        )
    )
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
                    task["run"],
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
