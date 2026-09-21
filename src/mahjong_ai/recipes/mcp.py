"""Tools of one recipe trial's Agent session. Transitions stay in recipes.service."""

import difflib
import functools
import json
from pathlib import Path, PurePosixPath
import subprocess
import sys
import time
from typing import Literal

from mcp.server.fastmcp import FastMCP

from mahjong_ai.persistence.artifacts import atomic_json, read_json
from mahjong_ai.recipes.contract import CONTRACT, budget_note, trial_brief
from mahjong_ai.recipes.service import (
    RecipeIteration,
    compact_metrics,
    copy_recipe,
    recipe_files,
    smoke_extension,
    tree_hash,
)

STATUSES = ("已验证", "已否定", "证据不足", "未有效验证")
SUFFIXES = (".py", ".json", ".md", ".txt", ".toml")
FILE_LIMIT = 200_000
TOTAL_LIMIT = 2_000_000
GUIDED = "projects/personal/selfplay/guided/"
REFERENCES = {
    "api": ("project", "recipes/API.md"),
    "adr": ("workspace", "docs/adr/0012-recipe-iteration-for-rl-training.md"),
    "trial": ("frozen", GUIDED + "trial.py"),
    "collection": ("frozen", GUIDED + "collection.py"),
    "environment": ("frozen", GUIDED + "environment.py"),
    "networks": ("frozen", GUIDED + "networks.py"),
    "learning": ("frozen", GUIDED + "learning.py"),
    "statistics": ("frozen", GUIDED + "statistics.py"),
    "config": ("frozen", GUIDED + "config.py"),
    "encoding": ("frozen", GUIDED + "encoding.py"),
    "engine": ("frozen", "src/mahjong_ai/domain/engine.py"),
}


def relative(path, suffixes=None):
    value = PurePosixPath(str(path or "").strip().removeprefix("./"))
    if (
        not value.parts
        or value.is_absolute()
        or ".." in value.parts
        or "__pycache__" in value.parts
        or (suffixes and value.suffix not in suffixes)
    ):
        raise ValueError(
            "路径必须是 recipe 包内的相对路径，扩展名限 " + " ".join(SUFFIXES)
        )
    return value.as_posix()


def lines_page(text, offset, limit):
    rows = text.splitlines()
    offset = max(0, int(offset))
    limit = max(1, min(int(limit), 800))
    return {
        "content": "\n".join(rows[offset : offset + limit]),
        "offset": offset,
        "total_lines": len(rows),
        "next_offset": offset + limit if offset + limit < len(rows) else None,
    }


def recipe_diff(before, after):
    output = []
    names = sorted(set(recipe_files(before)) | set(recipe_files(after)))
    for name in names:
        old = (
            (Path(before) / name).read_text(errors="replace").splitlines(keepends=True)
            if (Path(before) / name).exists()
            else []
        )
        new = (
            (Path(after) / name).read_text(errors="replace").splitlines(keepends=True)
            if (Path(after) / name).exists()
            else []
        )
        output.extend(
            difflib.unified_diff(old, new, "a/recipe/" + name, "b/recipe/" + name)
        )
    return "".join(output)


def server(workspace, experiment, trial_id, job_root):
    service = RecipeIteration(workspace)
    store = service.store
    job_root = Path(job_root)
    mcp = FastMCP("mahjong_recipes", instructions=CONTRACT)

    def current():
        s = store.read(experiment)
        return s, s["trials"][trial_id]

    def note():
        try:
            task = read_json(job_root / "task.json")
            path = job_root / "extension.json"
            extension = read_json(path)["seconds"] if path.exists() else 0.0
            return budget_note(task, extension)
        except (OSError, ValueError, KeyError):
            return None

    def tool(**kwargs):
        def decorate(fn):
            @functools.wraps(fn)
            def wrapped(*args, **keywords):
                result = fn(*args, **keywords)
                budget = note()
                if budget is not None and isinstance(result, dict):
                    return {**result, "budget": budget}
                return result

            return mcp.tool(**kwargs)(wrapped)

        return decorate

    def designing(trial):
        if trial["status"] != "designing":
            raise ValueError("本次尝试已提交或已结束，工作区不能再修改")

    def source_root(s, trial, source):
        line = s["lines"][trial["line"]]
        if source == "workspace":
            return Path(trial["workspace"])
        if source == "incumbent":
            return Path(line["versions"][-1]["recipe"])
        if source == "baseline":
            return Path(line["versions"][0]["recipe"])
        if source.startswith("trial:"):
            number = int(source.split(":", 1)[1])
            other = s["trials"].get(f"{trial['line']}-{number:03d}")
            if not other or not other.get("submitted"):
                raise ValueError("该尝试没有提交过方案")
            return Path(other["submitted"])
        raise ValueError("source 只能是 workspace、incumbent、baseline 或 trial:<编号>")

    def parent_root(s, trial):
        parent = trial["parent"]
        if parent["source"] == "incumbent":
            return Path(
                s["lines"][trial["line"]]["versions"][parent["version"]]["recipe"]
            )
        return source_root(s, trial, parent["source"])

    def unreviewed(s, trial):
        return [
            t["number"]
            for t in s["trials"].values()
            if t["line"] == trial["line"]
            and t["status"] == "completed"
            and not t.get("reviewed")
        ]

    ro = {"readOnlyHint": True, "destructiveHint": False, "openWorldHint": False}
    rw = {"readOnlyHint": False, "destructiveHint": False, "openWorldHint": False}

    @tool(annotations=ro)
    def recipe_status() -> dict:
        """本次尝试的实际状态：计划、工作区文件与哈希、冒烟记录、待复盘的尝试编号、是否已提交。"""
        s, trial = current()
        root = Path(trial["workspace"])
        digest = tree_hash(root)
        return {
            "trial": trial["number"],
            "line": trial["line"],
            "status": trial["status"],
            "parent": trial["parent"],
            "plan": trial.get("plan"),
            "workspace": {
                name: (root / name).stat().st_size for name in recipe_files(root)
            },
            "workspace_hash": digest,
            "smokes": [
                {k: x.get(k) for k in ("index", "ok", "state", "seconds", "hash")}
                for x in trial["smokes"]
            ],
            "current_content_smoke_passed": any(
                x["ok"] and x["hash"] == digest for x in trial["smokes"]
            ),
            "unreviewed": unreviewed(s, trial),
            "summary": trial.get("summary"),
        }

    @tool(annotations=ro)
    def recipe_history(query: str = "", offset: int = 0, limit: int = 5) -> dict:
        """按关键词分页检索本方案线已完成的尝试（新的在前），含计划、摘要、训练诊断与评估差值。"""
        s, trial = current()
        rows = [
            t
            for t in sorted(
                s["trials"].values(), key=lambda t: t["number"], reverse=True
            )
            if t["line"] == trial["line"] and t["status"] == "completed"
        ]
        if query:
            rows = [
                t for t in rows if query in str(t.get("plan")) + str(t.get("summary"))
            ]
        page = rows[offset : offset + max(1, min(limit, 20))]
        return {
            "trials": [trial_brief(t) | {"plan": t.get("plan")} for t in page],
            "next_offset": offset + len(page)
            if offset + len(page) < len(rows)
            else None,
        }

    @tool(annotations=ro)
    def recipe_read_trial(
        trial: int,
        part: Literal["diff", "metrics", "error"] = "diff",
        offset: int = 0,
        limit: int = 300,
    ) -> dict:
        """读取本方案线某次尝试的方案改动（diff）、逐次更新的训练诊断（metrics）或完整报错（error）。"""
        s, own = current()
        other = s["trials"].get(f"{own['line']}-{trial:03d}")
        if other is None:
            raise ValueError("本方案线没有这次尝试")
        if part == "diff":
            path = service.trial_root(s, other) / "diff.txt"
            return lines_page(path.read_text() if path.exists() else "", offset, limit)
        if part == "metrics":
            training = other.get("training") or {}
            path = (
                service.job_root(s, training["job"]) / "metrics.jsonl"
                if training.get("job")
                else None
            )
            rows = (
                [
                    compact_metrics(json.loads(x))
                    for x in path.read_text().splitlines()
                    if x.strip()
                ]
                if path and path.exists()
                else []
            )
            return {
                "rows": rows[offset : offset + max(1, min(limit, 200))],
                "total": len(rows),
            }
        training = other.get("training") or {}
        return {"error": other.get("error"), "training_error": training.get("error")}

    @tool(annotations=rw)
    def recipe_record_conclusion(
        hypothesis: str,
        status: Literal["已验证", "已否定", "证据不足", "未有效验证"],
        evidence: str,
        trials: list[int],
        record_id: str | None = None,
    ) -> dict:
        """新增或修订一条累计研究结论，并把引用的尝试标记为已复盘。evidence 写具体数字与机制；区分假设本身的效果和实现、超时、资源问题。修订时传 record_id。"""
        if status not in STATUSES or not hypothesis.strip() or not evidence.strip():
            raise ValueError("需要 hypothesis、evidence 和合法的 status")
        with store.edit(experiment) as (s, db):
            trial = s["trials"][trial_id]
            line = s["lines"][trial["line"]]
            referenced = []
            for number in trials:
                other = s["trials"].get(f"{trial['line']}-{int(number):03d}")
                if other is None or other["status"] != "completed":
                    raise ValueError(f"尝试 {number} 不存在或尚未完成")
                referenced.append(other)
            entry = {
                "hypothesis": hypothesis.strip()[:1000],
                "status": status,
                "evidence": evidence.strip()[:3000],
                "trials": sorted({int(n) for n in trials}),
                "at": time.time(),
                "by_trial": trial["number"],
            }
            if record_id:
                old = next(
                    (c for c in line["conclusions"] if c["id"] == record_id), None
                )
                if old is None:
                    raise ValueError("没有这条结论")
                revisions = old.get("revisions", []) + [
                    {k: old[k] for k in ("hypothesis", "status", "evidence", "at")}
                ]
                old.update(entry, revisions=revisions[-5:])
                record = old
            else:
                record = {"id": f"C{len(line['conclusions']) + 1:03d}", **entry}
                line["conclusions"].append(record)
            for other in referenced:
                other["reviewed"] = True
            store.event(db, s, "conclusion", {"line": line["id"], "id": record["id"]})
            return {"record": record, "unreviewed": unreviewed(s, trial)}

    @tool(annotations=rw)
    def recipe_declare_plan(assessment: str, direction: str, reason: str) -> dict:
        """写下本次尝试的研究判断（可重写）。必须先复盘完所有待复盘尝试；没有计划不能修改工作区。"""
        if not all(x.strip() for x in (assessment, direction, reason)):
            raise ValueError("assessment、direction、reason 都不能为空")
        with store.edit(experiment) as (s, db):
            trial = s["trials"][trial_id]
            designing(trial)
            pending = unreviewed(s, trial)
            if pending:
                raise ValueError(f"先用 recipe_record_conclusion 复盘尝试 {pending}")
            trial["plan"] = {
                "assessment": assessment.strip()[:3000],
                "direction": direction.strip()[:1500],
                "reason": reason.strip()[:3000],
                "at": time.time(),
            }
            store.event(db, s, "plan", {"trial": trial_id})
            return {"plan": trial["plan"]}

    @tool(annotations=ro)
    def recipe_list_files(source: str = "workspace") -> dict:
        """列出方案文件。source：workspace（本次工作区）、incumbent（现役）、baseline（初始基线）、trial:<编号>（某次提交）。"""
        s, trial = current()
        root = source_root(s, trial, source)
        return {name: (root / name).stat().st_size for name in recipe_files(root)}

    @tool(annotations=ro)
    def recipe_read_file(
        path: str, source: str = "workspace", offset: int = 0, limit: int = 400
    ) -> dict:
        """按行分页读取一个方案文件；source 同 recipe_list_files。"""
        s, trial = current()
        target = source_root(s, trial, source) / relative(path)
        if not target.is_file():
            raise ValueError("文件不存在")
        return {"path": relative(path)} | lines_page(target.read_text(), offset, limit)

    def check_sizes(root):
        total = sum((root / n).stat().st_size for n in recipe_files(root))
        if total > TOTAL_LIMIT:
            raise ValueError("方案总大小超过 2 MB")

    @tool(annotations=rw)
    def recipe_write_file(path: str, content: str) -> dict:
        """整文件写入工作区（新建或覆盖）。需要先写计划。"""
        s, trial = current()
        designing(trial)
        if not trial.get("plan"):
            raise ValueError("先用 recipe_declare_plan 写下计划")
        if len(content.encode()) > FILE_LIMIT:
            raise ValueError("单个文件不能超过 200 KB")
        root = Path(trial["workspace"])
        target = root / relative(path, SUFFIXES)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content)
        check_sizes(root)
        return {
            "written": relative(path),
            "bytes": len(content.encode()),
            "workspace_hash": tree_hash(root),
        }

    @tool(annotations=rw)
    def recipe_edit_file(
        path: str, old: str, new: str, replace_all: bool = False
    ) -> dict:
        """在工作区文件里做精确字符串替换；old 必须唯一匹配，除非 replace_all。需要先写计划。"""
        s, trial = current()
        designing(trial)
        if not trial.get("plan"):
            raise ValueError("先用 recipe_declare_plan 写下计划")
        root = Path(trial["workspace"])
        target = root / relative(path, SUFFIXES)
        if not target.is_file():
            raise ValueError("文件不存在")
        text = target.read_text()
        count = text.count(old)
        if not old or count == 0:
            raise ValueError("old 在文件中不存在")
        if count > 1 and not replace_all:
            raise ValueError(f"old 匹配了 {count} 处，请扩大上下文或设 replace_all")
        text = text.replace(old, new) if replace_all else text.replace(old, new, 1)
        if len(text.encode()) > FILE_LIMIT:
            raise ValueError("单个文件不能超过 200 KB")
        target.write_text(text)
        return {
            "edited": relative(path),
            "replacements": count if replace_all else 1,
            "workspace_hash": tree_hash(root),
        }

    @tool(annotations=rw)
    def recipe_delete_file(path: str) -> dict:
        """从工作区删除一个文件。"""
        s, trial = current()
        designing(trial)
        target = Path(trial["workspace"]) / relative(path)
        if relative(path) == "__init__.py":
            raise ValueError("recipe 包必须保留 __init__.py")
        if not target.is_file():
            raise ValueError("文件不存在")
        target.unlink()
        return {
            "deleted": relative(path),
            "workspace_hash": tree_hash(Path(trial["workspace"])),
        }

    @tool(annotations=rw)
    def recipe_reset(source: str) -> dict:
        """用 incumbent、baseline 或 trial:<编号> 的方案整体替换工作区，作为本次改动的起点（记为派生父代）。"""
        with store.edit(experiment) as (s, db):
            trial = s["trials"][trial_id]
            designing(trial)
            root = source_root(s, trial, source)
            copy_recipe(root, trial["workspace"])
            line = s["lines"][trial["line"]]
            trial["parent"] = (
                {"source": "incumbent", "version": line["versions"][-1]["version"]}
                if source == "incumbent"
                else {"source": source}
            )
            return {"parent": trial["parent"], "workspace_hash": tree_hash(root)}

    @tool(annotations=ro)
    def recipe_read_reference(
        name: Literal[tuple(REFERENCES)] = "api", offset: int = 0, limit: int = 400
    ) -> dict:
        """只读参考：api（方案接口，权威）、adr（设计与起点诊断）以及冻结框架源码 trial、collection、environment、networks、learning、statistics、config、encoding、engine。"""
        s, _ = current()
        base, path = REFERENCES[name]
        root = {
            "project": service.project,
            "workspace": store.workspace,
            "frozen": service.experiment_root(s) / "source",
        }[base]
        target = root / path
        if not target.is_file():
            raise ValueError("参考文件不存在")
        return {"name": name} | lines_page(target.read_text(), offset, limit)

    @tool(annotations=rw)
    def recipe_smoke() -> dict:
        """把当前工作区上传到方案线的主机，以很短时限实跑 recipe.train 并导出 Actor（约 2 分钟），不评估棋力。
        返回是否通过、报错和训练诊断。远端运行时间不计入思考时限。提交前当前内容必须冒烟通过。"""
        s, trial = current()
        designing(trial)
        if not trial.get("plan"):
            raise ValueError("先用 recipe_declare_plan 写下计划")
        root = Path(trial["workspace"])
        if not (root / "__init__.py").is_file():
            raise ValueError("recipe 包缺少 __init__.py")
        digest = tree_hash(root)
        index = len(trial["smokes"]) + 1
        jid, task = service.smoke_task(s, trial, index)
        job = service.job_root(s, jid)
        atomic_json(job / "task.json", task)
        started = time.time()
        with (job / "worker.log").open("ab") as log:
            try:
                subprocess.run(
                    [sys.executable, "-B", "-m", "guided.recipe_jobs", str(job)],
                    cwd=service.project,
                    env=service.environment("smoke"),
                    stdin=subprocess.DEVNULL,
                    stdout=log,
                    stderr=log,
                    timeout=task["smoke_seconds"] + 360,
                )
            except subprocess.TimeoutExpired:
                pass
        seconds = time.time() - started
        result = (
            read_json(job / "result.json") if (job / "result.json").exists() else {}
        )
        output = result.get("output") or {}
        status = output.get("status") or {}
        record = {
            "index": index,
            "hash": digest,
            "ok": bool(output.get("ok")) and tree_hash(root) == digest,
            "state": status.get("state") or result.get("status") or "timeout",
            "seconds": round(seconds, 1),
            "error": (status.get("error") or result.get("error") or "")[-4000:] or None,
            "metrics": [compact_metrics(r) for r in (output.get("metrics") or [])[-3:]],
            "training_seconds": status.get("training_seconds"),
            "overrun_seconds": status.get("overrun_seconds"),
        }
        infrastructure = result.get("status") == "failed"
        with store.edit(experiment) as (s, db):
            trial = s["trials"][trial_id]
            trial["smokes"].append(record)
            extension = smoke_extension(trial, s["config"]["agent"]["smoke_limit"])
        atomic_json(job_root / "extension.json", {"seconds": extension})
        return record | {
            "infrastructure_failure": infrastructure,
            "hint": "通过后如果再改工作区，需要重新冒烟才能提交"
            if record["ok"]
            else "修复后重新冒烟；infrastructure_failure 为真表示 SSH 或远端环境问题，不是方案本身的错",
        }

    @tool(annotations=rw)
    def recipe_submit(summary: str) -> dict:
        """冻结当前工作区为本次尝试的方案并排入限时训练与配对评估。要求：已写计划、已复盘、当前内容冒烟通过。提交后立即结束回复。"""
        if not summary.strip():
            raise ValueError("需要一句话摘要")
        with store.edit(experiment) as (s, db):
            trial = s["trials"][trial_id]
            if trial["status"] == "submitted":
                return {"submitted": True, "existing": True}
            designing(trial)
            if not trial.get("plan"):
                raise ValueError("先用 recipe_declare_plan 写下计划")
            pending = unreviewed(s, trial)
            if pending:
                raise ValueError(f"先复盘尝试 {pending}")
            root = Path(trial["workspace"])
            digest = tree_hash(root)
            if not any(x["ok"] and x["hash"] == digest for x in trial["smokes"]):
                raise ValueError("当前工作区内容没有通过冒烟检查，先 recipe_smoke")
            frozen = service.trial_root(s, trial) / "submitted" / "recipe"
            copy_recipe(root, frozen)
            if tree_hash(frozen) != digest:
                raise ValueError("冻结方案时内容发生变化，请重试")
            (service.trial_root(s, trial) / "diff.txt").write_text(
                recipe_diff(parent_root(s, trial), frozen)
            )
            trial.update(
                status="submitted",
                submitted_at=time.time(),
                summary=summary.strip()[:500],
                submitted=str(frozen),
                recipe_hash=digest,
            )
            store.event(db, s, "submitted", {"trial": trial_id})
            return {"submitted": True, "trial": trial["number"]}

    return mcp


if __name__ == "__main__":
    server(*sys.argv[1:5]).run(transport="stdio")
