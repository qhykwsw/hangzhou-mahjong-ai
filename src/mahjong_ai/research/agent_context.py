"""Bounded agent readouts; canonical evidence and web snapshots remain complete."""

from collections import Counter, OrderedDict
from copy import deepcopy
import json
from hashlib import sha256

CONDITION_FIELDS = (
    "experiment",
    "runtime",
    "cycle",
    "opponents",
    "game",
    "timeouts",
    "resources",
    "evaluation_games",
)
PAGE_CHARACTERS = 20000


def handoff_view(handoff):
    """Factor repeated conditions without deleting or shortening any conclusion."""
    result = deepcopy(handoff)
    conditions = {}
    for record in result.get("research_conclusions", []):
        evidence = record.get("evidence", {})
        context = {
            key: evidence.pop(key) for key in CONDITION_FIELDS if key in evidence
        }
        key = str(context.get("cycle", len(conditions)))
        if key in conditions and conditions[key] != context:
            raise ValueError("同一轮的研究条件不一致")
        conditions[key] = context
        evidence["condition_id"] = key
    result["research_conditions"] = conditions
    return result


def result_summary(result):
    # Raw per-group samples are available in immutable receipts, not in repeated
    # status/history responses. Paired statistics and error counters stay visible.
    return {
        key: deepcopy(value)
        for key, value in result.items()
        if not key.endswith("group_means") and key != "group_blocks"
    }


def candidate_summary(candidate):
    keys = (
        "id",
        "branch",
        "combination",
        "iteration",
        "parent",
        "hypothesis",
        "status",
        "error",
        "preflight",
        "available",
    )
    result = {key: deepcopy(candidate[key]) for key in keys if key in candidate}
    if candidate.get("last_result"):
        result["last_result"] = result_summary(candidate["last_result"])
    return result


def iteration_summary(iteration):
    keys = (
        "id",
        "branch",
        "cycle",
        "number",
        "status",
        "incumbent",
        "candidates",
        "representative",
        "improved",
        "error",
        "plan",
        "selection_rule",
        "execution_feedback",
    )
    result = {key: deepcopy(iteration[key]) for key in keys if key in iteration}
    if iteration.get("results"):
        result["results"] = {
            cid: result_summary(row) for cid, row in iteration["results"].items()
        }
    return result


def status_view(snapshot, branch):
    own = next(b for b in snapshot["branches"] if b["id"] == branch)
    iterations = [i for i in snapshot["iterations"] if i["branch"] == branch]
    current = iterations[-1] if iterations else None
    ids = {own.get("representative")}
    if current:
        ids.add(current.get("incumbent"))
        ids.update(current.get("candidates", []))
    candidates = [
        c
        for c in snapshot["candidates"]
        if c["branch"] == branch
        and (c["id"] in ids or (current and c.get("iteration") == current["id"]))
    ]
    records = snapshot["research_conclusions"]
    return dict(
        experiment=deepcopy(snapshot["experiment"]),
        config=deepcopy(snapshot["config"]),
        branch={
            key: deepcopy(own[key])
            for key in (
                "id",
                "combination",
                "model",
                "effort",
                "status",
                "representative",
                "completed",
                "successful",
                "improved",
                "failed",
            )
            if key in own
        },
        iteration=(
            {
                key: deepcopy(current[key])
                for key in (
                    "id",
                    "number",
                    "status",
                    "incumbent",
                    "candidates",
                    "representative",
                    "error",
                )
                if key in current
            }
            | {"plan_declared": bool(current.get("plan"))}
        )
        if current
        else None,
        candidates=[
            {
                key: deepcopy(c[key])
                for key in ("id", "parent", "status", "available", "error", "preflight")
                if key in c
            }
            for c in candidates
        ],
        ranking=deepcopy(snapshot.get("ranking", [])),
        pending_conclusion_ids=[
            r["record_id"] for r in records if r["assessment"] is None
        ],
        conclusion_counts=dict(
            Counter(
                (r.get("assessment") or {}).get("status", "pending") for r in records
            )
        ),
        history_note='这里只列待复盘 ID；正文已在开场交接中。按需用 research_history(kind="conclusions") 分页检索，research_read_conclusion 读取单条完整证据与修订历史。',
    )


def history_page(snapshot, branch, kind="candidates", query="", offset=0, limit=5):
    if offset < 0 or not 1 <= limit <= 50:
        raise ValueError("分页参数无效")
    if kind == "candidates":
        rows = [
            candidate_summary(c)
            for c in snapshot["candidates"]
            if c["branch"] == branch
        ]
    elif kind == "iterations":
        rows = [
            iteration_summary(i)
            for i in snapshot["iterations"]
            if i["branch"] == branch
        ]
    elif kind == "conclusions":
        rows = [
            deepcopy(r)
            for r in snapshot["research_conclusions"]
            if r["branch"] == branch
        ]
    else:
        raise ValueError("未知历史类型")
    rows = [
        row
        for row in rows
        if query.casefold() in json.dumps(row, ensure_ascii=False).casefold()
    ]
    page = []
    size = 0
    for row in rows[offset : offset + limit]:
        length = len(json.dumps(row, ensure_ascii=False))
        if page and size + length > PAGE_CHARACTERS:
            break
        page.append(row)
        size += length
    end = offset + len(page)
    return dict(
        kind=kind,
        total=len(rows),
        offset=offset,
        items=page,
        next_offset=end if end < len(rows) else None,
    )


def read_conclusion(snapshot, branch, record_id):
    record = next(
        (
            r
            for r in snapshot["research_conclusions"]
            if r["branch"] == branch and r["record_id"] == record_id
        ),
        None,
    )
    if record is None:
        raise ValueError("未找到本分支的研究记录")
    iteration = next(
        i for i in snapshot["iterations"] if i["id"] == record["iteration_id"]
    )
    return dict(
        record=deepcopy(record),
        revisions=deepcopy(iteration.get("conclusions", {}).get(record_id, [])),
    )


class StatusReadout:
    """Conditional status replies based on the revision the caller actually received."""

    def __init__(self, capacity=32):
        self.capacity = max(1, capacity)
        self.snapshots = OrderedDict()

    def reply(self, view, known_revision=None):
        revision = sha256(
            json.dumps(
                view, sort_keys=True, ensure_ascii=False, separators=(",", ":")
            ).encode()
        ).hexdigest()
        previous = self.snapshots.get(known_revision)
        self.snapshots[revision] = deepcopy(view)
        self.snapshots.move_to_end(revision)
        while len(self.snapshots) > self.capacity:
            self.snapshots.popitem(last=False)
        if known_revision == revision:
            return dict(mode="unchanged", revision=revision)
        if previous is None:
            return dict(mode="full", revision=revision, **deepcopy(view))
        changes = {
            key: deepcopy(value)
            for key, value in view.items()
            if key not in previous or previous[key] != value
        }
        return dict(
            mode="delta",
            base_revision=known_revision,
            revision=revision,
            changes=changes,
            removed=[key for key in previous if key not in view],
        )


def text_page(content, offset=0, limit=4000):
    """Bound escaped UTF-8 payloads, including long embedded-source lines."""
    if not 0 <= offset <= len(content) or not 1 <= limit <= 4000:
        raise ValueError("分页参数无效")
    low, high = offset, min(len(content), offset + limit)
    while low < high:
        end = (low + high + 1) // 2
        size = len(json.dumps(content[offset:end], ensure_ascii=False).encode())
        if size <= 8000:
            low = end
        else:
            high = end - 1
    return dict(
        content=content[offset:low],
        offset=offset,
        next_offset=low if low < len(content) else None,
        total_characters=len(content),
        revision=sha256(content.encode()).hexdigest(),
    )
