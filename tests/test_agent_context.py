import asyncio
from copy import deepcopy
import json
import pytest


@pytest.fixture
def snapshot():
    records = [
        dict(
            record_id=f"candidate:own-c{i}",
            branch="own",
            iteration_id=f"own-i{i}",
            iteration=i,
            candidate_id=f"own-c{i}",
            hypothesis=f"hypothesis-{i}",
            outcome="evaluated",
            assessment=dict(
                status="inconclusive",
                conclusion=f"conclusion-{i}",
                conditions=f"conditions-{i}",
                failure_kind="none",
                failure_reason="",
            )
            if i < 8
            else None,
            revision_count=1 if i < 8 else 0,
            evidence=dict(
                cycle=(i + 1) // 2,
                experiment="run",
                runtime="runtime",
                opponents=["opponent"],
                game={"rounds": 8},
                timeouts={"discard": 3},
                resources={"threads": 1},
                evaluation_games=1024,
                paired={"delta": 0.01, "standard_error": 0.02},
                failures=[],
            ),
        )
        for i in range(1, 9)
    ]
    return dict(
        experiment={"id": "run"},
        config={"candidate_limit": 1},
        branches=[{"id": "own", "representative": "own-c7"}],
        candidates=[
            dict(
                id=f"own-c{i}",
                branch="own",
                iteration=f"own-i{i}",
                hypothesis=f"hypothesis-{i}",
                status="evaluated",
                last_result={
                    "mean_utility": 1.5,
                    "field_group_means": [1.5] * 256,
                    "group_blocks": list(range(256)),
                },
            )
            for i in range(1, 9)
        ],
        iterations=[
            dict(
                id=f"own-i{i}",
                branch="own",
                number=i,
                status="completed",
                incumbent="own-c7",
                candidates=[f"own-c{i}"],
                conclusions={f"candidate:own-c{i}": [{"status": "inconclusive"}]},
            )
            for i in range(1, 9)
        ],
        history=[{"cycle": 1, "ranking": []}],
        ranking=[],
        research_conclusions=records,
    )


@pytest.fixture
def mcp(snapshot, monkeypatch):
    from mahjong_ai.interfaces import mcp as interface

    class Arena:
        def __init__(self, *args):
            pass

        def read(self, *args):
            return {"branches": {"own": {}}}

        def snapshot(self, *args):
            return deepcopy(snapshot)

    monkeypatch.setattr(interface, "Research", Arena)
    return interface.server("/unused", "run", "own")


def call(mcp, name, args):
    result = asyncio.run(mcp.call_tool(name, args))
    if isinstance(result, tuple):
        return result[1]
    return json.loads(result[0].text)


def test_history_query_does_not_attach_unrelated_ledger(mcp):
    result = call(mcp, "research_history", {"query": "no-match", "limit": 1})
    assert result["total"] == 0 and result["items"] == []
    assert "iterations" not in result and "research_conclusions" not in result
    assert len(json.dumps(result)) < 1000


def test_status_contains_only_pending_conclusions_and_current_candidates(mcp):
    result = call(mcp, "research_status", {})
    assert result["pending_conclusion_ids"] == ["candidate:own-c8"]
    assert "research_conclusions" not in result
    assert {c["id"] for c in result["candidates"]} == {"own-c7", "own-c8"}
    assert "field_group_means" not in json.dumps(result)


def test_history_pages_cover_all_conclusions_and_search(mcp, snapshot):
    offset = 0
    seen = []
    while offset is not None:
        page = call(
            mcp,
            "research_history",
            {"kind": "conclusions", "offset": offset, "limit": 2},
        )
        assert len(page["items"]) <= 2
        seen.extend(page["items"])
        offset = page["next_offset"]
    assert seen == snapshot["research_conclusions"]
    page = call(
        mcp, "research_history", {"kind": "conclusions", "query": "conclusion-3"}
    )
    assert [r["record_id"] for r in page["items"]] == ["candidate:own-c3"]


def test_history_size_limit_paginates_without_cutting_records(snapshot, monkeypatch):
    from mahjong_ai.research import agent_context

    monkeypatch.setattr(agent_context, "PAGE_CHARACTERS", 2000)
    for record in snapshot["research_conclusions"]:
        record["hypothesis"] = "long-" * 1000
    offset, seen = 0, []
    while offset is not None:
        page = agent_context.history_page(
            snapshot, "own", "conclusions", offset=offset, limit=50
        )
        assert len(page["items"]) == 1
        seen.extend(page["items"])
        offset = page["next_offset"]
    assert seen == snapshot["research_conclusions"]


def test_handoff_factors_conditions_without_losing_any_record(snapshot):
    from mahjong_ai.research.agent_context import handoff_view

    original = dict(
        branch="own", research_conclusions=deepcopy(snapshot["research_conclusions"])
    )
    compact = handoff_view(original)
    assert len(compact["research_conditions"]) < len(original["research_conclusions"])
    for before, after in zip(
        original["research_conclusions"], compact["research_conclusions"]
    ):
        reconstructed = deepcopy(after)
        condition = compact["research_conditions"][
            reconstructed["evidence"].pop("condition_id")
        ]
        reconstructed["evidence"].update(condition)
        assert reconstructed == before
    assert original["research_conclusions"] == snapshot["research_conclusions"]


def test_read_conclusion_preserves_evidence_and_revision_history(mcp, snapshot):
    result = call(mcp, "research_read_conclusion", {"record_id": "candidate:own-c2"})
    assert result["record"] == snapshot["research_conclusions"][1]
    assert (
        result["revisions"]
        == snapshot["iterations"][1]["conclusions"]["candidate:own-c2"]
    )
    with pytest.raises(Exception, match="本分支"):
        call(mcp, "research_read_conclusion", {"record_id": "candidate:other-c2"})


def test_readouts_do_not_include_another_branch(snapshot):
    from mahjong_ai.research.agent_context import history_page

    foreign = deepcopy(snapshot["research_conclusions"][0])
    foreign.update(branch="other", record_id="foreign-secret-record")
    snapshot["research_conclusions"].append(foreign)
    assert "foreign-secret-record" not in json.dumps(
        history_page(snapshot, "own", "conclusions")
    )


def test_iteration_history_keeps_statistics_without_raw_group_samples(snapshot):
    from mahjong_ai.research.agent_context import history_page

    snapshot["iterations"][0]["results"] = {
        "own-c1": {
            "mean_utility": 1.5,
            "paired": {"delta": 0.1},
            "group_blocks": list(range(256)),
            "field_group_means": [1.5] * 256,
        }
    }
    page = history_page(snapshot, "own", "iterations", query="own-i1", limit=1)
    assert page["items"][0]["results"]["own-c1"] == {
        "mean_utility": 1.5,
        "paired": {"delta": 0.1},
    }


def test_status_known_revision_returns_only_changes_and_recovers_lost_reply(
    mcp, snapshot
):
    first = call(mcp, "research_status", {})
    revision = first["revision"]
    same = call(mcp, "research_status", {"known_revision": revision})
    assert same == {"mode": "unchanged", "revision": revision}
    snapshot["candidates"][-1]["status"] = "draft"
    changed = call(mcp, "research_status", {"known_revision": revision})
    assert changed["mode"] == "delta"
    assert set(changed["changes"]) == {"candidates"}
    assert changed["changes"]["candidates"][-1]["status"] == "draft"
    # A lost reply must not advance the caller's cursor or suppress its changes.
    assert call(mcp, "research_status", {"known_revision": revision}) == changed
    recovered = call(
        mcp, "research_status", {"known_revision": "unavailable-old-server-revision"}
    )
    assert recovered["mode"] == "full"
    assert recovered["candidates"] == changed["changes"]["candidates"]


def test_status_does_not_repeat_handoff_or_authored_plan(mcp, snapshot):
    snapshot["iterations"][-1]["plan"] = {
        "assessment": "already-authored-plan",
        "direction": "already-authored-direction",
        "reason": "already-authored-reason",
    }
    result = call(mcp, "research_status", {})
    encoded = json.dumps(result)
    assert "hypothesis-8" not in encoded
    assert "conclusion-7" not in encoded
    assert "already-authored" not in encoded
    assert result["iteration"]["plan_declared"] is True
    assert result["pending_conclusion_ids"] == ["candidate:own-c8"]


def test_repeated_explicit_detail_read_is_not_suppressed(mcp):
    first = call(mcp, "research_read_conclusion", {"record_id": "candidate:own-c2"})
    assert (
        call(mcp, "research_read_conclusion", {"record_id": "candidate:own-c2"})
        == first
    )


def test_resume_prompt_does_not_send_handoff_again():
    from mahjong_ai.execution.agent import agent_prompt

    task = dict(
        brief="test",
        candidate_limit=1,
        thread_id="existing-thread",
        handoff={"research_conclusions": [{"hypothesis": "already-sent-hypothesis"}]},
    )
    prompt = agent_prompt(task)
    assert "already-sent-hypothesis" not in prompt
    assert "research_status" in prompt


def test_candidate_file_pages_are_bounded_and_lossless(mcp, monkeypatch):
    from mahjong_ai.interfaces import mcp as interface

    # Long lines and escaped embedded source occur in accumulated candidates.
    content = ('规则\\n"' * 7000) + "\nlast line\n"
    monkeypatch.setattr(
        interface.Research,
        "candidate",
        lambda *a, **kw: {"path": "strategy.py", "content": content},
        raising=False,
    )
    offset, chunks, revisions = 0, [], set()
    while offset is not None:
        page = call(
            mcp, "research_read_file", {"candidate_id": "own-c7", "offset": offset}
        )
        assert len(json.dumps(page, ensure_ascii=False).encode()) < 9000
        chunks.append(page["content"])
        revisions.add(page["revision"])
        offset = page["next_offset"]
    assert "".join(chunks) == content
    assert len(revisions) == 1
    assert (
        call(mcp, "research_read_file", {"candidate_id": "own-c7", "offset": 0})[
            "content"
        ]
        == chunks[0]
    )


def test_public_source_pages_cover_full_source_and_validate_offsets(mcp):
    from mahjong_ai.domain import rules
    from pathlib import Path

    offset, chunks = 0, []
    while offset is not None:
        page = call(mcp, "research_read_source", {"name": "rules", "offset": offset})
        assert len(json.dumps(page, ensure_ascii=False).encode()) < 9000
        chunks.append(page["content"])
        offset = page["next_offset"]
    assert "".join(chunks) == Path(rules.__file__).read_text()
    with pytest.raises(Exception, match="分页"):
        call(mcp, "research_read_source", {"name": "rules", "offset": -1})


def test_tool_replies_carry_the_remaining_iteration_budget(mcp, monkeypatch):
    """非交互会话只能靠工具返回提醒时间，所以每次调用都带 budget。"""
    import time

    monkeypatch.setenv("MAHJONG_ITERATION_BUDGET", "600")
    monkeypatch.setenv("MAHJONG_ITERATION_DEADLINE", str(int(time.time() + 500)))
    budget = call(mcp, "research_status", {})["budget"]
    assert budget["limit_seconds"] == 600 and 480 <= budget["remaining_seconds"] <= 500
    assert "留出" in budget["advice"]

    monkeypatch.setenv("MAHJONG_ITERATION_DEADLINE", str(int(time.time() + 60)))
    assert "最后提交窗口" in call(mcp, "research_status", {})["budget"]["advice"]

    monkeypatch.delenv("MAHJONG_ITERATION_DEADLINE")
    assert "budget" not in call(mcp, "research_status", {})


def test_agent_prompt_states_the_iteration_budget_and_submit_first_order():
    import time

    from mahjong_ai.execution.agent import agent_prompt

    task = {
        "brief": "牌效优先",
        "candidate_limit": 1,
        "deadline": time.time() + 600,
        "execution_limit_seconds": 600,
    }
    prompt = agent_prompt(task)
    assert "总预算 600 秒（10 分钟）" in prompt and "剩余约" in prompt
    assert "最小可行改动" in prompt and "budget.remaining_seconds" in prompt
    # 续跑调用沿用同一预算，也要带上剩余时间。
    assert "时间预算" in agent_prompt({**task, "continuation": 1})
    assert "时间预算" not in agent_prompt({k: v for k, v in task.items() if k != "deadline"})
