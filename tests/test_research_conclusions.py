from copy import deepcopy

import pytest

from mahjong_ai.research.config import configuration
from mahjong_ai.research.conclusions import ledger
from mahjong_ai.research.service import Research


@pytest.fixture
def conclusions(tmp_path):
    (tmp_path / "project.toml").write_text(
        '[project]\nid="conclusions"\nfamily="test"\n'
    )
    config = configuration(tmp_path, {"evaluation_games": 4})
    state = {
        "id": "current-experiment",
        "status": "running",
        "last_ranking": [],
        "branches": {},
        "iterations": {},
        "candidates": {},
        "cycles": [{"config": config, "reference": ["opponent-v1"]}],
    }
    for branch in ("own", "other"):
        state["branches"][branch] = {
            "id": branch,
            "combination": "heuristic-balanced",
            "status": "researching",
            "iteration": branch + "-current",
            "representative": branch + "-reference",
            "completed": 5,
            "improved": 0,
            "failed": 0,
        }
        state["candidates"][branch + "-reference"] = {"id": branch + "-reference"}
        for number in range(1, 6):
            iid, cid = f"{branch}-i{number}", f"{branch}-c{number}"
            state["candidates"][cid] = {
                "id": cid,
                "branch": branch,
                "iteration": iid,
                "hypothesis": f"Hypothesis {number}",
                "parent": branch + "-reference",
            }
            result = {
                "appearances": 4,
                "mean_utility": 1.4,
                "errors": 0,
                "timeouts": 0,
                "fallback_actions": 0,
            }
            state["iterations"][iid] = {
                "id": iid,
                "branch": branch,
                "number": number,
                "cycle": 1,
                "experiment": "past-experiment",
                "status": "completed",
                "incumbent": branch + "-reference",
                "candidates": [cid],
                "results": {
                    cid: result,
                    branch + "-reference": {
                        **result,
                        "mean_utility": 1.5,
                    },
                },
            }
        state["iterations"][branch + "-current"] = {
            "id": branch + "-current",
            "branch": branch,
            "number": 6,
            "status": "researching",
        }
    research = Research(tmp_path)
    research.store.create(state)
    return research


def record(research, record_id="candidate:own-c1", **patch):
    values = dict(
        status="inconclusive",
        conclusion="Insufficient evidence for this hypothesis.",
        conditions="Only opponent-v1, 4 paired games.",
    )
    return research.record_conclusion(
        "current-experiment", "own", record_id, **(values | patch)
    )


def test_cumulative_handoff_keeps_old_conditions_and_does_not_infer_refutation(
    conclusions,
):
    research = conclusions
    record(research, status="refuted", conclusion="No benefit under these conditions.")
    restarted = Research(research.project)
    entries = restarted.handoff(restarted.read(), "own")["research_conclusions"]
    assert len(entries) == 5  # includes evidence older than the last three iterations
    assert entries[0]["assessment"]["status"] == "refuted"
    assert entries[0]["evidence"]["experiment"] == "past-experiment"
    assert entries[0]["evidence"]["opponents"] == ["opponent-v1"]
    assert entries[1]["assessment"] is None  # losing is not an automatic refutation
    assert entries[1]["evidence"]["mean_delta"] == pytest.approx(-0.1)
    assert all(row["branch"] == "own" for row in entries)


@pytest.mark.parametrize("status", ["verified", "refuted"])
@pytest.mark.parametrize(
    "problem",
    [
        "missing_result",
        "incomplete",
        "errors",
        "timeouts",
        "fallback_actions",
        "reference_error",
    ],
)
def test_invalid_evaluation_cannot_validate_or_refute(conclusions, status, problem):
    with conclusions.edit() as (s, db):
        results = s["iterations"]["own-i1"]["results"]
        if problem == "missing_result":
            results.pop("own-c1")
            s["candidates"]["own-c1"]["error"] = "ImportError: missing dependency"
        elif problem == "incomplete":
            results["own-c1"]["appearances"] = 2
        elif problem == "reference_error":
            results["own-reference"]["errors"] = 1
        else:
            results["own-c1"][problem] = 1
    with pytest.raises(ValueError, match="完整配对评估"):
        record(conclusions, status=status)
    with pytest.raises(ValueError, match="失败类型"):
        record(conclusions, status="untested")
    result = record(
        conclusions,
        status="untested",
        failure_kind="implementation",
        failure_reason="Execution was unreliable; fix and rerun the hypothesis.",
    )
    assert result["assessment"]["status"] == "untested"


def test_no_candidate_failure_is_preserved(conclusions):
    with conclusions.edit() as (s, db):
        s["candidates"].pop("own-c1")
        s["iterations"]["own-i1"].update(
            candidates=[], results={}, error="provider unavailable"
        )
    entry = ledger(conclusions.read(), "own")[0]
    assert entry["record_id"] == "iteration:own-i1"
    assert entry["evidence"]["failures"][0]["reason"] == "provider unavailable"
    result = record(
        conclusions,
        entry["record_id"],
        status="untested",
        failure_kind="infrastructure",
        failure_reason="Provider unavailable; retry later.",
    )
    assert result["assessment"]["failure_kind"] == "infrastructure"


def test_review_gate_is_idempotent_revisioned_and_branch_scoped(conclusions):
    with pytest.raises(ValueError, match="research_record_conclusion"):
        conclusions.declare_plan(
            "current-experiment", "own", "assessment", "direction", "reason"
        )
    first = record(conclusions)
    assert record(conclusions) == first
    revised = record(conclusions, conclusion="A narrower interpretation after review.")
    assert revised["revision_count"] == 2
    assert (
        conclusions.read()["iterations"]["own-i1"]["conclusions"]["candidate:own-c1"][0]
        == first["assessment"]
    )
    with pytest.raises(ValueError, match="本分支"):
        record(conclusions, "candidate:other-c1")
    with pytest.raises(ValueError, match="当前实验"):
        conclusions.record_conclusion(
            "past-experiment",
            "own",
            "candidate:own-c1",
            "inconclusive",
            "judgement",
            "scope",
        )
    for number in range(2, 6):
        record(conclusions, f"candidate:own-c{number}")
    conclusions.declare_plan(
        "current-experiment", "own", "assessment", "direction", "reason"
    )
    assert "plan" in conclusions.read()["iterations"]["own-current"]


def test_repaired_checks_keep_failure_history_but_allow_valid_evidence(conclusions):
    conclusions.record_check(
        "current-experiment", "own", "own-c1", {"ok": False, "error": "SyntaxError"}
    )
    conclusions.record_check(
        "current-experiment", "own", "own-c1", {"ok": True, "error": None}
    )
    entry = ledger(conclusions.read(), "own")[0]
    assert entry["outcome"] == "evaluated"
    assert entry["evidence"]["failures"] == [
        {"stage": "preflight", "reason": "SyntaxError"}
    ]
    with pytest.raises(ValueError, match="失败类型"):
        record(conclusions)
    result = record(
        conclusions,
        status="refuted",
        failure_kind="implementation",
        failure_reason="Fixed syntax before the full clean evaluation; observed loss is separate.",
    )
    assert result["assessment"]["status"] == "refuted"


def test_returned_evidence_is_not_mutable_state(conclusions):
    before = deepcopy(conclusions.read())
    entry = ledger(before, "own")[0]
    entry["evidence"]["game"]["rounds"] = 999
    entry["evidence"]["opponents"].clear()
    assert before == conclusions.read()


def test_mcp_exposes_and_executes_conclusion_tool(conclusions):
    import asyncio
    from mahjong_ai.interfaces.mcp import server

    mcp = server(conclusions.project, "current-experiment", "own")
    tools = {tool.name: tool for tool in asyncio.run(mcp.list_tools())}
    schema = tools["research_record_conclusion"].inputSchema
    assert set(schema["required"]) == {
        "record_id",
        "status",
        "conclusion",
        "conditions",
    }
    assert not tools["research_check_candidate"].annotations.readOnlyHint
    asyncio.run(
        mcp.call_tool(
            "research_record_conclusion",
            {
                "record_id": "candidate:own-c1",
                "status": "inconclusive",
                "conclusion": "Too few independent groups.",
                "conditions": "opponent-v1 only",
            },
        )
    )
    assert (
        ledger(conclusions.read(), "own")[0]["assessment"]["status"] == "inconclusive"
    )


def test_recovered_job_failure_and_focal_diagnostics_remain_in_evidence(conclusions):
    with conclusions.edit() as (s, db):
        s["iterations"]["own-i1"]["execution_failures"] = [
            {"stage": "match", "candidate_id": "own-c1", "reason": "worker exited"},
        ]
        s["jobs"] = {
            "own-job": {
                "iteration": "own-i1",
                "candidate": "own-c1",
                "diagnostics": [
                    {"type": "TimeoutError", "message": "action deadline exceeded"},
                ],
            },
            "other-job": {
                "iteration": "other-i1",
                "candidate": "other-c1",
                "diagnostics": [
                    {
                        "type": "ValueError",
                        "message": "other branch implementation detail",
                    },
                ],
            },
        }
    failures = ledger(conclusions.read(), "own")[0]["evidence"]["failures"]
    assert {f["reason"] for f in failures} == {
        "worker exited",
        "action deadline exceeded",
    }
