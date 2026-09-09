"""Branch-scoped tools; research lifecycle owns all evaluation conditions."""

from pathlib import Path
import sys
from mcp.server.fastmcp import FastMCP
from mahjong_ai.research.service import Research
from mahjong_ai.research.contract import CONTRACT


def server(project, run_id, branch):
    arena = Research(project)
    if branch not in arena.read(run_id)["branches"]:
        raise ValueError("unknown branch")
    mcp = FastMCP("mahjong_research", instructions=CONTRACT)
    ro = {"readOnlyHint": True, "destructiveHint": False, "openWorldHint": False}
    rw = {"readOnlyHint": False, "destructiveHint": False, "openWorldHint": False}

    @mcp.tool(annotations=ro)
    def research_status() -> dict:
        """Read iteration allowance, own candidates and shared cycle ranking."""
        return arena.snapshot(run_id, branch)

    @mcp.tool(annotations=ro)
    def research_history(query: str = "", offset: int = 0, limit: int = 10) -> dict:
        """Search this branch's candidate hypotheses and completed competition evidence."""
        if offset < 0 or not 1 <= limit <= 50:
            raise ValueError("invalid page")
        state = arena.snapshot(run_id, branch)
        rows = [
            c
            for c in state["candidates"]
            if c["branch"] == branch and query.lower() in str(c).lower()
        ]
        return {
            "total": len(rows),
            "items": rows[offset : offset + limit],
            "cycles": state["history"][-3:],
            "iterations": state["iterations"],
        }

    @mcp.tool(annotations=ro)
    def research_read_source(name: str = "contract") -> dict:
        """Read contract, capabilities validator, or public types/rules/policy/training/simulation SDK."""
        if name == "contract":
            return {"content": CONTRACT}
        if name not in (
            "types",
            "rules",
            "policy",
            "candidate_worker",
            "shape",
            "simulation",
            "learning",
            "heuristic",
            "search",
            "trajectory",
            "capabilities",
        ):
            raise ValueError("unknown public SDK source")
        return {
            "content": (
                Path(__file__).parents[1]
                / {
                    "types": "domain/types.py",
                    "rules": "domain/rules.py",
                    "policy": "strategies/common.py",
                    "candidate_worker": "execution/candidate_worker.py",
                    "shape": "sdk/shape.py",
                    "simulation": "sdk/simulation.py",
                    "learning": "strategies/learning.py",
                    "heuristic": "strategies/heuristic.py",
                    "search": "strategies/search.py",
                    "trajectory": "sdk/trajectory.py",
                    "capabilities": "strategies/catalog.py",
                }[name]
            ).read_text()
        }

    @mcp.tool(annotations=rw)
    def research_create_candidate(
        candidate_id: str, hypothesis: str, parent: str | None = None
    ) -> dict:
        """Create an own-branch draft, optionally from one of its existing candidates."""
        return arena.fork(run_id, branch, candidate_id, hypothesis, parent)

    @mcp.tool(annotations=ro)
    def research_read_file(candidate_id: str, path: str = "strategy.py") -> dict:
        """Read one own-branch candidate file. Archived source remains available."""
        return arena.candidate(run_id, branch, candidate_id, path=path)

    @mcp.tool(annotations=rw)
    def research_write_file(candidate_id: str, path: str, content: str) -> dict:
        """Write a draft source file; frozen submissions cannot be edited."""
        return arena.candidate(run_id, branch, candidate_id, path=path, content=content)

    @mcp.tool(annotations=rw)
    def research_set_dependencies(candidate_id: str, requirements: list[str]) -> dict:
        """Declare exact name==version wheels for a draft."""
        return arena.dependencies(run_id, branch, candidate_id, requirements)

    @mcp.tool(annotations=rw)
    def research_submit_batch(
        candidates: list[str], training_games: int | None = None
    ) -> dict:
        """Freeze 1..4 own drafts and queue preparation plus mixed competition. End this turn after submitting."""
        return arena.submit(run_id, branch, candidates, training_games)

    return mcp


if __name__ == "__main__":
    server(*sys.argv[1:4]).run(transport="stdio")
