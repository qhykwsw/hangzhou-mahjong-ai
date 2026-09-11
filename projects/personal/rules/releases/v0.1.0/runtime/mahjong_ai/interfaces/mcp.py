"""分支作用域的研究工具；评估条件全部由研究生命周期拥有。"""

from pathlib import Path
import sys
from mcp.server.fastmcp import FastMCP
from mahjong_ai.research.service import Research
from mahjong_ai.research.contract import CONTRACT


def server(project, experiment_id, branch):
    arena = Research(project)
    if branch not in arena.read(experiment_id)["branches"]:
        raise ValueError("未知分支")
    mcp = FastMCP("mahjong_research", instructions=CONTRACT)
    ro = {"readOnlyHint": True, "destructiveHint": False, "openWorldHint": False}
    rw = {"readOnlyHint": False, "destructiveHint": False, "openWorldHint": False}

    @mcp.tool(annotations=ro)
    def research_status() -> dict:
        """读取本分支每次迭代的执行反馈（实际训练场数与经过秒数）、候选、额度与共同排名。"""
        return arena.snapshot(experiment_id, branch)

    @mcp.tool(annotations=ro)
    def research_history(query: str = "", offset: int = 0, limit: int = 10) -> dict:
        """读取本分支的候选历史、带执行反馈的迭代记录，以及已完成的共同竞技成绩。"""
        if offset < 0 or not 1 <= limit <= 50:
            raise ValueError("分页参数无效")
        state = arena.snapshot(experiment_id, branch)
        rows = [
            c
            for c in state["candidates"]
            if c["branch"] == branch and query.lower() in str(c).lower()
        ]
        return {
            "total": len(rows),
            "items": rows[offset : offset + limit],
            "rounds": state["history"][-3:],
            "iterations": state["iterations"],
        }

    @mcp.tool(annotations=ro)
    def research_read_source(name: str = "contract") -> dict:
        """读取契约、capabilities（含自由启发式的风格例外），或公开的 types/rules/policy/training/simulation SDK 源码。"""
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
            raise ValueError("未知的公开 SDK 源码名")
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
    def research_declare_plan(assessment: str, direction: str, reason: str) -> dict:
        """迭代的第一步，必须在写草稿之前完成：对现役做法与其轨迹的判断、本次尝试的方向和原因。展示在研究控制台上。"""
        return arena.declare_plan(experiment_id, branch, assessment, direction, reason)

    @mcp.tool(annotations=rw)
    def research_create_candidate(
        candidate_id: str, hypothesis: str, parent: str | None = None
    ) -> dict:
        """创建本分支的草稿，可指定从本分支已有的候选派生。"""
        return arena.fork(experiment_id, branch, candidate_id, hypothesis, parent)

    @mcp.tool(annotations=ro)
    def research_read_file(candidate_id: str, path: str = "strategy.py") -> dict:
        """读取本分支某个候选的文件；已归档的源码仍可读取。"""
        return arena.candidate(experiment_id, branch, candidate_id, path=path)

    @mcp.tool(annotations=rw)
    def research_write_file(candidate_id: str, path: str, content: str) -> dict:
        """写入草稿的源文件；已冻结的提交不能再修改。"""
        return arena.candidate(
            experiment_id, branch, candidate_id, path=path, content=content
        )

    @mcp.tool(annotations=rw)
    def research_set_dependencies(candidate_id: str, requirements: list[str]) -> dict:
        """为草稿声明精确 name==version 的 wheel 依赖。"""
        return arena.dependencies(experiment_id, branch, candidate_id, requirements)

    @mcp.tool(annotations=ro)
    def research_check_candidate(candidate_id: str) -> dict:
        """按真实任务的方式试跑草稿：导入、build、打一场训练。提交前必须修好所有失败，否则整批会被拒绝。"""
        return arena.check(experiment_id, branch, candidate_id)

    @mcp.tool(annotations=rw)
    def research_submit_batch(
        candidates: list[str], training_games: int | None = None
    ) -> dict:
        """冻结本分支草稿并排入准备与评估。heuristic 与 search 组合不训练：不要定义 strategy.train，也不要传
        training_games。可训练组合依据上次的执行反馈决定 training_games，省略则用配置默认值。提交后立即结束本次回复。"""
        return arena.submit(experiment_id, branch, candidates, training_games)

    return mcp


if __name__ == "__main__":
    server(*sys.argv[1:4]).run(transport="stdio")
