"""分支作用域的研究工具；评估条件全部由研究生命周期拥有。"""

from pathlib import Path
import sys
from typing import Literal
from mcp.server.fastmcp import FastMCP
from mahjong_ai.research.service import Research
from mahjong_ai.research.contract import CONTRACT
from mahjong_ai.research.agent_context import (
    status_view,
    history_page,
    read_conclusion,
    StatusReadout,
    text_page,
)


def server(project, experiment_id, branch):
    arena = Research(project)
    if branch not in arena.read(experiment_id)["branches"]:
        raise ValueError("未知分支")
    mcp = FastMCP("mahjong_research", instructions=CONTRACT)
    status_readout = StatusReadout()
    ro = {"readOnlyHint": True, "destructiveHint": False, "openWorldHint": False}
    rw = {"readOnlyHint": False, "destructiveHint": False, "openWorldHint": False}

    @mcp.tool(annotations=ro)
    def research_status(known_revision: str | None = None) -> dict:
        """读取必要状态和待复盘 ID，正文已在开场交接。首次省略 known_revision；后续传上次收到的 revision，仅返回变化或 unchanged。delta.changes 替换对应顶层字段，removed 删除字段。省略版本号可重新取得完整当前状态；版本不可用时自动返回 full。"""
        return status_readout.reply(
            status_view(arena.snapshot(experiment_id, branch), branch), known_revision
        )

    @mcp.tool(annotations=ro)
    def research_history(
        query: str = "",
        offset: int = 0,
        limit: int = 5,
        kind: Literal["candidates", "iterations", "conclusions"] = "candidates",
    ) -> dict:
        """按类型和关键词分页检索本分支历史，不附加无关历史。返回完整记录；以 next_offset 继续翻页，实际条数可能小于 limit。"""
        return history_page(
            arena.snapshot(experiment_id, branch), branch, kind, query, offset, limit
        )

    @mcp.tool(annotations=ro)
    def research_read_conclusion(record_id: str) -> dict:
        """按 record_id 读取本分支单条累计结论、完整机械证据和全部修订；包括没有候选的失败尝试。"""
        return read_conclusion(arena.snapshot(experiment_id, branch), branch, record_id)

    @mcp.tool(annotations=ro)
    def research_read_source(
        name: Literal[
            "contract",
            "capabilities",
            "types",
            "rules",
            "observations",
            "policy",
            "candidate_worker",
            "shape",
            "simulation",
            "learning",
            "heuristic",
            "search",
            "trajectory",
        ] = "contract",
        offset: int = 0,
        limit: int = 4000,
    ) -> dict:
        """按枚举名称分段读取公开源码，以 next_offset 继续，offset/limit 单位为字符。policy 对应 strategies/common.py（RulePolicy、risk_preference 等），不是 common。
        heuristic 提供定向规则与风险风格基线；types 为观察和动作类型，observations 为事件与财飘语义，
        rules 为判胡与番型，shape 为牌效，simulation 为假想采样；其余源码用 name 枚举中的名称。
        engine、development、策略组合 ID 不是可读源码名。"""
        if name == "contract":
            return text_page(CONTRACT, offset, limit)
        if name not in (
            "types",
            "rules",
            "observations",
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
        content = (
            Path(__file__).parents[1]
            / {
                "types": "domain/types.py",
                "rules": "domain/rules.py",
                "observations": "domain/observations.py",
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
        return text_page(content, offset, limit)

    @mcp.tool(annotations=rw)
    def research_record_conclusion(
        record_id: str,
        status: str,
        conclusion: str,
        conditions: str,
        failure_kind: str = "none",
        failure_reason: str = "",
    ) -> dict:
        """整理本分支已完成尝试的累计结论，record_id 来自开场 research_conclusions 或 research_status 的 pending_conclusion_ids。
        status: verified/refuted/inconclusive/untested。conditions 写明适用条件与局限。
        failure_kind: none/implementation/resource_limit/infrastructure/unknown；有失败须说明 failure_reason。
        未完整评估或评估含报错、超时、代打时，不得标 verified/refuted。重复同内容幂等，修改保留历史。
        声明新计划前须整理所有 assessment 为空的记录。"""
        return arena.record_conclusion(
            experiment_id,
            branch,
            record_id,
            status,
            conclusion,
            conditions,
            failure_kind,
            failure_reason,
        )

    @mcp.tool(annotations=rw)
    def research_declare_plan(assessment: str, direction: str, reason: str) -> dict:
        """先整理待复盘的累计研究结论，再声明现役判断、本次方向和原因；必须在写草稿之前完成。"""
        return arena.declare_plan(experiment_id, branch, assessment, direction, reason)

    @mcp.tool(annotations=rw)
    def research_create_candidate(
        candidate_id: str, hypothesis: str, parent: str | None = None
    ) -> dict:
        """创建本分支草稿。candidate_id 是 1..64 位字母、数字、下划线或连字符短名，不含分支前缀；parent 可指定本分支历史候选完整 ID。"""
        return arena.fork(experiment_id, branch, candidate_id, hypothesis, parent)

    @mcp.tool(annotations=ro)
    def research_read_file(
        candidate_id: str, path: str = "strategy.py", offset: int = 0, limit: int = 4000
    ) -> dict:
        """分段读取本分支候选文件，包括已归档源码。offset/limit 单位为字符，以 next_offset 继续；按 revision 核对文件有无变化。"""
        result = arena.candidate(experiment_id, branch, candidate_id, path=path)
        return {"path": result["path"], **text_page(result["content"], offset, limit)}

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

    @mcp.tool(annotations=rw)
    def research_check_candidate(candidate_id: str) -> dict:
        """试跑草稿并记录错误：导入、build；规则/搜索策略实际打一局，报错、超时、非法动作均拒绝；可训练策略打一场训练。提交前必须修好所有失败。"""
        return arena.check(experiment_id, branch, candidate_id)

    @mcp.tool(annotations=rw)
    def research_submit_batch(
        candidates: list[str], training_games: int | None = None
    ) -> dict:
        """冻结本分支本次迭代派生的草稿并排入准备与评估；历史草稿应作为 parent 新建候选。
        数量上限为 research_status.config.candidate_limit，不固定为四个。
        heuristic 与 search 组合不训练：不要定义 strategy.train，也不要传
        training_games。可训练组合依据上次的执行反馈决定 training_games，省略则用配置默认值。提交后立即结束本次回复。"""
        return arena.submit(experiment_id, branch, candidates, training_games)

    return mcp


if __name__ == "__main__":
    server(*sys.argv[1:4]).run(transport="stdio")
