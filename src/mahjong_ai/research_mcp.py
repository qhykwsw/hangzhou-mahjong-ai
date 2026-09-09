"""Direction-scoped candidate tools; the arena alone controls competition."""
from pathlib import Path
import sys
from mcp.server.fastmcp import FastMCP
from .arena import Arena
from .research_templates import CONTRACT


def server(project, run_id, direction):
    arena = Arena(project)
    if direction not in arena.read(run_id)['directions']:
        raise ValueError('unknown direction')
    mcp = FastMCP('mahjong_research', instructions=CONTRACT)
    ro = {'readOnlyHint': True, 'destructiveHint': False, 'openWorldHint': False}
    rw = {'readOnlyHint': False, 'destructiveHint': False, 'openWorldHint': False}

    @mcp.tool(annotations=ro)
    def research_status() -> dict:
        """Read remaining budget, own drafts/batches and shared period ranking."""
        return arena.snapshot(run_id, direction)

    @mcp.tool(annotations=ro)
    def research_history(query: str = '', offset: int = 0, limit: int = 10) -> dict:
        """Search this direction's candidate hypotheses and completed competition evidence."""
        if offset < 0 or not 1 <= limit <= 50:
            raise ValueError('invalid page')
        state = arena.snapshot(run_id, direction)
        rows = [c for c in state['candidates'] if c['direction'] == direction
                and query.lower() in str(c).lower()]
        return {'total': len(rows), 'items': rows[offset:offset+limit], 'periods': state['periods'][-3:]}

    @mcp.tool(annotations=ro)
    def research_read_source(name: str = 'contract') -> dict:
        """Read contract or the public types, rules, policy and candidate_worker SDK."""
        if name == 'contract':
            return {'content': CONTRACT}
        if name not in ('types', 'rules', 'policy', 'candidate_worker'):
            raise ValueError('unknown public SDK source')
        return {'content': Path(__file__).with_name(name + '.py').read_text()}

    @mcp.tool(annotations=rw)
    def research_create_candidate(candidate_id: str, hypothesis: str, parent: str | None = None) -> dict:
        """Create an own-direction draft, optionally from one of its existing candidates."""
        return arena.fork(run_id, direction, candidate_id, hypothesis, parent)

    @mcp.tool(annotations=ro)
    def research_read_file(candidate_id: str, path: str = 'strategy.py') -> dict:
        """Read one own-direction candidate file. Archived source remains available."""
        return arena.candidate(run_id, direction, candidate_id, path=path)

    @mcp.tool(annotations=rw)
    def research_write_file(candidate_id: str, path: str, content: str) -> dict:
        """Write a draft source file; frozen submissions cannot be edited."""
        return arena.candidate(run_id, direction, candidate_id, path=path, content=content)

    @mcp.tool(annotations=rw)
    def research_set_dependencies(candidate_id: str, requirements: list[str]) -> dict:
        """Declare exact name==version wheels for a draft."""
        return arena.dependencies(run_id, direction, candidate_id, requirements)

    @mcp.tool(annotations=rw)
    def research_submit_batch(candidates: list[str], training_games: int | None = None) -> dict:
        """Freeze 1..4 own drafts and queue preparation plus mixed competition. End this turn after submitting."""
        return arena.submit(run_id, direction, candidates, training_games)

    return mcp


if __name__ == '__main__':
    server(*sys.argv[1:4]).run(transport='stdio')
