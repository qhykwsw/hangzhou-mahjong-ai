"""Typed research tools. There is deliberately no private-file or shell endpoint."""
import asyncio
import sys
import time
from pathlib import Path

from mcp.server.fastmcp import FastMCP

from .research import Research
from .research_templates import CONTRACT


def server(root):
    research=Research(root)
    mcp=FastMCP('mahjong_research_mcp',instructions=(
        'Optimize terminal average four-player rank. One persistent rule OR RL run. '
        'Create multiple code candidates, submit a batch, wait for ALL experiments, then decide once. '
        'Query history summaries and details. You cannot edit evaluation rules or grant budget.\n'+CONTRACT))
    ro={'readOnlyHint':True,'destructiveHint':False,'openWorldHint':False}
    rw={'readOnlyHint':False,'destructiveHint':False,'openWorldHint':False}

    @mcp.tool(annotations=ro)
    def research_status()->dict:
        """Get objective, remaining budgets, baseline, population and batch/acceptance states."""
        return research.status()

    @mcp.tool(annotations=ro)
    def research_history(query:str='',offset:int=0,limit:int=10)->dict:
        """Search all persisted candidate hypotheses, outcomes, batches and decisions. Paginated."""
        return research.history(query,offset,limit)

    @mcp.tool(annotations=ro)
    def research_details(kind:str,object_id:str,offset:int=0,limit:int=10)->dict:
        """Read candidate, batch, acceptance summary or trace (public decision observations).
        Batch results stay hidden until all experiments finish. Trace object_id is candidate ID.
        """
        return research.details(kind,object_id,offset,limit)

    @mcp.tool(annotations=ro)
    def research_sample_parents(limit:int=4)->dict:
        """Recommend baseline, quality leaders and an alternative; you may choose any stored candidate."""
        return research.sample(limit)

    @mcp.tool(annotations=ro)
    def research_read_source(name:str)->dict:
        """Read contract or SDK source: contract, types, policy, rules, candidate_worker.
        Use policy for the complete initial RL algorithm, types for all legal observations.
        types includes catch_play, god_discarder_seat and the derived catch_restricted flag.
        """
        if name=='contract':return {'content':CONTRACT}
        if name not in ('types','policy','rules','candidate_worker'):raise ValueError('unknown SDK source')
        return {'name':name,'content':Path(__file__).with_name(name+'.py').read_text()}

    @mcp.tool(annotations=rw)
    def research_create_candidate(candidate_id:str,hypothesis:str,parent:str|None=None,references:list[str]|None=None)->dict:
        """Fork a candidate package for unrestricted strategy/algorithm editing; record its hypothesis and ancestry."""
        return research.create_candidate(candidate_id,hypothesis,parent,references)

    @mcp.tool(annotations=ro)
    def research_read_file(candidate_id:str,path:str='strategy.py')->dict:
        """Read a candidate source file, including archived versions. Relative path only."""
        return research.candidate_file(candidate_id,path)

    @mcp.tool(annotations=rw)
    def research_write_file(candidate_id:str,path:str,content:str)->dict:
        """Write a complete Python module/text file in a draft candidate. No limited expression schema."""
        return research.candidate_file(candidate_id,path,content)

    @mcp.tool(annotations=rw)
    def research_set_dependencies(candidate_id:str,requirements:list[str])->dict:
        """Declare exact name==version packages. Host prepares a separate environment and locks dependencies."""
        return research.dependencies(candidate_id,requirements)

    @mcp.tool(annotations=rw)
    def research_submit_batch(candidates:list[str],pairs:int=20,training_episodes:int=64,league_samples:int=8)->dict:
        """Freeze candidate code, train RL, run paired developer games and candidate league in parallel.
        pairs is old/new complete settlement pairs per candidate. All results form ONE decision snapshot.
        Bootstrap a new run first with candidates=['initial']. Does not spend formal acceptance slots.
        """
        return research.submit_batch(candidates,pairs,training_episodes,league_samples)

    @mcp.tool(annotations=ro)
    async def research_wait(seconds:int=20)->dict:
        """Wait up to 30 seconds for active batch/acceptance completion, then return status. Never evolve early."""
        end=time.monotonic()+max(0,min(seconds,30))
        while True:
            status=research.status()
            running=any(b['status']=='running' for b in status['batches']) or any(a['status']=='running' for a in status['acceptances'])
            if not running or time.monotonic()>=end:return status
            await asyncio.sleep(.5)

    @mcp.tool(annotations=rw)
    def research_cancel(kind:str,object_id:str)->dict:
        """Stop a batch or acceptance job and record cancellation. Acceptance slot stays spent."""
        return research.cancel(kind,object_id)

    @mcp.tool(annotations=rw)
    def research_decide_batch(batch_id:str,retain:list[str],conclusion:str)->dict:
        """After ALL results finish, record one evidence-based decision and retained population. Archive others, never delete history."""
        return research.decide(batch_id,retain,conclusion)

    @mcp.tool(annotations=rw)
    def research_submit_acceptance(candidate_id:str)->dict:
        """Spend one slot on independent old/new 1000 settlements each. Same code/conditions reuse evidence. No private traces returned."""
        return research.accept(candidate_id)

    return mcp


if __name__=='__main__':server(sys.argv[1]).run(transport='stdio')
