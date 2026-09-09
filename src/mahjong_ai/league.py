"""Resumable local league of frozen versions with a separate Elo ledger."""
from pathlib import Path
from .engine import MahjongEngine, derive_seed
from .ratings import EloLedger
from .storage import atomic_json, digest, exclusive_lock, read_json
from .types import RANK_OBJECTIVE


def run_league(policies, config, samples, seed, output):
    if len(policies) != 4 or samples < 1: raise ValueError('four policies and positive sample count required')
    root = Path(output)
    ids = [p.identity for p in policies]
    context = {'engine':MahjongEngine.name,'config':config.as_dict()}
    protocol = {'objective':RANK_OBJECTIVE,'context':context,'policies':ids,'samples':samples,'seed':seed}
    with exclusive_lock(root / '.league.lock'):
        path=root/'protocol.json'
        if path.exists() and read_json(path) != protocol: raise ValueError('league protocol changed')
        atomic_json(path,protocol)
        ledger=EloLedger(root/'elo.json',context)
        totals=[0.]*4
        for i in range(samples):
            path=root/f'sample-{i:06}.json'
            order=[(slot+i)%4 for slot in range(4)]
            if path.exists():
                row=read_json(path)
                checksum=row.pop('checksum',None)
                if checksum!=digest(row) or row['protocol_id']!=digest(protocol) or row['index']!=i:
                    raise ValueError('corrupt league sample')
            else:
                result=MahjongEngine(record_events=False).run_round([policies[j].build() for j in order],config,derive_seed(seed,'league',i))
                if not result.completed: raise RuntimeError('league settlement incomplete: '+result.metadata['reason'])
                row={'protocol_id':digest(protocol),'index':i,'order':order,'result':result.as_dict()}
                atomic_json(path,row|{'checksum':digest(row)})
            ranks=row['result']['rank_by_seat']
            ledger.record(digest([protocol,i]),[ids[j] for j in order],ranks,
                          completed=row['result']['completed'],source='research-league')
            for slot,j in enumerate(order): totals[j]+=ranks[slot]
        result={'status':'completed','objective':RANK_OBJECTIVE,'samples':samples,
                'table':ledger.table(),'policies':[{'version':p.version,'policy_id':ids[i],'mean_rank':totals[i]/samples} for i,p in enumerate(policies)],
                'note':'Descriptive Elo from local league only; no promotion or training.'}
        atomic_json(root/'summary.json',result)
        return result
