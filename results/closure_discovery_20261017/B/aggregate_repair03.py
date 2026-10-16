from pathlib import Path
import sys,time,json,os,hashlib
ROOT=Path(__file__).resolve().parents[3]
sys.path[:0]=[str(ROOT/'experiments'),str(ROOT/'src')]
import torch
import discover_closure as dc
from closure_discovery_support import ProcessLedger
out=ROOT/'private_data/closure_discovery_20261017'
torch.set_num_threads(1)
ledger=ProcessLedger(out/'ledger')
def snapshot():
 return {str(p):dc.sha(p) for mode in ('library','generated') for p in (out/'B'/mode).rglob('*') if p.is_file() and (p.name.startswith('model') or p.name.startswith('training') or p.name.startswith('checkpoint') or 'failure' in p.name)}
before=snapshot()
identity=dc.bind_execution_sources(out)
def forbidden_fit(*args,**kwargs):raise RuntimeError('repair03 aggregation cannot add adaptation updates')
dc.fit=forbidden_fit
started=time.perf_counter()
with ledger.run('B','cached-aggregation-repair03') as receipt:
 result=dc.stage_b(out,time.perf_counter()+min(1800,ledger.remaining('B')-300))
 assert snapshot()==before,'cached aggregation changed scientific checkpoint bytes'
 result['cached_aggregation_only']=True
 result['new_fit_calls']=0
 result['implementation_identity_sha256']=identity
 dc.write(out/'B/repair03_aggregation_result.json',result)
 compact={k:v for k,v in result.items() if k!='selected'}
 compact['selected']={k:({a:b for a,b in v.items() if a!='score'}|{'score':{a:b for a,b in v['score'].items() if a!='events'}} if isinstance(v,dict) and 'score' in v else v) for k,v in result['selected'].items()}
 dc.write(out/'B/repair03_aggregation_compact.json',compact)
 print(json.dumps({'completed':True,'receipt_id':receipt,'seconds':time.perf_counter()-started,'summary':{k:v for k,v in compact.items() if k!='selected'}},default=str),flush=True)
