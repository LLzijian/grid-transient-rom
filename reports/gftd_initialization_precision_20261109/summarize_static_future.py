"""Summarize closed static future receipts; no scientific solves or truth reads."""
import json,pathlib,hashlib
paths=sorted(pathlib.Path('results/gftd_initialization_precision_20261109/candidate/full/future').glob('static_*.json')); assert len(paths)==297
rs=[json.loads(p.read_bytes()) for p in paths]
assert all(r['integrals'][s]['completed'] for r in rs for s in ['main','fine'])
refs=[r for r in rs if r['single_radau_reference'] is not None]; assert len(refs)==251 and all(r['single_radau_reference']['status']=='confirmed' for r in refs)
def mx(rows,func):
 r=max(rows,key=func);return {'value':func(r),'index':r['index'],'event_id':r['event_id']}
d={'scope':'297 closed static 15-second pairs; no full-run conclusion','sources':{p.as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},'pairs_completed':297,'new_single_radau_confirmed':251,'observed_difference_max':{t:{m:mx(rs,lambda r:r['differences'][t]['observed'][m]) for m in ['rms','peak']} for t in ['1','2','5','8','15']},'new_single_radau_difference_max':{m:mx(refs,lambda r:r['single_radau_reference']['difference']['observed'][m]) for m in ['rms','peak']},'costs':{'actual_rhs_calls':sum(r['counts']['actual_rhs_calls'] for r in rs),'integrations_attempted':sum(r['counts']['integrations_attempted'] for r in rs),'integrations_completed':sum(r['counts']['integrations_completed'] for r in rs),'worker_cpu_seconds':sum(r['worker_cpu_seconds'] for r in rs),'worker_wall_seconds_sum':sum(r['worker_wall_seconds'] for r in rs),'main_cpu_seconds_sum':sum(r['integrals']['main']['cpu_seconds'] for r in rs),'fine_cpu_seconds_sum':sum(r['integrals']['fine']['cpu_seconds'] for r in rs)}}
pathlib.Path('results/gftd_initialization_precision_20261109/candidate_static_future_digest.json').write_text(json.dumps(d,indent=2),encoding='utf-8');print(json.dumps({k:v for k,v in d.items() if k!='sources'},indent=2))
