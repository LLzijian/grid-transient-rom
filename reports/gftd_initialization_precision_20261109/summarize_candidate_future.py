"""Frozen full-run receipt analysis only; never imports model code or solves ODEs."""
import json,pathlib,hashlib,collections
base=pathlib.Path('results/gftd_initialization_precision_20261109'); paths=sorted((base/'candidate/full/future').glob('*.json'));assert len(paths)==579
rs=[json.loads(p.read_bytes()) for p in paths]; init=json.load(open(base/'candidate_initialization_digest.json')); labels={r['id']:r['label'] for r in init['rows']}
def ident(r):return f"{r['mode']}_{r['index']:03d}"
def mx(rows,t,m):
 r=max(rows,key=lambda r:r['differences'][t]['observed'][m]);return dict(value=r['differences'][t]['observed'][m],id=ident(r))
summ={}
for mode in ['static','modulated']:
 rr=[r for r in rs if r['mode']==mode]; summ[mode]={'horizons':{}}
 for t in ['1','2','5','8','15']:
  avail=[r for r in rr if r['differences'][t] is not None];summ[mode]['horizons'][t]={'n':len(avail),'stable':sum(labels[ident(r)]=='stable' for r in avail),'unstable':sum(labels[ident(r)]=='unstable' for r in avail),'rms':mx(avail,t,'rms'),'peak':mx(avail,t,'peak')}
failed=[];unresolved=[]
for r in rs:
 if not all(r['integrals'][s]['completed'] for s in ['main','fine']):
  failed.append({'id':ident(r),'event_id':r['event_id'],'label':labels[ident(r)],'integrals':{s:{k:r['integrals'][s].get(k) for k in ['last_accepted_time','first_failure_interval','failure','component_ranges']} for s in ['main','fine']},'single_reference':r['single_radau_reference']})
 ref=r['single_radau_reference']
 if ref and ref['status']!='confirmed':unresolved.append({'id':ident(r),'event_id':r['event_id'],'label':labels[ident(r)],'reference':ref,'integral':r['integrals']['radau'],'pair15_completed':all(r['integrals'][s]['completed'] for s in ['main','fine'])})
exe=[json.load(open(base/f'candidate/{phase}/execution.json')) for phase in ['pilot','full']];assert all(e['status']=='COMPLETED' for e in exe)
cost={k:sum(e['new_costs'][k] for e in exe) for k in exe[0]['new_costs']};cost['host_wall_seconds_sum']=sum(e['host_wall_seconds'] for e in exe)
d={'sources':{p.as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},'summary':summ,'long_failures':failed,'reference_unresolved':unresolved,'candidate_pilot_plus_full_costs':cost}
(base/'candidate_future_scientific_digest.json').write_text(json.dumps(d,indent=2),encoding='utf-8');print(json.dumps(summ,indent=2));print('FAILURES',[(r['id'],[(s,r['integrals'][s]['first_failure_interval'],r['integrals'][s]['failure']['error']) for s in ['main','fine']]) for r in failed]);print('UNRESOLVED',[(r['id'],r['label'],r['integral']['last_accepted_time'],r['integral']['actual_rhs_calls'],r['integral']['error'],r['pair15_completed']) for r in unresolved]);print('COST',cost)
