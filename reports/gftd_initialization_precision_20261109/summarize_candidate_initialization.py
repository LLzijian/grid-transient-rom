"""Read-only initialization summary; reproduces the executed PowerShell stdin analysis.
No model imports or scientific integrations. Outputs the initialization digest.
"""
import json,gzip,hashlib,collections,pathlib,numpy as np
paths=list(pathlib.Path('.').glob('results/gftd_initialization_precision_20261109/candidate/*/initialize/*.json')); assert len(paths)==594
rows=[]; hashes={}
for p in sorted(paths):
 b=p.read_bytes(); hashes[p.as_posix()]=hashlib.sha256(b).hexdigest(); x=json.loads(b); q=pathlib.Path(x['diagnostics']['path']); b=q.read_bytes(); assert hashlib.sha256(b).hexdigest()==x['diagnostics']['sha256']; hashes[q.as_posix()]=x['diagnostics']['sha256']; d=json.loads(gzip.decompress(b)); stages={}
 for s in ['stage1','stage2']:
  z=d.get(s)
  if z: stages[s]={k:z.get(k) for k in ['status','reason','actual_residual_calls','optimization_sufficiency','observation_projected_gradient_kkt','projected_gradient_kkt','optimizer_residual_divisor','scipy_status']}
 sel=x['stage']; z=d.get(sel) or d['stage1']; u=np.array(z['best']['u']); lo=np.array(z['lower_bounds']); hi=np.array(z['upper_bounds']); low=np.flatnonzero(u<=lo+1e-7); high=np.flatnonzero(u>=hi-1e-7); a=u[:10]*.1 if len(u)==20 else np.zeros(10)
 rows.append(dict(id=f"{x['mode']}_{x['index']:03d}",mode=x['mode'],label=x['label'],old_passed=x['old_gates']['passed'],passed=x['passed'],status=x['status'],stage=sel,history_mse=x['history_mse'],stages=stages,a_lower=int(sum(low<10)) if len(u)==20 else 0,a_upper=int(sum(high<10)) if len(u)==20 else 0,a_fixed=len(u)!=20,a=a.tolist(),r_lower=int(sum(low>=(10 if len(u)==20 else 0))),r_upper=int(sum(high>=(10 if len(u)==20 else 0))),stage1_passed=x['stage1_gates']['passed'],final_gates=x['final_gates'],error=x['stage1_gates'].get('error') if x['final_gates'] is None else x['final_gates'].get('error')))
assert len({r['id'] for r in rows})==594
stats={}
def rng(xs):
 xs=[v for v in xs if v is not None];return [min(xs),max(xs)] if xs else None
for mode in ['static','modulated']:
 rs=[r for r in rows if r['mode']==mode]; good=[r for r in rs if r['passed']]
 stats[mode]={'groups':{label:{'total':sum(r['label']==label for r in rs),'old':sum(r['old_passed'] for r in rs if r['label']==label),'new':sum(r['passed'] for r in rs if r['label']==label)} for label in ['stable','unstable']},'old_success_retained':sum(r['old_passed'] and r['passed'] for r in rs),'regressions':[r['id'] for r in rs if r['old_passed'] and not r['passed']],'stage_status':{s:dict(collections.Counter(r['stages'][s]['status'] for r in rs if s in r['stages'])) for s in ['stage1','stage2']},'history_change_range':rng(r['history_mse']['change'] for r in rs),'qualified_a_lower_total':sum(r['a_lower'] for r in good),'qualified_a_upper_total':sum(r['a_upper'] for r in good),'qualified_a_lower_more_events':sum(r['a_lower']>r['a_upper'] for r in good),'qualified_a_upper_more_events':sum(r['a_lower']<r['a_upper'] for r in good),'qualified_a_equal_events':sum(r['a_lower']==r['a_upper'] for r in good),'qualified_a_active_range':rng(r['a_lower']+r['a_upper'] for r in good),'selected_observation_kkt_range':rng(r['stages'][r['stage']]['observation_projected_gradient_kkt'] for r in rs),'selected_kkt_missing':sum(r['stages'][r['stage']]['observation_projected_gradient_kkt'] is None for r in rs)}
model=json.load(open('artifacts/gftd_initialization_precision_20261109/candidate/pilot/modulated_model.json'))['functional']; gamma={k:v for k,v in model.items() if 'gamma' in k};print('gamma',gamma)
digest={'scope':'594 closed initialization packets only; no ODE, optimization, reference, spectrum, or future truth read','boundary_tolerance_u':1e-7,'a_scale':.1,'gamma':gamma,'sources':hashes,'summary':stats,'rows':rows};pathlib.Path('results/gftd_initialization_precision_20261109/candidate_initialization_digest.json').write_text(json.dumps(digest,ensure_ascii=False,indent=2),encoding='utf-8');print(json.dumps(stats,indent=2));print('failures',[(r['id'],r['error']) for r in rows if not r['passed']])
