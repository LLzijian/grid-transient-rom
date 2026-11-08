"""Independent read-only QA for completed Loop0077 artifacts."""
from __future__ import annotations
import csv, hashlib, json
from pathlib import Path
import h5py
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
IDS = ("boundary_0000", "boundary_0002", "boundary_0015")
ARMS = ("repeat_0025_tol1e-7", "refine_00125_tol1e-7", "refine_000625_tol1e-7", "tol_0025_tol1e-9")
S_DELTA, S_OMEGA, OMEGA_S = .47675314588683515, 1.0033667249602924, 2*np.pi*60.

def _load(root: Path):
    splits = json.loads((root/'splits.json').read_text())
    if splits.get('validation') or splits.get('test') or not set(IDS).issubset(set(splits.get('train', []))):
        raise AssertionError('selected source is not all-TRAIN')
    rows = {}
    with (root/'dataset_index.csv').open(newline='') as f:
        rows = {r['event_id']: r for r in csv.DictReader(f)}
    old = {}
    with h5py.File(root/'dataset.h5','r') as f:
        mass = np.asarray(f['inertia'][...], float)
        for eid in IDS:
            if eid not in f['events'] or eid not in rows: raise AssertionError('missing selected old payload')
            g=f['events'][eid]
            old[eid]={'attrs': {k: g.attrs[k].item() if hasattr(g.attrs[k], 'item') else g.attrs[k] for k in g.attrs},
                      'arrays': {k: np.asarray(g[k][...]) for k in g}}
    return old, mass

def _float(v):
    try: return float(v)
    except (TypeError, ValueError): return None

def _metrics(a, b, mass, strict_repeat=False):
    if a is None or b is None: return {'status':'UNKNOWN','structural_checks':{},'missing':a is None or b is None}
    aa,bb=a['attrs'],b['attrs']; xa,xb=a['arrays'],b['arrays']
    n=min(len(xa.get('t',[])),len(xb.get('t',[])))
    checks={'label': aa.get('label') == bb.get('label') and aa.get('label') in ('stable','unstable'),
            'signature': aa.get('postfault_signature') == bb.get('postfault_signature'),
            'origin': (_float(aa.get('record_start_delay_s')) is not None and _float(bb.get('record_start_delay_s')) is not None and abs(_float(aa.get('record_start_delay_s'))-_float(bb.get('record_start_delay_s'))) <= 1e-8),
            'grid_prefix': 't' in xa and 't' in xb and n > 0 and np.allclose(xa['t'][:n],xb['t'][:n],atol=1e-8,rtol=0),
            'finite': all(np.isfinite(xa[k]).all() and np.isfinite(xb[k]).all() for k in set(xa)&set(xb))}
    checks['strict_grid_length'] = len(xa.get('t',[])) == len(xb.get('t',[]))
    checks['strict_end_time'] = bool(n and abs(float(xa['t'][n-1])-float(xb['t'][n-1])) <= 1e-8)
    if strict_repeat:
        checks['strict_repeat'] = checks['strict_grid_length'] and checks['strict_end_time']
    result={'structural_checks':checks,'shared_points':n}
    if n:
        d=xa['delta'][:n]-xb['delta'][:n]; w=(xa['omega'][:n]-xb['omega'][:n])*OMEGA_S
        coi=lambda x: x-(x*mass).sum(axis=1,keepdims=True)/mass.sum()
        dc,wc=coi(d),coi(w); common=(w*mass[None,:]).sum(axis=1)/mass.sum()
        result.update(delta_rmse_normalized=float(np.sqrt(np.mean((dc/S_DELTA)**2))), omega_rmse_normalized=float(np.sqrt(np.mean((wc/S_OMEGA)**2))), combined_rmse_normalized=float(np.sqrt(.5*np.mean((dc/S_DELTA)**2+(wc/S_OMEGA)**2))), common_frequency_rmse_normalized=float(np.sqrt(np.mean((common/S_OMEGA)**2))))
        if aa.get('label') == 'unstable':
            ca=_float(aa.get('clearing_time_s')); cb=_float(bb.get('clearing_time_s'))
            ea=_float(aa.get('event_time_s')); eb=_float(bb.get('event_time_s'))
            da=None if ea is None or eb is None else abs((ea-ca)-(eb-cb))
            result['los_since_clearing_difference_s']=da
    result['status']='OBSERVED'; return result

def _case(path, eid, arm):
    meta=json.loads((path/f'{eid}__{arm}.json').read_text())
    if meta.get('status') != 'completed': return None, {'status':meta.get('status'),'error':meta.get('error')}
    payload={'attrs':meta['metadata'],'arrays':{k:v for k,v in np.load(path/f'{eid}__{arm}.npz',allow_pickle=False).items()}}
    return payload, {'status':'completed','diagnostics':meta.get('diagnostics',{})}

def run(source=ROOT/'private_data/ieee39_boundary_fine_0075', output=ROOT/'private_data/task2_boundary_probe_0077'):
    old,mass=_load(source); statuses={}; comparisons=[]
    for eid in IDS:
        statuses[eid]={}; cases={}
        for arm in ARMS:
            cases[arm],statuses[eid][arm]=_case(output,eid,arm)
        comparisons.append({'event_id':eid,'comparison':'old_fine_vs_repeat',**_metrics(old[eid],cases[ARMS[0]],mass,strict_repeat=True)})
        for left,right in ((ARMS[0],ARMS[1]),(ARMS[1],ARMS[2]),(ARMS[0],ARMS[3])):
            comparisons.append({'event_id':eid,'comparison':f'{left}_vs_{right}',**_metrics(cases[left],cases[right],mass)})
    report=json.loads((output/'report.json').read_text())
    report_rows={(r['event_id'],r['comparison']):r for r in report['comparisons']}
    metric_discrepancies=[]
    for r in comparisons:
        if r['comparison']=='old_fine_vs_repeat': name='old0075_fine_vs_new_repeat'
        else: name=r['comparison']
        q=report_rows.get((r['event_id'],name),{})
        for key in ('combined_rmse_normalized','common_frequency_rmse_normalized'):
            if key in r and key in q: metric_discrepancies.append(abs(float(r[key])-float(q[key])))
    timing=[]
    for eid in IDS:
        for arm in ARMS:
            d=statuses[eid][arm].get('diagnostics',{})
            timing.append({'event_id':eid,'arm':arm,'status':statuses[eid][arm]['status'],
                           'attempted_steps':d.get('attempted_steps'),'step_size_min_s':d.get('step_size_min_s'),
                           'step_size_max_s':d.get('step_size_max_s'),'initial_f_max_abs':d.get('initial_f',{}).get('max_abs'),
                           'initial_g_max_abs':d.get('initial_g',{}).get('max_abs'),'event_brackets':d.get('event_brackets'),
                           'switch_callbacks':d.get('switch_callbacks')})
    result={'scope':'independent selected-TRAIN payload QA','source_event_ids':list(IDS),'source_train_only_asserted':True,'old_payloads_read':3,'statuses':statuses,'comparisons':comparisons,'timing_and_residual_summary':timing,'runner_metric_max_abs_discrepancy':max(metric_discrepancies,default=0.0),'source_hashes':{n:hashlib.sha256((source/n).read_bytes()).hexdigest() for n in ('dataset.h5','dataset_index.csv','splits.json')},'test_read':False}
    (output/'independent_qa.json').write_text(json.dumps(result,indent=2,allow_nan=False),encoding='utf-8'); return result

if __name__=='__main__':
    print(json.dumps(run(),indent=2))
