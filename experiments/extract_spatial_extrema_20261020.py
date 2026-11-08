"""Read-only JSON arithmetic for exactly six fixed 512-update train queues.

No model imports, binary artifacts, data readers, inference or statistical tests.
Cohorts use the frozen385 identity file, never event-name prefixes.
Only JSON is read; no training/evaluation/model code is imported.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT/'private_data/closure_mechanism_20261019/P2/spatial_extrema01'
COHORT_PATH = ROOT/'results/quadratic_memory_edge_20261020/cohort_identity.json'
COHORT_SHA = '0d2b3ed030cd742ae0809114008c31e3a99c35c56d0238d2d03c5473abcecbf5'
SEEDS = (6101, 6102, 6103)
VARIANTS = ('O', 'S')


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def finite(v):
    return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)


def risk(values):
    if not values or not all(finite(v) for v in values):
        return None
    n = len(values); tail = math.ceil(.1*n)
    return .8*sum(values)/n+.2*sum(sorted(values, reverse=True)[:tail])/tail


def identity_map(rows):
    result = {r['id']: {'group': str(r['group']), 'label': r['label']} for r in rows}
    require(len(result) == len(rows), 'duplicate source identity')
    return result


def load_cohorts(payload):
    rows = payload['cohorts']
    original, new = identity_map(rows['original366']), identity_map(rows['new19'])
    require(len(original) == 366 and len(new) == 19 and not set(original) & set(new), 'frozen 366/19 cohorts invalid')
    full = original | new
    require(sum(r['label'] == 'stable' for r in full.values()) == 342 and sum(r['label'] == 'unstable' for r in full.values()) == 43, '385 labels must be342/43')
    return {'common385': set(full), 'original366': set(original), 'new19': set(new)}, full


def validate_evidence(evidence, full):
    extras = evidence['extras']; ids = [r['event_id'] for r in extras]
    require(evidence['event_count'] == 385 and len(ids) == len(set(ids)) == 385 and set(ids) == set(full), 'queue IDs missing/extra/duplicated')
    require(evidence.get('test_read') is False, 'evidence must declare test_read=false')
    require(all({'group': str(r['group']), 'label': r['label']} == full[r['event_id']] for r in extras), 'queue source label/group mismatch')
    for horizon in ('1', '2', '5', '8', 'prefix'):
        rows = [r for r in evidence['rows']['objective_point_error'] if str(r['horizon']) == horizon]
        require(len(rows) == 385 and {r['event_id'] for r in rows} == set(full), 'point-error horizon IDs missing or duplicated')


def pair_ok(p):
    return (p.get('finite_to_endpoint') is True and p.get('los_pair_pass') is True and
            all(finite(p.get(k)) and p[k] <= .001 for k in ('coi_normalized_rmse', 'common_normalized_rmse')))


def qualified15(e):
    a = e['actual_interval15']
    return (a.get('available') is True and a.get('finite') is True and a.get('numerically_admissible') is True
            and pair_ok(e['classification_15s']) and
            all(finite(a.get(k)) and a[k] <= .001 for k in ('coi_normalized_rmse', 'common_normalized_rmse')))


def pooled(extras, family, horizon=None):
    sse = 0.; count = 0; bad = []
    for e in extras:
        actual = horizon is None
        p = e['actual_interval15'] if actual else e['physical_1_2_5_8s'][horizon]
        valid = qualified15(e) if actual else e['scientific_numeric_8s'] is True
        if actual:
            n = p.get('n_time_points', 0)*(10 if family == 'mechanical_power' else 1)
            rmse = p.get('power_rmse_system_pu' if family == 'mechanical_power' else 'frequency_rmse_rad_s')
            value = rmse*rmse*n if finite(rmse) else None
        else:
            f = p[family]; n = f.get('n_scalar_points', 0); value = f.get('physical_sse')
            valid = valid and f.get('finite') is True
        if not valid or p.get('available') is not True or not finite(value) or value < 0 or not isinstance(n, int) or n <= 0:
            bad.append(e['event_id'])
        else:
            sse += value; count += n
    complete = bool(extras) and not bad
    rmse = math.sqrt(sse/count) if complete and count else None
    return {'events': len(extras), 'unresolved_ids': bad, 'sse': sse if complete else None,
            'n_scalar_points': count if complete else None, 'rmse': rmse,
            'rmse_MW': 100*rmse if family == 'mechanical_power' and rmse is not None else None,
            'unit': 'system pu (100 MVA)' if family == 'mechanical_power' else 'rad/s',
            'sse_origin': 'event_rmse_squared_times_actual_scalar_points' if horizon is None else 'stored_physical_sse'}


def classifications(extras, seconds):
    out = {k: 0 for k in ('stable_denominator', 'unstable_denominator', 'missed', 'false_los', 'undetermined', 'numerical_unresolved')}
    out['categories'] = {}
    for e in extras:
        out[e['label']+'_denominator'] += 1
        c = e[f'classification_{seconds}s']['fine_classification']
        ok = qualified15(e) if seconds == 15 else e['scientific_numeric_8s'] is True
        category = c['category'] if ok else 'numerical_unresolved'
        out['categories'][category] = out['categories'].get(category, 0)+1
        if not ok or category == 'numerical_unresolved':
            out['numerical_unresolved'] += 1
            continue
        if category == 'window_no_los_unsettled': out['undetermined'] += 1
        if c.get('predicted_los') is None or category in ('latency_uncovered', 'undetermined', 'nonfinite'):
            out['undetermined'] += 1
            continue
        out['missed'] += int(e['label'] == 'unstable' and c['predicted_los'] is False)
        out['false_los'] += int(e['label'] == 'stable' and c['predicted_los'] is True)
    out['note'] = 'unsettled no-LOS counts as undetermined and, for unstable truth, missed; counts can overlap'
    return out


def base_summary(evidence, ids):
    extras = [e for e in evidence['extras'] if e['event_id'] in ids]
    byid = {e['event_id']: e for e in extras}
    require(set(byid) == ids, 'summary cohort incomplete')
    stable = [e for e in extras if e['label'] == 'stable']; unstable = [e for e in extras if e['label'] == 'unstable']
    out = {'event_count': len(ids), 'stable_count': len(stable), 'unstable_count': len(unstable), 'stable': {}}
    for h in ('1', '2', '5', '8'):
        rows = [r for r in evidence['rows']['objective_point_error'] if r['event_id'] in ids and r['label'] == 'stable' and str(r['horizon']) == h]
        require(len(rows) == len(stable) and {r['event_id'] for r in rows} == {r['event_id'] for r in stable}, 'stable rows incomplete')
        bad = [r['event_id'] for r in rows if not (r.get('available') is True and r.get('finite') is True and r.get('numerically_admissible') is True and byid[r['event_id']]['scientific_numeric_8s'] is True and finite(r.get('mean_point_error')))]
        values = [r['mean_point_error'] for r in rows]
        out['stable'][h] = {'events': len(rows), 'unresolved_ids': bad,
            'mean_event_point_error': None if bad or not values else sum(values)/len(values),
            **{f: pooled(stable, f, h) for f in ('frequency', 'mechanical_power')}}
        if h == '5': out['stable'][h]['ceil_risk'] = None if bad else risk(values)
    bad = [e['event_id'] for e in unstable if not qualified15(e) or not finite(e['actual_interval15'].get('mean_point_error')) or e['actual_interval15'].get('n_time_points', 0) <= 0]
    points = sum(e['actual_interval15']['n_time_points'] for e in unstable) if not bad else 0
    out['unstable_actual_interval15'] = {'events': len(unstable), 'unresolved_ids': bad,
        'mean_event_point_error': None if bad or not unstable else sum(e['actual_interval15']['mean_point_error'] for e in unstable)/len(unstable),
        'time_pooled_point_error': None if bad or not points else sum(e['actual_interval15']['mean_point_error']*e['actual_interval15']['n_time_points'] for e in unstable)/points,
        **{f: pooled(unstable, f) for f in ('frequency', 'mechanical_power')}}
    out['classification_8s'] = classifications(extras, 8); out['classification_15s'] = classifications(extras, 15)
    return out


SCHEDULE_SHA = 'a5ea62c3685ff443c0032e80e0eab951c20c3fa3d56bebdcf0faaa9526b7f4bb'
IMPORTED_HASHES = ('749c8f15cd72d08a253d9381766abec82d04b55c69bc04ccf50ce6fdd2bba39d',
                  'bafcc320fdc8c19395f535d9dc7ab801c7f857677e4f7c3e8ca33a45d50251d7',
                  'de35c95ac18ac70538a7176df25d67f077509155c2bc280fd51788c1985bc428')


def role_path(seed,variant):
    base = BASE.parent/'temporal_objective01' if (seed,variant)==(6101,'O') else BASE
    return (base/f'models/{seed}_{variant}/checkpoints/step000512/model.json').resolve()


def checked_manifest(manifest):
    expected = {(s,v) for s in SEEDS for v in VARIANTS}
    rows = manifest['endpoints']; result = {}
    require(manifest['endpoint_count']==6 and len(rows)==6 and manifest['schedule_sha256']==SCHEDULE_SHA and
            manifest['test_read'] is False and manifest['val_read'] is False, 'exact six-role frozen manifest required')
    for row in rows:
        role = (row['seed'],row['variant'])
        require(role in expected and role not in result and row['stage_step']==512, 'duplicate/invalid frozen role')
        path = Path(row['model_path']).resolve()
        require(path==role_path(*role), 'frozen source path differs from exact role')
        require(Path(row['checkpoint_path']).resolve()==path.parent/'checkpoint.json' and
                Path(row['training_path']).resolve()==path.parent/'training.pt', 'frozen sibling source paths differ')
        hashes = tuple(row[k] for k in ('model_sha256','training_sha256','checkpoint_sha256'))
        require(all(isinstance(x,str) and len(x)==64 and all(c in '0123456789abcdef' for c in x) for x in hashes), 'invalid frozen source hashes')
        imported = role==(6101,'O')
        require(row['imported_control'] is imported and (not imported or hashes==IMPORTED_HASHES), 'imported control provenance differs')
        result[role]=row
    require(set(result)==expected,'incomplete frozen roles')
    return result


def spatial_facts(extras):
    metrics = ('combined','angle_normalized_squared_error','speed_normalized_squared_error')
    bad=[]; sums={k:0. for k in metrics}
    for e in extras:
        s=e.get('spatial',{}); n=e['actual_interval15']['n_time_points']
        require(s.get('actual_future_points')==n,'spatial interval differs from genuine endpoint')
        if not qualified15(e) or s.get('finite') is not True or s.get('numerically_admissible') is not True or not all(finite(s.get(k)) for k in metrics):
            bad.append(e['event_id']); continue
        for k in metrics: sums[k]+=s[k]
    return {'events':len(extras),'unresolved_ids':bad,'qualification':'same-event full actual15 plus pair15; finite alone never qualifies',
            **{k:sums[k]/len(extras) if extras and not bad else None for k in metrics}}


def coi_summary(evidence,ids,label,horizon):
    extras={r['event_id']:r for r in evidence['extras'] if r['event_id'] in ids and (label=='all' or r['label']==label)}
    rows=[r for r in evidence['rows']['records'] if r['event_id'] in extras and str(r['horizon'])==horizon]
    require(len(rows)==len(extras) and {r['event_id'] for r in rows}==set(extras),'COI records incomplete/duplicate')
    numerical=[r for r in evidence['rows']['numerical'] if r['event_id'] in extras and str(r['horizon'])==horizon]
    require(len(numerical)==len(extras) and {r['event_id'] for r in numerical}==set(extras),'COI qualification rows incomplete')
    nums={r['event_id']:r for r in numerical}; eligible=[r for r in rows if r['available']]
    bad=[r['event_id'] for r in eligible if not (r.get('finite') is True and r.get('numerically_admissible') is True and extras[r['event_id']]['scientific_numeric_8s'] is True and pair_ok(nums[r['event_id']]) and finite(r.get('rmse')) and finite(r.get('normalized_sse')) and r.get('n',0)>0)]
    values=[r['rmse'] for r in eligible]; okay=bool(eligible) and not bad
    tail=math.ceil(.1*len(values))
    return {'all_events':len(rows),'eligible_events':len(eligible),'unavailable_events':len(rows)-len(eligible),'unresolved_ids':bad,
            'mean_event_normalized_rmse':sum(values)/len(values) if okay else None,
            'pooled_normalized_rmse':math.sqrt(sum(r['normalized_sse'] for r in eligible)/sum(r['n'] for r in eligible)) if okay else None,
            'worst10_mean_event_normalized_rmse':sum(sorted(values,reverse=True)[:tail])/tail if okay else None,
            'definition':'stored COI relative delta/omega rmse; pooled=sqrt(sum(normalized_sse)/sum(n)); ceil event tail; not point_error'}


def summarize(evidence,ids):
    result=base_summary(evidence,ids)
    extras=[e for e in evidence['extras'] if e['event_id'] in ids]
    result['spatial_actual15']={label:spatial_facts([e for e in extras if label=='all' or e['label']==label]) for label in ('all','stable','unstable')}
    result['coi']={label+'_'+h:coi_summary(evidence,ids,label,h) for label in ('all','stable') for h in ('1','2','5','8')}
    return result


def numeric_metrics(summary):
    out={}
    for h,cell in summary['stable'].items():
        out[f'stable_{h}.pointmean']=cell['mean_event_point_error']
        if h=='5': out['stable_5.ceilrisk']=cell['ceil_risk']
        for f in ('frequency','mechanical_power'): out[f'stable_{h}.{f}.rmse']=cell[f]['rmse']
    cell=summary['unstable_actual_interval15']
    for k in ('mean_event_point_error','time_pooled_point_error'): out['unstable15.'+k]=cell[k]
    for f in ('frequency','mechanical_power'): out['unstable15.'+f+'.rmse']=cell[f]['rmse']
    out['unstable15.power_rmse_MW']=cell['mechanical_power']['rmse_MW']
    for s in (8,15):
        for k in ('missed','false_los','undetermined','numerical_unresolved'): out[f'classification_{s}.{k}']=summary[f'classification_{s}s'][k]
    for group,cell in summary['spatial_actual15'].items():
        for k in ('combined','angle_normalized_squared_error','speed_normalized_squared_error'): out[f'spatial15.{group}.{k}']=cell[k]
    return out


def write_new(path,value):
    with Path(path).open('x',encoding='utf-8') as f: json.dump(value,f,indent=2,allow_nan=False)


def extract(output,manifest_sha256):
    output=Path(output); event_output=BASE/'extracted_event_rows_step000512.json'
    require(not output.exists() and not event_output.exists(),'new output files required')
    manifest_path=BASE/'frozen_evaluation_endpoints.json'
    require(sha(manifest_path)==manifest_sha256,'root-bound frozen manifest hash changed')
    roles=checked_manifest(read(manifest_path))
    require(sha(COHORT_PATH)==COHORT_SHA,'frozen cohort identity changed')
    groups,full=load_cohorts(read(COHORT_PATH))
    queues={role:BASE/f'evaluation/{role[0]}_{role[1]}/step000512/queue' for role in roles}
    for role,path in queues.items():
        complete=read(path/'complete.json')
        require(complete['status']=='COMPLETED' and complete['event_count']==385 and complete['evidence_sha256']==sha(path/'evidence.json'),'all six complete385 evidence hashes required')
    # Source bytes are not read: each provenance hash must match the exact role
    # frozen by the production check_endpoint validator before evaluation.
    models={}; events={}; sources={}; checked={}
    for role,path in queues.items():
        evidence=read(path/'evidence.json'); validate_evidence(evidence,full)
        p=evidence['provenance']; seed,v=role; entry=roles[role]
        require((p['seed'],p['variant'],p['stage_step'],p['model_sha256'])==(seed,v,512,entry['model_sha256']),'queue differs from frozen role/source')
        checked[role]=evidence
    for role,path in queues.items():
        evidence=checked[role]; seed,v=role; entry=roles[role]
        key=f'{seed}_{v}'; models[key]={name:summarize(evidence,ids) for name,ids in groups.items()}
        local=read(path/'local.json')
        models[key]['local']={k:local[k] for k in ('source_sha256','quotient','linear_field','energy_curvature')}
        models[key]['local']['origin']={k:local['origin'][k] for k in ('equilibrium_pass','energy_stationary_pass','quotient_eligible','f0_norm_inf','gradH0_norm_inf')}
        events[key]={'objective_point_error':evidence['rows']['objective_point_error'],'records':evidence['rows']['records'],'extras':evidence['extras']}
        sources[key]={'evidence_path':str(path/'evidence.json'),'evidence_sha256':sha(path/'evidence.json'),'local_sha256':sha(path/'local.json'),'frozen_source':entry}
    pairs={}
    for seed in SEEDS:
        pairs[str(seed)]={}
        for group in groups:
            s,o=numeric_metrics(models[f'{seed}_S'][group]),numeric_metrics(models[f'{seed}_O'][group])
            pairs[str(seed)][group]={k:s[k]-o[k] if finite(s[k]) and finite(o[k]) else None for k in s}
    result={'status':'COMPLETED','split':'train','stage_step':512,'models':models,'S_minus_O':pairs,'sources':sources,
            'manifest_sha256':manifest_sha256,'cohort_sha256':COHORT_SHA,'local_event_rows':str(event_output),
            'scope':'fixed train JSON facts only; imported6101_O remains old training; no adoption/significance/stability claim'}
    write_new(event_output,events); write_new(output,result)


def main():
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('--output',type=Path,required=True)
    p.add_argument('--manifest-sha256',required=True); a=p.parse_args()
    try: extract(a.output,a.manifest_sha256)
    except (OSError,ValueError,KeyError,TypeError) as error:
        if a.output.exists(): raise
        write_new(a.output,{'status':'UNRESOLVED','split':'train','error':repr(error),'scope':'no missing event/queue filled; no new inference'})
        raise SystemExit(2)


if __name__=='__main__': main()


