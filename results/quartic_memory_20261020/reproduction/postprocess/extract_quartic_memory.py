"""N/Q1024 stdlib-only JSON facts. Execute only after all six evaluations."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import runpy

BASE=Path(__file__).resolve().parents[1]
ROOT=BASE.parents[3]
SEEDS=(6101,6102,6103);VARIANTS=('N','Q')
SCHEDULE_SHA='07fd838bd2a9d136db9ce0b789562c26c93098cf5b6bba40af8f67d674bf2b0c'
COHORT_SHA='0d2b3ed030cd742ae0809114008c31e3a99c35c56d0238d2d03c5473abcecbf5'
SOURCE_SHA='b8c525f9b87709528c74f1b64724aace4b653e4a6948765da8471599cc1d0af4'
PURE_SHA='e4f5494e2cd6feb14e53cef1ccbfe8f70eee1bbdfae9961fc2bfb85ffa4ae9ca'
QUARTIC={'channel':'node','expression':{'op':'pow4','arg':{'op':'var','name':'q'}}}
def variable(name):return {'op':'var','name':name}
N_TERMS=[{'channel':'node','expression':{'op':'mul','args':[variable('p'),variable('q')]}},
         {'channel':'edge','expression':{'op':'mul','args':[{'op':'add','args':[variable('qi'),
          {'op':'mul','args':[{'op':'const','value':-1},variable('qj')]}]},variable('s')]}}]
N_TERMS += [{'channel':'port','expression':variable(name)} for name in ('w','p','q','ec','nq')]
_pure=None


def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def require(ok,message):
    if not ok:raise ValueError(message)
def finite(x):return isinstance(x,(int,float)) and not isinstance(x,bool) and math.isfinite(x)


def pure():
    global _pure
    if _pure is None:
        p=ROOT/'experiments/extract_spatial_extrema_20261020.py'
        require(sha(p)==PURE_SHA,'tested pure extractor source hash changed')
        _pure=runpy.run_path(str(p))
    return _pure


def role_path(seed,v):return (BASE/f'models/{seed}_{v}/checkpoints/step001024/model.json').resolve()


def checked_manifest(m):
    require(m.get('test_read') is False and m.get('val_read') is False and m['schedule_sha256']==SCHEDULE_SHA,'nontrain/wrong schedule manifest')
    require(m['endpoint_count']==6 and len(m['endpoints'])==6,'six endpoints required')
    result={}
    for row in m['endpoints']:
        role=(row['seed'],row['variant']);require(role in {(s,v) for s in SEEDS for v in VARIANTS} and role not in result,'unknown/duplicate role')
        p=role_path(*role)
        require(Path(row['model_path']).resolve()==p and row['stage_step']==1024,'exact1024 role model path required')
        require(Path(row['training_path']).resolve()==p.parent/'training.pt' and Path(row['checkpoint_path']).resolve()==p.parent/'checkpoint.json','exact1024 role siblings required')
        require(all(isinstance(row[k],str) and len(row[k])==64 and all(c in '0123456789abcdef' for c in row[k]) for k in ('model_sha256','training_sha256','checkpoint_sha256')),'invalid frozen hash')
        result[role]=row
    return result


def canonical(ast):
    if isinstance(ast,list):return [canonical(x) for x in ast]
    if not isinstance(ast,dict):return ast
    d={k:canonical(v) for k,v in ast.items()}
    if d.get('op') in ('add','mul'):d['args']=sorted(d['args'],key=lambda x:json.dumps(x,sort_keys=True,separators=(',',':')))
    return d


def term_key(term):return json.dumps(canonical(term),sort_keys=True,separators=(',',':'))


def flatten(x):
    if isinstance(x,list):
        for v in x:yield from flatten(v)
    else:
        require(finite(x),'nonfinite saved parameter');yield float(x)


def parameter_map(payload):
    require(payload['coefficients']==payload['state_dict']['coefficients'],'coefficient copies differ')
    keys=[term_key(t) for t in payload['terms']]
    require(len(keys)==len(set(keys))==len(payload['coefficients']),'term/coefficient identity differs')
    result={}
    for name in payload['trainable_parameters']:
        if name=='coefficients':
            for key,value in zip(keys,payload['coefficients']):
                require(finite(value),'nonfinite coefficient');result['coefficients:'+key]=float(value)
        else:
            for i,value in enumerate(flatten(payload['state_dict'][name])):result[f'{name}[{i}]']=value
    return result


def validate_saved_terms(payload,variant):
    actual=[term_key(t) for t in payload['terms']]
    expected=[term_key(t) for t in N_TERMS+([QUARTIC] if variant=='Q' else [])]
    require(len(actual)==len(expected) and set(actual)==set(expected),'saved exact N/Q AST set differs')


def scalar_stats(rows,key):
    values=[r['quartic'].get(key) for r in rows]
    good=[v for v in values if finite(v)];bad=[v for v in values if v is not None and not finite(v)]
    return {'actual_rows':len(values),'finite_count':len(good),'missing_count':sum(v is None for v in values),'nonfinite_count':len(bad),
        'finite_min':min(good) if good else None,'finite_max':max(good) if good else None,
        'finite_mean':sum(good)/len(good) if good else None,'positive_count':sum(v>0 for v in good),
        'negative_count':sum(v<0 for v in good),'zero_count':sum(v==0 for v in good)}


def trace_statistics(trace):
    names=('risk','mean','local','latent_consistency','torque','complexity')
    for row in trace:
        require(all(finite(row.get('components',{}).get(k)) for k in names),'original components missing/nonfinite')
    phases={}
    for h in (1.,2.,5.):
        rows=[r for r in trace if r['horizon_s']==h];cell={'actual_rows':len(rows)}
        for name,selected in (('first64',rows[:64]),('last64',rows[-64:])):
            cell[name]={'actual_rows':len(selected),'first_stage_step':selected[0]['stage_step'],'last_stage_step':selected[-1]['stage_step'],
                'loss_mean':sum(r['loss'] for r in selected)/len(selected),
                'component_means':{k:sum(r['components'][k] for r in selected)/len(selected) for k in names}}
        phases[str(h)]=cell
    fields=('raw_gradient','gradient','before_update','before','after','adam_delta','projected_update')
    scopes={'all1024':trace,'stage1_512':trace[:512],'stage513_1024':trace[512:]}
    scopes.update({'h'+str(h):[r for r in trace if r['horizon_s']==h] for h in (1.,2.,5.)})
    return {'by_horizon':phases,'quartic_scalar_statistics':{scope:{k:scalar_stats(rows,k) for k in fields} for scope,rows in scopes.items()},
        'stage_counts':{scope:{'actual_rows':len(rows),'projection_count':sum(r['quartic'].get('projected') is True for r in rows),'clip_count':sum(r['gradient_norm']>100 for r in rows)} for scope,rows in scopes.items()},
        'gradient_definition':'raw_gradient before clip100; gradient is clipped gradient used by Adam',
        'interpretation':'same-horizon first/last64 actual updates; changing samples; no convergence claim'}


def checked_local(local,model_sha):
    require(local['source_sha256']==model_sha,'local analysis source differs from frozen endpoint')
    return {k:local[k] for k in ('source_sha256','origin','quotient','linear_field','energy_curvature')}


def shared_update(previous_n,current_n,previous_q,current_q):
    pn,cn,pq,cq=[parameter_map(x) for x in (previous_n,current_n,previous_q,current_q)]
    require(set(pn)==set(cn) and set(pq)==set(cq) and set(pn)<=set(pq),'shared parameter identities changed')
    require(set(pq)-set(pn)=={'coefficients:'+term_key(QUARTIC)},'Q must add only the quartic parameter')
    keys=sorted(pn);dn=[cn[k]-pn[k] for k in keys];dq=[cq[k]-pq[k] for k in keys]
    return {'shared_scalar_count':len(keys),'N_update_l2':math.sqrt(sum(x*x for x in dn)),
            'Q_common_update_l2':math.sqrt(sum(x*x for x in dq)),
            'QminusN_update_l2':math.sqrt(sum((y-x)**2 for x,y in zip(dn,dq))),
            'scope':'saved checkpoint interval updates; identity-aligned, no optimizer replay'}


def summarize(evidence,ids):
    result=pure()['summarize'](evidence,ids)
    for h,cell in result['stable'].items():
        values=[r['mean_point_error'] for r in evidence['rows']['objective_point_error'] if r['event_id'] in ids and r['label']=='stable' and str(r['horizon'])==h]
        cell['ceil_risk']=None if cell['unresolved_ids'] else pure()['risk'](values)
    return result


def metric_values(summary):
    result=pure()['numeric_metrics'](summary)
    for h,cell in summary['stable'].items():result[f'stable_{h}.ceilrisk']=cell['ceil_risk']
    for group,cell in summary['coi'].items():
        for name in ('mean_event_normalized_rmse','pooled_normalized_rmse','worst10_mean_event_normalized_rmse'):
            result['coi.'+group+'.'+name]=cell[name]
    return result


def gate_facts(models):
    p=pure();facts={};strict=[]
    keys=('stable_5.ceilrisk','stable_5.frequency.rmse','unstable15.mean_event_point_error',
          'unstable15.frequency.rmse','unstable15.mechanical_power.rmse',
          'classification_8.missed','classification_8.false_los','classification_15.missed','classification_15.false_los')
    for seed in SEEDS:
        n,q=[models[f'{seed}_{v}']['common385'] for v in VARIANTS]
        left,right=p['numeric_metrics'](q),p['numeric_metrics'](n)
        no_increase={k:None if not finite(left[k]) or not finite(right[k]) else left[k]<=right[k] for k in keys}
        numeric=all(x[f'classification_{s}s']['numerical_unresolved']==0 for x in (n,q) for s in (8,15)) and all(v is not None for v in no_increase.values())
        fewer=None if not finite(left['classification_15.missed']) or not finite(right['classification_15.missed']) else left['classification_15.missed']<right['classification_15.missed']
        if fewer is True:strict.append(seed)
        facts[str(seed)]={'all_relevant_numeric_resolved':numeric,'no_increase':no_increase,'strictly_fewer_actual15_missed':fewer}
    return {'per_seed':facts,'strict_missed_reduction_seeds':strict,'at_least_two_including6101':len(strict)>=2 and 6101 in strict,
            'scope':'protocol arithmetic booleans only, no tolerance or adoption decision; root must also verify physical/source/power evidence'}


def training_facts(roles,source_entries):
    result={};payloads={}
    for role,entry in roles.items():
        seed,v=role;folder=role_path(*role).parent
        require(sha(folder/'checkpoint.json')==entry['checkpoint_sha256'],'final checkpoint JSON changed')
        meta=read(folder/'checkpoint.json');identity=meta['identity'];trace=meta['trace']
        require(meta['step']==1024 and len(trace)==1024 and (identity['seed'],identity['variant'])==role,'trace1024 role mismatch')
        require(identity['schedule_sha256']==SCHEDULE_SHA and identity['source_model_sha256']==source_entries[seed]['model_sha256'] and identity['source_training_sha256']==source_entries[seed]['training_sha256'],'trace source/schedule mismatch')
        counts={1.:0,2.:0,5.:0};projections=0;clips=0;projection_rows=[]
        for step,row in enumerate(trace,1):
            require(row['stage_step']==step and row['horizon_s']==(1.,2.,5.)[(step-1)%3],'trace phase mismatch')
            require(finite(row['loss']) and finite(row['gradient_norm']),'nonfinite trace')
            require(row['learning_rates']=={k:x*(.3 if step>512 else 1.) for k,x in {'skeleton':.0009,'closure':.00009,'encoder':.000027}.items()},'registered513 learning-rate transition differs')
            counts[row['horizon_s']]+=1;clips+=row['gradient_norm']>100
            q=row['quartic'];projections+=q.get('projected') is True
            projection_rows.append({'stage_step':step,'horizon_s':row['horizon_s'],'loss':row['loss'],'gradient_norm':row['gradient_norm'],'quartic':q})
        require(counts=={1.:342,2.:341,5.:341},'wrong phase denominators')
        key=f'{seed}_{v}';payloads[key]={}
        for step in range(0,1025,32):
            path=folder.parent/f'step{step:06d}/model.json';data=read(path);values=parameter_map(data)
            validate_saved_terms(data,v)
            require(('coefficients:'+term_key(QUARTIC) in values)==(v=='Q'),'saved role term identity differs')
            if v=='Q':require(values['coefficients:'+term_key(QUARTIC)]>=0,'negative saved quartic')
            payloads[key][step]=data
        require(sha(folder/'model.json')==entry['model_sha256'],'final model JSON differs')
        result[key]={'updates':1024,'phase_counts':counts,'clip_count':clips,'projection_count':projections,
            'training_statistics':trace_statistics(trace),
            'final_a':None if v=='N' else parameter_map(payloads[key][1024])['coefficients:'+term_key(QUARTIC)],
            'saved_a_by_step':{str(step):None if v=='N' else parameter_map(data)['coefficients:'+term_key(QUARTIC)] for step,data in payloads[key].items()},
            'trace_rows_local_only':projection_rows}
    intervals={}
    for seed in SEEDS:
        intervals[str(seed)]=[{'start':step-32,'end':step,**shared_update(payloads[f'{seed}_N'][step-32],payloads[f'{seed}_N'][step],payloads[f'{seed}_Q'][step-32],payloads[f'{seed}_Q'][step])} for step in range(32,1025,32)]
    return result,intervals


def write_new(path,value):
    with Path(path).open('x',encoding='utf-8') as f:json.dump(value,f,indent=2,allow_nan=False)


def extract(output,manifest_sha256,after_complete=False):
    require(after_complete,'root must authorize --after-complete after all six frozen evaluations')
    output=Path(output);private=BASE/'postprocess/event_and_trace_rows.json'
    require(not output.exists() and not private.exists(),'outputs must be new')
    manifest=BASE/'frozen_evaluation_endpoints.json';require(sha(manifest)==manifest_sha256,'frozen manifest changed')
    roles=checked_manifest(read(manifest));p=pure()
    cohort=ROOT/'results/quadratic_memory_edge_20261020/cohort_identity.json';require(sha(cohort)==COHORT_SHA,'cohort changed')
    groups,full=p['load_cohorts'](read(cohort))
    source_manifest=ROOT/'results/spatial_extrema_20261020/frozen_evaluation_endpoints.json'
    require(sha(source_manifest)==SOURCE_SHA,'parent source manifest changed')
    source_entries={e['seed']:e for e in read(source_manifest)['endpoints'] if e['variant']=='O'}
    require(set(source_entries)==set(SEEDS),'three fixed O sources required')
    queues={r:BASE/f'evaluation/{r[0]}_{r[1]}/step001024/queue' for r in roles}
    for role,queue in queues.items():
        marker=read(queue/'complete.json')
        require(marker['status']=='COMPLETED' and marker['event_count']==385 and marker['evidence_sha256']==sha(queue/'evidence.json'),'all six complete/hash checks required')
    checked={}
    for role,queue in queues.items():
        e=read(queue/'evidence.json');p['validate_evidence'](e,full);pv=e['provenance']
        require((pv['seed'],pv['variant'],pv['stage_step'],pv['model_sha256'])==(*role,1024,roles[role]['model_sha256']),'queue frozen-role provenance differs')
        checked[role]=e
    models={};events={};sources={}
    for role,e in checked.items():
        key=f'{role[0]}_{role[1]}';queue=queues[role]
        models[key]={name:summarize(e,ids) for name,ids in groups.items()}
        local=read(queue/'local.json');models[key]['local']=checked_local(local,roles[role]['model_sha256'])
        models[key]['local']['origin']={k:local['origin'][k] for k in ('equilibrium_pass','energy_stationary_pass','quotient_eligible','f0_norm_inf','gradH0_norm_inf')}
        events[key]={'extras':e['extras'],'rows':e['rows']};sources[key]={'evidence_path':str(queue/'evidence.json'),'sha256':sha(queue/'evidence.json'),
            'complete_sha256':sha(queue/'complete.json'),'local_sha256':sha(queue/'local.json'),'frozen':roles[role]}
    training,intervals=training_facts(roles,source_entries)
    trace_rows={key:row.pop('trace_rows_local_only') for key,row in training.items()}
    differences={}
    for seed in SEEDS:
        differences[str(seed)]={}
        for name in groups:
            n,q=[metric_values(models[f'{seed}_{v}'][name]) for v in VARIANTS]
            differences[str(seed)][name]={k:q[k]-n[k] if finite(q[k]) and finite(n[k]) else None for k in n}
    result={'status':'COMPLETED','split':'train','stage_step':1024,'models':models,'Q_minus_N':differences,'protocol_gate_facts':gate_facts(models),
        'training':training,'shared_parameter_update_intervals':intervals,'sources':sources,'manifest_sha256':manifest_sha256,
        'cohort_sha256':COHORT_SHA,'schedule_sha256':SCHEDULE_SHA,'pure_source_sha256':PURE_SHA,'private_event_trace_rows':str(private),
        'scope':'JSON facts only; root owns conclusions, physical/source/power checks and adoption; no new prediction or optimization'}
    write_new(private,{'events':events,'training_trace':trace_rows});write_new(output,result)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--manifest-sha256',required=True);parser.add_argument('--after-complete',action='store_true');a=parser.parse_args()
    try:extract(a.output,a.manifest_sha256,a.after_complete)
    except (OSError,ValueError,KeyError,TypeError) as error:
        if a.output.exists():raise
        write_new(a.output,{'status':'UNRESOLVED','split':'train','error':repr(error),'scope':'no missing role/event substituted'})
        raise SystemExit(2)


if __name__=='__main__':main()
