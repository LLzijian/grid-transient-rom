"""Post-freeze W/N/Q facts from JSON only. No inference or adoption decisions."""
import argparse,hashlib,json,math,runpy,time
from pathlib import Path
BASE=Path(__file__).resolve().parents[1]
ROOT=BASE.parents[3]
SEEDS=(6101,6102,6103)
NQ_SHA='4f53916743184e269541f42797a829bbcf938b7a8047b18978147f7d9f107820'
NQ_MANIFEST_SHA='a1b411dae2c00867611afe390d934b8cc718d55d3103dd81aba0b6229aa34842'
COHORT_SHA='0d2b3ed030cd742ae0809114008c31e3a99c35c56d0238d2d03c5473abcecbf5'
SCHEDULE_SHA='07fd838bd2a9d136db9ce0b789562c26c93098cf5b6bba40af8f67d674bf2b0c'
SOURCE_SHA='b8c525f9b87709528c74f1b64724aace4b653e4a6948765da8471599cc1d0af4'
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def require(ok,message):
    if not ok:raise ValueError(message)
def finite(x):return isinstance(x,(int,float)) and not isinstance(x,bool) and math.isfinite(x)
def digest(x):return isinstance(x,str) and len(x)==64 and all(c in '0123456789abcdef' for c in x)
def write(p,v):
    with Path(p).open('x',encoding='utf-8') as f:json.dump(v,f,indent=2,allow_nan=False)
def roles(manifest,variants,base):
    require(manifest.get('test_read') is False and manifest.get('val_read') is False and manifest.get('schedule_sha256')==SCHEDULE_SHA,'nontrain/schedule manifest')
    result={};expected={(s,v) for s in SEEDS for v in variants}
    require(manifest['endpoint_count']==len(expected) and len(manifest['endpoints'])==len(expected),'incomplete freeze')
    for e in manifest['endpoints']:
        key=(e['seed'],e['variant']);require(key in expected and key not in result and e['stage_step']==1024,'wrong/duplicate role')
        model=(base/f'models/{key[0]}_{key[1]}/checkpoints/step001024/model.json').resolve()
        for kind,path in [('model',model),('training',model.parent/'training.pt'),('checkpoint',model.parent/'checkpoint.json')]:
            require(Path(e[kind+'_path']).resolve()==path and digest(e[kind+'_sha256']),'role/path/hash contract')
        result[key]=e
    return result
def algebra(extras,scope,pure):
    actual=scope=='actual_interval15';selected=extras if actual else [e for e in extras if e['label']=='stable']
    bad=[];items=[]
    for e in selected:
        row=e.get('algebraic',{}).get(scope) or {};summary=row.get('summary')
        okay=pure['qualified15'](e) if actual else e.get('scientific_numeric_8s') is True
        n=e['actual_interval15'].get('n_time_points') if actual else int(scope)*100
        good=okay and row.get('available') is True and row.get('numerically_admissible') is True and row.get('future_points')==n and isinstance(n,int) and n>0 and isinstance(summary,dict)
        cells=[summary.get(k) for k in ('port_power','omega_field_norm')] if isinstance(summary,dict) else []
        good=good and len(cells)==2 and all(isinstance(c,dict) and all(finite(x) for x in c.values()) for c in cells)
        if not good:bad.append(e['event_id'])
        else:items.append(summary)
    complete=bool(selected) and not bad
    result={'events':len(selected),'unresolved_ids':bad,'complete':complete,'summary':None,'power_unit':'H-unit/s','omega_field_norm_unit':'rad/s^2'}
    if complete:
        keys={k:set(items[0][k]) for k in ('port_power','omega_field_norm')}
        require(all(set(x[k])==keys[k] for x in items for k in keys),'algebra summary schema changed')
        result['summary']={k:{name:sum(x[k][name] for x in items)/len(items) for name in keys[k]} for k in keys}
        result['summary_definition']='event arithmetic means of each stored time statistic; denominators retained; no finite subset'
    return result
def training(meta):
    trace=meta['trace'];require(len(trace)==1024,'training trace incomplete')
    phases={};counts={1.:0,2.:0,5.:0}
    for i,t in enumerate(trace,1):
        require(t['stage_step']==i and t['horizon_s']==(1.,2.,5.)[(i-1)%3],'training phase changed')
        require(t['learning_rates']=={k:v*(.3 if i>512 else 1.) for k,v in {'skeleton':.0009,'closure':.00009,'encoder':.000027}.items()},'LR513 identity')
        require(finite(t['loss']) and finite(t['gradient_norm']) and finite(t['frequency_dissipation']['after']) and t['frequency_dissipation']['after']<=0,'trace finite/projection identity')
        counts[t['horizon_s']]+=1
    require(counts=={1.:342,2.:341,5.:341},'wrong phase denominators')
    for h in (1.,2.,5.):
        rows=[t for t in trace if t['horizon_s']==h];phases[str(h)]={}
        for name,rr in [('first64',rows[:64]),('last64',rows[-64:])]:
            keys=set(rr[0]['components']);require(all(set(t['components'])==keys and all(finite(v) for v in t['components'].values()) for t in rr),'missing components')
            phases[str(h)][name]={'updates':len(rr),'loss_mean':sum(t['loss'] for t in rr)/len(rr),'components':{k:sum(t['components'][k] for t in rr)/len(rr)}}
    return {'final_b_W':trace[-1]['frequency_dissipation']['after'],'projection_count':sum(t['frequency_dissipation']['projected'] is True for t in trace),'clip_count':sum(t['gradient_norm']>100 for t in trace),'phase_counts':counts,'loss_by_horizon':phases,'scope':'JSON trace only; no optimizer replay or convergence claim'}
def extract(a):
    require(finite(a.max_seconds) and 0<a.max_seconds<=40,'metadata self limit must be at most40 seconds')
    deadline=time.perf_counter()+a.max_seconds
    def check():
        if time.perf_counter()>=deadline:raise TimeoutError('JSON metadata deadline')
    require(a.after_complete,'root --after-complete required')
    require(a.output.resolve().is_relative_to(ROOT) and not a.output.exists(),'new workspace output required')
    helper=ROOT/'private_data/closure_mechanism_20261019/P2/quartic_memory01/postprocess/extract_quartic_memory.py'
    require(sha(helper)==a.helper_sha256,'pure helper identity changed');q=runpy.run_path(str(helper));p=q['pure']()
    cohort=ROOT/'results/quadratic_memory_edge_20261020/cohort_identity.json';require(sha(cohort)==COHORT_SHA,'cohort changed');groups,full=p['load_cohorts'](read(cohort))
    require(sha(a.baseline_metrics)==NQ_SHA and sha(a.baseline_manifest)==NQ_MANIFEST_SHA,'baseline evidence changed')
    baseline=read(a.baseline_metrics);require(baseline['status']=='COMPLETED' and baseline['split']=='train' and baseline['stage_step']==1024 and baseline['cohort_sha256']==COHORT_SHA and baseline['schedule_sha256']==SCHEDULE_SHA,'baseline scope differs')
    nq=roles(read(a.baseline_manifest),('N','Q'),BASE.parent/'quartic_memory01')
    for key,e in nq.items():require(baseline['sources'][f'{key[0]}_{key[1]}']['frozen']==e,'baseline frozen provenance differs')
    require(a.manifest.resolve()==BASE/'frozen_evaluation_endpoints.json' and sha(a.manifest)==a.manifest_sha256,'W freeze changed');wr=roles(read(a.manifest),('W',),BASE)
    source=ROOT/'results/spatial_extrema_20261020/frozen_evaluation_endpoints.json';require(sha(source)==SOURCE_SHA,'O512 identity changed');sources={e['seed']:e for e in read(source)['endpoints'] if e['variant']=='O'}
    evidence={};metas={};provenance={}
    for (seed,v),entry in wr.items():
        check()
        queue=BASE/f'evaluation/{seed}_W/step001024/queue';ep=queue/'evidence.json';complete=read(queue/'complete.json')
        require(complete['status']=='COMPLETED' and complete['event_count']==385 and complete['evidence_sha256']==sha(ep),'three complete385 queues required')
        e=read(ep);p['validate_evidence'](e,full);pv=e['provenance']
        require((pv['seed'],pv['variant'],pv['stage_step'],pv['model_sha256'])==(seed,'W',1024,entry['model_sha256']),'W queue provenance differs')
        meta_path=Path(entry['checkpoint_path']);require(sha(meta_path)==entry['checkpoint_sha256'],'checkpoint JSON changed');meta=read(meta_path);identity=meta['identity']
        require(meta['step']==1024 and meta['model_sha256']==entry['model_sha256'] and meta['training_sha256']==entry['training_sha256'] and (identity['seed'],identity['variant'])==(seed,'W'),'checkpoint role/hash differs')
        require(identity['schedule_sha256']==SCHEDULE_SHA and identity['source_model_sha256']==sources[seed]['model_sha256'] and identity['source_training_sha256']==sources[seed]['training_sha256'],'W source/schedule differs')
        require(len(e['prediction_files'])==25 and all(digest(x['sha256']) for x in e['prediction_files']),'prediction identities incomplete')
        evidence[seed]=e;metas[seed]=meta;provenance[str(seed)]={'frozen':entry,'evidence_path':str(ep),'evidence_sha256':sha(ep),'complete_sha256':sha(queue/'complete.json')}
    models={};differences={};train={}
    for seed,e in evidence.items():
        check()
        key=f'{seed}_W';models[key]={};differences[str(seed)]={}
        for name,ids in groups.items():
            check()
            summary=q['summarize'](e,ids);selected=[r for r in e['extras'] if r['event_id'] in ids]
            summary['frequency_dissipation']={'stable':{h:algebra(selected,h,p) for h in ('1','2','5','8')},'actual15':{label:algebra([r for r in selected if label=='all' or r['label']==label],'actual_interval15',p) for label in ('all','stable','unstable')}}
            summary['classification_rows']=[{'event_id':r['event_id'],'label':r['label'],'scientific_numeric_8s':r['scientific_numeric_8s'],'actual15_qualified':p['qualified15'](r),'classification_8s':r['classification_8s'],'classification_15s':r['classification_15s']} for r in selected]
            models[key][name]=summary;values=q['metric_values'](summary);differences[str(seed)][name]={}
            for v in ('N','Q'):
                reference=q['metric_values'](baseline['models'][f'{seed}_{v}'][name]);require(set(values)==set(reference),'metric schema mismatch')
                differences[str(seed)][name]['W_minus_'+v]={k:values[k]-reference[k] if finite(values[k]) and finite(reference[k]) else None for k in values}
        if a.with_training:train[key]=training(metas[seed])
    write(a.output,{'status':'COMPLETED','split':'train','stage_step':1024,'models':models,'paired_differences':differences,'training':train,'sources':provenance,'baseline_metrics_sha256':NQ_SHA,'manifest_sha256':a.manifest_sha256,'cohort_sha256':COHORT_SHA,'scope':'JSON facts only; nulls retained; no scientific gate, inference or promotion decision'})
def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('output','manifest','baseline-metrics','baseline-manifest'):parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--manifest-sha256',required=True);parser.add_argument('--helper-sha256',required=True);parser.add_argument('--max-seconds',type=float,default=40.);parser.add_argument('--after-complete',action='store_true');parser.add_argument('--with-training',action='store_true');a=parser.parse_args()
    try:extract(a)
    except (ValueError,KeyError,TypeError,OSError,TimeoutError) as exc:
        if a.output.exists():raise
        write(a.output,{'status':'UNRESOLVED','error':repr(exc),'scope':'no missing role/event substituted'});raise SystemExit(2)
if __name__=='__main__':main()
