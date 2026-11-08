"""Finite D/W preparation and execution; old numerical kernels are reused.

Import reads no data. Every executable phase requires the independently bound
root release in weak_force_execution_20261029. No phase authorizes itself.
"""
from __future__ import annotations
import copy
import json
import math
from pathlib import Path
import random
import sys
import time
import uuid

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'experiments'),str(ROOT/'src')]
SEEDS=(29311,29312)
VARIANTS=('D','W')
LIMITS=dict(qa=72,formal=768,repair=72,replay=64)
TOTAL=976
PARENT=ROOT/'private_data/state_gated_memory_20261028/study01'
SOURCE=PARENT/'study/A_28311/checkpoints/step000096'
SOURCE_SHA={'model.json':'ec8c78cbff0b6dd65bd3ba6dfc5a0ed65e7fa418227c90adc3a703311aebadf4',
 'checkpoint.json':'249cc7fab5226cfbd52046efa2dc0cd8e21c0c4dda1f23c71092d8e6e86b0b12',
 'training.pt':'15cc4be8ba8bf39e755f8fafcadf0b78ec8badd65db6a8dfeb6b248283ed2ecc'}
EVIDENCE_SHA={'qualification/R_N_fit297/evidence.json':'140af6d7d30d663f66a95f31b92348903283d973e0091f9121d6e1548cfbeb19',
 'study/A_28311/fit000096/evidence.json':'22e2b30d9c4e6dc4d6162d11e3eeb2707a9a6b5d461fef956a3aff8e4e15c5fb'}


def deps():
    import numpy as np
    import torch
    import state_gated_memory_study_20261028 as sgm
    import weak_force_identification_20261029 as weakcore
    return np,torch,sgm,weakcore


def admitted(base):
    from experiments import weak_force_execution_20261029 as execution
    execution.require_release()
    execution.check_binding(execution.read(Path(base)/'binding.json'),base)


def counts(base):
    calls={}; categories={k:dict(attempted=0,completed=0) for k in LIMITS}
    path=Path(base)/'optimizer_calls.jsonl'
    if path.exists():
        for line in path.read_text(encoding='utf-8').splitlines():
            r=json.loads(line); cid=r['call_id']; category=r['category']
            if category not in LIMITS: raise ValueError('unknown optimizer category')
            if r['event']=='optimizer_step_attempt' and cid not in calls:
                calls[cid]=dict(r,completed=False); categories[category]['attempted']+=1
            elif r['event']=='optimizer_step_completed' and cid in calls and not calls[cid]['completed']:
                if any(r[k]!=calls[cid][k] for k in ('run_id','category','step')):
                    raise ValueError('optimizer completion identity mismatch')
                calls[cid]['completed']=True; categories[category]['completed']+=1
            else: raise ValueError('duplicate or unpaired optimizer event')
    total=sum(v['attempted'] for v in categories.values())
    if total>TOTAL or any(v['attempted']>LIMITS[k] for k,v in categories.items()):
        raise ValueError('registered 976/category optimizer bound exceeded')
    return dict(categories=categories,attempted_total=total,
                completed_total=sum(v['completed'] for v in categories.values()),
                pending=[r for r in calls.values() if not r['completed']])


def journal(base,event,category,run_id,step,cid):
    if category not in ('qa','formal'):
        raise ValueError('reserved repair/replay is not automatically enabled')
    c=counts(base)
    if event=='optimizer_step_attempt':
        if c['pending'] or c['attempted_total']>=TOTAL or c['categories'][category]['attempted']>=LIMITS[category]:
            raise ValueError('pending optimizer attempt or finite allowance exhausted')
        if category=='formal' and (run_id not in {f'{v}_{s}' for v in VARIANTS for s in SEEDS} or not 1<=step<=192):
            raise ValueError('unregistered formal run or step')
    row=dict(event=event,category=category,run_id=run_id,step=step,call_id=cid,time_unix=time.time())
    with (Path(base)/'optimizer_calls.jsonl').open('a',encoding='utf-8') as stream:
        stream.write(json.dumps(row,allow_nan=False)+'\n');stream.flush()


def source_identity():
    _,_,sgm,_=deps()
    actual={name:sgm.sha(SOURCE/name) for name in SOURCE_SHA}
    if actual!=SOURCE_SHA: raise ValueError('frozen A96 three assets changed')
    return dict(logical_name='SGM_A_28311_step96',files=actual,ancestry='N_M192 -> SGM_A96',old_Adam_loaded=False)


def build_model(variant,base,device='cpu'):
    _,torch,sgm,_=deps()
    source_identity()
    m=sgm.StateGatedMemoryROM.from_export(SOURCE/'model.json')
    if m.gate_mode!='A' or m.history_dt!=.0025: raise ValueError('same frozen gate A required')
    if variant=='W':
        result=sgm.read(Path(base)/'weak/summary.json')
        if result.get('training_candidate') is not True: raise ValueError('weak candidate not admitted')
        raw=result['weak_solution']['raw_coordinates']
        with torch.no_grad():
            m.kappa_raw.copy_(torch.tensor(raw['kappa_raw'],dtype=torch.float64))
            m.lambda_raw.copy_(torch.tensor(raw['lambda_raw'],dtype=torch.float64))
            m.a_bar.copy_(torch.tensor(raw['a_bar'],dtype=torch.float64))
        m.clear_cache()
    elif variant!='D': raise ValueError('D/W only')
    return m.to(sgm.resolve_device(device)).double()


def checked_preparation(base,fit_only=True):
    admitted(base)
    _,_,sgm,_=deps(); base=Path(base); p=sgm.read(base/'prepare.json')
    if p['source']!=source_identity() or p['schedule_sha256']!=sgm.sha(base/'schedule.json'):
        raise ValueError('frozen source/schedule changed')
    old,events=sgm.checked_preparation(PARENT,fit_only=fit_only)
    if old['fit_ids']!=p['fit_ids']: raise ValueError('fit297 identity changed')
    return p,events


def prepare(base):
    admitted(base)
    np,_,sgm,_=deps();base=Path(base)
    parent,events=sgm.checked_preparation(PARENT,fit_only=True)
    if len(events)!=297 or sum(e['label']=='unstable' for e in events)!=30:
        raise ValueError('fixed fit297/30 identity required')
    schedules={}
    for seed in SEEDS:
        rng=np.random.default_rng(seed);cov=dict(cover_order=rng.permutation(297).tolist(),cursor=0)
        initial=dict(rng=copy.deepcopy(rng.bit_generator.state),coverage=copy.deepcopy(cov));rows=[]
        for step in range(1,193):
            _,sampling=sgm.fg.sample_step(events,rng,cov)
            rows.append(dict(step=step,horizon_s=sgm.horizon_for(step),**sampling,
                rng_after=copy.deepcopy(rng.bit_generator.state),coverage_after=copy.deepcopy(cov)))
        schedules[str(seed)]=dict(initial=initial,rows=rows)
    sgm.write(base/'schedule.json',schedules,exclusive=True)
    record={k:parent[k] for k in ('fit_ids','no_update_ids','original366_ids','fresh19_ids','fit_groups','no_update_groups',
                                  'fit_cache_sha256','no_update_cache_sha256','panel_ids','cost12_ids')}
    record.update(status='COMPLETED',source=source_identity(),schedule_sha256=sgm.sha(base/'schedule.json'),
                  source_exposed_to_train385=True,no_update_payload_read=False,val_read=False,test_read=False)
    sgm.write(base/'prepare.json',record,exclusive=True);return record


def weak(base,device='cpu'):
    admitted(base)
    _,_,sgm,core=deps();base=Path(base);output=base/'weak';output.mkdir(exist_ok=False)
    tick=time.perf_counter()
    result=dict(status='CANDIDATE_NOT_ADMITTED',training_candidate=False,real_nnls_attempts=0,val_read=False,test_read=False)
    try:
        fixtures=core.qualify_fixtures(output/'fixtures')
        result['fixtures_passed']=fixtures['passed']
        if not fixtures['passed']:
            result['reason']='fixed_known_ROM_qualification_failed'
        else:
            prepared,events=checked_preparation(base)
            skeleton=core.Skeleton.from_model(build_model('D',base))
            observations=[core.Observations(e['id'],e['label']=='unstable',e['x']) for e in events]
            matrices=core.build_matrices(observations,skeleton,'A')
            core.write_new_json(output/'coverage.json',matrices.coverage)
            import numpy as np
            with (output/'matrices.npz').open('xb') as stream:
                np.savez(stream,A=matrices.A,b=matrices.b,A2=matrices.A2,b2=matrices.b2,unstable_rows=matrices.unstable_rows)
            def real_event(event,detail):
                if event=='attempt':
                    if (output/'real_nnls_attempt.json').exists(): raise ValueError('sole real NNLS attempt already consumed')
                    core.write_new_json(output/'real_nnls_attempt.json',dict(event=event,**detail))
                    result['real_nnls_attempts']=1
                else:
                    core.write_new_json(output/('real_nnls_'+event+'.json'),dict(event=event,**detail))
            solution=core.solve_once(matrices,nnls_event=real_event)
            core.write_new_json(output/'solution.json',solution)
            result.update(weak_solution=solution,training_candidate=solution['success'],
                          status='COMPLETED' if solution['success'] else 'CANDIDATE_NOT_ADMITTED',
                          reason=None if solution['success'] else solution.get('reason','weak_KKT_or_conversion_failed'))
    except BaseException as exc:
        result.update(status='NUMERICAL_UNRESOLVED',reason=repr(exc),failure_is_physical_instability=False,seconds=time.perf_counter()-tick)
        core.write_new_json(output/'summary.json',result)
        if not isinstance(exc,Exception):raise
        return result
    result['seconds']=time.perf_counter()-tick
    core.write_new_json(output/'summary.json',result);return result


def one_update(model,opt,store,samples,horizon,*,base,category,run_id,step):
    _,torch,sgm,_=deps()
    sgm.synchronize(model.mass.device);tick=time.perf_counter();opt.zero_grad(set_to_none=True)
    with sgm.graph(model):
        J,terms=sgm.objective(model,store,samples,horizon,method='rk4')
        if not torch.isfinite(J):raise FloatingPointError('nonfinite J')
        J.backward()
    params=[p for p in model.parameters() if p.requires_grad]
    if any(p.grad is None or not torch.isfinite(p.grad).all() for p in params):raise FloatingPointError('invalid gradient')
    norm=torch.nn.utils.clip_grad_norm_(params,100.)
    if not torch.isfinite(norm):raise FloatingPointError('nonfinite clipped norm')
    before={n:p.detach().clone() for n,p in model.named_parameters() if p.requires_grad};cid=uuid.uuid4().hex
    journal(base,'optimizer_step_attempt',category,run_id,step,cid);opt.step()
    journal(base,'optimizer_step_completed',category,run_id,step,cid)
    model.project_runtime_();sgm.synchronize(model.mass.device)
    if any(not torch.isfinite(p).all() for p in params) or not sgm.old._adam_states_finite(opt.state.values()):
        raise FloatingPointError('nonfinite parameter/Adam after completed update')
    return dict(step=step,horizon_s=horizon,method='rk4',**{k:float(v.detach()) for k,v in terms.items()},
        gradient_norm=float(norm),seconds=time.perf_counter()-tick,
        block_updates={n:dict(relative_l2=float((p.detach()-before[n]).norm()/before[n].norm().clamp_min(1.)))
                       for n,p in model.named_parameters() if p.requires_grad})


def device_comparison(variant,base):
    np,torch,sgm,_=deps();events=sgm.old.synthetic_fixture()
    models=[build_model(variant,base,d) for d in ('cpu','cuda')]
    h=np.stack([e['x'][:11] for e in events]);states=np.column_stack((h[:,-1],np.full((len(h),10),1e-5)))
    values=[];gradients=[]
    for m in models:
        t=dict(field=m.field(m._tensor(states)),initialize=m.initialize(h),short_RK4=m.rollout(h,4,.01))
        with sgm.graph(m):
            J,_=sgm.objective(m,sgm.EventStore(events,m.mass.device),[(e,10) for e in events],.02,method='rk4');J.backward()
        t['J']=J;values.append({k:v.detach().cpu() for k,v in t.items()})
        gradients.append({n:p.grad.detach().cpu() for n,p in m.named_parameters() if p.requires_grad and p.grad is not None})
        if set(gradients[-1])!={n for n,p in m.named_parameters() if p.requires_grad}:raise ValueError('missing raw gradient')
    for key in values[0]:torch.testing.assert_close(values[0][key],values[1][key],rtol=1e-8,atol=1e-10)
    for key in gradients[0]:
        torch.testing.assert_close(gradients[0][key],gradients[1][key],rtol=1e-8,atol=1e-10)
        if not torch.isfinite(gradients[1][key]).all():raise ValueError('nonfinite gradient')
    return dict(status='PASS',atol=1e-10,rtol=1e-8,all_raw_gradients=True)


def qualify(base,device='cuda'):
    admitted(base)
    np,torch,sgm,_=deps();base=Path(base);output=base/'qualification';output.mkdir(exist_ok=False)
    result=dict(status='CANDIDATE_NOT_ADMITTED',training_admission=False,eligible_variants=[],per_variant={},val_read=False,test_read=False)
    if not sgm.read(base/'weak/summary.json').get('training_candidate'):
        result['reason']='weak_candidate_not_admitted';sgm.write(output/'summary.json',result,exclusive=True);return result
    prepared,fit=checked_preparation(base)
    cpu=sgm.read(ROOT/'results/state_gated_memory_20261028/model_pre_freeze_qa.json')
    if cpu.get('shared_checks_qualified') is not True or cpu.get('per_variant_qualified',{}).get('A') is not True:
        raise ValueError('unchanged gate A mathematical qualification required')
    for path,digest in EVIDENCE_SHA.items():
        if sgm.sha(PARENT/path)!=digest:raise ValueError('frozen source/reference evidence changed')
    result['R_N_profile']=sgm.read(PARENT/'qualification/summary.json')['R_N_profile']
    result['A96_profile']=sgm.read(PARENT/'study/A_28311/fit000096/state_gated_profile.json')
    selected=[e for eid in prepared['cost12_ids'] for e in fit if e['id']==eid]
    if len(selected)!=12:raise ValueError('registered cost12 missing')
    for variant in VARIANTS:
        tick_variant=time.perf_counter();row=dict(training_admission=False)
        try:
            parity=device_comparison(variant,base)
            model=build_model(variant,base)
            evidence=sgm.evaluate(model,fit,output/f'{variant}_zero_fit297',prepared)
            if variant=='D':
                old_evidence=sgm.read(PARENT/'study/A_28311/fit000096/evidence.json')
                a=sgm._fine_cache(old_evidence,fit,prepared['fit_ids'])
                d=sgm._fine_cache(evidence,fit,prepared['fit_ids'])
                byid={e['id']:e for e in fit}
                rows=[sgm._cross_event(model,byid[eid],a[eid],d[eid]) for eid in prepared['fit_ids']]
                cross=dict(qualified=len(rows)==297 and all(r['qualified'] for r in rows),
                    record_count=len(rows),event_ids=list(prepared['fit_ids']),
                    scope='A96_D0_fit297_bidirectional_RK4_fine005_base01_8s_actual15',
                    bidirectional=True,history_rows=11,initial_observation_index=10,
                    fine_dt=.005,output_dt=.01,coarse_stride=2,
                    original_schema_aliases=dict(R_N='frozen_A96',A='current_D0'),rows=rows,
                    source_A96_prediction_recomputed=False,source_evidence_sha256=EVIDENCE_SHA['study/A_28311/fit000096/evidence.json'])
                sgm.write(output/'D_A96_compatibility.json',cross,exclusive=True)
                row['D_A96_compatibility_qualified']=cross['qualified']
                row['D_A96_compatibility']=dict(path=str((output/'D_A96_compatibility.json').resolve()),
                                               sha256=sgm.sha(output/'D_A96_compatibility.json'))
                if not cross['qualified']:raise ValueError('D0/A96 original bidirectional compatibility failed')
            m=build_model(variant,base,device);store=sgm.EventStore(selected,m.mass.device)
            sgm.synchronize(m.mass.device);tick=time.perf_counter()
            with sgm.graph(m):
                J,_=sgm.objective(m,store,[(e,10) for e in selected],8.,method='rk4');J.backward()
            sgm.synchronize(m.mass.device)
            finite=bool(torch.isfinite(J)) and all(p.grad is not None and torch.isfinite(p.grad).all() for p in m.parameters() if p.requires_grad)
            row.update(cpu_qualified=True,cpu_cuda_pair_pass=True,device_comparison=parity,
                finite_12x8_backward=finite,cost12_seconds=time.perf_counter()-tick,
                fullfit297_qualified=sgm.valid(evidence['decision_profile']),zero_profile=evidence['decision_profile'])
            row['training_admission']=finite and row['fullfit297_qualified']
        except Exception as exc:row.update(error=repr(exc),failure_is_physical_instability=False)
        row['qualification_seconds']=time.perf_counter()-tick_variant
        result['per_variant'][variant]=row
        sgm.write(output/(variant+'_result.json'),row,exclusive=True)
    result['training_admission']=all(result['per_variant'][v]['training_admission'] for v in VARIANTS)
    result.update(status='COMPLETED' if result['training_admission'] else 'NUMERICAL_UNRESOLVED',
        eligible_variants=list(VARIANTS) if result['training_admission'] else [],
        training_method='rk4',history_dt_main=.0025,history_dt_reference=.00125)
    sgm.write(output/'summary.json',result,exclusive=True);return result


def qa(base,device='cuda'):
    admitted(base)
    np,torch,sgm,_=deps();base=Path(base);output=base/'qa';output.mkdir(exist_ok=False)
    eligible=sgm.read(base/'qualification/summary.json')['eligible_variants'];results=[]
    for variant in eligible:
        tick_variant=time.perf_counter()
        folder=output/variant;folder.mkdir();events=sgm.old.synthetic_fixture();store=sgm.EventStore(events,device)
        identity=dict(variant=variant,fixture_only=True,source=source_identity())
        def initial():
            m=build_model(variant,base,device);opt=sgm.make_optimizer(m);sgm.fg._seed(SEEDS[0],m.mass.device)
            rng=np.random.default_rng(SEEDS[0]);return m,opt,rng,dict(cover_order=rng.permutation(len(events)).tolist(),cursor=0)
        def advance(m,opt,rng,cov,start,end,label):
            trace=[]
            for step in range(start+1,end+1):
                samples,_=sgm.fg.sample_step(events,rng,cov);random.random();np.random.random();torch.rand((),device=m.mass.device)
                trace.append(one_update(m,opt,store,samples,.02,base=base,category='qa',run_id=f'qa_{variant}_{label}',step=step))
            return trace
        try:
            a,oa,ra,ca=initial();advance(a,oa,ra,ca,0,18,'continuous');final_rng=sgm.old._cpu_snapshot(sgm.fg._rng_state(ra,ca))
            b,ob,rb,cb=initial();trace=advance(b,ob,rb,cb,0,16,'restored')
            snapshot=sgm.old._cpu_snapshot(dict(model=b.state_dict(),Adam=ob.state_dict(),RNG=sgm.fg._rng_state(rb,cb),layout=sgm.old._parameter_layout(b,ob)))
            if not any(bool(torch.count_nonzero(s['exp_avg'])) for s in ob.state.values()):raise ValueError('nonzero Adam moments required')
            sgm.save_checkpoint(folder/'checkpoints',b,ob,rb,cb,16,identity,trace)
            c,oc,rc,cc,_=sgm.restore_checkpoint(folder/'checkpoints',device,identity)
            sgm.old._exact_nested(snapshot,sgm.old._cpu_snapshot(dict(model=c.state_dict(),Adam=oc.state_dict(),RNG=sgm.fg._rng_state(rc,cc),layout=sgm.old._parameter_layout(c,oc))))
            advance(c,oc,rc,cc,16,18,'restored')
            pe=sgm.old._qa_compare(sgm.old._cpu_snapshot(a.state_dict()),sgm.old._cpu_snapshot(c.state_dict()))
            sa,sc=sgm.old._cpu_snapshot(oa.state_dict()),sgm.old._cpu_snapshot(oc.state_dict())
            sgm.old._exact_nested(sa['param_groups'],sc['param_groups'])
            ae=max(sgm.old._qa_compare(sa['state'][k],sc['state'][k]) for k in sa['state'])
            sgm.old._exact_nested(final_rng,sgm.old._cpu_snapshot(sgm.fg._rng_state(rc,cc)))
            row=dict(variant=variant,status='PASS',actual_optimizer_calls=36,parameter_maxabs=pe,Adam_maxabs=ae)
        except BaseException as exc:
            row=dict(variant=variant,status='NUMERICAL_UNRESOLVED',error=repr(exc),failure_is_physical_instability=False,seconds=time.perf_counter()-tick_variant)
            sgm.write(folder/'result.json',row,exclusive=True);results.append(row)
            if not isinstance(exc,Exception):raise
            break
        row['seconds']=time.perf_counter()-tick_variant
        sgm.write(folder/'result.json',row,exclusive=True);results.append(row)
    passed=len(results)==2 and all(r['status']=='PASS' for r in results)
    # This immutable projection binds only QA calls. The append-only global ledger
    # necessarily grows during formal training and must not become a frozen input.
    records=[]
    ledger=base/'optimizer_calls.jsonl'
    if ledger.exists():
        records=[json.loads(line) for line in ledger.read_text(encoding='utf-8').splitlines()]
    projection=[r for r in records if r['category']=='qa']
    sgm.write(output/'optimizer_projection.json',projection,exclusive=True)
    result=dict(status='COMPLETED' if passed else 'CANDIDATE_NOT_ADMITTED',models=results,
        eligible_variants=list(VARIANTS) if passed else [],training_admission=passed,calls=counts(base),
        qa_journal_projection=dict(path=str((output/'optimizer_projection.json').resolve()),
                                   sha256=sgm.sha(output/'optimizer_projection.json')),val_read=False,test_read=False)
    sgm.write(output/'summary.json',result,exclusive=True);return result


def timing_comparison(profiles):
    ids=set.intersection(*(set(p['timing_actual15']) for p in profiles.values()));common=[]
    for eid in sorted(ids):
        rows=[p['timing_actual15'][eid] for p in profiles.values()]
        if all(r['numerically_qualified'] and all(r[k]['time_s'] is not None for k in ('truth','fine','coarse')) for r in rows):common.append(eid)
    if not common:return dict(passed=False,common_event_ids=[],reason='empty_common_T')
    metrics={name:dict(E=sum(abs(p['timing_actual15'][i]['fine']['time_s']-p['timing_actual15'][i]['truth']['time_s']) for i in common)/len(common),
        u=max(abs(p['timing_actual15'][i]['fine']['time_s']-p['timing_actual15'][i]['coarse']['time_s']) for i in common)) for name,p in profiles.items()}
    passed=all(metrics['W']['E']<=m['E']+.01+metrics['W']['u']+m['u'] for name,m in metrics.items() if name!='W')
    return dict(passed=passed,metrics=metrics,common_event_ids=common)


def retention(profiles,optimization_unresolved=False):
    _,_,sgm,_=deps()
    if set(profiles)!={'D','W','R_N'} or optimization_unresolved or not all(sgm.valid(p) for p in profiles.values()):
        return dict(second_seed=False,reason='missing_or_unresolved_control')
    w,d,r=profiles['W'],profiles['D'],profiles['R_N']; timing=timing_comparison(profiles)
    fewer=w['classification_actual15s']['missed_resolved_unstable']<=d['classification_actual15s']['missed_resolved_unstable']-1
    risk=w['unstable_actual15_risk']<=.95*d['unstable_actual15_risk'] and d['unstable_actual15_risk']-w['unstable_actual15_risk']>w['unstable_actual15_numerical_score_perturbation']+d['unstable_actual15_numerical_score_perturbation']
    guards=True
    for p in (d,r):
        guards &= all(w[key][metric]<=p[key][metric] for key in ('classification_8s','classification_actual15s')
                      for metric in ('missed_resolved_unstable','false_los_resolved_stable'))
        guards &= all(w['horizon_risks'][h]['value']<=1.05*p['horizon_risks'][h]['value'] for h in ('5','8'))
        guards &= all(w[k]<=1.05*p[k] for k in ('common_frequency_rmse_8s','normalized_p_rmse_8s'))
    guards &= w['unstable_actual15_risk']<=r['unstable_actual15_risk']
    def vector(p):
        return [p['classification_actual15s']['missed_resolved_unstable'],p['classification_8s']['missed_resolved_unstable'],
            p['classification_actual15s']['false_los_resolved_stable'],p['unstable_actual15_risk'],
            *[p['horizon_risks'][h]['value'] for h in ('5','8')],p['common_frequency_rmse_8s'],p['normalized_p_rmse_8s']]
    dominated=any(all(a<=b for a,b in zip(vector(p),vector(w))) and any(a<b for a,b in zip(vector(p),vector(w))) for p in (d,r))
    return dict(second_seed=bool((fewer or risk) and guards and timing['passed'] and not dominated),
        fewer_misses=bool(fewer),risk_gain=bool(risk),guards=bool(guards),dominated=dominated,timing=timing)


def formal_run(base,variant,seed,prepared,fit,panel,device,prior_candidate_seconds=0.):
    np,_,sgm,_=deps();base=Path(base);run_id=f'{variant}_{seed}';folder=base/'study'/run_id;folder.mkdir(exist_ok=False)
    m=build_model(variant,base,device);opt=sgm.make_optimizer(m);sgm.fg._seed(seed,m.mass.device)
    schedule=sgm.read(base/'schedule.json')[str(seed)];rng=np.random.default_rng();rng.bit_generator.state=schedule['initial']['rng']
    cov=copy.deepcopy(schedule['initial']['coverage']);trace=[];profiles={};full={};cost=[]
    identity=dict(variant=variant,seed=seed,source=prepared['source'],schedule_sha256=prepared['schedule_sha256'])
    store=sgm.EventStore(fit,m.mass.device);lookup={e['id']:e for e in fit};started=time.perf_counter()
    qualified=sgm.read(base/'qualification/summary.json')['per_variant'][variant]
    qa_row=next(r for r in sgm.read(base/'qa/summary.json')['models'] if r['variant']==variant)
    prelude=qualified['qualification_seconds']+qa_row['seconds']+prior_candidate_seconds
    if variant=='W':prelude+=sgm.read(base/'weak/summary.json')['seconds']
    try:
        profiles['0']=sgm.score_panel(m,panel);sgm.write(folder/'panel000000.json',profiles['0'],exclusive=True)
        if not sgm.valid(profiles['0']):raise ValueError('initial panel unresolved')
        sgm.save_checkpoint(folder/'checkpoints',m,opt,rng,cov,0,identity,trace);end=96
        full['0']=sgm.read(base/f'qualification/{variant}_zero_fit297/state_gated_profile.json')
        cost.append(dict(step=0,run_cumulative_seconds=time.perf_counter()-started,
                         candidate_cumulative_seconds=prelude+time.perf_counter()-started))
        for row in schedule['rows']:
            step=row['step']
            if step>end:break
            trace.append(one_update(m,opt,store,[(lookup[eid],10) for eid in row['sample_ids']],row['horizon_s'],base=base,category='formal',run_id=run_id,step=step))
            rng.bit_generator.state=row['rng_after'];cov=copy.deepcopy(row['coverage_after'])
            if step%16==0:sgm.save_checkpoint(folder/'checkpoints',m,opt,rng,cov,step,identity,trace)
            if step in (48,96,192):
                profiles[str(step)]=sgm.score_panel(m,panel);sgm.write(folder/f'panel{step:06d}.json',profiles[str(step)],exclusive=True)
            if step in (96,192):
                full[str(step)]=sgm.evaluate(m,fit,folder/f'fit{step:06d}',prepared)['decision_profile']
                cost.append(dict(step=step,run_cumulative_seconds=time.perf_counter()-started,
                                 candidate_cumulative_seconds=prelude+time.perf_counter()-started))
            if step==96:
                decision=sgm.extension_decision(profiles['48'],profiles['96'],trace)
                sgm.write(folder/'extension_decision.json',decision,exclusive=True)
                if decision['extend']:end=192
        result=dict(status='COMPLETED',run_id=run_id,variant=variant,seed=seed,updates=len(trace),
            endpoint=str(folder/'checkpoints'/f'step{len(trace):06d}'),fit_endpoints=full,cost_milestones=cost,
            run_total_seconds=time.perf_counter()-started,timing_scope='wall seconds including registered evaluation; not device occupation',
            optimization_unresolved_at96=sgm.optimization(trace[:96])['optimization_unresolved'],finite_optimization_scope=True)
    except BaseException as exc:
        quarantine=sgm.old.quarantine_failure(folder,m,opt,rng,cov,len(trace),trace,exc)
        result=dict(status='NUMERICAL_UNRESOLVED',run_id=run_id,variant=variant,seed=seed,updates=len(trace),
            error=repr(exc),quarantine=quarantine,failure_is_physical_instability=False,resumable=False)
        sgm.write(folder/'result.json',result,exclusive=True)
        if not isinstance(exc,Exception):raise
        return result
    sgm.write(folder/'result.json',result,exclusive=True);return result


def study(base,device='cuda'):
    admitted(base)
    _,_,sgm,_=deps();base=Path(base);output=base/'study';output.mkdir(exist_ok=False)
    eligible=sgm.read(base/'qa/summary.json')['eligible_variants'];runs=[];decisions=[];reports=[]
    if eligible==list(VARIANTS):
        prepared,fit=checked_preparation(base);panel=[e for eid in prepared['panel_ids'] for e in fit if e['id']==eid]
        reference=sgm.read(base/'qualification/summary.json')['R_N_profile']
        for seed in SEEDS:
            group=[]
            for variant in VARIANTS:
                prior=sum(r.get('run_total_seconds',0.) for r in runs if r['variant']==variant)
                r=formal_run(base,variant,seed,prepared,fit,panel,device,prior);group.append(r);runs.append(r)
                if r['status']!='COMPLETED':break
            profiles={r['variant']:r['fit_endpoints']['96'] for r in group if r['status']=='COMPLETED'}
            profiles['R_N']=reference
            decision=retention(profiles,any(r.get('optimization_unresolved_at96',True) for r in group));decisions.append(dict(seed=seed,**decision))
            sgm.write(output/f'retention_{seed}.json',decisions[-1],exclusive=True)
            if not decision['second_seed']:break
        sgm.write(output/'fit_decisions_closed.json',dict(decisions=decisions,run_ids=[r['run_id'] for r in runs]),exclusive=True)
        qualification=sgm.read(base/'qualification/summary.json')
        efficiency=[]
        for run in runs:
            if run['status']!='COMPLETED':continue
            records=[]
            paired=[r for r in runs if r['seed']==run['seed'] and r['status']=='COMPLETED']
            common_full=set.intersection(*(set(r['fit_endpoints']) for r in paired)) if len(paired)==2 else set()
            for milestone in run['cost_milestones']:
                step=str(milestone['step'])
                if step not in common_full:continue
                p=run['fit_endpoints'][step]
                target=efficiency_target(p,qualification['A96_profile'],qualification['R_N_profile'])
                records.append(dict(**milestone,target=target))
            first=next((r for r in records if r['target']['passed']),None)
            efficiency.append(dict(run_id=run['run_id'],milestones=records,first_attained=first,
                                   right_censored=first is None,no_interpolation=True))
        sgm.write(output/'efficiency.json',dict(runs=efficiency,cost_scope='per-candidate full prelude + registered run wall time; common preparation and native subtree cost separate',
            same_cost_winner_claimed=False,unmeasured_capability_not_interpolated=True),exclusive=True)
        # Only after every fit decision is fixed may train88 payload be loaded.
        _,all_events=checked_preparation(base,fit_only=False);held=[e for e in all_events if e['id'] in prepared['no_update_ids']]
        for r in runs:
            if r['status']!='COMPLETED':continue
            m=sgm.StateGatedMemoryROM.from_export(Path(r['endpoint'])/'model.json');folder=output/r['run_id']
            sgm.evaluate(m,all_events,folder/'evaluation385',prepared)
            sgm.evaluate(m,held,folder/'noise88',prepared,noisy=True)
            reports.append(dict(run_id=r['run_id'],report385=True,noise88_report_only=True))
    result=dict(status='COMPLETED' if len(runs)>=2 and all(r['status']=='COMPLETED' for r in runs) else 'CANDIDATE_NOT_ADMITTED',
        runs=runs,decisions=decisions,reports=reports,calls=counts(base),default_model=None,scientific_model_adoption=False,val_read=False,test_read=False)
    sgm.write(output/'summary.json',result,exclusive=True);return result


def efficiency_target(candidate,A96,R_N):
    """Frozen absolute capability target, only at existing full-fit milestones."""
    _,_,sgm,_=deps()
    if not all(sgm.valid(p) for p in (candidate,A96,R_N)):
        return dict(passed=False,reason='unresolved_profile')
    w=candidate;timing=timing_comparison(dict(W=w,A96=A96,R_N=R_N));guards=True
    for p in (A96,R_N):
        guards &= all(w[key][metric]<=p[key][metric] for key in ('classification_8s','classification_actual15s')
            for metric in ('missed_resolved_unstable','false_los_resolved_stable'))
    guards &= w['unstable_actual15_risk']<=R_N['unstable_actual15_risk']
    guards &= all(w['horizon_risks'][h]['value']<=1.05*R_N['horizon_risks'][h]['value'] for h in ('5','8'))
    guards &= all(w[k]<=1.05*R_N[k] for k in ('common_frequency_rmse_8s','normalized_p_rmse_8s'))
    fewer=w['classification_actual15s']['missed_resolved_unstable']<=20
    gain=w['unstable_actual15_risk']<=.95*A96['unstable_actual15_risk'] and A96['unstable_actual15_risk']-w['unstable_actual15_risk']>A96['unstable_actual15_numerical_score_perturbation']+w['unstable_actual15_numerical_score_perturbation']
    return dict(passed=bool(guards and (fewer or gain) and timing['passed']),timing=timing,
                fewer_misses=bool(fewer),risk_gain=bool(gain),guards=bool(guards))


def summary(base):
    admitted(base)
    _,_,sgm,_=deps();base=Path(base)
    result=dict(status=sgm.read(base/'study/summary.json')['status'],calls=counts(base),model_adopted=False,
        val_read=False,test_read=False,no_process_quotas=True,scientific_decision_owner='root')
    sgm.write(base/'summary.json',result,exclusive=True);return result
