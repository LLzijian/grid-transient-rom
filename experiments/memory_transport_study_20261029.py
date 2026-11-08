"""Finite A/S/P adaptation. Import never reads event/source payloads or runs work."""
from __future__ import annotations
import copy,json,math,random,sys,time,uuid
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'experiments'),str(ROOT/'src')]
VARIANTS=('A','S','P');SEEDS=(30311,30312)
LIMITS=dict(formal=1536,qa=108,repair=108,replay=96);TOTAL=1848
SGM=ROOT/'private_data/state_gated_memory_20261028/study01'
SOURCE=SGM/'study/A_28311/checkpoints/step000096'
SOURCE_SHA={'model.json':'ec8c78cbff0b6dd65bd3ba6dfc5a0ed65e7fa418227c90adc3a703311aebadf4',
 'checkpoint.json':'249cc7fab5226cfbd52046efa2dc0cd8e21c0c4dda1f23c71092d8e6e86b0b12',
 'training.pt':'15cc4be8ba8bf39e755f8fafcadf0b78ec8badd65db6a8dfeb6b248283ed2ecc'}
PLANE=ROOT/'private_data/memory_transport_20261029/source_plane01/result.json'
PLANE_SHA='3dee4225e77e5bf6ab6ad58a73ab4ad9c8a263e647d053fcb6842cc4a7281941'
PLANE_PUBLIC=ROOT/'results/memory_transport_20261029/source_plane_public.json'
PLANE_PUBLIC_SHA='c40bb7a6ecc31e8d7228f77f6e71937ac8e3fed66f305700520d35bc6b4bfdda'
EVIDENCE_SHA={'qualification/summary.json':'a169a7e798b87ce244b0c45eb27b6ce75e38f55ed014e28382e71d918c113288',
 'qualification/R_N_fit297/evidence.json':'140af6d7d30d663f66a95f31b92348903283d973e0091f9121d6e1548cfbeb19',
 'study/A_28311/fit000096/evidence.json':'22e2b30d9c4e6dc4d6162d11e3eeb2707a9a6b5d461fef956a3aff8e4e15c5fb'}
PROTOCOL=ROOT/'reports/memory_transport_20261029_protocol.md'


def deps():
    import numpy as np
    import torch
    import state_gated_memory_study_20261028 as sgm
    import memory_transport_model_20261029 as mt
    return np,torch,sgm,mt


def admitted(base):
    from experiments import memory_transport_execution_20261029 as ex
    ex.require_release();ex.check_binding(ex.read(Path(base)/'binding.json'),base)


def code_identity():
    from experiments.r_memory_execution_20261025 import sha
    paths=[Path(__file__),ROOT/'experiments/memory_transport_execution_20261029.py',
           ROOT/'experiments/memory_transport_model_20261029.py',PROTOCOL]
    return {p.relative_to(ROOT).as_posix():sha(p) for p in paths}


def source_identity():
    from experiments.r_memory_execution_20261025 import sha,read
    actual={name:sha(SOURCE/name) for name in SOURCE_SHA}
    if actual!=SOURCE_SHA or sha(PLANE)!=PLANE_SHA or sha(PLANE_PUBLIC)!=PLANE_PUBLIC_SHA:
        raise ValueError('frozen A96/plane bytes changed; never recompute plane')
    plane=read(PLANE)
    if plane.get('status')!='COMPLETED' or plane.get('cutoff_pass') is not True or plane.get('eigh_count')!=1 or plane.get('source_sha256')!=SOURCE_SHA['model.json']:
        raise ValueError('sole source plane is not qualified')
    return dict(logical_name='SGM_A_28311_step96',files=actual,plane_sha256=PLANE_SHA,
                plane_public_sha256=PLANE_PUBLIC_SHA,ancestry='N_M192 -> SGM A96',old_Adam_loaded=False)


def build_model(variant,device='cpu'):
    if variant not in VARIANTS:raise ValueError('A/S/P only')
    _,_,sgm,mt=deps();identity=source_identity();plane=sgm.read(PLANE)
    model=mt.MemoryTransportROM.from_a_export(SOURCE/'model.json',variant,u=plane['u'],v=plane['v'],
        expected_sha256=SOURCE_SHA['model.json'],history_dt=.0025,plane_identity=dict(source_plane_sha256=PLANE_SHA))
    return model.to(sgm.resolve_device(device)).double()


def counts(base):
    calls={};categories={k:dict(attempted=0,completed=0) for k in LIMITS};path=Path(base)/'optimizer_calls.jsonl'
    if path.exists():
        for line in path.read_text(encoding='utf-8').splitlines():
            r=json.loads(line);cid=r['call_id'];category=r['category']
            if category not in LIMITS:raise ValueError('unknown optimizer category')
            if r['event']=='optimizer_step_attempt' and cid not in calls:
                calls[cid]=dict(r,completed=False);categories[category]['attempted']+=1
            elif r['event']=='optimizer_step_completed' and cid in calls and not calls[cid]['completed']:
                if any(r[k]!=calls[cid][k] for k in ('category','run_id','step')):raise ValueError('optimizer completion identity differs')
                calls[cid]['completed']=True;categories[category]['completed']+=1
            else:raise ValueError('duplicate/unpaired optimizer event')
    total=sum(r['attempted'] for r in categories.values())
    if total>TOTAL or any(r['attempted']>LIMITS[k] for k,r in categories.items()):raise ValueError('1848/category bound exceeded')
    return dict(categories=categories,attempted_total=total,completed_total=sum(r['completed'] for r in categories.values()),
                pending=[r for r in calls.values() if not r['completed']])


def journal(base,event,category,run_id,step,cid):
    if category not in ('formal','qa'):raise ValueError('reserved repair/replay not automatically enabled')
    c=counts(base)
    if event=='optimizer_step_attempt':
        if any(r['run_id']==run_id for r in c['pending']) or c['attempted_total']>=TOTAL or c['categories'][category]['attempted']>=LIMITS[category]:
            raise ValueError('unreconciled run or finite call allowance exhausted')
        if category=='formal' and (run_id not in {f'{v}_{s}' for v in VARIANTS for s in SEEDS} or not 1<=step<=256):
            raise ValueError('unregistered formal identity/update')
    elif event!='optimizer_step_completed':raise ValueError('unregistered optimizer event')
    with (Path(base)/'optimizer_calls.jsonl').open('a',encoding='utf-8') as stream:
        stream.write(json.dumps(dict(event=event,category=category,run_id=run_id,step=step,call_id=cid,time_unix=time.time()),allow_nan=False)+'\n');stream.flush()


def horizon_for(step):
    if type(step) is not int or not 1<=step<=256:raise ValueError('registered step1..256 required')
    return 5. if step%2 else 8.


def make_schedule(events,seed):
    np,_,sgm,_=deps()
    if seed not in SEEDS or len(events)!=297:raise ValueError('registered seed/fit297 required')
    rng=np.random.default_rng(seed);coverage=dict(cover_order=rng.permutation(297).tolist(),cursor=0)
    initial=dict(rng=copy.deepcopy(rng.bit_generator.state),coverage=copy.deepcopy(coverage));rows=[]
    for step in range(1,257):
        _,sampling=sgm.fg.sample_step(events,rng,coverage)
        rows.append(dict(step=step,horizon_s=horizon_for(step),**sampling,
            rng_after=copy.deepcopy(rng.bit_generator.state),coverage_after=copy.deepcopy(coverage)))
    return dict(initial=initial,rows=rows)


def prepare(base):
    admitted(base);_,_,sgm,_=deps();base=Path(base);parent,events=sgm.checked_preparation(SGM,fit_only=True)
    if len(events)!=297 or sum(e['label']=='unstable' for e in events)!=30:raise ValueError('fit297/30 identity changed')
    sgm.write(base/'schedule.json',{str(seed):make_schedule(events,seed) for seed in SEEDS},exclusive=True)
    record={k:parent[k] for k in ('fit_ids','no_update_ids','original366_ids','fresh19_ids','fit_groups','no_update_groups',
        'fit_cache_sha256','no_update_cache_sha256','panel_ids','cost12_ids')}
    record.update(status='COMPLETED',source=source_identity(),code=code_identity(),schedule_sha256=sgm.sha(base/'schedule.json'),
        source_plane_recomputed=False,fixed_mathematical_suite_rerun=False,source_exposed_to_train385=True,
        no_update_payload_read=False,val_read=False,test_read=False)
    sgm.write(base/'prepare.json',record,exclusive=True);return record


def checked_preparation(base,*,fit_only=True):
    admitted(base);_,_,sgm,_=deps();base=Path(base);p=sgm.read(base/'prepare.json')
    if p['source']!=source_identity() or p['code']!=code_identity() or p['schedule_sha256']!=sgm.sha(base/'schedule.json'):
        raise ValueError('source/code/schedule changed')
    old,events=sgm.checked_preparation(SGM,fit_only=fit_only)
    if p['fit_ids']!=old['fit_ids']:raise ValueError('fit identity changed')
    return p,events


def make_optimizer(model):
    return deps()[2].make_optimizer(model)


def restore_checkpoint(output,device,identity):
    """Actual new model + Adam/all RNG restore; no legacy StateGated loader."""
    np,torch,sgm,mt=deps();output=Path(output);pointer=sgm.read(output/'latest.json');name=pointer['directory']
    if Path(name).name!=name:raise ValueError('checkpoint must be immediate child')
    folder=output/name
    if sgm.sha(folder/'checkpoint.json')!=pointer['checkpoint_sha256']:raise ValueError('metadata changed')
    meta=sgm.read(folder/'checkpoint.json');device=sgm.resolve_device(device)
    if meta['identity']!=identity or not meta['resumable'] or meta['runtime']!=sgm.old.runtime_identity(device):raise ValueError('resume identity/runtime differs')
    for name,key in (('model.json','model_sha256'),('training.pt','training_sha256')):
        if sgm.sha(folder/name)!=meta[key]:raise ValueError('checkpoint asset changed')
    model=mt.MemoryTransportROM.from_export(folder/'model.json').to(device).double();opt=make_optimizer(model)
    if model.history_dt!=.0025 or model.variant!=identity['variant'] or sgm.old._parameter_layout(model,opt)!=meta['parameter_layout']:
        raise ValueError('transport variant/history/Adam layout differs')
    state=torch.load(folder/'training.pt',map_location='cpu',weights_only=False)
    if state['step']!=meta['step']:raise ValueError('checkpoint step differs')
    opt.load_state_dict(state['optimizer'])
    if meta['step'] and (not opt.state or any(int(r['step'])!=meta['step'] for r in opt.state.values())):raise ValueError('actual Adam counters differ')
    if not sgm.old._adam_states_finite(opt.state.values()):raise ValueError('nonfinite restored Adam')
    rng=np.random.default_rng();rng.bit_generator.state=state['numpy_rng'];np.random.set_state(state['numpy_global_rng'])
    random.setstate(state['python_rng']);torch.set_rng_state(state['torch_rng'])
    if device.type=='cuda':
        if state['cuda_rng'] is None:raise ValueError('missing actual CUDA RNG')
        torch.cuda.set_rng_state_all(state['cuda_rng'])
    return model,opt,rng,state['coverage'],meta


def one_update(model,opt,store,samples,horizon,*,base,category,run_id,step):
    _,torch,sgm,_=deps();sgm.synchronize(model.mass.device);tick=time.perf_counter();opt.zero_grad(set_to_none=True)
    with sgm.graph(model):
        J,terms=sgm.objective(model,store,samples,horizon,method='rk4')
        if not torch.isfinite(J):raise FloatingPointError('nonfinite J')
        J.backward()
    params=[p for p in model.parameters() if p.requires_grad]
    if any(p.grad is None or not torch.isfinite(p.grad).all() for p in params):raise FloatingPointError('missing/nonfinite gradient')
    norm=torch.nn.utils.clip_grad_norm_(params,100.)
    if not torch.isfinite(norm):raise FloatingPointError('nonfinite gradient norm')
    before={n:p.detach().clone() for n,p in model.named_parameters() if p.requires_grad};cid=uuid.uuid4().hex
    journal(base,'optimizer_step_attempt',category,run_id,step,cid);opt.step()
    journal(base,'optimizer_step_completed',category,run_id,step,cid);model.project_runtime_();sgm.synchronize(model.mass.device)
    if any(not torch.isfinite(p).all() for p in params) or not sgm.old._adam_states_finite(opt.state.values()):raise FloatingPointError('post-step nonfinite')
    return dict(step=step,horizon_s=horizon,method='rk4',**{k:float(v.detach()) for k,v in terms.items()},gradient_norm=float(norm),
        block_updates={n:dict(relative_l2=float((p.detach()-before[n]).norm()/before[n].norm().clamp_min(1.))) for n,p in model.named_parameters() if p.requires_grad},
        seconds=time.perf_counter()-tick)


def optimization(trace):
    recent=trace[-64:];groups={name for r in recent for name in r['block_updates']}
    clips=sum(r['gradient_norm']>100 for r in recent)
    jumps={name:sum(r['block_updates'].get(name,{}).get('relative_l2',0)>.25 for r in recent) for name in groups}
    return dict(optimization_unresolved=len(recent)!=64 or clips>16 or any(n>=16 for n in jumps.values()),
                clip_count_last64=clips,large_change_counts_last64=jumps)


def extension_decision(before,after,trace):
    _,_,sgm,_=deps();opt=optimization(trace)
    if not sgm.valid(before) or not sgm.valid(after):return dict(opt,extend=False,reason='required_numerics_unresolved')
    a,b=before['unstable_actual15_risk'],after['unstable_actual15_risk']
    gain=(after['classification_actual15s']['missed_resolved_unstable']<=before['classification_actual15s']['missed_resolved_unstable']-1
          or b<=.99*a and a-b>before['unstable_actual15_numerical_score_perturbation']+after['unstable_actual15_numerical_score_perturbation'])
    guards=b<=1.05*a
    for key in ('classification_8s','classification_actual15s'):
        guards &= all(after[key][k]<=before[key][k] for k in ('missed_resolved_unstable','false_los_resolved_stable'))
    guards &= all(after['horizon_risks'][h]['value']<=1.05*before['horizon_risks'][h]['value'] for h in ('5','8'))
    guards &= all(after[k]<=1.05*before[k] for k in ('common_frequency_rmse_8s','normalized_p_rmse_8s'))
    return dict(opt,extend=bool(gain and guards and not opt['optimization_unresolved']),registered_actual15_gain=bool(gain),nonworsening_guards=bool(guards))


def timing_comparison(profiles,candidate='S'):
    if not profiles or candidate not in profiles:return dict(passed=False,common_event_ids=[],reason='missing_candidate')
    common=[];ids=set.intersection(*(set(p['timing_actual15']) for p in profiles.values()))
    for eid in sorted(ids):
        rows=[p['timing_actual15'][eid] for p in profiles.values()]
        if all(r['numerically_qualified'] and all(r[k]['time_s'] is not None for k in ('truth','fine','coarse')) for r in rows):common.append(eid)
    if not common:return dict(passed=False,common_event_ids=[],reason='empty_common_T')
    metrics={name:dict(E=sum(abs(p['timing_actual15'][i]['fine']['time_s']-p['timing_actual15'][i]['truth']['time_s']) for i in common)/len(common),
        u=max(abs(p['timing_actual15'][i]['fine']['time_s']-p['timing_actual15'][i]['coarse']['time_s']) for i in common)) for name,p in profiles.items()}
    passed=all(metrics[candidate]['E']<=m['E']+.01+metrics[candidate]['u']+m['u'] for name,m in metrics.items() if name!=candidate)
    return dict(passed=bool(passed),metrics=metrics,common_event_ids=common)


def retention(profiles,optimization_unresolved=False):
    _,_,sgm,_=deps()
    if set(profiles)!={'A','S','P','R_N'} or optimization_unresolved or not all(sgm.valid(p) for p in profiles.values()):
        return dict(second_seed=False,reason='missing_or_unresolved_control',mechanism='UNRESOLVED')
    s=profiles['S'];improvements={};guards=True
    for name in ('A','P'):
        p=profiles[name];fewer=s['classification_actual15s']['missed_resolved_unstable']<=p['classification_actual15s']['missed_resolved_unstable']-1
        risk=s['unstable_actual15_risk']<=.95*p['unstable_actual15_risk'] and p['unstable_actual15_risk']-s['unstable_actual15_risk']>p['unstable_actual15_numerical_score_perturbation']+s['unstable_actual15_numerical_score_perturbation']
        improvements[name]=dict(fewer_misses=bool(fewer),risk_gain=bool(risk),passed=bool(fewer or risk))
    for name in ('A','P','R_N'):
        p=profiles[name]
        guards &= s['classification_actual15s']['missed_resolved_unstable']<=p['classification_actual15s']['missed_resolved_unstable']
        guards &= s['classification_8s']['missed_resolved_unstable']<=p['classification_8s']['missed_resolved_unstable']
        guards &= all(s['horizon_risks'][h]['value']<=1.05*p['horizon_risks'][h]['value'] for h in ('5','8'))
        guards &= all(s[k]<=1.05*p[k] for k in ('common_frequency_rmse_8s','normalized_p_rmse_8s'))
    guards &= s['classification_8s']['missed_resolved_unstable']<=3 # frozen A96 protection
    guards &= all(s[k]['false_los_resolved_stable']==0 for k in ('classification_8s','classification_actual15s'))
    guards &= s['unstable_actual15_risk']<=profiles['R_N']['unstable_actual15_risk']
    timing=timing_comparison(profiles)
    passed=all(r['passed'] for r in improvements.values()) and guards and timing['passed']
    return dict(second_seed=bool(passed),mechanism='RETAINED' if passed else 'NOT_RETAINED',
        improvements=improvements,guards=bool(guards),timing=timing,P_independent_capacity_matched=False)


def efficiency_target(candidate,A96,R_N):
    _,_,sgm,_=deps()
    if not all(sgm.valid(p) for p in (candidate,A96,R_N)):return dict(passed=False,reason='unresolved_profile')
    w=candidate;timing=timing_comparison(dict(candidate=w,A96=A96,R_N=R_N),candidate='candidate')
    guards=w['classification_actual15s']['missed_resolved_unstable']<=21 and w['classification_8s']['missed_resolved_unstable']<=3
    guards &= all(w[k]['false_los_resolved_stable']==0 for k in ('classification_8s','classification_actual15s'))
    guards &= w['unstable_actual15_risk']<=R_N['unstable_actual15_risk']
    guards &= all(w['horizon_risks'][h]['value']<=1.05*R_N['horizon_risks'][h]['value'] for h in ('5','8'))
    guards &= all(w[k]<=1.05*R_N[k] for k in ('common_frequency_rmse_8s','normalized_p_rmse_8s'))
    fewer=w['classification_actual15s']['missed_resolved_unstable']<=20
    gain=w['unstable_actual15_risk']<=.95*A96['unstable_actual15_risk'] and A96['unstable_actual15_risk']-w['unstable_actual15_risk']>A96['unstable_actual15_numerical_score_perturbation']+w['unstable_actual15_numerical_score_perturbation']
    return dict(passed=bool(guards and (fewer or gain) and timing['passed']),fewer_misses=bool(fewer),risk_gain=bool(gain),guards=bool(guards),timing=timing)


def device_comparison(variant):
    np,torch,sgm,_=deps();events=sgm.old.synthetic_fixture();models=[build_model(variant,d) for d in ('cpu','cuda')]
    h=np.stack([e['x'][:11] for e in events]);states=np.c_[h[:,-1],np.full((len(h),10),1e-5)];values=[];gradients=[]
    for m in models:
        tensors=dict(field=m.field(m._tensor(states)),initialize=m.initialize(h),short_RK4=m.rollout(h,4,.01))
        with sgm.graph(m):
            J,_=sgm.objective(m,sgm.EventStore(events,m.mass.device),[(e,10) for e in events],.02,method='rk4');J.backward()
        tensors['J']=J;values.append({k:v.detach().cpu() for k,v in tensors.items()})
        gradients.append({n:p.grad.detach().cpu() for n,p in m.named_parameters() if p.requires_grad and p.grad is not None})
        if set(gradients[-1])!={n for n,p in m.named_parameters() if p.requires_grad}:raise ValueError('all raw gradients required')
        if not all(torch.isfinite(v).all() for v in gradients[-1].values()):raise ValueError('nonfinite gradient')
    for k in values[0]:torch.testing.assert_close(values[0][k],values[1][k],atol=1e-10,rtol=1e-8)
    for k in gradients[0]:torch.testing.assert_close(gradients[0][k],gradients[1][k],atol=1e-10,rtol=1e-8)
    return dict(status='PASS',atol=1e-10,rtol=1e-8,all_raw_gradients=True)


def performance_pair(variant,events,output,device):
    """Exactly once uncached then cached 12x8 full J/backward, no optimizer."""
    _,torch,sgm,mt=deps();output=Path(output);output.mkdir(exist_ok=False)
    class UncachedReference(mt.MemoryTransportROM):
        def _context(self):return self._new_context()
    base=build_model(variant);reference=UncachedReference(base._source_config,variant,u=base.u,v=base.v,source_identity=base.source_identity,plane_identity=base.plane_identity)
    reference.load_state_dict(base.state_dict());models=[('uncached',reference.to(device).double()),('cached',base.to(device).double())]
    values=[];gradients=[];rows=[]
    for name,m in models:
        marker=output/(name+'_attempt.json');sgm.write(marker,dict(variant=variant,implementation=name,optimizer_calls=0),exclusive=True)
        m.zero_grad(set_to_none=True);store=sgm.EventStore(events,m.mass.device)
        torch.cuda.reset_peak_memory_stats(m.mass.device);sgm.synchronize(m.mass.device);tick=time.perf_counter()
        with sgm.graph(m):
            J,_=sgm.objective(m,store,[(e,10) for e in events],8.,method='rk4');J.backward()
        sgm.synchronize(m.mass.device);seconds=time.perf_counter()-tick
        gradients.append({n:p.grad.detach().cpu().clone() for n,p in m.named_parameters() if p.requires_grad and p.grad is not None})
        if not torch.isfinite(J) or set(gradients[-1])!={n for n,p in m.named_parameters() if p.requires_grad} or not all(torch.isfinite(v).all() for v in gradients[-1].values()):
            raise FloatingPointError('finite complete J/all gradients required; no replacement implementation')
        values.append(J.detach().cpu());row=dict(implementation=name,seconds=seconds,peak_cuda_bytes=torch.cuda.max_memory_allocated(m.mass.device),J=float(J),all_raw_gradients=True,
            gradients={key:value.tolist() for key,value in gradients[-1].items()})
        rows.append(row);sgm.write(output/(name+'_result.json'),row,exclusive=True)
    torch.testing.assert_close(values[0],values[1],atol=1e-10,rtol=1e-8)
    for k in gradients[0]:torch.testing.assert_close(gradients[0][k],gradients[1][k],atol=1e-10,rtol=1e-8)
    result=dict(status='PASS',variant=variant,ordered_runs=rows,equivalent=True,atol=1e-10,rtol=1e-8,optimizer_calls=0,
        event_ids=[e['id'] for e in events],horizon_s=8.,method='rk4',dtype='float64',
        interpretation='one fixed uncached-then-cached measurement; no general speed claim')
    sgm.write(output/'summary.json',result,exclusive=True);return result


def qualify(base,device='cuda'):
    admitted(base);_,_,sgm,_=deps();base=Path(base);output=base/'qualification';output.mkdir(exist_ok=False)
    dev=sgm.resolve_device(device)
    if dev.type!='cuda':raise ValueError('actual CUDA required')
    prepared,fit=checked_preparation(base);selected=[e for eid in prepared['cost12_ids'] for e in fit if e['id']==eid]
    if len(selected)!=12 or any(e['label']!='stable' or len(e['x'])-11<800 for e in selected):raise ValueError('fixed twelve complete8 stable events required')
    for path,digest in EVIDENCE_SHA.items():
        if sgm.sha(SGM/path)!=digest:raise ValueError('frozen reusable evidence changed')
    result=dict(status='COMPLETED',per_variant={},eligible_variants=[],R_N_profile=sgm.read(SGM/'qualification/summary.json')['R_N_profile'],
        A96_profile=sgm.read(SGM/'study/A_28311/fit000096/state_gated_profile.json'),training_method='rk4',history_dt_main=.0025,history_dt_reference=.00125,val_read=False,test_read=False)
    for variant in VARIANTS:
        tick=time.perf_counter();row=dict(training_admission=False)
        try:
            row['device_comparison']=device_comparison(variant);row['cpu_cuda_pair_pass']=True
            performance_path=output/(variant+'_performance')/'summary.json'
            perf=performance_pair(variant,selected,performance_path.parent,dev);row['performance_equivalent']=perf['equivalent']
            row['performance_evidence']=dict(path=str(performance_path.resolve()),sha256=sgm.sha(performance_path))
            current_model=build_model(variant)
            evidence=sgm.evaluate(current_model,fit,output/f'{variant}_zero_fit297',prepared)
            if variant=='A':
                old=sgm.read(SGM/'study/A_28311/fit000096/evidence.json');a=sgm._fine_cache(old,fit,prepared['fit_ids']);b=sgm._fine_cache(evidence,fit,prepared['fit_ids']);lookup={e['id']:e for e in fit}
                rows=[sgm._cross_event(current_model,lookup[eid],a[eid],b[eid]) for eid in prepared['fit_ids']]
                cross=dict(qualified=len(rows)==297 and all(r['qualified'] for r in rows),record_count=len(rows),event_ids=list(prepared['fit_ids']),
                    scope='A96_A0_fit297_bidirectional_RK4_fine005_base01_8s_actual15',bidirectional=True,history_rows=11,initial_observation_index=10,fine_dt=.005,output_dt=.01,coarse_stride=2,
                    original_schema_aliases=dict(R_N='frozen_A96',A='current_A0'),rows=rows,source_A96_prediction_recomputed=False,source_evidence_sha256=EVIDENCE_SHA['study/A_28311/fit000096/evidence.json'])
                path=output/'A_A96_compatibility.json';sgm.write(path,cross,exclusive=True)
                row.update(A_A96_compatibility_qualified=cross['qualified'],A_A96_compatibility=dict(path=str(path.resolve()),sha256=sgm.sha(path)))
                if not cross['qualified']:raise ValueError('A0/A96 original compatibility failed')
            row.update(cpu_qualified=True,finite_12x8_backward=True,fullfit297_qualified=sgm.valid(evidence['decision_profile']),zero_profile=evidence['decision_profile'])
            row['training_admission']=row['fullfit297_qualified'] and row['performance_equivalent']
            if row['training_admission']:result['eligible_variants'].append(variant)
        except Exception as exc:row.update(error=repr(exc),failure_is_physical_instability=False)
        row['qualification_seconds']=time.perf_counter()-tick;result['per_variant'][variant]=row
        sgm.write(output/(variant+'_result.json'),row,exclusive=True)
    result['training_admission']=bool(result['eligible_variants']);result['full_matched_comparison_qualified']=result['eligible_variants']==list(VARIANTS)
    result['actual_performance_attempts']=len(list(output.glob('*_performance/*_attempt.json')))
    result['registered_performance_attempts_upper_bound']=6
    sgm.write(output/'summary.json',result,exclusive=True);return result


def qa(base,device='cuda'):
    admitted(base);np,torch,sgm,_=deps();base=Path(base);output=base/'qa';output.mkdir(exist_ok=False)
    eligible=sgm.read(base/'qualification/summary.json')['eligible_variants'];results=[]
    for variant in eligible:
        tick=time.perf_counter();folder=output/variant;folder.mkdir();events=sgm.old.synthetic_fixture();store=sgm.EventStore(events,device)
        identity=dict(variant=variant,fixture_only=True,source=source_identity(),code=code_identity())
        def initial():
            m=build_model(variant,device);opt=make_optimizer(m);sgm.fg._seed(SEEDS[0],m.mass.device);rng=np.random.default_rng(SEEDS[0])
            return m,opt,rng,dict(cover_order=rng.permutation(len(events)).tolist(),cursor=0)
        def advance(m,opt,rng,cov,start,end,label):
            trace=[]
            for step in range(start+1,end+1):
                samples,_=sgm.fg.sample_step(events,rng,cov);random.random();np.random.random();torch.rand((),device=m.mass.device)
                trace.append(one_update(m,opt,store,samples,.02,base=base,category='qa',run_id=f'qa_{variant}_{label}',step=step))
            return trace
        current=None;trace=[]
        try:
            a,oa,ra,ca=initial();current=(a,oa,ra,ca);trace=advance(a,oa,ra,ca,0,18,'continuous');final_rng=sgm.old._cpu_snapshot(sgm.fg._rng_state(ra,ca))
            b,ob,rb,cb=initial();current=(b,ob,rb,cb);trace=advance(b,ob,rb,cb,0,16,'restored')
            snapshot=sgm.old._cpu_snapshot(dict(model=b.state_dict(),Adam=ob.state_dict(),RNG=sgm.fg._rng_state(rb,cb),layout=sgm.old._parameter_layout(b,ob)))
            if not any(bool(torch.count_nonzero(s['exp_avg'])) for s in ob.state.values()):raise ValueError('actual nonzero Adam moments required')
            sgm.save_checkpoint(folder/'checkpoints',b,ob,rb,cb,16,identity,trace)
            c,oc,rc,cc,_=restore_checkpoint(folder/'checkpoints',device,identity);current=(c,oc,rc,cc)
            sgm.old._exact_nested(snapshot,sgm.old._cpu_snapshot(dict(model=c.state_dict(),Adam=oc.state_dict(),RNG=sgm.fg._rng_state(rc,cc),layout=sgm.old._parameter_layout(c,oc))))
            trace+=advance(c,oc,rc,cc,16,18,'restored')
            pe=sgm.old._qa_compare(sgm.old._cpu_snapshot(a.state_dict()),sgm.old._cpu_snapshot(c.state_dict()))
            sa,sc=sgm.old._cpu_snapshot(oa.state_dict()),sgm.old._cpu_snapshot(oc.state_dict());sgm.old._exact_nested(sa['param_groups'],sc['param_groups'])
            ae=max(sgm.old._qa_compare(sa['state'][k],sc['state'][k]) for k in sa['state']);sgm.old._exact_nested(final_rng,sgm.old._cpu_snapshot(sgm.fg._rng_state(rc,cc)))
            row=dict(variant=variant,status='PASS',actual_optimizer_calls=36,parameter_maxabs=pe,Adam_maxabs=ae,restore_atol=2e-9,restore_rtol=2e-9)
        except BaseException as exc:
            quarantine=None if current is None else sgm.old.quarantine_failure(folder,*current[:3],current[3],len(trace),trace,exc)
            row=dict(variant=variant,status='NUMERICAL_UNRESOLVED',error=repr(exc),quarantine=quarantine,failure_is_physical_instability=False)
            if not isinstance(exc,Exception):sgm.write(folder/'result.json',row,exclusive=True);raise
        row['seconds']=time.perf_counter()-tick;results.append(row);sgm.write(folder/'result.json',row,exclusive=True)
    records=[];ledger=base/'optimizer_calls.jsonl'
    if ledger.exists():records=[json.loads(line) for line in ledger.read_text(encoding='utf-8').splitlines()]
    path=output/'optimizer_projection.json';sgm.write(path,[r for r in records if r['category']=='qa'],exclusive=True)
    passed=[r['variant'] for r in results if r['status']=='PASS']
    result=dict(status='COMPLETED',models=results,eligible_variants=passed,training_admission=bool(passed),calls=counts(base),
        qa_journal_projection=dict(path=str(path.resolve()),sha256=sgm.sha(path)),val_read=False,test_read=False)
    sgm.write(output/'summary.json',result,exclusive=True);return result


def formal_run(base,variant,seed,prepared,fit,panel,device,prior_candidate_seconds=0.):
    np,_,sgm,_=deps();base=Path(base);run_id=f'{variant}_{seed}';folder=base/'study'/run_id;folder.mkdir(exist_ok=False)
    m=build_model(variant,device);opt=make_optimizer(m);sgm.fg._seed(seed,m.mass.device)
    schedule=sgm.read(base/'schedule.json')[str(seed)];rng=np.random.default_rng();rng.bit_generator.state=schedule['initial']['rng'];cov=copy.deepcopy(schedule['initial']['coverage'])
    identity=dict(variant=variant,seed=seed,source=prepared['source'],code=prepared['code'],schedule_sha256=prepared['schedule_sha256'])
    store=sgm.EventStore(fit,m.mass.device);lookup={e['id']:e for e in fit};trace=[];panels={};full={};cost=[];started=time.perf_counter()
    qualified=sgm.read(base/'qualification/summary.json')['per_variant'][variant];qa_row=next(r for r in sgm.read(base/'qa/summary.json')['models'] if r['variant']==variant)
    prelude=qualified['qualification_seconds']+qa_row['seconds']+prior_candidate_seconds
    try:
        panels['0']=sgm.score_panel(m,panel);sgm.write(folder/'panel000000.json',panels['0'],exclusive=True)
        if not sgm.valid(panels['0']):raise ValueError('initial panel unresolved')
        sgm.save_checkpoint(folder/'checkpoints',m,opt,rng,cov,0,identity,trace);end=128
        full['0']=sgm.read(base/f'qualification/{variant}_zero_fit297/state_gated_profile.json')
        cost.append(dict(step=0,run_cumulative_seconds=time.perf_counter()-started,candidate_cumulative_seconds=prelude+time.perf_counter()-started))
        for row in schedule['rows']:
            step=row['step']
            if step>end:break
            trace.append(one_update(m,opt,store,[(lookup[eid],10) for eid in row['sample_ids']],row['horizon_s'],base=base,category='formal',run_id=run_id,step=step))
            rng.bit_generator.state=row['rng_after'];cov=copy.deepcopy(row['coverage_after'])
            if step%16==0:sgm.save_checkpoint(folder/'checkpoints',m,opt,rng,cov,step,identity,trace)
            if step in (64,128,256):panels[str(step)]=sgm.score_panel(m,panel);sgm.write(folder/f'panel{step:06d}.json',panels[str(step)],exclusive=True)
            if step==128:
                full[str(step)]=sgm.evaluate(m,fit,folder/f'fit{step:06d}',prepared)['decision_profile']
                cost.append(dict(step=step,run_cumulative_seconds=time.perf_counter()-started,candidate_cumulative_seconds=prelude+time.perf_counter()-started))
            if step==128:
                decision=extension_decision(panels['64'],panels['128'],trace);sgm.write(folder/'extension_decision.json',decision,exclusive=True)
                if decision['extend']:end=256
        result=dict(status='COMPLETED',run_id=run_id,variant=variant,seed=seed,updates=len(trace),endpoint=str(folder/'checkpoints'/f'step{len(trace):06d}'),
            fit_endpoints=full,cost_milestones=cost,run_total_seconds=time.perf_counter()-started,candidate_prelude_seconds=prelude,extension_decision=decision,
            optimization_unresolved_at128=optimization(trace[:128])['optimization_unresolved'],
            upper_bound_still_improving=extension_decision(panels['128'],panels['256'],trace)['extend'] if len(trace)==256 else False,
            endpoint_extra_parameter=float(m.s.detach()) if variant=='S' else float(m.d.detach()) if variant=='P' else 0.,finite_optimization_scope=True)
    except BaseException as exc:
        quarantine=sgm.old.quarantine_failure(folder,m,opt,rng,cov,len(trace),trace,exc)
        result=dict(status='NUMERICAL_UNRESOLVED',run_id=run_id,variant=variant,seed=seed,updates=len(trace),error=repr(exc),quarantine=quarantine,failure_is_physical_instability=False,resumable=False)
        sgm.write(folder/'result.json',result,exclusive=True)
        if not isinstance(exc,Exception):raise
        return result
    sgm.write(folder/'result.json',result,exclusive=True);return result


def study(base,device='cuda'):
    admitted(base);_,_,sgm,mt=deps();base=Path(base);output=base/'study';output.mkdir(exist_ok=False)
    eligible=sgm.read(base/'qa/summary.json')['eligible_variants'];prepared,fit=checked_preparation(base);panel=[e for eid in prepared['panel_ids'] for e in fit if e['id']==eid]
    reference=sgm.read(base/'qualification/summary.json');runs=[];decisions=[]
    for seed in SEEDS:
        group=[]
        for variant in eligible:
            prior=sum(r.get('run_total_seconds',0.) for r in runs if r['variant']==variant)
            r=formal_run(base,variant,seed,prepared,fit,panel,device,prior);runs.append(r);group.append(r)
        # A full-fit256 is registered only when the complete comparison reaches
        # that milestone. Individual extenders retain panels/checkpoints without
        # adding an unpaired full-fit evaluation.
        if {r['variant'] for r in group if r['status']=='COMPLETED' and r['updates']==256}==set(VARIANTS):
            for r in group:
                tick=time.perf_counter();folder=output/r['run_id']
                endpoint=mt.MemoryTransportROM.from_export(Path(r['endpoint'])/'model.json')
                evidence=sgm.evaluate(endpoint,fit,folder/'fit000256',prepared)
                seconds=time.perf_counter()-tick;r['fit_endpoints']['256']=evidence['decision_profile'];r['run_total_seconds']+=seconds
                point=dict(step=256,run_cumulative_seconds=r['run_total_seconds'],candidate_cumulative_seconds=r['candidate_prelude_seconds']+r['run_total_seconds'])
                r['cost_milestones'].append(point)
                sgm.write(folder/'common256_fit_link.json',dict(profile_sha256=sgm.sha(folder/'fit000256/state_gated_profile.json'),
                    additional_fullfit_evaluation_seconds=seconds,cost=point,all_three_reached256=True),exclusive=True)
        profiles={r['variant']:r['fit_endpoints']['128'] for r in group if r['status']=='COMPLETED'};profiles['R_N']=reference['R_N_profile']
        decision=retention(profiles,optimization_unresolved=any(r.get('optimization_unresolved_at128',True) for r in group))
        decisions.append(dict(seed=seed,**decision));sgm.write(output/f'retention_{seed}.json',decisions[-1],exclusive=True)
        if not decision['second_seed']:break
    sgm.write(output/'fit_decisions_closed.json',dict(decisions=decisions,run_ids=[r['run_id'] for r in runs]),exclusive=True)
    efficiency=[]
    for r in runs:
        if r['status']!='COMPLETED':continue
        group=[x for x in runs if x['seed']==r['seed'] and x['status']=='COMPLETED'];common=set.intersection(*(set(x['fit_endpoints']) for x in group)) if len(group)==3 else set()
        milestones=[dict(c,target=efficiency_target(r['fit_endpoints'][str(c['step'])],reference['A96_profile'],reference['R_N_profile'])) for c in r['cost_milestones'] if str(c['step']) in common]
        first=next((v for v in milestones if v['target']['passed']),None)
        efficiency.append(dict(run_id=r['run_id'],milestones=milestones,first_attained=first,right_censored=first is None))
    sgm.write(output/'efficiency.json',dict(runs=efficiency,same_cost_winner_claimed=False,no_interpolation=True,
        scope='candidate qualification/QA/training/evaluation wall time; common preparation/plane/math and native subtree separate'),exclusive=True)
    reports=[]
    if any(r['status']=='COMPLETED' for r in runs):
        _,all_events=checked_preparation(base,fit_only=False);held=[e for e in all_events if e['id'] in prepared['no_update_ids']]
        for r in runs:
            if r['status']!='COMPLETED':continue
            m=mt.MemoryTransportROM.from_export(Path(r['endpoint'])/'model.json');folder=output/r['run_id']
            sgm.evaluate(m,all_events,folder/'evaluation385',prepared);sgm.evaluate(m,held,folder/'noise88',prepared,noisy=True)
            reports.append(dict(run_id=r['run_id'],evaluation385_report_only=True,noise88_report_only=True))
    result=dict(status='COMPLETED' if len(runs)>=3 and all(r['status']=='COMPLETED' for r in runs) else 'UNRESOLVED',runs=runs,
        decisions=decisions,reports=reports,calls=counts(base),default_model=None,scientific_model_adoption=False,val_read=False,test_read=False)
    sgm.write(output/'summary.json',result,exclusive=True);return result


def summary(base):
    admitted(base);_,_,sgm,_=deps();base=Path(base)
    result=dict(status=sgm.read(base/'study/summary.json')['status'],calls=counts(base),model_adopted=False,val_read=False,test_read=False,no_process_quotas=True)
    sgm.write(base/'summary.json',result,exclusive=True);return result
