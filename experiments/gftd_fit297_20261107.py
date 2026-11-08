"""Single, release-bound train fit297 Run3; import performs no payload reads."""
from __future__ import annotations
import copy
import hashlib
import io
import json
import math
import os
from pathlib import Path
import platform
import re
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'src'), str(ROOT/'experiments')]
import numpy as np
import scipy
from kiki.models.base import RunContext
from kiki.models.gftd_kuramoto import GFTDKuramotoModel
from kiki.models import gftd_fitting as engine, gftd_discovery as core
from kiki.physics.kuramoto_functional import GeneratingFunctionalKuramoto, StructuralGates
import gftd_fit297_l1_20261107 as l1

OUTPUT = ROOT/'results/gftd_fit297_20261107'
PRIVATE = ROOT/'private_data/gftd_fit297_20261107'
PROTOCOL = 'reports/gftd_fit297_20261107_protocol.md'
PROTOCOL_SHA256 = 'ca22743c53fe4fe398960e111838a373f8c4686eab0974c5daf6d9c28435062f'
REVIEW = 'reports/gftd_fit297_20261107_design_review.md'
REVIEW_SHA256 = 'e9cf79e5f6a82101e811ac722cfb4192066378c6fa90864c9fe9cbb4dd864b6a'
PARENT = '66f1308c25ee38c3d7217bd86ee43b61a042a5b6'
RN_EXPORT = 'private_data/force_graph_20261028/study01/study/N6101_M_28102/checkpoints/step000192/model.json'
RN_SHA256 = '937f00b740875913923487b8d772f40d2a8145a88eba2cdaf61f2d889715bfc9'
MODES = ('static', 'modulated')
REQUIRED_SOURCES = (
    'experiments/gftd_fit297_20261107.py', 'experiments/gftd_fit297_l1_20261107.py',
    'experiments/gftd_fit297_analysis_20261107.py', 'src/kiki/models/base.py',
    'src/kiki/models/gftd_kuramoto.py', 'src/kiki/models/gftd_fit297.py',
    'src/kiki/models/gftd_fitting.py', 'src/kiki/models/gftd_discovery.py',
    'src/kiki/physics/kuramoto_functional.py', 'src/kiki/physics/graph.py',
    'experiments/closure_probe_20261030.py', 'experiments/state_gated_memory_study_20261028.py',
    'experiments/force_graph_study_20261028.py', 'experiments/r_memory_study_20261025.py',
    'experiments/force_graph_model_20261028.py', 'experiments/evaluate_structured_rom.py',
    'experiments/evaluate_long_latent.py', 'experiments/evaluate_l1_latent.py',
    'experiments/frequency_dissipation_20261021.py', 'src/kiki/data/private_dataset.py')


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def safe(value):
    if isinstance(value, dict): return {str(k): safe(v) for k,v in value.items()}
    if isinstance(value, (list,tuple)): return [safe(v) for v in value]
    if isinstance(value, np.ndarray): return safe(value.tolist())
    if isinstance(value, (np.bool_,)): return bool(value)
    if isinstance(value, (np.integer,)): return int(value)
    if isinstance(value, (float,np.floating)): return float(value) if np.isfinite(value) else None
    return value


def atomic_bytes(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name+'.tmp')
    with temp.open('xb') as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temp, path)
    return dict(path=str(path.relative_to(ROOT)), sha256=digest(path), bytes=len(data))


def write_json(path, data):
    return atomic_bytes(path, (json.dumps(safe(data),ensure_ascii=False,indent=2,allow_nan=False)+'\n').encode('utf-8'))


def write_npz(path, **arrays):
    stream=io.BytesIO()
    np.savez_compressed(stream, **arrays)
    return atomic_bytes(path, stream.getvalue())


def environment():
    import torch
    return dict(python_executable=str(Path(sys.executable).resolve()), python_version=platform.python_version(),
                numpy=np.__version__, scipy=scipy.__version__, torch=torch.__version__, device='cpu', dtype='float64')


def dependency_sources():
    """Static imports only; root can use this to construct the source freeze."""
    import closure_probe_20261030 as loader
    import gftd_fit297_analysis_20261107
    from kiki.models import gftd_fit297
    from kiki.data import private_dataset
    loader._deps()
    files=set(REQUIRED_SOURCES)
    for module in list(sys.modules.values()):
        file=getattr(module,'__file__',None)
        if not file: continue
        path=Path(file).resolve()
        if path.suffix=='.py' and path.is_relative_to(ROOT):
            name=path.relative_to(ROOT).as_posix()
            if name.startswith(('src/kiki/','experiments/')): files.add(name)
    return {name:digest(ROOT/name) for name in sorted(files)}


def required_asset_paths():
    import closure_probe_20261030 as loader
    import state_gated_memory_study_20261028 as sgm
    fg=sgm.fg
    old=fg.old
    paths={RN_EXPORT, 'results/gftd_fit297_20261107/static_identity_audit.json',
        'private_data/gftd_fit297_20261107/reference/ieee39_full.xlsx',
        'private_data/task2_fullorder_descriptor_0035/descriptor.npz',
        'private_data/task2_fullorder_descriptor_0035/evidence.json',
        'private_data/task2_verified_modes_0035b/evidence.json'}
    for base in (loader.SGM_STUDY,sgm.PARENT,fg.OLD_BASE):
        for name in ('prepare.json','schedule.json'): paths.add((base/name).relative_to(ROOT).as_posix())
    paths.add((fg.OLD_BASE/'fit_events.npz').relative_to(ROOT).as_posix())
    paths.add(old.SOURCE.relative_to(ROOT).as_posix())
    for folder in (*fg.SOURCES.values(),sgm.SOURCE):
        for name in ('model.json','checkpoint.json','training.pt'):
            paths.add((folder/name).relative_to(ROOT).as_posix())
    return sorted(paths)


def verify_release(release, *, current_environment=None):
    if release.get('released') is not True: raise PermissionError('root scientific release required')
    if not isinstance(release.get('action_id'),str) or not re.fullmatch(r'[A-Za-z0-9_.-]+',release['action_id']):
        raise ValueError('one explicit safe action_id required')
    for key,value in (('parent_commit',PARENT),('protocol_sha256',PROTOCOL_SHA256),('design_review_sha256',REVIEW_SHA256)):
        if release.get(key)!=value: raise ValueError('release binding mismatch: '+key)
    if digest(ROOT/PROTOCOL)!=PROTOCOL_SHA256 or digest(ROOT/REVIEW)!=REVIEW_SHA256:
        raise ValueError('frozen design changed')
    if release.get('environment') != (environment() if current_environment is None else current_environment):
        raise ValueError('frozen runtime environment changed')
    for field in ('source_sha256','asset_sha256'):
        mapping=release.get(field)
        if not isinstance(mapping,dict) or not mapping: raise ValueError('missing '+field)
        for name,sha in mapping.items():
            path=(ROOT/name).resolve()
            if not path.is_relative_to(ROOT) or not re.fullmatch('[0-9a-f]{64}',sha):
                raise ValueError('invalid frozen path/digest')
            # Payload bytes are first opened after the exclusive job claim.
            if Path(name).name!='fit_events.npz' and digest(path)!=sha: raise ValueError('frozen bytes changed: '+name)
    if not set(required_asset_paths())<=set(release['asset_sha256']): raise ValueError('release omits mandatory static/data assets')
    if not set(REQUIRED_SOURCES)<=set(release['source_sha256']): raise ValueError('release omits execution sources')
    if release['asset_sha256'].get(RN_EXPORT)!=RN_SHA256: raise ValueError('frozen R_N missing')
    if release['asset_sha256'].get('results/gftd_fit297_20261107/static_identity_audit.json')!=digest(OUTPUT/'static_identity_audit.json'):
        raise ValueError('static identity audit missing from release')
    for field,name in (('parent_archive_sha256','parent_archive_verified.json'),('design_adoption_sha256','design_adoption.json')):
        if release.get(field)!=digest(OUTPUT/name): raise ValueError('missing parent/adoption receipt binding')
    adoption=read_json(OUTPUT/'design_adoption.json')
    archive=read_json(OUTPUT/'parent_archive_verified.json')
    if adoption['parent_commit']!=PARENT or archive['local_head']!=PARENT or archive['remote_head']!=PARENT:
        raise ValueError('parent archive mismatch')
    actual_sources=dependency_sources()
    if any(release['source_sha256'].get(name)!=sha for name,sha in actual_sources.items()):
        raise ValueError('release omits an imported project dependency')
    return True


def adaptation_indices(n):
    if type(n) is not int or n<=0: raise ValueError('positive legal observation count required')
    m=min(24,n)
    indices=[j*n//m for j in range(m)]
    assert len(set(indices))==m and all(0<=i<n for i in indices)
    return indices


def prepare_packets(events, registered_ids, delta_star, *, expected_delta_sha256=None):
    ids=[str(e['id']) for e in events]
    if len(ids)!=297 or len(set(ids))!=297 or set(ids)!={str(i) for i in registered_ids}:
        raise ValueError('fit297 identity changed')
    if sum(e['label']=='stable' for e in events)!=267 or sum(e['label']=='unstable' for e in events)!=30:
        raise ValueError('original 267/30 denominator changed')
    by_id={str(e['id']):e for e in events}
    packets,legal,invalid=[],[],[]
    for eid in sorted(ids):
        event=by_id[eid]
        try:
            x=np.asarray(event['x'])
            if x.dtype!=np.float64 or x.ndim!=2 or x.shape[1]!=30 or len(x)<11:
                raise ValueError('canonical float64 [time,30], eleven history points required')
            meta=event['classification_meta']
            raw_delta=np.asarray(meta['equilibrium_delta'],dtype=np.float64)
            if expected_delta_sha256 is not None and hashlib.sha256(raw_delta.tobytes()).hexdigest()!=expected_delta_sha256:
                raise ValueError('classification equilibrium differs from the bound static source')
            shift=raw_delta-delta_star
            if shift.shape!=(10,) or not np.isfinite(shift).all() or np.max(np.abs(shift-shift.mean()))>1e-12:
                raise ValueError('registered equilibrium differs beyond the common angle gauge')
            if float(meta.get('omega_s',2*np.pi*60))!=float(2*np.pi*60):
                raise ValueError('canonical omega units differ')
            packet=dict(theta_history=x[:11,:10].copy()+delta_star,
                        omega_history=x[:11,10:20].copy(),p0=x[10,20:30].copy(),
                        history_times=np.linspace(-.1,0.,11),sigma=0.)
            core.validate_observation({('p_end' if k=='p0' else k):v for k,v in packet.items()},10)
            packets.append(packet); legal.append(eid)
        except (ValueError,KeyError,TypeError) as error:
            invalid.append(dict(event_id=eid,type=type(error).__name__,message=str(error)))
    indices=adaptation_indices(len(legal)) if legal else []
    receipt=dict(ordered_all_ids=sorted(ids),legal_ids=legal,invalid=invalid,adaptation_indices=indices,
        adaptation_ids=[legal[i] for i in indices],denominators=l1.DENOMINATORS.copy())
    return by_id,packets,receipt


def frozen_model():
    """Read only the registered frozen export; this does not fit or integrate."""
    if digest(ROOT/RN_EXPORT)!=RN_SHA256: raise ValueError('R_N export changed')
    payload=read_json(ROOT/RN_EXPORT)
    if payload.get('format')!='force_graph_model_20261028' or payload.get('mode')!='M':
        raise ValueError('registered R_N JSON format required')
    state=payload['state_dict']
    def a(name):
        value=np.asarray(state[name],dtype=np.float64)
        shape=(45,) if name=='reference_coupling45' else (10,)
        if value.shape!=shape or not np.isfinite(value).all(): raise ValueError('invalid frozen buffer: '+name)
        return value.copy()
    mass,delta=a('mass'),a('delta_star')
    kappa=float(state['kappa_reference'])*math.exp(float(state['kappa_raw']))
    lam=float(state['lambda_reference'])*math.exp(float(state['lambda_raw']))
    positive=a('reference_coupling45');positive=positive[positive>0]
    if len(positive)!=26 or kappa<=0 or lam<=0: raise ValueError('R_N constants changed')
    k_ref=float(np.median(positive))
    model=GFTDKuramotoModel(n_nodes=10,mass=mass,damping=a('damping'))
    model.delta_star=delta
    model.functional=GeneratingFunctionalKuramoto(mass=mass,damping=a('damping'),k_edge=np.full(45,k_ref),
        eta_edge=model.incidence.T@delta,gamma_amp=np.full(10,.5),lambda_amp=np.ones(10),
        lambda_mem=np.full(10,1/kappa),tau_amp=np.full(10,5.),tau_mem_sync=np.full(10,1/(kappa*lam)),
        tau_mem_coi=1.,j_omega_mem=np.full(10,kappa),storage_p=a('storage_base'),gain_p=a('jpi_p'),
        rate_p=a('rpp'),gates=StructuralGates(np.ones(45),np.zeros(45),np.ones(10),0.))
    model.gates=model.functional.gates
    scales={key:a(key).tolist() for key in ('input_scale_delta','input_scale_omega','rstar','p_scale')}
    return model,scales,dict(k_ref=k_ref,kappa=kappa,lambda_=lam,source_sha256=RN_SHA256,
                            physical_mapping='jpi_p=G,rpp=R_p,storage_base=S_p,Q=lambda I')


def integrate(functional, state, step, counts):
    n=int(round(15./step))
    path=np.full((n+1,50),np.nan,dtype=np.float64)
    path[0]=state
    failure=None
    with np.errstate(over='raise',invalid='raise',divide='raise'):
        for i in range(n):
            counts['rk4_attempted']+=1
            try:
                path[i+1]=core.rk4(functional,path[i],step)
                if not np.isfinite(path[i+1]).all(): raise FloatingPointError('nonfinite autonomous state')
                counts['rk4_completed']+=1
            except (FloatingPointError,ValueError,OverflowError) as error:
                failure=dict(stage='future_integration',type=type(error).__name__,message=str(error),
                             step=float(step),first_failed_step=i+1)
                break
    return path,failure


def execution_budget(counts):
    totals={}
    for row in (*counts['fit'].values(),counts['initialization'],*counts.get('coverage',{}).values()):
        for key,value in row.items():
            if isinstance(value,(int,float)) and not isinstance(value,bool): totals[key]=totals.get(key,0)+value
    totals['rk4_steps_attempted']=totals.get('rk4_steps_attempted',0)+counts['rk4_attempted']
    totals['rk4_steps_completed']=totals.get('rk4_steps_completed',0)+counts['rk4_completed']
    limits=dict(rk4_steps_attempted=49229640,actual_residual_calls=1146000,optimization_tasks=1572,
                algebra_terms_actual=57337,nnls_attempts=57337,reconstructions=594)
    exceeded={k:dict(actual=totals.get(k,0),limit=v) for k,v in limits.items() if totals.get(k,0)>v}
    return dict(actual=totals,limits=limits,exceeded=exceeded,within_limits=not exceeded,
        note='information_history_steps is a tagged subset of rk4_steps_attempted, never added twice')


def model_analysis_budget_checks(results, modes):
    checks=[]
    def check(name,actual,limit):
        checks.append(dict(name=name,actual=actual,limit=limit,within_limit=None if actual is None else actual<=limit))
    for mode in modes:
        item=results[mode]
        check(mode+'_L2_radii',item['l2_radius_candidates_actual'],13)
        check(mode+'_L2_point_calls',item['l2_point_calls_actual'],297)
        for key,limit in (('newton_iterations',20),('gradient_calls',1000),('eig_count',2),('rhs_calls',217)):
            check(mode+'_ROM_'+key,(item['rom_counts'] or {}).get(key),limit)
    for key,actual in results.get('kron_counts',{}).items(): check('Kron_'+key,actual,1)
    if modes: check('FOM_QZ',results.get('fom_qz_actual'),2)
    return checks


def run(release_path=None):
    release_path=Path(release_path or OUTPUT/'execution_release.json')
    release=read_json(release_path)
    verify_release(release)
    from kiki.models import gftd_fit297 as fit_engine
    if any(getattr(fit_engine,key,{}) for key in ('_CACHE','_PREPARATION_FAILURES','_LEDGERS')):
        raise RuntimeError('fresh process required')
    OUTPUT.mkdir(parents=True,exist_ok=True)
    PRIVATE.mkdir(parents=True,exist_ok=True)
    claim=dict(action_id=release['action_id'],release_sha256=digest(release_path),status='RUNNING',
               parent_commit=PARENT,protocol_sha256=PROTOCOL_SHA256,started_unix=time.time(),
               environment=environment(),source_sha256=release['source_sha256'])
    # An existing claim is an unknown/completed job, never an automatic resume.
    with (OUTPUT/'started.json').open('x',encoding='utf-8',newline='\n') as stream:
        json.dump(claim,stream,indent=2,allow_nan=False);stream.flush();os.fsync(stream.fileno())
    counts=dict(fits_attempted=0,fits_completed=0,initializations_attempted=0,initializations_completed=0,
        future_pairs_attempted=0,future_pairs_persisted=0,future_pairs_completed=0,future_pairs_failed=0,rk4_attempted=0,rk4_completed=0,
        scoring_attempted=0,scoring_completed=0,fit={},initialization={},coverage={})
    receipt=dict(claim,status='RUNNING',counts=counts,current=None,event_payload_read=False,
        denominators=l1.DENOMINATORS.copy(),artifacts=[],costs={},analysis={},automatic_resume=False)
    begin=time.perf_counter()
    def persist(): write_json(OUTPUT/'execution.partial.json',receipt)
    def at(stage,mode=None,event_id=None):
        receipt['current']=dict(stage=stage,mode=mode,event_id=event_id);persist()
    try:
        at('verify_payload_binding')
        for name,sha in release['asset_sha256'].items():
            if Path(name).name=='fit_events.npz' and digest(ROOT/name)!=sha: raise ValueError('fit cache bytes changed')
        at('load_frozen_constants')
        template,scales,constants=frozen_model()
        receipt['artifacts'].append(write_json(PRIVATE/'constants.json',dict(constants=constants,scales=scales)))
        import closure_probe_20261030 as loader
        at('read_train_fit297')
        receipt['event_payload_read']=True;persist()
        preparation,events=loader.load_frozen_events()
        audit=read_json(OUTPUT/'static_identity_audit.json')
        expected_delta=audit['shards'][0]['equilibrium_arrays']['delta']['sha256']
        by_id,packets,input_receipt=prepare_packets(events,preparation['fit_ids'],template.delta_star,expected_delta_sha256=expected_delta)
        receipt['artifacts'].append(write_json(PRIVATE/'inputs.json',dict(packets=packets,receipt=input_receipt)))
        receipt['artifacts'].append(write_json(PRIVATE/'preparation.json',preparation))
        receipt['input_identity']=dict(legal_count=len(packets),invalid=input_receipt['invalid'],
            legal_ids=input_receipt['legal_ids'],adaptation_indices=input_receipt['adaptation_indices'])
        persist()
        if not packets: raise engine.FitFailure('no legal fit297 packet',dict(status='no_legal_input'))
        config=dict(format='gftd_fit297_v1',protocol_sha256=PROTOCOL_SHA256,
            adaptation_indices=input_receipt['adaptation_indices'],k_ref=constants['k_ref'],
            source_sha256={RN_EXPORT:RN_SHA256,PROTOCOL:PROTOCOL_SHA256},
            **{k:scales[k] for k in ('input_scale_delta','input_scale_omega','rstar')})
        models,states,failures={}, {},{}
        for mode in MODES:
            at('fit',mode);counts['fits_attempted']+=1;persist();tick=time.perf_counter()
            receipt['costs'][mode]=dict(fit_seconds=None,initialization_seconds=0.,integration_seconds=0.)
            model=copy.deepcopy(template);model.fit_mode=mode
            fit_error=None
            try:
                model.fit(packets,[],RunContext(gftd_fit297_v1=config))
                counts['fit'][mode]=copy.deepcopy(model.fit_state.get('counts',{}))
                counts['fits_completed']+=1
            except engine.FitFailure as error:
                counts['fit'][mode]=copy.deepcopy(error.diagnostics.get('counts',{}))
                fit_error=error
                failures[mode]=dict(stage='fit',type=type(error).__name__,message=str(error))
            finally:
                receipt['costs'][mode]['fit_seconds']=time.perf_counter()-tick
            if fit_error is not None:
                receipt['artifacts'].append(write_json(PRIVATE/(mode+'_fit_failure.json'),fit_error.diagnostics))
                persist();continue
            persist()
            receipt['artifacts'].append(write_json(PRIVATE/(mode+'_fit_details.json'),model.fit_private_details))
            checkpoint=PRIVATE/(mode+'_model.json');temp=checkpoint.with_suffix('.tmp')
            model.save(temp);os.replace(temp,checkpoint)
            receipt['artifacts'].append(dict(path=str(checkpoint.relative_to(ROOT)),sha256=digest(checkpoint)))
            models[mode]=model;states[mode]=[];persist()
        for mode,model in models.items():
            for index,(eid,packet) in enumerate(zip(input_receipt['legal_ids'],packets)):
                at('initialize',mode,eid);counts['initializations_attempted']+=1;persist();tick=time.perf_counter()
                state=None
                diag=None
                try:
                    state=model.initialize({k:v for k,v in packet.items() if k!='sigma'})
                    diag=model.last_initialization or {}
                    counts['initializations_completed']+=1
                except engine.InitializationFailure as error:
                    diag=error.diagnostics
                    failures[(mode,eid)]=dict(stage='initialization',type=type(error).__name__,message=str(error))
                finally:
                    if diag is not None:
                        for key,value in diag.get('counts',{}).items():
                            if isinstance(value,(int,float)) and not isinstance(value,bool):
                                counts['initialization'][key]=counts['initialization'].get(key,0)+value
                    receipt['costs'][mode]['initialization_seconds']+=time.perf_counter()-tick
                states[mode].append(state)
                artifact=None if state is None else write_npz(PRIVATE/'initializations'/f'{mode}_{index:03d}.npz',state=state)
                detail=write_json(PRIVATE/'initializations'/f'{mode}_{index:03d}.json',dict(event_id=eid,diagnostics=diag,state=artifact))
                receipt['artifacts'].append(detail);persist()
        # Every model and every deployment initializer is frozen before any future score.
        frozen=write_json(PRIVATE/'all_initializations_frozen.json',dict(models={mode:digest(PRIVATE/(mode+'_model.json')) for mode in models},
            counts=counts,legal_ids=input_receipt['legal_ids'],success={mode:[s is not None for s in states[mode]] for mode in states}))
        receipt['artifacts'].append(frozen);persist()
        validated=[core.validate_observation({('p_end' if k=='p0' else k):v for k,v in p.items()},10) for p in packets]
        for mode,model in models.items():
            at('coverage_information',mode)
            coverage=fit_engine.coverage_information(model.functional,validated,states[mode],model.inference_config)
            counts['coverage'][mode]=coverage.get('counts',{})
            receipt['artifacts'].append(write_json(PRIVATE/(mode+'_coverage_information.json'),coverage))
        records={mode:[] for mode in MODES}
        for mode in MODES:
            model=models.get(mode)
            for index,eid in enumerate(input_receipt['ordered_all_ids']):
                event=by_id[eid]
                if model is None or eid not in input_receipt['legal_ids']:
                    records[mode].append(l1.failure_event(event,'fit' if model is None else 'input',RuntimeError(str(failures.get(mode,'illegal input')))))
                    continue
                pos=input_receipt['legal_ids'].index(eid);state=states[mode][pos]
                if state is None:
                    records[mode].append(l1.failure_event(event,'initialization',RuntimeError(str(failures[(mode,eid)]))))
                    continue
                at('future_pair',mode,eid);counts['future_pairs_attempted']+=1;persist();tick=time.perf_counter()
                coarse,ce=integrate(model.functional,state,.01,counts)
                fine,fe=integrate(model.functional,state,.005,counts)
                receipt['costs'][mode]['integration_seconds']+=time.perf_counter()-tick
                counts['future_pairs_completed' if ce is None and fe is None else 'future_pairs_failed']+=1
                arrays=write_npz(PRIVATE/'predictions'/f'{mode}_{index:03d}.npz',coarse=coarse,fine=fine)
                receipt['artifacts'].append(arrays);counts['future_pairs_persisted']+=1
                at('score',mode,eid);counts['scoring_attempted']+=1;persist()
                record=l1.score_event(event,fine,coarse,delta_star=model.delta_star,mass=model.mass,p_scale=scales['p_scale'])
                record['integration_errors']=[e for e in (ce,fe) if e is not None]
                record['trajectory']=arrays
                records[mode].append(record);counts['scoring_completed']+=1
                receipt['artifacts'].append(write_json(OUTPUT/'events'/f'{mode}_{index:03d}.json',record));persist()
        for mode in MODES:
            for index,record in enumerate(records[mode]):
                if record.get('failure') is not None:
                    receipt['artifacts'].append(write_json(OUTPUT/'events'/f'{mode}_{index:03d}.json',record))
        paired=l1.paired_summary(records['static'],records['modulated'])
        receipt['artifacts'].append(write_json(OUTPUT/'l1_summary.json',paired))
        at('model_analysis')
        import gftd_fit297_analysis_20261107 as analysis
        for mode,model in models.items():
            certificate=analysis.l2_certificate(model.functional,model.delta_star)
            receipt['analysis'][mode]=dict(certificate_status=certificate.get('status'),
                l2_radius_candidates_actual=len(certificate.get('candidates',[])),l2_point_calls_actual=0)
            points=[]
            for state in states[mode]:
                if state is None:
                    points.append(None)
                else:
                    point=analysis.l2_point(certificate,model.functional,state)
                    receipt['analysis'][mode]['l2_point_calls_actual']+=1
                    points.append(point)
            modes=analysis.rom_modes(model.functional,model.delta_star,certificate)
            receipt['analysis'][mode].update(point_count=len(points),mode_status=modes.get('status'),rom_counts=modes.get('counters'))
            receipt['artifacts'].append(write_json(PRIVATE/(mode+'_l2_l3.json'),dict(certificate=certificate,points=points,modes=modes)))
        if models:
            reference=analysis.kron_reference(PRIVATE/'reference/ieee39_full.xlsx')
            receipt['analysis']['kron_counts']={k:reference.get(k) for k in ('svd_count','solve_count')}
            receipt['artifacts'].append(write_json(PRIVATE/'kron_reference.json',reference))
            identity=release.get('fom_expected_identity',dict(equilibrium_identity_verified=False))
            fom=analysis.fom_modes(ROOT/'private_data/task2_fullorder_descriptor_0035/descriptor.npz',
                ROOT/'private_data/task2_fullorder_descriptor_0035/evidence.json',ROOT/'private_data/task2_verified_modes_0035b/evidence.json',
                template.mass,expected_identity=identity)
            receipt['analysis']['fom_qz_actual']=fom.get('qz_count',fom.get('checks',{}).get('qz_count'))
            receipt['artifacts'].append(write_json(PRIVATE/'fom_modes.json',fom))
            for mode,model in models.items():
                content=read_json(PRIVATE/(mode+'_l2_l3.json'))
                matched=analysis.match_modes(fom,content['modes'])
                receipt['artifacts'].append(write_json(PRIVATE/(mode+'_modal_match.json'),matched))
            for mode,model in models.items():
                correspondence=analysis.graph_correspondence(reference,model.functional,model.delta_star)
                receipt['artifacts'].append(write_json(PRIVATE/(mode+'_graph_correspondence.json'),correspondence))
            key=next(iter(models.values())).fit_state.get('preparation_key')
            receipt['analysis_budget']=fit_engine.analysis_budget_snapshot(key)
        checks=model_analysis_budget_checks(receipt['analysis'],models)
        receipt['analysis_budget_checks']=checks
        if any(c['within_limit'] is False for c in checks): raise RuntimeError('registered model analysis budget exceeded')
        receipt['execution_budget']=execution_budget(counts)
        if not receipt['execution_budget']['within_limits']: raise RuntimeError('registered execution budget exceeded')
        receipt['status']='COMPLETED'
    except Exception as error:
        receipt.update(status='INTERRUPTED_RUNTIME',error=dict(type=type(error).__name__,message=str(error)),
                       traceback=traceback.format_exc())
        if hasattr(error,'diagnostics'):
            receipt['artifacts'].append(write_json(PRIVATE/'interruption_diagnostics.json',error.diagnostics))
        raise
    finally:
        receipt['execution_budget']=execution_budget(counts)
        receipt['wall_seconds']=time.perf_counter()-begin
        receipt['source_unchanged']=all(digest(ROOT/p)==sha for p,sha in release['source_sha256'].items())
        receipt['release_unchanged']=digest(release_path)==claim['release_sha256']
        receipt['evidence_binding_valid']=bool(receipt['source_unchanged'] and receipt['release_unchanged'])
        if not receipt['evidence_binding_valid']: receipt['status']='INTERRUPTED_SOURCE_CHANGED'
        for cost in receipt['costs'].values():
            cost['deployment_total_seconds']=cost['initialization_seconds']+cost['integration_seconds']
            cost['same_precision_conditions']=dict(device='cpu',dtype='float64',future_steps=[.01,.005],duration_s=15.,batch=1)
        receipt['FOM_speedup']='未比较'
        persist();write_json(OUTPUT/'execution.json',receipt)
    return receipt


if __name__=='__main__':
    run()
