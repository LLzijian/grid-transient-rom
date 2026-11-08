"""Frozen Run2 artificial modulation experiment; execution requires a root release."""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
import scipy

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from kiki.models.base import RunContext
from kiki.models.gftd_kuramoto import GFTDKuramotoModel
from kiki.models import gftd_discovery as core, gftd_fitting as engine
from kiki.physics.kuramoto_functional import GeneratingFunctionalKuramoto, StructuralGates

OUTPUT = ROOT/'results/gftd_modulation_20261107'
PROTOCOL = 'reports/gftd_modulation_20261107_protocol.md'
PROTOCOL_SHA256 = '54622ad9da5fea1afb326140c6d4288a51bddca2aae896ef10926f951b09dcb0'
DESIGN_REVIEW_SHA256 = '1d57115c4e962d8cba4823c81b7c246f9b125c72dacadfc83430c6b14290471a'
STEPS = (.0025, .00125)
SIGMAS = (0., 1e-4)
FAMILIES = ('Static', 'Modulated', 'Memory', 'Equilibrium')
MODES = ('static', 'modulated')
REQUIRED_SOURCES = ('experiments/gftd_modulation_20261107.py', 'src/kiki/models/gftd_kuramoto.py',
                    'src/kiki/models/gftd_fitting.py', 'src/kiki/models/gftd_discovery.py',
                    'src/kiki/models/base.py', 'src/kiki/physics/kuramoto_functional.py',
                    'src/kiki/physics/graph.py')
LIMITS = dict(algebra_terms_actual=9832, nnls_attempts=9816, optimization_tasks=480,
              actual_residual_calls=163200)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    def serial(item):
        if isinstance(item, np.ndarray): return item.tolist()
        if isinstance(item, np.generic): return item.item()
        raise TypeError(type(item).__name__)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(value, default=serial, indent=2, allow_nan=False)+'\n', encoding='utf-8')


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def add_counts(target, source):
    for key, value in source.items():
        target[key] = target.get(key, 0)+value


def functional(k=None, beta=None):
    k = np.full(6,.2) if k is None else np.asarray(k, dtype=np.float64)
    beta = np.zeros(6) if beta is None else np.asarray(beta, dtype=np.float64)
    modulation = np.divide(beta,k,out=np.zeros(6),where=k>0)
    return GeneratingFunctionalKuramoto(mass=np.array([1.,1.3,1.7,2.]),damping=np.full(4,.2),
        k_edge=k,eta_edge=np.zeros(6),gamma_amp=np.full(4,.5),lambda_amp=np.ones(4),
        lambda_mem=np.full(4,1/.7),tau_amp=np.full(4,5.),tau_mem_sync=np.full(4,1/(.7*.3)),
        tau_mem_coi=1/(.7*.3),j_omega_mem=np.full(4,.7),storage_p=np.ones(4),gain_p=np.ones(4),
        rate_p=np.full(4,.3),gates=StructuralGates(np.ones(6),modulation,np.ones(4)))


def conditions():
    k = np.array([.7,0.,0.,.9,0.,1.1])
    excitations = [np.r_[[.12,-.08,.03,-.04],[.03,-.02,.01,0.],[.02,-.01,.005,0.],np.zeros(4),[.015,-.02,.005,.01]],
                   np.r_[[-.03,.11,-.07,.05],[-.01,.04,-.03,.02],[-.01,.015,0.,.005],np.zeros(4),[.02,.01,-.015,-.005]],
                   np.r_[[.04,-.02,.13,-.09],[.02,.01,-.04,.03],[.005,0.,-.015,.02],np.zeros(4),[-.01,.02,.01,-.015]]]
    amplitudes = ([.03,-.02,.01,-.01],[-.02,.01,.04,-.03],[.01,.03,-.02,.02])
    result = []
    for family in FAMILIES:
        for event, initial in enumerate(excitations):
            state = initial.copy()
            beta = .5*k if family in ('Modulated','Equilibrium') else np.zeros(6)
            if family == 'Modulated': state[12:16] = amplitudes[event]
            if family == 'Memory': state[16:20] *= 3
            if family == 'Equilibrium': state[:] = 0
            result.append(dict(index=len(result),family=family,event=event,k=k.copy(),beta=beta,state=state))
    return result


def integrate(f, initial, step, duration, counts, *, backward=False):
    counts['integrations_attempted'] = counts.get('integrations_attempted',0)+1
    state = np.asarray(initial, dtype=np.float64).copy()
    states = [state.copy()]
    for _ in range(int(round(duration/step))):
        counts['rk4_steps_attempted'] = counts.get('rk4_steps_attempted',0)+1
        state = core.rk4(f,state,-step if backward else step)
        counts['rk4_steps_completed'] = counts.get('rk4_steps_completed',0)+1
        states.append(state.copy())
    counts['integrations_completed'] = counts.get('integrations_completed',0)+1
    return np.asarray(states[::-1] if backward else states)


def generate(output, counts):
    truth, observed, plan = [], {}, []
    for spec in conditions():
        f = functional(spec['k'],spec['beta'])
        record = dict(spec, history={}, future={})
        for step in STEPS:
            record['history'][str(step)] = integrate(f,spec['state'],step,.1,counts)
            counts['generated_history_grids'] = counts.get('generated_history_grids',0)+1
        end = record['history'][str(STEPS[1])][-1]
        for step in (.01,.005):
            record['future'][str(step)] = integrate(f,end,step,1.,counts)
            counts['generated_future_grids'] = counts.get('generated_future_grids',0)+1
        fine = record['history'][str(STEPS[1])][::8]
        noise = np.random.Generator(np.random.PCG64(160000+spec['index'])).standard_normal((11,8))
        for sigma in SIGMAS:
            key = f'obs_{len(observed):03d}'
            measured = fine[:,:8]+.1*sigma*noise
            observed[key] = dict(theta_history=measured[:,:4],omega_history=measured[:,4:8],
                                 history_times=np.linspace(-.1,0.,11),p0=end[8:12].copy(),sigma=sigma)
            plan.append(dict(input_id=key,family=spec['family'],event=spec['event'],trajectory=spec['index'],
                             sigma=sigma,seed=160000+spec['index']))
            counts['observation_packages'] = counts.get('observation_packages',0)+1
        truth.append(record)
        write_json(output/'generation.partial.json',dict(truth=truth,counts=counts))
    write_json(output/'isolated_truth.json',truth)
    write_json(output/'observations.json',observed)
    write_json(output/'observation_plan.json',plan)
    return output/'isolated_truth.json', output/'observations.json', output/'observation_plan.json'


def qualify_A(truth, counts):
    records = []
    for event in truth:
        f = functional(event['k'],event['beta'])
        histories, roundtrips, weak = [], [], []
        for step in STEPS:
            states = np.asarray(event['history'][str(step)])
            histories.append(core.observations(states[::int(round(.01/step))], f.mass))
            back = integrate(f,states[-1],step,.1,counts,backward=True)
            roundtrips.append(float(np.max(np.abs((back[0]-states[0])/.1))))
            t = np.linspace(-.1,0.,len(states))
            v,dv = core.weak_test_functions(t)
            weights = core.boole_weights(t)
            weighted = v*weights
            theta,omega,p,a,r = np.split(states,5,axis=1)
            h = (a*f.gamma_amp)@np.abs(f.incidence)
            force = (np.sin(theta@f.incidence)-np.sin(f.eta_edge))*(f.k_edge-f.k_edge*f.gates.s_mod+np.exp(h)*f.k_edge*f.gates.s_mod)
            lhs = np.vstack([f.incidence*row for row in weighted@force]).sum(axis=1)/.1
            rhs = (((dv*weights)@(omega*f.mass))-weighted@(omega*f.damping)+
                   weighted@(p*f.gain_p*f.storage_p)-weighted@(r*f.j_omega_mem*f.lambda_mem)).reshape(-1)/.1
            residual = lhs-rhs
            weak.append(float(np.max(np.abs(residual))))
        history_delta = histories[0]-histories[1]
        futures = [core.observations(np.asarray(event['future'][str(step)])[::int(round(.01/step))],f.mass)
                   for step in (.01,.005)]
        delta = futures[0]-futures[1]
        record = dict(trajectory=event['index'],roundtrip_scaled_max=roundtrips,
            weak_scaled_max=weak,history_main_fine_rms=float(np.sqrt(np.mean(history_delta**2))),
            future_main_fine_rms=float(np.sqrt(np.mean(delta**2))),future_main_fine_peak=float(np.max(np.abs(delta))))
        record['passed'] = bool(np.isfinite(np.r_[roundtrips,weak,record['history_main_fine_rms'],record['future_main_fine_rms'],record['future_main_fine_peak']]).all()
            and max(roundtrips)<=1e-7 and max(weak)<=1e-7 and record['history_main_fine_rms']<=1e-7
            and record['future_main_fine_rms']<=1e-5 and record['future_main_fine_peak']<=1e-4)
        records.append(record)
    return dict(records=records,passed=bool(records) and all(row['passed'] for row in records))


def batches(observed, plan):
    if len(observed)!=24 or len(plan)!=24:
        raise ValueError('Run2 requires exactly 24 registered observation packages')
    grouped = []
    for family in FAMILIES:
        for sigma in SIGMAS:
            ids = [row['input_id'] for row in sorted(plan,key=lambda row:row['event'])
                   if row['family']==family and row['sigma']==sigma]
            events = [row['event'] for row in plan if row['family']==family and row['sigma']==sigma]
            if len(ids)!=3 or len(set(ids))!=3 or sorted(events)!=[0,1,2]:
                raise ValueError('each Run2 batch requires exactly E1/E2/E3')
            if any(float(observed[key]['sigma'])!=sigma for key in ids):
                raise ValueError('packet sigma differs from its registered batch')
            grouped.append(dict(batch_id=len(grouped),family=family,sigma=sigma,packet_ids=ids))
    if sorted(key for batch in grouped for key in batch['packet_ids']) != sorted(observed):
        raise ValueError('observation plan must partition the legal packages')
    return grouped


def fit_B(observed, plan, output, counts):
    if engine._BATCH_CACHE:
        raise RuntimeError('Run2 must start in a fresh process with an empty preparation cache')
    records, preparations = [], {}
    for batch in batches(observed,plan):
        for mode in MODES:
            row = dict(batch,mode=mode,fit_status='failed',best_validation_loss=None,val_case_ids=[],
                       model_path=None,model_sha256=None,fit_state=None)
            model = GFTDKuramotoModel(n_nodes=4,fit_mode=mode)
            model.functional = functional()
            model.gates = model.functional.gates
            model.mass,model.damping = model.functional.mass.copy(),model.functional.damping.copy()
            counts['fits_attempted'] += 1
            try:
                result = model.fit([observed[key] for key in batch['packet_ids']],[],RunContext())
                if result.best_validation_loss is not None or result.val_case_ids:
                    raise AssertionError('closed val must remain null/empty')
                row.update(fit_status=result.status,fit_state=model.fit_state,
                           rhs_probe=model.functional.rhs(np.linspace(-.01,.01,20)))
                path = output/'models'/f'batch_{batch["batch_id"]}_{mode}.json'
                model.save(path)
                row.update(model_path=str(path.relative_to(output)),model_sha256=digest(path))
                counts['fits_completed'] += 1
            except engine.FitFailure as error:
                row['fit_state'] = error.diagnostics
                row['error'] = str(error)
            state = row['fit_state']
            add_counts(counts['fit'],state.get('counts',{}))
            if 'preparation_key' in state:
                key = state['preparation_key']
                prep = dict(preparation_key=key,counts=state['shared_preparation_counts'],
                    initial_static=state['initial_static'],initial_information=state['initial_information'],
                    static_enumeration_references=state['static_enumeration_references'],
                    source='src/kiki/models/gftd_fitting.py:_prepare_batch',report_only=True)
                if not state['preparation_cache_hit']:
                    if key in preparations: raise AssertionError('repeated preparation charged twice')
                    prep_path = output/'preparations'/f'batch_{batch["batch_id"]}.json'
                    write_json(prep_path,prep)
                    preparations[key] = dict(path=str(prep_path.relative_to(output)),sha256=digest(prep_path),batch_id=batch['batch_id'])
                elif key not in preparations:
                    raise AssertionError('cache hit without a frozen first preparation')
                row['preparation_reference'] = preparations[key]
                row['preparation_cache_hit'] = state['preparation_cache_hit']
            records.append(row)
            write_json(output/'fits.partial.json',dict(records=records,counts=counts))
    result = dict(records=records,preparations=preparations,counts=copy.deepcopy(counts['fit']))
    write_json(output/'fits.json',result)
    frozen = dict(fits_sha256=digest(output/'fits.json'),models={row['model_path']:row['model_sha256'] for row in records if row['model_path']},
                  preparations=preparations)
    write_json(output/'fits_frozen.json',frozen)
    return frozen


def verify_frozen(output, frozen):
    if digest(output/'fits.json') != frozen['fits_sha256']:
        raise ValueError('fit selection changed after freezing')
    for path, expected in frozen['models'].items():
        if digest(output/path)!=expected: raise ValueError('frozen model changed')
    for item in frozen['preparations'].values():
        if digest(output/item['path'])!=item['sha256']: raise ValueError('frozen preparation changed')


def compare_saved_model(model, content):
    f = model.functional
    from dataclasses import fields
    for field in fields(f):
        if field.name == 'gates': continue
        actual,expected = getattr(f,field.name),content['functional'][field.name]
        if actual is None or expected is None:
            if actual is not expected: raise AssertionError('saved functional parameter changed')
        elif not np.array_equal(actual,expected): raise AssertionError('saved functional parameter changed')
    for key,value in content['gates'].items():
        if not np.array_equal(getattr(f.gates,key),value): raise AssertionError('saved gate changed')
    # Deterministic non-trajectory point: verifies decoding without another initialization.
    point = np.linspace(-.01,.01,5*f.n_nodes)
    reference = GeneratingFunctionalKuramoto(**content['functional'],gates=StructuralGates(**content['gates']))
    np.testing.assert_array_equal(f.rhs(point),reference.rhs(point))


def initialize_frozen(output, frozen, observed, counts):
    verify_frozen(output,frozen)
    fits = read_json(output/'fits.json')
    rows = []
    for fit in fits['records']:
        model = None
        if fit['model_path']:
            path = output/fit['model_path']
            model = GFTDKuramotoModel.load(path)
            compare_saved_model(model,read_json(path))
            np.testing.assert_array_equal(model.functional.rhs(np.linspace(-.01,.01,20)),fit['rhs_probe'])
        for input_id in fit['packet_ids']:
            row = dict(batch_id=fit['batch_id'],mode=fit['mode'],input_id=input_id,state=None,
                       status='blocked_by_fit',diagnostics=None,future_main=None,future_fine=None)
            if model is not None:
                counts['initializations_attempted'] += 1
                initial = {key:value for key,value in observed[input_id].items() if key!='sigma'}
                try:
                    state = model.initialize(initial)
                    row.update(state=state,status='completed',diagnostics=model.last_initialization)
                    for step,name in ((.01,'future_main'),(.005,'future_fine')):
                        row[name] = integrate(model.functional,state,step,1.,counts['evaluation'])[::int(round(.01/step))]
                    counts['initializations_completed'] += 1
                except engine.InitializationFailure as error:
                    row.update(status=error.diagnostics.get('status','failed'),diagnostics=error.diagnostics,error=str(error))
                add_counts(counts['initialization'],row['diagnostics'].get('counts',{}))
            rows.append(row)
            write_json(output/'initializations.partial.json',dict(rows=rows,counts=counts))
    verify_frozen(output,frozen)
    result = dict(rows=rows,counts=copy.deepcopy(counts['initialization']))
    write_json(output/'initializations.json',result)
    sha = digest(output/'initializations.json')
    write_json(output/'initializations_frozen.json',dict(initializations_sha256=sha,fits_sha256=frozen['fits_sha256']))
    return sha


def support_score(support, truth):
    if support is None: return dict(exact_support=None,false_positive=None,false_negative=None)
    return dict(exact_support=set(support)==set(truth),false_positive=len(set(support)-set(truth)),false_negative=len(set(truth)-set(support)))


def score_frozen(output, frozen, initializations_sha256, truth_path, plan):
    verify_frozen(output,frozen)
    if digest(output/'initializations.json')!=initializations_sha256:
        raise ValueError('initialized outputs changed before truth scoring')
    truth = read_json(truth_path)
    assignments = {row['input_id']:row for row in plan}
    fits = read_json(output/'fits.json')['records']
    initializations = read_json(output/'initializations.json')['rows']
    fit_rows, prediction_rows = [], []
    for fit in fits:
        event = truth[assignments[fit['packet_ids'][0]]['trajectory']]
        state = fit['fit_state']
        row = dict(batch_id=fit['batch_id'],family=fit['family'],sigma=fit['sigma'],mode=fit['mode'],fit_status=fit['fit_status'])
        row['edges'] = support_score(state.get('support'),np.flatnonzero(np.asarray(event['k'])>0))
        row['modulation'] = support_score(None if fit['mode']=='static' else state.get('modulation_support'),np.flatnonzero(np.asarray(event['beta'])>0))
        for key in ('k','beta'):
            row['max_'+key+'_error'] = None if state.get(key) is None else float(np.max(np.abs(np.asarray(state[key])-event[key])))
        row['information'] = state.get('information')
        row['joint_identifiability'] = state.get('joint_identifiability','unknown')
        fit_rows.append(row)
    fit_by_key = {(row['batch_id'],row['mode']):row for row in fits}
    for initialized in initializations:
        assignment = assignments[initialized['input_id']]
        event = truth[assignment['trajectory']]
        row = dict(batch_id=initialized['batch_id'],family=assignment['family'],sigma=assignment['sigma'],mode=initialized['mode'],
                   input_id=initialized['input_id'],status=initialized['status'],rms=None,peak=None,prediction_passed=False,
                   common_omega_rmse=None,p_normalized_rmse=None,a0_scaled_error=None,r0_scaled_error=None)
        if initialized['state'] is not None:
            reference = np.asarray(event['future']['0.005'])[::2]
            prediction = np.asarray(initialized['future_main'])
            model_spec = read_json(output/fit_by_key[(row['batch_id'],row['mode'])]['model_path'])
            mass = np.asarray(model_spec['functional']['mass'])
            difference = core.observations(prediction,mass)-core.observations(reference,mass)
            row['rms'],row['peak'] = float(np.sqrt(np.mean(difference**2))),float(np.max(np.abs(difference)))
            row['prediction_passed'] = row['rms']<=1e-3 and row['peak']<=1e-2
            row['common_omega_rmse'] = float(np.sqrt(np.mean(((prediction[:,4:8]-reference[:,4:8])@(mass/mass.sum()))**2)))
            row['p_normalized_rmse'] = float(np.sqrt(np.mean(((prediction[:,8:12]-reference[:,8:12])/.1)**2)))
            end = np.asarray(event['history']['0.00125'])[-1]
            row['a0_scaled_error'] = (np.asarray(initialized['state'])[12:16]-end[12:16])/.1
            row['r0_scaled_error'] = (np.asarray(initialized['state'])[16:20]-end[16:20])/.1
        prediction_rows.append(row)
    mechanism = [row for row in fit_rows if row['sigma']==0 and row['mode']=='modulated']
    mechanism_passed = len(mechanism)==4 and all(
        (row['edges']['exact_support'] is True and row['modulation']['exact_support'] is True and
         row['max_k_error'] is not None and row['max_k_error']<=.05 and row['max_beta_error'] is not None and row['max_beta_error']<=.05)
        if row['family']=='Modulated' else
        (row['edges']['exact_support'] is None and row['modulation']['exact_support'] is None)
        if row['family']=='Equilibrium' else row['modulation']['exact_support'] is True for row in mechanism)
    return dict(fits=fit_rows,predictions=prediction_rows,prediction_denominator=48,mechanism_gate_passed=mechanism_passed,
                scope='registered artificial K4 only; joint identifiability remains unknown',real_data_reads=False)


def algebra_record_counts(record, *, enumeration=False):
    solutions = record.get('enumeration',[]) if enumeration else [item['solution'] for item in record.get('comparisons',[])]
    solutions = list(solutions)
    if record.get('debias') is not None: solutions.append(record['debias'])
    actual = len(solutions)
    skipped = record.get('unused_algebra_items',{}).get('count',0)
    return dict(algebra_terms_actual=actual,algebra_terms_skipped=skipped,algebra_terms_planned=actual+skipped,
                nnls_attempts=sum(item['actual_nnls_calls'] for item in solutions),
                nnls_completed=sum(item['nnls_completed'] for item in solutions),
                nnls_qualified=sum(item['nnls_qualified'] for item in solutions))


def algebra_breakdown(fits):
    result=dict(initial_static={},static_enumeration={},shared_proposals={})
    seen=set()
    for row in fits['records']:
        state=row['fit_state']
        if state.get('preparation_key') is not None and not state['preparation_cache_hit']:
            key=state['preparation_key']
            if key in seen: raise AssertionError('preparation was recomputed')
            seen.add(key)
            add_counts(result['initial_static'],algebra_record_counts(state['initial_static']))
            for reference in state['static_enumeration_references'].values():
                add_counts(result['static_enumeration'],algebra_record_counts(reference,enumeration=True))
        for proposal in state.get('trace',[]):
            add_counts(result['shared_proposals'],algebra_record_counts(proposal['coefficients']))
    return result


def validate_counts(counts, fits):
    total = {}
    for category in ('fit','initialization'): add_counts(total,counts[category])
    for key,limit in LIMITS.items():
        if total.get(key,0)>limit: raise AssertionError('finite workload exceeded: '+key)
    if counts['fit'].get('actual_residual_calls',0)>76800 or counts['initialization'].get('actual_residual_calls',0)>86400:
        raise AssertionError('proxy/deployment budgets cannot be transferred')
    if counts['fits_attempted']!=16 or len(fits['records'])!=16:
        raise AssertionError('fit denominator changed')
    for category in ('fit',):
        values=counts[category]
        if values.get('algebra_terms_actual',0)+values.get('algebra_terms_skipped',0)!=values.get('algebra_terms_planned',0):
            raise AssertionError('unaccounted algebra terms')
    keys={row['fit_state'].get('preparation_key') for row in fits['records']}
    for key,expected in (('generated_history_grids',24),('generated_future_grids',24),('observation_packages',24)):
        if counts['generation'].get(key)!=expected: raise AssertionError('generation workload changed: '+key)
    if counts['initializations_attempted']!=3*counts['fits_completed']:
        raise AssertionError('initialization denominator changed')
    complete=counts['fits_completed']==16
    if complete and (None in keys or len(keys)!=8 or counts['fit'].get('reconstructions')!=48):
        raise AssertionError('shared preparation accounting changed')
    breakdown=algebra_breakdown(fits)
    recomputed={}
    for category in breakdown.values(): add_counts(recomputed,category)
    for key,value in recomputed.items():
        if value!=counts['fit'].get(key,0): raise AssertionError('algebra trace/count mismatch: '+key)
    return dict(total=total,algebra_breakdown=breakdown,limits=LIMITS,prepared_batches=len(keys-{None}),
                static_grid_references=2*len(keys-{None}),maximum_algebra_terms=9832,maximum_nnls_calls=9816)


def main():
    if digest(ROOT/PROTOCOL)!=PROTOCOL_SHA256: raise ValueError('protocol changed')
    release_path = OUTPUT/'execution_release.json'
    release = read_json(release_path)
    if release.get('released') is not True: raise PermissionError('root release required')
    if release.get('protocol_sha256')!=PROTOCOL_SHA256 or release.get('design_review_sha256')!=DESIGN_REVIEW_SHA256:
        raise ValueError('release does not bind the frozen protocol and review')
    if not set(REQUIRED_SOURCES)<=set(release['source_sha256']): raise ValueError('release omits required sources')
    for path,sha in release['source_sha256'].items():
        if digest(ROOT/path)!=sha: raise ValueError('frozen source changed: '+path)
    if engine._BATCH_CACHE: raise RuntimeError('fresh process with empty cache required')
    OUTPUT.mkdir(parents=True,exist_ok=True)
    release_hash=digest(release_path)
    with (OUTPUT/'started.json').open('x',encoding='utf-8') as stream:
        json.dump(dict(action_id=release['action_id'],status='RUNNING',release_sha256=release_hash,
                       protocol_sha256=PROTOCOL_SHA256,source_sha256=release['source_sha256']),stream)
    counts=dict(fits_attempted=0,fits_completed=0,initializations_attempted=0,initializations_completed=0,
                fit={},initialization={},evaluation={},generation={},qualification={})
    begin=time.perf_counter()
    receipt=dict(action_id=release['action_id'],status='FAILED',counts=counts,release_sha256=release_hash,
                 numpy=np.__version__,scipy=scipy.__version__,real_data_reads=False)
    try:
        if np.__version__!='2.5.2' or scipy.__version__!='1.18.0': raise ValueError('registered library versions required')
        truth_path,obs_path,plan_path=generate(OUTPUT,counts['generation'])
        gate=qualify_A(read_json(truth_path),counts['qualification'])
        write_json(OUTPUT/'qualification_A.json',gate)
        if not gate['passed']:
            receipt['status']='STOPPED_A_FAILED_B_NOT_RUN'
            return
        observed,plan=read_json(obs_path),read_json(plan_path)
        inputs=dict(observations=digest(obs_path),plan=digest(plan_path))
        frozen=fit_B(observed,plan,OUTPUT,counts)
        frozen['input_sha256']=inputs
        write_json(OUTPUT/'fits_frozen.json',frozen)
        initialized_sha=initialize_frozen(OUTPUT,frozen,observed,counts)
        if digest(obs_path)!=inputs['observations'] or digest(plan_path)!=inputs['plan']: raise ValueError('legal inputs changed')
        accounting=validate_counts(counts,read_json(OUTPUT/'fits.json'))
        write_json(OUTPUT/'accounting.json',accounting)
        summary=score_frozen(OUTPUT,frozen,initialized_sha,truth_path,plan)
        write_json(OUTPUT/'summary.json',summary)
        receipt.update(status='COMPLETED',fits_sha256=frozen['fits_sha256'],initializations_sha256=initialized_sha,
                       summary_sha256=digest(OUTPUT/'summary.json'))
    except Exception as error:
        receipt['error']=repr(error)
        raise
    finally:
        receipt.update(wall_seconds=time.perf_counter()-begin,release_unchanged=digest(release_path)==release_hash,
                       source_unchanged=all(digest(ROOT/path)==sha for path,sha in release['source_sha256'].items()),
                       protocol_unchanged=digest(ROOT/PROTOCOL)==PROTOCOL_SHA256)
        if receipt['status']=='COMPLETED' and not all(receipt[key] for key in ('release_unchanged','source_unchanged','protocol_unchanged')):
            receipt['status']='INVALID_FROZEN_INPUT_CHANGED'
        write_json(OUTPUT/'execution_receipt.json',receipt)
        print(json.dumps(receipt),flush=True)


if __name__=='__main__':
    main()
