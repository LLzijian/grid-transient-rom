"""Small controlled compiler/discovery checks, never a replacement for real fits.

Observed-coordinate cases use linear-in-coefficient field identification followed
by autonomous validation. The partial-observation case fits coefficients/GRU
against trajectories. Their different information privileges are explicit.
"""
from __future__ import annotations
from pathlib import Path
import time
import numpy as np
import torch


FIXTURE_SEED = 4801
OBSERVATION_NOISE_LEVELS = (0.0, 0.01)
STATE_TRAIN_COUNT = 128
PARTIAL_TRAJECTORY_COUNT = 8
PARTIAL_HISTORY_STEPS = 10
PARTIAL_FUTURE_STEPS = 20
PARTIAL_DT_SECONDS = 0.01
PARTIAL_OTHER_INITIAL_STD = 0.2
PARTIAL_Q_INITIAL_STD = 0.8
PARTIAL_UPDATES_PER_CANDIDATE = 128


def v(name):return {'op':'var','name':name}
def mul(a,b):return {'op':'mul','args':[a,b]}
def power(a,n):return a if n==1 else {'op':f'pow{n}','arg':a}


def toy_source():
    from structured_rom import StructuredROM
    from local_memory_rom_0098 import LocalMemoryROM,_inv_softplus
    from symbolic_closure_rom import SymbolicClosureROM
    D=np.zeros((10,10))
    for i in range(10):D[i,i]=1;D[i,(i+1)%10]=-1
    base=StructuredROM(mass=np.ones(10),D=D,coupling=np.full(10,.2),
          delta_star=np.linspace(-.05,.05,10),damping=np.full(10,.15),gain=np.ones(10),
          jpi_p=np.full(10,.3),rpp=np.full(10,.4),n_q=0)
    base.rate=torch.full((10,),.4,dtype=torch.float64)
    old=LocalMemoryROM(base,input_scale=np.ones(20),p_scale=np.ones(10),seed=4801)
    with torch.no_grad():
        old.template_mask[:]=0;old.template_mask[3]=old.template_mask[5]=1
        old._template_scalars[3].fill_(.3)
        old._template_scalars[5].copy_(_inv_softplus(torch.tensor(.4,dtype=torch.float64)))
    return SymbolicClosureROM.from_source(old).with_terms([])


def pool():
    """Use the production expression generator, including one parent-tree expansion.

    Candidate generation sees syntax only, never fixture targets or fitted errors.
    The component test enumerates proposals before the budgeted beam screening.
    """
    from closure_discovery_support import _candidate_terms,fixed_library,term_key
    first=_candidate_terms(fixed_library())
    node_parents=[term for term in first if term['channel']=='node']
    second=_candidate_terms(node_parents)
    unique={term_key(t):t for t in first+second}
    return [unique[k] for k in sorted(unique)]


def path(model,initial,steps=100,dt=.01):
    from train_local_memory_0098 import graph
    state=initial;rows=[state]
    with graph(model):
        for _ in range(steps):state=model.rk4(state,dt);rows.append(state)
    return torch.stack(rows,1)


def _observe_with_noise(clean,scales,fraction,rng):
    """Add fixed train-scale observation noise, shared across cases/coordinates."""
    clean=np.asarray(clean,dtype=np.float64)
    scales=np.asarray(scales,dtype=np.float64)
    if clean.shape[-1]!=scales.shape[-1]:raise ValueError('noise scales must match observed variables')
    if fraction==0:return clean.copy()
    return clean+float(fraction)*rng.normal(size=clean.shape)*scales


def _partial_truth_fixture(truth,seed=FIXTURE_SEED):
    """Generate ten 100-ms autonomous history steps and twenty future steps."""
    rng=np.random.default_rng(seed)
    initial=rng.normal(0,PARTIAL_OTHER_INITIAL_STD,
                       size=(PARTIAL_TRAJECTORY_COUNT,40))
    initial[:,30:]=rng.normal(0,PARTIAL_Q_INITIAL_STD,
                              size=(PARTIAL_TRAJECTORY_COUNT,10))
    initial_tensor=torch.as_tensor(initial,dtype=torch.float64)
    with torch.no_grad():
        trajectory=path(truth,initial_tensor,
                        PARTIAL_HISTORY_STEPS+PARTIAL_FUTURE_STEPS,
                        dt=PARTIAL_DT_SECONDS)
    full=trajectory.detach().cpu().numpy()
    history=full[:,:PARTIAL_HISTORY_STEPS+1,:30]
    future=full[:,PARTIAL_HISTORY_STEPS+1:,:30]
    if history.shape[1]!=11 or future.shape[1]!=PARTIAL_FUTURE_STEPS:
        raise AssertionError('partial fixture did not retain the registered history/future lengths')
    return initial,full,history,future


def identify(base,observed_states,clean_total_rhs,candidates):
    """Fit a residual against clean RHS labels and the baseline at observed states."""
    observed_states=torch.as_tensor(observed_states,dtype=torch.float64)
    clean_total_rhs=torch.as_tensor(clean_total_rhs,dtype=torch.float64)
    with torch.no_grad():
        y=clean_total_rhs-base.field(observed_states)
    scores=[]
    for term in candidates:
        m=base.with_terms([term])
        with torch.no_grad():
            m.coefficients.fill_(1.);b=m.field(observed_states)-base.field(observed_states)
            c=(b*y).sum()/b.square().sum().clamp_min(1e-20)
            error=float((y-c*b).square().mean())
        scores.append((error,term,float(c)))
    best=min(scores,key=lambda r:r[0]);recovered=base.with_terms([best[1]])
    with torch.no_grad():recovered.coefficients.fill_(best[2])
    return recovered,best


def controlled_benchmarks(output,deadline=None):
    from discover_closure import write
    from symbolic_closure_rom import term_key
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    rng=np.random.default_rng(FIXTURE_SEED);base=toy_source();candidates=pool()
    node={'channel':'node','expression':mul(power(v('p'),2),power(v('q'),2))}
    edge={'channel':'edge','expression':mul({'op':'add','args':[v('qi'),mul({'op':'const','value':-1.},v('qj'))]},v('s'))}
    saturating={'channel':'port','expression':{'op':'tanh','scale':2.,'arg':v('q')}}
    partial_pool=[t for t in candidates if t['channel']=='port' and
                  ((t['expression']['op']=='var' and t['expression']['name'] in ('p','w','q')) or
                   (t['expression']['op']=='tanh' and t['expression']['arg']==v('q')))]
    if len(partial_pool)!=6:raise AssertionError('the registered partial-observation candidate set changed')
    write(output/'truths_before_fits.json',{'seed':FIXTURE_SEED,'cases':[node,edge,saturating],
            'outside':'port=q/(1+abs(q)); excluded from candidate syntax',
            'pool':candidates,'partial_blind_candidates':partial_pool,
            'observation_noise':{'levels':list(OBSERVATION_NOISE_LEVELS),
                'state_noise_location':'observed 40-D input states only; RHS labels remain noiseless',
                'state_scale':'per-variable standard deviation over fixed 128-state train fixture'},
            'partial_saturation':{'history_steps':PARTIAL_HISTORY_STEPS,'dt_seconds':PARTIAL_DT_SECONDS,
                'history_seconds':PARTIAL_HISTORY_STEPS*PARTIAL_DT_SECONDS,
                'future_steps':PARTIAL_FUTURE_STEPS,'observed_coordinates':30,
                'future_seconds':PARTIAL_FUTURE_STEPS*PARTIAL_DT_SECONDS,
                'other_initial_std':PARTIAL_OTHER_INITIAL_STD,'hidden_q_initial_std':PARTIAL_Q_INITIAL_STD,
                'noise_scale':'per-variable standard deviation over clean train history only',
                'q_target_used':False,'future_inputs_used':False},
            'outside_grammar':{'noise_levels':list(OBSERVATION_NOISE_LEVELS),
                'state_initial_std':1.5,
                'noise_scale':'per-variable standard deviation over its fixed clean train states',
                'target':'clean q/(1+abs(q))','exact_recovery_claim':False},
            'scope':'controlled implementation evidence; no train/val/test generalization claim'})

    states_np=rng.normal(0,.3,size=(STATE_TRAIN_COUNT,40))
    state_train_scales=states_np.std(axis=0)
    state_noise_rng=np.random.default_rng(FIXTURE_SEED+1)
    observed_states_1pct=_observe_with_noise(states_np,state_train_scales,.01,state_noise_rng)
    state_initials=torch.as_tensor(rng.normal(0,.2,size=(8,40)),dtype=torch.float64)
    states=torch.as_tensor(states_np,dtype=torch.float64)
    with torch.no_grad():
        clean_total_rhs={}
        for name,term,coefficient in (('node',node,.2),('edge',edge,.15)):
            truth=base.with_terms([term])
            truth.coefficients.fill_(coefficient)
            clean_total_rhs[name]=truth.field(states).detach().cpu().numpy()
            truth.export(output/f'truth_{name}.json')

    fixture_arrays={
        'state_train_clean':states_np,
        'state_train_scales':state_train_scales,
        'state_noise_draw_seed_4802':(observed_states_1pct-states_np)/(.01*state_train_scales),
        'state_train_observed_1pct':observed_states_1pct,
        'state_total_rhs_node_clean':clean_total_rhs['node'],
        'state_total_rhs_edge_clean':clean_total_rhs['edge'],
        'state_autonomous_initials':state_initials.numpy(),
    }
    rows=[]
    for name,term,coefficient in (('node',node,.2),('edge',edge,.15)):
        truth=base.with_terms([term])
        with torch.no_grad():truth.coefficients.fill_(coefficient)
        with torch.no_grad():target=path(truth,state_initials)
        for noise in OBSERVATION_NOISE_LEVELS:
            if deadline and time.perf_counter()>deadline-60:raise TimeoutError('controlled cases incomplete')
            observed_states=states_np if noise==0 else observed_states_1pct
            labels=clean_total_rhs[name]
            recovered,best=identify(base,observed_states,labels,candidates)
            with torch.no_grad():prediction=path(recovered,state_initials)
            exact=(term_key(term)==term_key(best[1]))
            error=float((prediction-target).square().mean())
            # Symmetrically equivalent edge terms can differ by a scalar factor.
            field_error=float((recovered.field(states)-truth.field(states)).square().mean().detach())
            rows.append({'case':name,'noise':noise,'recovered_term':best[1],'coefficient':best[2],
                'same_canonical_term':exact,'field_mse':field_error,'autonomous_mse':error,
                'identification_train_mse':best[0],
                'label_noise':0.0,'input_observation_noise':noise,
                'train_state_scales':state_train_scales.tolist(),
                'information':'complete 40-D observed states and known clean RHS labels; controlled field-identification fixture, not a partial observer or generalization test'})
    # Partial observations are generated from the autonomous truth ODE; hidden q
    # affects the future but is neither an input coordinate nor a training target.
    truth=base.with_terms([saturating])
    with torch.no_grad():truth.coefficients.fill_(.6)
    truth_payload=truth.export(output/'truth_partial_saturation.json')
    truth_payload.update({
        'history_seconds':PARTIAL_HISTORY_STEPS*PARTIAL_DT_SECONDS,
        'future_seconds':PARTIAL_FUTURE_STEPS*PARTIAL_DT_SECONDS,
        'trajectory_seed':FIXTURE_SEED,
    })
    write(output/'truth_partial_saturation.json',truth_payload)
    partial_initial,partial_trajectory,history_clean,future_clean=_partial_truth_fixture(truth)
    partial_train_scales=history_clean.std(axis=(0,1))
    partial_noise_rng=np.random.default_rng(FIXTURE_SEED+2)
    history_noisy_1pct=_observe_with_noise(history_clean,partial_train_scales,.01,partial_noise_rng)
    future_noisy_1pct=_observe_with_noise(future_clean,partial_train_scales,.01,partial_noise_rng)
    fixture_arrays.update({
        'partial_initial_full_state':partial_initial,
        'partial_truth_trajectory_full':partial_trajectory,
        'partial_clean_history_observed_30':history_clean,
        'partial_clean_future_observed_30':future_clean,
        'partial_clean_train_history_scales':partial_train_scales,
        'partial_history_noise_draw_seed_4803':(history_noisy_1pct-history_clean)/(.01*partial_train_scales),
        'partial_future_noise_draw_seed_4803':(future_noisy_1pct-future_clean)/(.01*partial_train_scales),
        'partial_history_observed_1pct':history_noisy_1pct,
        'partial_future_observed_1pct':future_noisy_1pct,
    })
    for noise in OBSERVATION_NOISE_LEVELS:
        history_np=history_clean if noise==0 else history_noisy_1pct
        target_np=future_clean if noise==0 else future_noisy_1pct
        history=torch.as_tensor(history_np,dtype=torch.float64)
        target_noisy=torch.as_tensor(target_np,dtype=torch.float64)
        fitted=[]
        for proposal in partial_pool:
            m=base.with_terms([proposal]);opt=torch.optim.Adam([p for p in m.parameters() if p.requires_grad],lr=.001)
            loss0=None
            for i in range(PARTIAL_UPDATES_PER_CANDIDATE):
                if deadline and time.perf_counter()>deadline-60:raise TimeoutError('partial observation controlled case incomplete')
                opt.zero_grad(set_to_none=True)
                initial=torch.cat((history[:,-1],m.encode(history)),-1)
                pred=path(m,initial,PARTIAL_FUTURE_STEPS,dt=PARTIAL_DT_SECONDS)[:,1:,:30]
                loss=(pred-target_noisy).square().mean()
                if loss0 is None:loss0=float(loss.detach())
                loss.backward();torch.nn.utils.clip_grad_norm_(m.parameters(),100.);opt.step()
            with torch.no_grad():
                initial=torch.cat((history[:,-1],m.encode(history)),-1)
                pred=path(m,initial,PARTIAL_FUTURE_STEPS,dt=PARTIAL_DT_SECONDS)[:,1:,:30]
                final_mse=float((pred-target_noisy).square().mean())
            fitted.append({'proposal':proposal,'initial_mse':loss0,'final_mse':final_mse,
                           'updates':PARTIAL_UPDATES_PER_CANDIDATE})
        best=min(fitted,key=lambda r:r['final_mse'])
        rows.append({'case':'partial_saturation','noise':noise,**best,'all_candidates':fitted,
                     'same_canonical_term':term_key(best['proposal'])==term_key(saturating),
                     'candidate_count':len(partial_pool),'updates_per_candidate':PARTIAL_UPDATES_PER_CANDIDATE,
                     'history_steps':PARTIAL_HISTORY_STEPS,'dt_seconds':PARTIAL_DT_SECONDS,
                     'history_seconds':PARTIAL_HISTORY_STEPS*PARTIAL_DT_SECONDS,
                     'future_steps':PARTIAL_FUTURE_STEPS,'observed_coordinates':30,
                     'future_seconds':PARTIAL_FUTURE_STEPS*PARTIAL_DT_SECONDS,
                     'q_target_used':False,'future_inputs_used':False,
                     'train_history_scales':partial_train_scales.tolist(),
                     'information':'truth autonomous ODE: 10-step 100-ms physical history and 20-step autonomous future; only first 30 coordinates observed; hidden q is neither input nor target',
                     'scope':'train-only finite blind candidate comparison; no train/val/test generalization claim'})
    # Out-of-grammar probe uses a deliberately omitted rational law.  The
    # rational target stays clean while only its observed state is perturbed.
    outside_states_np=rng.normal(0,1.5,size=(STATE_TRAIN_COUNT,40))
    outside_train_scales=outside_states_np.std(axis=0)
    outside_noise_rng=np.random.default_rng(FIXTURE_SEED+3)
    outside_states_1pct=_observe_with_noise(outside_states_np,outside_train_scales,
                                             .01,outside_noise_rng)
    outside_target=outside_states_np[:,30:]/(1+np.abs(outside_states_np[:,30:]))
    fixture_arrays.update({
        'outside_grammar_state_train_clean':outside_states_np,
        'outside_grammar_state_train_scales':outside_train_scales,
        'outside_grammar_state_noise_draw_seed_4804':
            (outside_states_1pct-outside_states_np)/(.01*outside_train_scales),
        'outside_grammar_state_train_observed_1pct':outside_states_1pct,
        'outside_grammar_target_clean_q_rational':outside_target,
    })
    for noise in OBSERVATION_NOISE_LEVELS:
        observed=outside_states_np if noise==0 else outside_states_1pct
        observed_tensor=torch.as_tensor(observed,dtype=torch.float64)
        target=torch.as_tensor(outside_target,dtype=torch.float64)
        errors=[]
        for term in candidates:
            if term['channel']!='port':continue
            candidate=base.with_terms([term])
            with torch.no_grad():
                candidate.coefficients.fill_(1.)
                b=candidate.port_values(candidate.interface_to_state(observed_tensor))
                c=(b*target).sum()/b.square().sum().clamp_min(1e-20)
                errors.append((float((target-c*b).square().mean()),term,float(c)))
        best_error,best_term,best_coefficient=min(errors,key=lambda row:row[0])
        rows.append({'case':'outside_grammar','noise':noise,'best_field_mse':best_error,
                     'best_candidate':best_term,'coefficient':best_coefficient,
                     'train_state_scales':outside_train_scales.tolist(),
                     'target_noise':0.0,'input_observation_noise':noise,
                     'exact_recovery_claim':False,
                     'scope':'train-only approximation diagnostic for a known omitted rational q law; no generalization claim'})
    implementation_pass=all(np.isfinite(r.get('autonomous_mse',r.get('final_mse',r.get('best_field_mse')))) for r in rows)
    clean=[r for r in rows if r.get('noise')==0 and r['case'] in ('node','edge')]
    implementation_pass=implementation_pass and all(r['field_mse']<1e-16 and r['autonomous_mse']<1e-16 for r in clean)
    result={'implementation_pass':bool(implementation_pass),'cases':rows,
        'scope':'known clean RHS with noisy full-state observations for node/edge identification, plus train-only partial-observer autonomous fitting; no generalization claim',
        'test_read':False}
    fixture_arrays['seed_sequence'] = np.asarray([FIXTURE_SEED,FIXTURE_SEED+1,
                                                   FIXTURE_SEED+2,FIXTURE_SEED+3],dtype=np.int64)
    np.savez_compressed(output/'training_fixture.npz',**fixture_arrays)
    write(output/'results.json',result);return result
