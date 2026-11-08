"""Run3 deterministic algebra/mock qualification; never opens fit297 payloads."""
import copy
import json

import numpy as np
import pytest

from kiki.models import gftd_fit297 as new, gftd_fitting as old, gftd_discovery as core
from kiki.physics.kuramoto_functional import (
    GeneratingFunctionalKuramoto, StructuralGates, latent_coupling_modulation,
    bregman_kuramoto_potential, bregman_kuramoto_force)


def functional(n=10):
    e = n*(n-1)//2
    return GeneratingFunctionalKuramoto(mass=np.linspace(1., 2., n), damping=np.full(n, .2),
        k_edge=np.full(e, .7), eta_edge=np.linspace(-.01, .02, e), gamma_amp=np.full(n, .5),
        lambda_amp=np.ones(n), lambda_mem=np.full(n, 1/.7), tau_amp=np.full(n, 5.),
        tau_mem_sync=np.full(n, 1/(.7*.3)), tau_mem_coi=1., j_omega_mem=np.full(n, .7),
        gates=StructuralGates(s_edge=np.ones(e), s_mod=np.full(e, .25), s_mem=np.ones(n), s_coi=0.),
        storage_p=np.ones(n), gain_p=np.ones(n), rate_p=np.full(n, .3))


def packet(n=10):
    return dict(theta_history=np.zeros((11,n)), omega_history=np.zeros((11,n)),
        p_end=np.arange(n)*.001, history_times=np.linspace(-.1,0,11), sigma=0.)


def request(f, n):
    m = min(24,n)
    return dict(format=new.FIT_FORMAT, protocol_sha256=new.PROTOCOL_SHA256,
        adaptation_indices=[j*n//m for j in range(m)], input_scale_delta=np.linspace(.2,.7,f.n_nodes).tolist(),
        input_scale_omega=np.linspace(.9,2.3,f.n_nodes).tolist(), rstar=np.linspace(.23,.53,f.n_nodes).tolist(),
        k_ref=1.7, source_sha256={'synthetic_constant_attestation':'a'*64})


@pytest.fixture(autouse=True)
def isolated_cache():
    new._CACHE.clear(); new._LEDGERS.clear(); new._PREPARATION_FAILURES.clear()
    yield
    new._CACHE.clear(); new._LEDGERS.clear(); new._PREPARATION_FAILURES.clear()


def test_request_indices_scales_and_unknown_fields_are_strict():
    f = functional()
    config = request(f, 100)
    assert new.validate_fit_config(f,config,100)['adaptation_indices'] == [j*100//24 for j in range(24)]
    for bad in (dict(config,labels=[]), dict(config,adaptation_indices=list(range(24))),
                dict(config,rstar=[0.]*10), dict(config,source_sha256={})):
        with pytest.raises(ValueError):
            new.validate_fit_config(f,bad,100)


def test_new_config_fingerprint_and_legacy_dispatch(monkeypatch):
    f = functional()
    config = new.make_config(f,'modulated',request(f,2),2,'cache-key')
    assert old.validate_config(f, json.loads(json.dumps(config))) == config
    assert old.validate_config(f,old._config(f,'static',0.))['format'] == 'gftd_causal_two_stage_v1'
    bad = copy.deepcopy(config); bad['r_scale'][0] += .01
    with pytest.raises(old.InitializationFailure): old.validate_config(f,bad)
    f.rate_p[0] += .01
    with pytest.raises(old.InitializationFailure): new.validate_config(f,config)
    f = functional()
    monkeypatch.setattr(old,'_initialize_validated',lambda *args: ('state',{'status':'mock'}))
    assert old.initialize_complete(f,packet(),config) == ('state',{'status':'mock'})
    with pytest.raises(old.InitializationFailure): old.initialize_complete(f,dict(packet(),sigma=1e-4),config)


def test_rstar_mapping_scales_and_full_common_omega_are_preserved():
    f, obs = functional(), packet()
    config = new.make_config(f,'modulated',request(f,2),2,'cache-key')
    u = np.linspace(-1.,1.,20)
    state = new.endpoint(obs,f,u,'modulated','smooth',config)
    np.testing.assert_array_equal(state[20:30],obs['p_end'])
    np.testing.assert_allclose(state[30:40],.1*u[:10])
    np.testing.assert_allclose(state[40:50],np.asarray(config['r_scale'])*u[10:])
    predicted = np.tile(state,(11,1))
    predicted[:,:10] += .5
    predicted[:,10:20] += .2
    residual = new.history_residual(f,predicted,obs,config).reshape(11,20)
    np.testing.assert_allclose(residual[:,:10],0.,atol=3e-16)
    np.testing.assert_allclose(residual[:,10:],np.tile(.2/np.asarray(config['history_omega_scale']),(11,1)))


def test_block_projection_subset_uses_its_own_row_count():
    systems, factors = [], []
    for index in range(3):
        A = np.arange(15,dtype=float).reshape(5,3)*(index+1)
        Z = np.eye(5)[:,:1]*(1.+index)
        system = dict(A=A,Z=Z,b=np.arange(5,dtype=float))
        U,s,Vt = np.linalg.svd(Z,full_matrices=False)
        systems.append(system); factors.append(dict(U=U,s=s,Vt=Vt))
    full = new.project_cached(systems,factors)
    subset = new.project_cached(systems[:1],factors[:1])
    expected = old.project_systems(systems[:1])
    np.testing.assert_allclose(subset['X'],expected['X'],atol=1e-15)
    np.testing.assert_allclose(full['X'][:5]*np.sqrt(3),subset['X'],atol=2e-15)
    assert full['total_rows'] == 15 and subset['total_rows'] == 5


def test_single_event_information_never_performs_column_projections():
    counts = new._counts(); ledger = dict.fromkeys(new._LIMITS,0)
    info = new._information(np.zeros((40,90)),dict.fromkeys(map(str,core.RANK_THRESHOLDS),10),ledger,counts,event=True)
    assert info['dimension_insufficient'] and info['projected_dimension_bound'] == 30
    assert ledger['information_svd'] == 1 and ledger['conditional_projections'] == 0
    assert 'columns_qualified' not in info


def test_analysis_limit_is_unknown_and_does_not_extend_budget(monkeypatch):
    f = functional(2)
    new._LEDGERS['key'] = dict(new._LIMITS)
    monkeypatch.setattr(new,'_candidate_systems',lambda *args: [dict(A=np.ones((4,2)),A_static=np.ones((4,1)),
        explanation=np.ones((4,2)),Z=np.zeros((4,0)),b=np.zeros(4))])
    prepared = dict(factors={.0025:[dict(U=np.empty((4,0)),s=np.zeros(0),Vt=np.empty((0,0)))]})
    info = new._audit(f,prepared,{},[0],'modulated','key',new._counts())
    assert info['status'] == 'unknown' and new.analysis_budget_snapshot('key')['information_svd']['actual'] == 1878


def test_thresholds_scale_with_k_ref_once():
    counts = old._counts()
    result = old.backward_select(np.ones((2,1)),np.full(2,.07),0.,np.ones(1,dtype=bool),np.zeros(1,dtype=bool),counts,k_ref=2.)
    assert result['raw_k'].tolist() == [0.] and result['edge_mask'].tolist() == [False]


def test_full_preparation_and_subset_proxy_are_reused_without_trajectory_reads(monkeypatch):
    f, n = functional(), 25
    packets = [packet() for _ in range(n)]
    for i,p in enumerate(packets): p['theta_history'][0,0] = i*.001
    calls = []
    def prepare(packet,f,step):
        calls.append(step)
        return dict(A_static=np.zeros((40,45)),Z=np.eye(40)[:,:10],b=np.zeros(40))
    monkeypatch.setattr(core,'prepare_weak_history',prepare)
    monkeypatch.setattr(new,'_information',lambda *args,**kwargs: dict(columns_qualified=[False]*45))
    def select(X,y,sigma,edges,mods,counts,**kwargs):
        capacity = 2*len(edges)*(2*len(edges)+1)+1
        counts['algebra_terms_planned'] += capacity; counts['algebra_terms_skipped'] += capacity
        return dict(status='insufficient_information')
    monkeypatch.setattr(old,'backward_select',select)
    proxy_sizes = []
    def proxy(f,subset,*args,**kwargs):
        proxy_sizes.append(len(subset))
        return dict(status='completed',counts=old._counts(),objective=1.,events=[])
    monkeypatch.setattr(old,'_proxy_candidate',proxy)
    monkeypatch.setattr(new,'_candidate_systems',lambda *args: [])
    monkeypatch.setattr(new,'_project_candidate',lambda *args: dict(X=np.zeros((960,90)),y=np.zeros(960)))
    monkeypatch.setattr(new,'_audit',lambda *args: dict(k_qualified=False,beta_qualified=False))
    a = new.fit_shared(f,packets,mode='static',config=request(f,n))
    b = new.fit_shared(f,packets,mode='modulated',config=request(f,n))
    assert len(calls) == 2*n and proxy_sizes == [24,24]
    assert not a['fit_state']['preparation_cache_hit'] and b['fit_state']['preparation_cache_hit']
    assert a['fit_state']['counts']['reconstructions'] == 50 and b['fit_state']['counts']['reconstructions'] == 0
    assert a['fit_state']['counts']['algebra_terms_planned']+b['fit_state']['counts']['algebra_terms_planned'] == 57337
    assert a['fit_state']['support'] is None and b['fit_state']['modulation_support'] is None
    json.dumps(a['fit_state'],allow_nan=False)
    assert 'dense_history' not in json.dumps(a['fit_state'])


def test_proxy_failure_retains_consumed_counts(monkeypatch):
    f = functional(2)
    prepared = dict(counts=new._counts(),initial_static=dict(status='insufficient_information'),initial_information={})
    monkeypatch.setattr(new,'_prepare',lambda *args: (prepared,True,'key'))
    consumed = old._counts(); consumed.update(optimization_tasks=2,actual_residual_calls=17,rk4_steps_attempted=680)
    monkeypatch.setattr(old,'_proxy_candidate',lambda *args,**kwargs: dict(status='no_output',counts=consumed))
    with pytest.raises(old.FitFailure) as caught:
        new.fit_shared(f,[packet(2)],mode='modulated',config=request(f,1))
    assert caught.value.diagnostics['counts']['actual_residual_calls'] == 17
    assert caught.value.diagnostics['counts']['optimization_tasks'] == 2


@pytest.mark.parametrize('gated',[False,True])
def test_vectorized_rhs_matches_independent_original_loop(gated):
    f = functional()
    if gated:
        f.gate_weight_amp = np.full((10,3),.1); f.gate_weight_mem = np.full((10,3),-.1)
    state = np.linspace(-.08,.1,50)
    theta,w,p,a,r = f.split_state(state)
    mod,di,dj = latent_coupling_modulation(a,f.edges,f.gamma_amp,f.gates)
    active = f.gates.s_edge*f.k_edge
    psi = bregman_kuramoto_potential(f.incidence.T@theta,f.eta_edge)
    ga = f.lambda_amp*a
    for edge,(i,j) in enumerate(f.edges):
        ga[i] += active[edge]*di[edge]*psi[edge]
        ga[j] += active[edge]*dj[edge]*psi[edge]
    expected_grad = (f.incidence@(active*mod*bregman_kuramoto_force(f.incidence.T@theta,f.eta_edge)),f.mass*w,f.storage_p*p,ga,f.lambda_mem*r)
    for actual, expected in zip(f.gradients(state),expected_grad): np.testing.assert_allclose(actual,expected,atol=3e-17,rtol=2e-14)
    ra,rm = f.dissipative_operators(state)
    j = f.gates.s_mem*f.j_omega_mem
    expected_rhs = f.pack_state(w,(-expected_grad[0]-f.damping*w+f.gain_p*expected_grad[2]-j*expected_grad[4])/f.mass,
        -f.gain_p*w-f.rate_p*expected_grad[2],-ra*ga,j*w-rm@expected_grad[4])
    np.testing.assert_allclose(f.rhs(state),expected_rhs,atol=3e-17,rtol=2e-14)
    audit = f.power_balance_audit(state)
    assert audit['power_leakage_abs'] < 1e-14 and audit['strictly_dissipative']
    with pytest.raises(ValueError): f.dissipative_operators(np.full(50,np.nan))
    with pytest.raises(ValueError): f.rhs(np.zeros(40))


def test_local_optimizer_uses_new_scale_mapping_and_actual_budget(monkeypatch):
    f,obs = functional(2),packet(2)
    config = new.make_config(f,'static',request(f,1),1,'key')
    def integrate(f,state,*args,**kwargs):
        states = np.tile(state,(41,1)); states[:,0] += state[-2]-.01
        return states
    monkeypatch.setattr(old,'_integrate',integrate)
    def optimizer(fun,x,jac,**kwargs):
        jac(x)
        pytest.fail('actual three-call budget must interrupt Jac')
    monkeypatch.setattr(old,'least_squares',optimizer)
    result = old.infer_endpoint(f,obs,mode='static',role='proxy',call_limit=3,config=config)
    assert result['actual_residual_calls'] == 3 and result['status'] == 'budget_limited'
    assert result['projected_gradient_kkt'] is None


def test_coverage_uses_only_actual_successful_states_and_never_fills_failures(monkeypatch):
    f,packets = functional(2),[packet(2) for _ in range(3)]
    req = request(f,3)
    key = new._key(f,packets,req)
    config = new.make_config(f,'modulated',req,3,key)
    new._CACHE[key] = {}
    state = f.pack_state(np.zeros(2),np.zeros(2),packets[0]['p_end'],np.zeros(2),np.zeros(2))
    calls = []
    def integrate(f,state,step,duration,counts,**kwargs):
        calls.append(state.copy())
        counts['rk4_steps_attempted'] += 40
        return np.tile(state,(41,1))
    monkeypatch.setattr(old,'_integrate',integrate)
    def audit(f,prepared,candidate,indices,*args):
        assert indices == [0,2] and len(candidate['events']) == 2
        return dict(status='completed',indices=indices)
    monkeypatch.setattr(new,'_audit',audit)
    result = new.coverage_information(f,packets,[state,None,state],config)
    assert result['excluded_indices'] == [1] and len(calls) == 2
    assert result['counts']['information_history_steps'] == 80
    new._CACHE.clear()
    assert new.coverage_information(f,packets,[state,None,state],config)['status'] == 'unknown'


def test_preparation_failure_is_sticky_preserves_cost_and_never_replays(monkeypatch):
    f,packets = functional(2),[packet(2),packet(2)]
    calls = []
    def prepare(*args):
        calls.append(1)
        if len(calls) == 2: raise FloatingPointError('mock partial preparation failure')
        return dict(Z=np.eye(4)[:,:2])
    monkeypatch.setattr(core,'prepare_weak_history',prepare)
    with pytest.raises(old.FitFailure) as first:
        new._prepare(f,packets,request(f,2))
    key = first.value.diagnostics['preparation_key']
    ledger = copy.deepcopy(new._LEDGERS[key])
    with pytest.raises(old.FitFailure) as second:
        new._prepare(f,packets,request(f,2))
    assert len(calls) == 2 and new._LEDGERS[key] == ledger
    assert first.value.diagnostics['counts']['reconstruction_attempts'] == 2
    assert first.value.diagnostics['counts']['reconstructions'] == 1
    assert second.value.diagnostics['counts']['reconstruction_attempts'] == 0
    assert second.value.diagnostics['original_preparation_counts'] == first.value.diagnostics['counts']


def test_coverage_reintegration_failure_is_unknown_with_consumed_steps(monkeypatch):
    f,packets = functional(2),[packet(2)]
    req = request(f,1); key = new._key(f,packets,req)
    new._CACHE[key] = {}; new._LEDGERS[key] = dict.fromkeys(new._LIMITS,0)
    state = f.pack_state(np.zeros(2),np.zeros(2),packets[0]['p_end'],np.zeros(2),np.zeros(2))
    def fail(f,state,step,duration,counts,**kwargs):
        counts['integrations_attempted'] += 1
        counts['rk4_steps_attempted'] += 7
        counts['rk4_steps_completed'] += 6
        raise FloatingPointError('mock information reintegration failure')
    monkeypatch.setattr(old,'_integrate',fail)
    result = new.coverage_information(f,packets,[state],new.make_config(f,'modulated',req,1,key))
    assert result['status'] == 'unknown' and result['failure_index'] == 0
    assert result['counts']['rk4_steps_attempted'] == result['counts']['information_history_steps'] == 7
    assert result['counts']['rk4_steps_completed'] == 6


@pytest.mark.parametrize('failure_type',[ValueError,RuntimeError])
def test_coverage_fine_reintegration_failure_preserves_main_and_partial_fine_cost(monkeypatch,failure_type):
    f,packets = functional(2),[packet(2)]
    req = request(f,1); key = new._key(f,packets,req)
    factor = dict(U=np.empty((40,0)),s=np.empty(0),Vt=np.empty((0,0)))
    grids = {step:[dict(t=np.linspace(-.1,0.,int(round(.1/step))+1))] for step in core.HISTORY_STEPS}
    new._CACHE[key] = dict(grids=grids,factors={step:[factor] for step in core.HISTORY_STEPS})
    new._LEDGERS[key] = dict.fromkeys(new._LIMITS,0)
    state = f.pack_state(np.zeros(2),np.zeros(2),packets[0]['p_end'],np.zeros(2),np.zeros(2))
    calls = []
    def integrate(f,state,step,duration,counts,**kwargs):
        calls.append(step)
        counts['integrations_attempted'] += 1
        if step == .00125:
            counts['rk4_steps_attempted'] += 7
            counts['rk4_steps_completed'] += 6
            raise failure_type('mock fine-grid information integration failure')
        counts['integrations_completed'] += 1
        counts['rk4_steps_attempted'] += 40
        counts['rk4_steps_completed'] += 40
        return np.tile(state,(41,1))
    monkeypatch.setattr(old,'_integrate',integrate)
    static = np.eye(40)[:,:1]
    system = dict(A_static=static,A=np.hstack((static,static)),
        explanation=np.hstack((static,np.zeros_like(static))),Z=np.empty((40,0)),b=np.zeros(40))
    monkeypatch.setattr(core,'conditional_dictionary',lambda *args,**kwargs:system)
    result = new.coverage_information(f,packets,[state],new.make_config(f,'modulated',req,1,key))
    assert calls == [.0025,.00125] and result['status'] == 'unknown'
    assert failure_type.__name__ in result['reason'] and '0.0025' in result['grids']
    assert result['counts']['information_history_steps'] == result['counts']['rk4_steps_attempted'] == 47
    assert result['counts']['rk4_steps_completed'] == 46
    assert result['counts']['integrations_attempted'] == 2 and result['counts']['integrations_completed'] == 1
