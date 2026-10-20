"""Synthetic-only reference arithmetic and original-objective equivalence."""
import copy
from pathlib import Path
import sys

import numpy as np
import pytest
import torch

sys.path[:0] = [str(Path(__file__).resolve().parents[1]/'experiments'), str(Path(__file__).resolve().parents[1]/'src')]
import temporal_objective_runtime_20261020 as temporal
from closure_training_runtime import TrainingClosureROM, TensorEventStore, PreparedRollout, objective as original_objective
from closure_discovery_benchmarks import toy_source
from linear_cache_runtime_20261019 import enable_linear_cache
from symbolic_closure_rom import _BASE_TERMS
from train_local_memory_0098 import graph, risk
from train_structured_rom import _point_error

DEVICES = ['cpu'] + (['cuda'] if torch.cuda.is_available() else [])


def model_fixture(device):
    base = toy_source().with_terms([_BASE_TERMS[i] for i in (0,1,3,4,5,7,8)])
    model = TrainingClosureROM.from_config(base._export_payload()['config'], terms=base.terms)
    model.load_state_dict(base.state_dict())
    flags = {n:p.requires_grad for n,p in base.named_parameters()}
    for n,p in model.named_parameters(): p.requires_grad_(flags[n])
    with torch.no_grad(): model.coefficients.copy_(torch.linspace(-.02,.03,7,dtype=torch.float64))
    return enable_linear_cache(model.to(device))


def event_fixture(device):
    rng = np.random.default_rng(20301)
    events = [{'id': f'synthetic_{i}', 'group': 'synthetic', 'label': 'synthetic',
               'x': rng.normal(0,.025,size=(length,30)), 'torque': rng.normal(0,.01,size=(length,10))}
              for i,length in enumerate((14,18,28))]
    store = TensorEventStore(events,device)
    samples = [(e,11+i) for i,e in enumerate(events)]
    return events, store, samples


def grads(model):
    return {n:None if p.grad is None else p.grad.clone() for n,p in model.named_parameters()}


def assert_grads(a,b):
    assert a.keys() == b.keys()
    for n in a:
        assert (a[n] is None) == (b[n] is None), n
        if a[n] is not None: torch.testing.assert_close(a[n],b[n],rtol=2e-10,atol=2e-10,msg=n)


class ScalarDynamics(torch.nn.Module):
    def __init__(self,device):
        super().__init__(); self.alpha = torch.nn.Parameter(torch.tensor(.7,dtype=torch.float64,device=device)); self.calls = 0
        self.register_buffer('direction',torch.nn.functional.one_hot(torch.tensor(0,device=device),40).double())
    def encode(self,h):
        self.calls += 1; return h.new_zeros((len(h),10))
    def rk4(self,z,dt):
        return z + dt*self.alpha*self.direction


@pytest.mark.parametrize('device', DEVICES)
def test_independent_hand_weights_ragged_state_and_parameter_gradients(monkeypatch,device):
    import train_structured_rom
    monkeypatch.setattr(train_structured_rom,'_point_error',lambda z,t,m:(z[:,0]-t[:,0]).square())
    counts = [1,4,5,6,11]
    events = []
    for i,n in enumerate(counts):
        x = np.zeros((11+n,30)); x[11:,0] = np.arange(1,n+1)*.2
        events.append({'id':str(i),'x':x,'torque':np.zeros((len(x),10))})
    store = TensorEventStore(events,device); batch = store.prepare([(e,10) for e in events],1.)
    model = ScalarDynamics(device)
    initial = torch.zeros((5,40),dtype=torch.float64,device=device,requires_grad=True)
    v,a,b = temporal.temporal_rollout_errors(model,batch,initial)
    alpha = model.alpha.detach().clone().requires_grad_(); start = initial.detach().clone().requires_grad_()
    av,bv,vv = [],[],[]
    for i,n in enumerate(counts):
        errors = [(start[i,0]+k*.01*alpha-k*.2).square() for k in range(1,n+1)]
        mean = sum(errors)/n; tail_size = (n+4)//5; tail = sum(errors[-tail_size:])/tail_size
        av.append(mean); bv.append(tail); vv.append(.5*mean+.5*tail)
    for actual,expected in ((a,av),(b,bv),(v,vv)):
        torch.testing.assert_close(actual,torch.stack(expected),rtol=1e-12,atol=1e-12)
    assert v[0] == a[0] and v[0] == b[0]
    # n=6 has a two-point tail across the segment ending at5: both5 and6 count.
    v.sum().backward(); sum(vv).backward()
    torch.testing.assert_close(model.alpha.grad,alpha.grad,rtol=1e-12,atol=1e-12)
    torch.testing.assert_close(initial.grad,start.grad,rtol=1e-12,atol=1e-12)
    assert model.calls == 0
    temporal.temporal_rollout_errors(model,batch)
    assert model.calls == 1


@pytest.mark.parametrize('device', DEVICES)
def test_original_values_components_gradients_and_real_adam_match(device):
    torch.set_num_threads(1)
    a = model_fixture(device); b = copy.deepcopy(a)
    _,store,samples = event_fixture(device)
    oa = torch.optim.Adam([p for p in a.parameters() if p.requires_grad],lr=9e-5)
    ob = torch.optim.Adam([p for p in b.parameters() if p.requires_grad],lr=9e-5)
    before_cpu = torch.get_rng_state().clone()
    before_cuda = torch.cuda.get_rng_state().clone() if device == 'cuda' else None
    with graph(a):
        old,ot = original_objective(a,store,samples,.12,.01); old.backward()
    with graph(b):
        new,nt = temporal.objective(b,store,samples,.12,.01,False); new.backward()
    torch.testing.assert_close(old,new,rtol=0,atol=0)
    for k in ot: torch.testing.assert_close(ot[k],nt[k],rtol=0,atol=0)
    assert nt['time_mean'] == nt['mean']
    assert_grads(grads(a),grads(b))
    for m,opt in ((a,oa),(b,ob)):
        torch.nn.utils.clip_grad_norm_([p for p in m.parameters() if p.requires_grad],100.); opt.step()
    for (n,p),(nr,q) in zip(a.named_parameters(),b.named_parameters()):
        assert n == nr; torch.testing.assert_close(p,q,rtol=2e-12,atol=2e-12)
    for pa,pb in zip(oa.state.values(),ob.state.values()):
        for k in pa: torch.testing.assert_close(pa[k],pb[k],rtol=2e-12,atol=2e-12)
    assert torch.equal(torch.get_rng_state(),before_cpu)
    if device == 'cuda': assert torch.equal(torch.cuda.get_rng_state(),before_cuda)


@pytest.mark.parametrize('device', DEVICES)
def test_one_update_matches_old_update_adam_and_clears_cache(device):
    from linear_cache_cost_gate_20261019 import one_update as old_update
    a = model_fixture(device); b = copy.deepcopy(a)
    _,store,samples = event_fixture(device)
    oa = torch.optim.Adam([p for p in a.parameters() if p.requires_grad],lr=9e-5)
    ob = torch.optim.Adam([p for p in b.parameters() if p.requires_grad],lr=9e-5)
    expected = old_update(a,oa,store,samples,.12,.01)
    actual = temporal.one_update(b,ob,store,samples,.12,.01,False)
    assert expected['loss'] == actual['loss'] and expected['gradient_norm'] == actual['gradient_norm']
    for k,v in expected['components'].items(): assert actual['components'][k] == v
    for p,q in zip(a.parameters(),b.parameters()): torch.testing.assert_close(p,q,rtol=2e-12,atol=2e-12)
    for pa,pb in zip(oa.state.values(),ob.state.values()):
        for k in pa: torch.testing.assert_close(pa[k],pb[k],rtol=2e-12,atol=2e-12)
    assert b._runtime_cache is None and b._linear_cache is None


def independent_long(model,events,store,horizon):
    values = []; means = []; tails = []
    # Separate single-event integration, without PreparedRollout or its masks.
    for e in events:
        history = store.events[e['id']][0][:11].unsqueeze(0)
        z = torch.cat((history[:,-1],model.encode(history)),-1)
        n = min(round(horizon/.01),len(e['x'])-11); errors = []
        for k in range(n):
            z = model.rk4(z,.01)
            errors.append(_point_error(z,store.events[e['id']][0][11+k].unsqueeze(0),model)[0])
        a = torch.stack(errors).mean(); b = torch.stack(errors[-((n+4)//5):]).mean()
        values.append(.5*a+.5*b); means.append(a); tails.append(b)
    return torch.stack(values),torch.stack(means),torch.stack(tails)


@pytest.mark.parametrize('device', DEVICES)
def test_tail_objective_against_single_event_reference_and_unchanged_auxiliaries(device):
    a = model_fixture(device); b = copy.deepcopy(a)
    events,store,samples = event_fixture(device)
    with graph(a):
        _,original_terms = original_objective(a,store,samples,.12,.01)
        long,means,tails = independent_long(a,events,store,.12)
        expected = 10*risk(long)+original_terms['local']+.1*original_terms['latent_consistency']+.1*original_terms['torque']
        expected.backward()
    with graph(b):
        actual,terms = temporal.objective(b,store,samples,.12,.01,True); actual.backward()
    torch.testing.assert_close(actual,expected,rtol=2e-10,atol=2e-10)
    for k in ('local','latent_consistency','torque','complexity'):
        torch.testing.assert_close(terms[k],original_terms[k],rtol=0,atol=0)
    torch.testing.assert_close(terms['time_mean'],means.mean(),rtol=2e-12,atol=2e-12)
    torch.testing.assert_close(terms['tail_mean'],tails.mean(),rtol=2e-12,atol=2e-12)
    assert_grads(grads(a),grads(b))


def test_auxiliary_gradients_and_encoding_calls_stay_separate(monkeypatch):
    a = model_fixture('cpu'); b = copy.deepcopy(a)
    events,store,samples = event_fixture('cpu')
    calls = []; original_encode = b.encode
    def encode(h):
        calls.append(h.detach().clone()); return original_encode(h)
    monkeypatch.setattr(b,'encode',encode)
    with graph(a): _,ot = original_objective(a,store,samples,.12,.01)
    with graph(b): _,nt = temporal.objective(b,store,samples,.12,.01,True)
    assert len(calls) == 3  # one long, original local, original adjacent helper
    torch.testing.assert_close(calls[0],store.histories([(e,10) for e in events]),rtol=0,atol=0)
    torch.testing.assert_close(calls[1],store.histories(samples),rtol=0,atol=0)
    torch.testing.assert_close(calls[2],store.histories(samples,adjacent=True),rtol=0,atol=0)
    for k in ('local','latent_consistency','torque'):
        a.zero_grad(set_to_none=True); b.zero_grad(set_to_none=True)
        ot[k].backward(retain_graph=True); nt[k].backward(retain_graph=True)
        assert_grads(grads(a),grads(b))


def test_one_update_errors_clear_cache_and_do_not_step(monkeypatch):
    model = model_fixture('cpu'); _,store,samples = event_fixture('cpu')
    opt = torch.optim.Adam(model.parameters(),lr=9e-5)
    before = {n:p.clone() for n,p in model.named_parameters()}
    for horizon,force,flag in ((0,.01,False),(.1,0,False),(.1,.01,'T')):
        with pytest.raises(ValueError): temporal.one_update(model,opt,store,samples,horizon,force,flag)
        assert model._runtime_cache is None and model._linear_cache is None
    def bad(*args,**kwargs):
        raise RuntimeError('synthetic objective failure')
    monkeypatch.setattr(temporal,'objective',bad)
    with pytest.raises(RuntimeError,match='synthetic objective failure'):
        temporal.one_update(model,opt,store,samples,.1,.01,True)
    assert model._runtime_cache is None and model._linear_cache is None
    for n,p in model.named_parameters(): assert torch.equal(before[n],p)


@pytest.mark.parametrize('fault', ['value', 'gradient'])
def test_nonfinite_update_stops_before_adam_and_closes_cache(monkeypatch,fault):
    model = model_fixture('cpu'); _,store,samples = event_fixture('cpu')
    opt = torch.optim.Adam(model.parameters(),lr=9e-5)
    before = {n:p.clone() for n,p in model.named_parameters()}
    handle = None
    if fault == 'value':
        def bad(*args,**kwargs):
            v = model.coefficients.sum()*float('nan'); return v,{'risk':v}
        monkeypatch.setattr(temporal,'objective',bad)
    else:
        handle = model.coefficients.register_hook(lambda grad: grad*float('nan'))
    try:
        with pytest.raises(FloatingPointError): temporal.one_update(model,opt,store,samples,.12,.01,True)
    finally:
        if handle is not None: handle.remove()
    assert model._runtime_cache is None and model._linear_cache is None
    assert not opt.state
    for n,p in model.named_parameters(): assert torch.equal(before[n],p)


def test_bad_genuine_counts_and_no_implicit_padding():
    model = ScalarDynamics('cpu')
    h = torch.zeros((1,11,30),dtype=torch.float64)
    truth = torch.zeros((1,1,30),dtype=torch.float64); ix = torch.tensor([0])
    for counts in (torch.tensor([0.]),torch.tensor([1.5]),torch.tensor([2.])):
        with pytest.raises(ValueError):
            temporal.temporal_rollout_errors(model,PreparedRollout(h,[(ix,truth)],counts))
    assert model.calls == 0


def test_noncanonical_segments_rejected_even_when_total_coverage_matches():
    model = ScalarDynamics('cpu')
    h = torch.zeros((2,11,30),dtype=torch.float64)
    counts = torch.tensor([1.,2.],dtype=torch.float64)
    # Row1 appears twice in interval1 and row0 appears late in interval2.
    # Total index-add coverage still equals [1,2], hiding the broken timeline.
    late = [(torch.tensor([1,1]),torch.zeros((1,2,30),dtype=torch.float64)),
            (torch.tensor([0]),torch.zeros((1,1,30),dtype=torch.float64))]
    with pytest.raises(ValueError,match='ascending active rows'):
        temporal.temporal_rollout_errors(model,PreparedRollout(h,late,counts))
    # Correct membership but a reversed row order also violates store.prepare.
    reversed_rows = [(torch.tensor([1,0]),torch.zeros((1,2,30),dtype=torch.float64)),
                     (torch.tensor([1]),torch.zeros((1,1,30),dtype=torch.float64))]
    with pytest.raises(ValueError,match='ascending active rows'):
        temporal.temporal_rollout_errors(model,PreparedRollout(h,reversed_rows,counts))
    assert model.calls == 0
