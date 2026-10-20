"""Synthetic-only physical extrema, gradients, causal integration and Adam QA."""
import copy
import math
from pathlib import Path
import sys
import numpy as np
import pytest
import torch

sys.path[:0] = [str(Path(__file__).resolve().parents[1]/'experiments'),str(Path(__file__).resolve().parents[1]/'src')]
import spatial_extrema_runtime_20261020 as spatial
from closure_training_runtime import TrainingClosureROM,TensorEventStore,objective as original_objective
from closure_discovery_benchmarks import toy_source
from linear_cache_runtime_20261019 import enable_linear_cache
from symbolic_closure_rom import _BASE_TERMS
from train_local_memory_0098 import graph
from train_structured_rom import METRIC

DEVICES = ['cpu']+(['cuda'] if torch.cuda.is_available() else [])


def model_fixture(device):
    base = toy_source().with_terms([_BASE_TERMS[i] for i in (0,1,3,4,5,7,8)])
    model = TrainingClosureROM.from_config(base._export_payload()['config'],terms=base.terms)
    model.load_state_dict(base.state_dict())
    flags = {n:p.requires_grad for n,p in base.named_parameters()}
    for n,p in model.named_parameters(): p.requires_grad_(flags[n])
    with torch.no_grad(): model.coefficients.copy_(torch.linspace(-.02,.03,7,dtype=torch.float64))
    return enable_linear_cache(model.to(device))


def fixtures(device):
    rng = np.random.default_rng(20391)
    events = [{'id':str(i),'x':rng.normal(0,.03,(n,30)),'torque':rng.normal(0,.01,(n,10))} for i,n in enumerate((12,16,28))]
    store = TensorEventStore(events,device)
    samples = [(e,i) for e,i in zip(events,(10,11,13))]
    eq = torch.linspace(-.2,.3,10,dtype=torch.float64,device=device)
    return events,store,samples,eq


def compare_grads(a,b):
    for (n,p),(nn,q) in zip(a.named_parameters(),b.named_parameters()):
        assert n == nn and (p.grad is None) == (q.grad is None),n
        if p.grad is not None: torch.testing.assert_close(p.grad,q.grad,rtol=2e-10,atol=2e-10,msg=n)


def hand_point(pred,truth,mass,eq):
    def extrema(row):
        angles = [row[i]+eq[i] for i in range(10)]
        common = sum(row[10+i]*mass[i] for i in range(10))/sum(mass)
        return max(angles)-min(angles),max(abs(row[10+i]-common) for i in range(10))
    a,w=extrema(pred); at,wt=extrema(truth)
    x=((a-at)/(2*METRIC[0]))**2; y=((w-wt)/METRIC[1])**2
    return .5*(x+y),x,y


@pytest.mark.parametrize('device',DEVICES)
def test_physical_eq_coi_shift_permutation_and_hand_formula(device):
    rng=np.random.default_rng(821)
    pred=torch.tensor(rng.normal(size=(3,30)),dtype=torch.float64,device=device)
    truth=torch.tensor(rng.normal(size=(3,30)),dtype=torch.float64,device=device)
    mass=torch.linspace(1.,3.,10,dtype=torch.float64,device=device)
    eq=torch.linspace(-.7,.9,10,dtype=torch.float64,device=device)
    actual=spatial.extrema_error(pred,truth,mass,eq)
    expected=[hand_point(p,t,mass.tolist(),eq.tolist()) for p,t in zip(pred.tolist(),truth.tolist())]
    for i,value in enumerate(actual): torch.testing.assert_close(value,value.new_tensor([x[i] for x in expected]),rtol=2e-12,atol=2e-12)
    shifted_pred,shifted_truth=pred.clone(),truth.clone()
    shifted_pred[:,:10]+=1.7; shifted_truth[:,:10]+=1.7
    for a,b in zip(actual,spatial.extrema_error(shifted_pred,shifted_truth,mass,eq)): torch.testing.assert_close(a,b,rtol=2e-12,atol=2e-12)
    perm=torch.tensor([6,1,8,3,0,9,4,7,2,5],device=device)
    ix=torch.cat([perm,perm+10,perm+20])
    for a,b in zip(actual,spatial.extrema_error(pred[:,ix],truth[:,ix],mass[perm],eq[perm])): torch.testing.assert_close(a,b,rtol=2e-12,atol=2e-12)
    p=torch.zeros((1,30),dtype=torch.float64,device=device); t=p.clone(); t[:,:10]=-eq
    _,angle,_=spatial.extrema_error(p,t,mass,eq)
    torch.testing.assert_close(angle,angle.new_tensor([((.9-(-.7))/(2*METRIC[0]))**2]),rtol=2e-12,atol=2e-12)


def test_ties_share_symmetric_subgradient_abs_zero_and_finite_difference():
    mass=torch.ones(10,dtype=torch.float64); eq=torch.zeros(10,dtype=torch.float64)
    p=torch.zeros((1,30),dtype=torch.float64); p[0,:4]=torch.tensor([1.,1.,-1.,-1.]); p[0,10:12]=torch.tensor([1.,-1.]); p.requires_grad_()
    truth=torch.zeros_like(p)
    spatial.extrema_error(p,truth,mass,eq)[0].sum().backward()
    assert p.grad[0,0] == p.grad[0,1] and p.grad[0,2] == p.grad[0,3]
    torch.testing.assert_close(p.grad[0,10],-p.grad[0,11],rtol=0,atol=1e-14)
    zero=torch.zeros((1,30),dtype=torch.float64,requires_grad=True)
    spatial.extrema_error(zero,truth,mass,eq)[0].sum().backward()
    assert torch.count_nonzero(zero.grad) == 0
    rng=np.random.default_rng(73)
    p=torch.tensor(rng.normal(size=(1,30)),dtype=torch.float64,requires_grad=True)
    t=torch.tensor(rng.normal(size=(1,30)),dtype=torch.float64)
    spatial.extrema_error(p,t,mass,eq)[0].sum().backward()
    for j in range(20):
        plus,minus=p.detach().clone(),p.detach().clone(); plus[0,j]+=1e-6; minus[0,j]-=1e-6
        fd=(spatial.extrema_error(plus,t,mass,eq)[0]-spatial.extrema_error(minus,t,mass,eq)[0])/(2e-6)
        torch.testing.assert_close(p.grad[0,j],fd[0],rtol=2e-6,atol=2e-8)


@pytest.mark.parametrize('device',DEVICES)
def test_o_original_components_all_gradients_and_adam_exact(device):
    torch.set_num_threads(1)
    a=model_fixture(device); b=copy.deepcopy(a); _,store,samples,eq=fixtures(device)
    oa=torch.optim.Adam(a.parameters(),lr=9e-5); ob=torch.optim.Adam(b.parameters(),lr=9e-5)
    cpu_rng=torch.get_rng_state().clone()
    cuda_rng=torch.cuda.get_rng_state().clone() if device=='cuda' else None
    with graph(a): old,ot=original_objective(a,store,samples,.07,.01); old.backward()
    with graph(b): new,nt=spatial.objective(b,store,samples,.07,.01,eq,False); new.backward()
    torch.testing.assert_close(old,new,rtol=0,atol=0)
    for k in ot: torch.testing.assert_close(ot[k],nt[k],rtol=0,atol=0)
    compare_grads(a,b)
    for m,opt in ((a,oa),(b,ob)):
        torch.nn.utils.clip_grad_norm_([p for p in m.parameters() if p.requires_grad],100.); opt.step()
    for p,q in zip(a.parameters(),b.parameters()): torch.testing.assert_close(p,q,rtol=2e-12,atol=2e-12)
    for sa,sb in zip(oa.state.values(),ob.state.values()):
        for k in sa: torch.testing.assert_close(sa[k],sb[k],rtol=2e-12,atol=2e-12)
    assert torch.equal(torch.get_rng_state(),cpu_rng)
    if device=='cuda': assert torch.equal(torch.cuda.get_rng_state(),cuda_rng)


@pytest.mark.parametrize('device',DEVICES)
def test_s_single_long_manual_per_event_and_auxiliary_match(device,monkeypatch):
    a=model_fixture(device); b=copy.deepcopy(a); events,store,samples,eq=fixtures(device)
    per_event=[]; counts=[]
    with torch.no_grad(),graph(a):
        for e in events:
            h=store.events[e['id']][0][:11][None]; z=torch.cat((h[:,-1],a.encode(h)),-1)
            n=min(7,len(e['x'])-11); errors=[]; counts.append(n)
            for k in range(n):
                z=a.rk4(z,.01)
                errors.append(hand_point(z[0].tolist(),store.events[e['id']][0][11+k].tolist(),a.mass.tolist(),eq.tolist())[0])
            per_event.append(sum(errors)/n)
    expected=.8*sum(per_event)/len(per_event)+.2*max(per_event)
    calls=[]; original_encode=b.encode
    def encode(h): calls.append(h.detach().clone()); return original_encode(h)
    monkeypatch.setattr(b,'encode',encode)
    with graph(a): old,ot=original_objective(a,store,samples,.07,.01)
    with graph(b): new,nt=spatial.objective(b,store,samples,.07,.01,eq,True)
    assert counts == [1,5,7] and len(calls) == 3
    torch.testing.assert_close(new,old+10*old.new_tensor(expected),rtol=2e-12,atol=2e-12)
    for k in ('local','latent_consistency','torque'):
        torch.testing.assert_close(ot[k],nt[k],rtol=0,atol=0)
        a.zero_grad(set_to_none=True); b.zero_grad(set_to_none=True)
        ot[k].backward(retain_graph=True); nt[k].backward(retain_graph=True); compare_grads(a,b)
    torch.testing.assert_close(calls[0],store.histories([(e,10) for e in events]),rtol=0,atol=0)
    torch.testing.assert_close(calls[1],store.histories(samples),rtol=0,atol=0)
    torch.testing.assert_close(calls[2],store.histories(samples,adjacent=True),rtol=0,atol=0)
    batch=store.prepare([(e,10) for e in events],.07)
    with graph(b):
        shared=spatial.spatial_rollout_errors(b,batch,eq)[1]
        batched=spatial.spatial_rollout_errors(b,batch,eq.expand(3,-1))[1]
    torch.testing.assert_close(shared,batched,rtol=0,atol=0)


def test_five_seconds_is_capped_by_real_points():
    events=[{'id':str(n),'x':np.zeros((n,30)),'torque':np.zeros((n,10))} for n in (12,510,520)]
    batch=TensorEventStore(events).prepare([(e,10) for e in events],5.)
    assert batch.counts.tolist() == [1.,499.,500.]


@pytest.mark.parametrize('fault',['eq','parameter','gradient'])
def test_bad_physical_inputs_and_nonfinite_updates_clear_cache(monkeypatch,fault):
    model=model_fixture('cpu'); _,store,samples,eq=fixtures('cpu'); opt=torch.optim.Adam(model.parameters(),lr=9e-5)
    hook=None
    if fault=='eq': eq[0]=float('nan')
    elif fault=='parameter':
        with torch.no_grad(): model.coefficients[0]=float('nan')
    else: hook=model.coefficients.register_hook(lambda g:g*float('nan'))
    try:
        with pytest.raises((ValueError,FloatingPointError)):
            spatial.one_update(model,opt,store,samples,.07,.01,eq,True)
    finally:
        if hook is not None: hook.remove()
    assert not opt.state and model._runtime_cache is None and model._linear_cache is None
    with pytest.raises(ValueError): spatial.extrema_error(torch.zeros(2,30),torch.zeros(2,30),torch.ones(10),None)
