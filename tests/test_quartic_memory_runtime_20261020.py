"""Synthetic-only quartic energy, projected Adam, portability and causal QA."""
import copy
import json
from pathlib import Path
import sys
import numpy as np
import pytest
import torch
sys.path[:0]=[str(Path(__file__).resolve().parents[1]/'experiments'),str(Path(__file__).resolve().parents[1]/'src')]
import quartic_memory_runtime_20261020 as qrt
from closure_discovery_benchmarks import toy_source
from closure_training_runtime import TrainingClosureROM,TensorEventStore
from symbolic_closure_rom import SymbolicClosureROM,numpy_field,term_key
from train_closure import make_optimizer
from train_local_memory_0098 import graph

DEVICES=['cpu']+(['cuda'] if torch.cuda.is_available() else [])


def source(device):
    base=toy_source().with_terms(qrt.TERMS['N'])
    m=TrainingClosureROM.from_config(base._export_payload()['config'],terms=base.terms)
    m.load_state_dict(base.state_dict())
    flags={n:p.requires_grad for n,p in base.named_parameters()}
    for n,p in m.named_parameters():p.requires_grad_(flags[n])
    with torch.no_grad():m.coefficients.copy_(torch.linspace(-.03,.05,7,dtype=torch.float64))
    return m.to(device)


def generic(model):
    payload=copy.deepcopy(model).cpu()._export_payload()
    m=SymbolicClosureROM.from_config(payload['config'],terms=payload['terms']);m.load_state_dict(model.state_dict())
    flags={n:p.requires_grad for n,p in model.named_parameters()}
    for n,p in m.named_parameters():p.requires_grad_(flags[n])
    return m.to(model.mass.device)


def inputs(device):
    rng=np.random.default_rng(202611)
    events=[{'id':str(i),'x':rng.normal(0,.08,(n,30)),'torque':rng.normal(0,.01,(n,10))} for i,n in enumerate((12,17,25))]
    return events,TensorEventStore(events,device),[(e,10) for e in events]


def set_a(model,a):
    with torch.no_grad():model.coefficients[qrt.quartic_index(model)].fill_(a)


@pytest.mark.parametrize('device',DEVICES)
def test_nonzero_energy_field_numpy_finite_difference_and_origin(device):
    n=qrt.build_variant(source(device),'N');q=qrt.build_variant(source(device),'Q');set_a(q,.17)
    z=torch.randn(3,40,dtype=torch.float64,device=device)*.22
    ref=generic(q);x=q.interface_to_state(z).detach().requires_grad_()
    hn,gn=n.energy_and_gradient(x);hq,gq=q.energy_and_gradient(x)
    torch.testing.assert_close(hq-hn,.17*x[...,30:].pow(4).sum(-1),rtol=2e-11,atol=2e-12)
    expected=torch.zeros_like(x);expected[...,30:]=4*.17*x[...,30:].pow(3)
    torch.testing.assert_close(gq-gn,expected,rtol=2e-11,atol=2e-12)
    torch.testing.assert_close(torch.autograd.grad(hq.sum(),x)[0],gq,rtol=2e-11,atol=2e-11)
    torch.testing.assert_close(q.field(z),ref.field(z),rtol=2e-11,atol=2e-11)
    np.testing.assert_allclose(q.field(z).detach().cpu(),numpy_field(copy.deepcopy(ref).cpu(),z.cpu()),rtol=2e-10,atol=2e-10)
    a=q.coefficients[qrt.quartic_index(q)].detach().clone();eps=1e-6
    set_a(q,a+eps);plus=q.field(z);set_a(q,a-eps);minus=q.field(z);set_a(q,a)
    grad=torch.autograd.grad(q.field(z).sum(),q.coefficients)[0][qrt.quartic_index(q)]
    torch.testing.assert_close(grad,((plus-minus)/(2*eps)).sum(),rtol=2e-6,atol=2e-8)
    zero=torch.zeros(40,dtype=torch.float64,device=device,requires_grad=True)
    def delta_energy(x):return q.energy_and_gradient(x)[0]-n.energy_and_gradient(x)[0]
    torch.testing.assert_close(torch.autograd.functional.jacobian(delta_energy,zero),torch.zeros_like(zero),rtol=0,atol=2e-12)
    torch.testing.assert_close(torch.autograd.functional.hessian(delta_energy,zero),zero.new_zeros(40,40),rtol=0,atol=2e-12)
    shifted=z.clone();shifted[:,:10]+=.7
    torch.testing.assert_close(q.field(z),q.field(shifted),rtol=2e-11,atol=2e-11)
    perm=torch.tensor([3,1,8,0,5,9,2,7,6,4],device=device)
    xp=x.detach().clone();xp[...,30:]=x.detach()[...,30:][:,perm]
    dh=hq-hn;dgp=q.energy_and_gradient(xp)[1]-n.energy_and_gradient(xp)[1]
    torch.testing.assert_close(q.energy_and_gradient(xp)[0]-n.energy_and_gradient(xp)[0],dh,rtol=2e-11,atol=2e-12)
    torch.testing.assert_close(dgp[...,30:],expected[...,30:][:,perm],rtol=2e-11,atol=2e-12)


@pytest.mark.parametrize('device',DEVICES)
def test_zero_a_unlocked_and_common_raw_gradients_only(device):
    src=source(device);n=qrt.build_variant(src,'N');q=qrt.build_variant(src,'Q')
    events,store,samples=inputs(device)
    z=torch.randn(3,40,dtype=torch.float64,device=device)*.3
    torch.testing.assert_close(n.field(z),q.field(z),rtol=0,atol=0)
    torch.testing.assert_close(n.encode(store.histories(samples)),q.encode(store.histories(samples)),rtol=0,atol=0)
    for m in (n,q):
        with graph(m):loss,_=qrt.objective(m,store,samples,.06,.01);loss.backward()
    pn,pq=dict(n.named_parameters()),dict(q.named_parameters())
    for name in pn:
        if name=='coefficients':continue
        assert (pn[name].grad is None)==(pq[name].grad is None)
        if pn[name].grad is not None:torch.testing.assert_close(pn[name].grad,pq[name].grad,rtol=2e-10,atol=2e-10)
    for i,t in enumerate(n.terms):
        j=next(j for j,u in enumerate(q.terms) if term_key(u)==term_key(t))
        torch.testing.assert_close(n.coefficients.grad[i],q.coefficients.grad[j],rtol=2e-10,atol=2e-10)
    q.zero_grad(set_to_none=True)
    g=torch.autograd.grad(q.field(z).square().sum(),q.coefficients)[0][qrt.quartic_index(q)]
    assert abs(float(g))>1e-12


@pytest.mark.parametrize('device',DEVICES)
def test_fast_generic_full_raw_clip_adam_projection_match(device):
    fast=qrt.build_variant(source(device),'Q');set_a(fast,.07);ref=generic(fast)
    _,store,samples=inputs(device);oa,ob=make_optimizer(fast),make_optimizer(ref)
    for m in (fast,ref):
        with graph(m):value,_=qrt.objective(m,store,samples,.06,.01);value.backward()
    for p,q in zip(fast.parameters(),ref.parameters()):
        assert (p.grad is None)==(q.grad is None)
        if p.grad is not None:torch.testing.assert_close(p.grad,q.grad,rtol=2e-9,atol=2e-9)
    for m,opt in ((fast,oa),(ref,ob)):
        torch.nn.utils.clip_grad_norm_([p for p in m.parameters() if p.requires_grad],100.);opt.step();qrt.project_quartic(m,opt)
    for p,q in zip(fast.parameters(),ref.parameters()):torch.testing.assert_close(p,q,rtol=2e-9,atol=2e-9)
    for sa,sb in zip(oa.state.values(),ob.state.values()):
        for k in sa:torch.testing.assert_close(sa[k],sb[k],rtol=2e-9,atol=2e-9)


def test_projection_identity_order_group_moments_and_prefinite(tmp_path):
    model=qrt.build_variant(source('cpu'),'Q').with_terms([qrt.QUARTIC_TERM]+list(reversed(qrt.TERMS['N'])))
    i=qrt.quartic_index(model);assert i==0
    opt=make_optimizer(model)
    assert next(g['block'] for g in opt.param_groups if any(p is model.coefficients for p in g['params']))=='closure'
    model.coefficients.sum().backward();opt.step()
    old=model.coefficients.detach().clone();moments=copy.deepcopy(opt.state_dict())
    result=qrt.project_quartic(model,opt)
    assert result['projected'] and result['after']==0.
    assert torch.equal(model.coefficients[1:],old[1:])
    for key,value in opt.state_dict()['state'].items():
        for name,t in value.items():assert torch.equal(t,moments['state'][key][name])
    model.export(tmp_path/'model.json');p=json.loads((tmp_path/'model.json').read_text())
    assert p['coefficients']==p['state_dict']['coefficients']
    bad=copy.deepcopy(p);bad['coefficients'][i]=.2;(tmp_path/'bad.json').write_text(json.dumps(bad))
    with pytest.raises(ValueError):qrt.QuarticMemoryRuntime.from_export(tmp_path/'bad.json')
    set_a(model,-.1)
    with pytest.raises(FloatingPointError):model.export(tmp_path/'negative.json')
    set_a(model,float('-inf'))
    with pytest.raises(FloatingPointError):qrt.project_quartic(model,opt)


@pytest.mark.parametrize('device',DEVICES)
def test_roundtrip_nonempty_adam_next_update_and_single_encode(tmp_path,device,monkeypatch):
    model=qrt.build_variant(source(device),'Q');set_a(model,.03);opt=make_optimizer(model)
    _,store,samples=inputs(device)
    qrt.one_update(model,opt,store,samples,.06,.01)
    assert opt.state
    model.export(tmp_path/'model.json');torch.save(opt.state_dict(),tmp_path/'adam.pt')
    loaded=qrt.QuarticMemoryRuntime.from_export(tmp_path/'model.json').to(device);other=make_optimizer(loaded)
    other.load_state_dict(torch.load(tmp_path/'adam.pt',map_location=device,weights_only=False))
    a=qrt.one_update(model,opt,store,samples,.06,.01);b=qrt.one_update(loaded,other,store,samples,.06,.01)
    assert a['quartic']['after']>=0 and b['quartic']['after']>=0
    for p,q in zip(model.parameters(),loaded.parameters()):torch.testing.assert_close(p,q,rtol=2e-10,atol=2e-10)
    calls=[];enc=loaded.encode
    def encode(h):calls.append(h.shape);return enc(h)
    monkeypatch.setattr(loaded,'encode',encode)
    with graph(loaded):loaded.rollout(store.histories(samples),2,.01)
    assert len(calls)==1
    calls.clear()
    with graph(loaded):qrt.objective(loaded,store,samples,.06,.01)
    assert len(calls)==3
    assert store.prepare([(samples[0][0],10)],5.).counts.tolist()==[1.]


def test_runtime_activation_cache_and_zero_source_immutable():
    src=source('cpu');state={k:v.clone() for k,v in src.state_dict().items()}
    q=qrt.build_variant(src,'Q');ids=[id(p) for p in q.parameters()]
    qrt.enable_runtime(q);assert ids==[id(p) for p in q.parameters()]
    for k,v in src.state_dict().items():assert torch.equal(v,state[k])
    q.begin_graph_cache()
    cached=q._quartic_cache['quartic_coefficient']
    assert cached.requires_grad and cached.grad_fn is not None
    assert cached.item()==0.
    with pytest.raises(RuntimeError):qrt.project_quartic(q,make_optimizer(q))
    q.end_graph_cache();q.load_state_dict(q.state_dict());assert q._quartic_cache is None
    set_a(q,.12);q.begin_graph_cache()
    assert q._quartic_cache['quartic_coefficient'] is not cached
    assert q._quartic_cache['quartic_coefficient'].item()==pytest.approx(.12)
    q.end_graph_cache()


@pytest.mark.parametrize('fault',['parameter','gradient','moment'])
def test_prefinite_failure_never_hidden_by_projection(fault):
    model=qrt.build_variant(source('cpu'),'Q');opt=make_optimizer(model)
    _,store,samples=inputs('cpu')
    hook=None
    if fault=='parameter':set_a(model,float('nan'))
    elif fault=='gradient':hook=model.coefficients.register_hook(lambda g:g*float('nan'))
    else:
        model.coefficients.sum().backward();opt.step()
        opt.state[model.coefficients]['exp_avg'][0]=float('nan')
    try:
        with pytest.raises(FloatingPointError):qrt.one_update(model,opt,store,samples,.06,.01)
    finally:
        if hook is not None:hook.remove()
    assert model._runtime_cache is None and model._quartic_cache is None
