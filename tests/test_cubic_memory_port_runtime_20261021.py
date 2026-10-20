"""Synthetic D9 QA; runtime execution belongs to the root native QA worker."""
import copy
import json
from pathlib import Path
import sys
import numpy as np
import pytest
import torch
sys.path[:0]=[str(Path(__file__).resolve().parent),str(Path(__file__).resolve().parents[1]/'experiments')]
import cubic_memory_port_runtime_20261021 as d
from test_quartic_memory_runtime_20261020 import source,inputs,generic
from quartic_memory_runtime_20261020 import build_variant,quartic_index
from symbolic_closure_rom import numpy_field,term_key
from train_closure import make_optimizer
from train_local_memory_0098 import graph

DEVICES=['cpu']+(['cuda'] if torch.cuda.is_available() else [])


def values(model,a,b):
    with torch.no_grad():
        ix=d.coefficient_indices(model);model.coefficients[ix['a']]=a;model.coefficients[ix['b']]=b


def assert_gradients(a,b,common=False):
    left,right=dict(a.named_parameters()),dict(b.named_parameters())
    for name,p in left.items():
        q=right[name]
        if common and name=='coefficients':
            for i,t in enumerate(a.terms):
                j=next(j for j,u in enumerate(b.terms) if term_key(u)==term_key(t))
                torch.testing.assert_close(p.grad[i],q.grad[j],rtol=2e-9,atol=2e-9)
        else:
            assert (p.grad is None)==(q.grad is None),name
            if p.grad is not None:torch.testing.assert_close(p.grad,q.grad,rtol=2e-9,atol=2e-9,msg=name)


@pytest.mark.parametrize('device',DEVICES)
@pytest.mark.parametrize('b',[-.19,.23])
def test_fast_generic_numpy_gradients_and_physical_port_units(device,b):
    model=d.build_D(source(device));values(model,.17,b);ref=generic(model)
    z=(torch.randn(3,40,dtype=torch.float64,device=device)*.27).requires_grad_()
    zr=z.detach().clone().requires_grad_()
    torch.testing.assert_close(model.field(z),ref.field(zr),rtol=2e-11,atol=2e-11)
    np.testing.assert_allclose(model.field(z).detach().cpu(),numpy_field(copy.deepcopy(ref).cpu(),z.detach().cpu()),rtol=2e-10,atol=2e-10)
    model.field(z).square().sum().backward();ref.field(zr).square().sum().backward()
    assert_gradients(model,ref);torch.testing.assert_close(z.grad,zr.grad,rtol=2e-9,atol=2e-9)
    without=copy.deepcopy(model);values(without,.17,0.)
    difference=model.field(z)-without.field(z);expected=torch.zeros_like(difference)
    expected[:,10:20]=b*z[:,30:].pow(3)/model.mass
    torch.testing.assert_close(difference,expected,rtol=2e-11,atol=2e-11)
    x=model.interface_to_state(z);_,e=model.energy_and_gradient(x)
    energy,_=model.energy_and_gradient(x)
    torch.testing.assert_close(torch.autograd.grad(energy.sum(),z,retain_graph=True)[0],
        e*torch.cat((model.mass.new_ones(10),model.mass,model.mass.new_ones(20))),rtol=2e-11,atol=2e-11)
    added=x[:,10:20]/model.mass*b*x[:,30:].pow(3)
    torch.testing.assert_close(d.extra_port_power(model,z),added.sum(-1),rtol=2e-12,atol=2e-12)
    # Full internal power identity includes every old and new B port.
    J,R=model.matrices();B=model.port_matrix();u=model.port_values(x)
    actual=(e*model.interface_to_state(model.field(z))).sum(-1)
    expected_power=-(e*(e@R.T)).sum(-1)+((e@B)*u).sum(-1)
    torch.testing.assert_close(actual,expected_power,rtol=2e-10,atol=2e-10)
    external=torch.randn(3,10,dtype=torch.float64,device=device)
    torch.testing.assert_close(model.field(z,external),ref.field(zr,external),rtol=2e-11,atol=2e-11)
    shifted=z.detach().clone();shifted[:,:10]+=.9
    torch.testing.assert_close(model.field(z.detach()),model.field(shifted),rtol=2e-11,atol=2e-11)


@pytest.mark.parametrize('device',DEVICES)
@pytest.mark.parametrize('reference_variant',['N','Q'])
def test_zero_port_embedding_shared_raw_gradients_and_new_coefficients_unlocked(device,reference_variant):
    src=source(device);ref=build_variant(src,reference_variant);model=d.build_D(src)
    if reference_variant=='Q':
        with torch.no_grad():ref.coefficients[quartic_index(ref)]=.08
        values(model,.08,0.)
    _,store,samples=inputs(device)
    z=torch.randn(3,40,dtype=torch.float64,device=device)*.3
    torch.testing.assert_close(ref.field(z),model.field(z),rtol=2e-11,atol=2e-11)
    for m in (ref,model):
        with graph(m):loss,_=d.objective(m,store,samples,.06,.01);loss.backward()
    assert_gradients(ref,model,common=True)
    model.zero_grad(set_to_none=True)
    grad=torch.autograd.grad(model.field(z).square().sum(),model.coefficients)[0]
    indices=d.coefficient_indices(model)
    assert grad[indices['a']].abs()>1e-12 and grad[indices['b']].abs()>1e-12


@pytest.mark.parametrize('device',DEVICES)
def test_full_adam_projection_and_nonempty_restore(device,tmp_path):
    model=d.build_D(source(device));values(model,.03,-.04);ref=generic(model)
    opt,other=make_optimizer(model),make_optimizer(ref);_,store,samples=inputs(device)
    for m in (model,ref):
        with graph(m):loss,_=d.objective(m,store,samples,.06,.01);loss.backward()
    assert_gradients(model,ref)
    for m,o in ((model,opt),(ref,other)):
        torch.nn.utils.clip_grad_norm_([p for p in m.parameters() if p.requires_grad],100.);o.step();d.project_a(m,o)
    for p,q in zip(model.parameters(),ref.parameters()):torch.testing.assert_close(p,q,rtol=2e-9,atol=2e-9)
    model.export(tmp_path/'model.json');torch.save(opt.state_dict(),tmp_path/'adam.pt')
    restored=d.DRuntime.from_export(tmp_path/'model.json').to(device);restored_opt=make_optimizer(restored)
    restored_opt.load_state_dict(torch.load(tmp_path/'adam.pt',map_location='cpu',weights_only=False))
    a=d.one_update(model,opt,store,samples,.06,.01);b=d.one_update(restored,restored_opt,store,samples,.06,.01)
    for p,q in zip(model.parameters(),restored.parameters()):torch.testing.assert_close(p,q,rtol=2e-10,atol=2e-10)
    assert 'projected' not in a['cubic_port'] and a['quartic']['after']>=0
    assert b['cubic_port']['after']==pytest.approx(a['cubic_port']['after'])


def test_term_reordering_only_a_projection_moments_and_export_contract(tmp_path):
    model=d.build_D(source('cpu')).with_terms(list(reversed(d.TERMS)))
    indices=d.coefficient_indices(model);assert indices['b']==0
    opt=make_optimizer(model)
    assert next(g['block'] for g in opt.param_groups if any(p is model.coefficients for p in g['params']))=='closure'
    model.coefficients.sum().backward();opt.step();values(model,-.2,-.3)
    state=copy.deepcopy(opt.state_dict());before=model.coefficients.detach().clone();trace=d.project_a(model,opt)
    assert trace['a']['projected'] and trace['b']['before']==trace['b']['after']==-.3
    for i in range(9):
        if i!=indices['a']:assert torch.equal(model.coefficients[i],before[i])
    for key,row in opt.state_dict()['state'].items():
        for name,value in row.items():assert torch.equal(value,state['state'][key][name])
    model.export(tmp_path/'model.json');payload=json.loads((tmp_path/'model.json').read_text())
    assert payload['coefficients']==payload['state_dict']['coefficients']
    payload['coefficients'][0]+=.2;(tmp_path/'bad.json').write_text(json.dumps(payload))
    with pytest.raises(ValueError):d.DRuntime.from_export(tmp_path/'bad.json')
    with pytest.raises(ValueError):model.with_terms(d.TERMS[:-1])
    values(model,-.1,.1)
    with pytest.raises(FloatingPointError):model.export(tmp_path/'negative.json')
    values(model,float('-inf'),-.3)
    with pytest.raises(FloatingPointError):d.project_a(model,opt)


def test_graph_cache_identity_connections_one_encode_and_genuine_n1(monkeypatch):
    model=d.build_D(source('cpu'));_,store,samples=inputs('cpu')
    identities=[id(p) for p in model.parameters()];d.enable_runtime(model)
    assert identities==[id(p) for p in model.parameters()]
    model.begin_graph_cache()
    for key in ('quartic_coefficient','cubic_coefficient'):
        assert model._d_cache[key].requires_grad and model._d_cache[key].grad_fn is not None
    model.end_graph_cache();assert model._d_cache is None
    calls=[];old=model.encode
    def encode(h):calls.append(h.shape);return old(h)
    monkeypatch.setattr(model,'encode',encode)
    model.rollout(store.histories(samples),2,.01);assert len(calls)==1
    calls.clear()
    with graph(model):d.objective(model,store,samples,.06,.01)
    assert len(calls)==3 and store.prepare([(samples[0][0],10)],5.).counts.tolist()==[1.]
    model.begin_graph_cache();model.load_state_dict(model.state_dict());assert model._d_cache is None
    model.begin_graph_cache();model.to('cpu');assert model._d_cache is None


def test_new_contributions_machine_permutation_and_signed_power():
    model=d.build_D(source('cpu'));values(model,.2,-.3)
    z=torch.randn(4,40,dtype=torch.float64)*.3
    perm=torch.tensor([3,1,8,0,5,9,2,7,6,4])
    rotated=torch.cat([part[:,perm] for part in z.split(10,-1)],-1)
    torch.testing.assert_close(d.extra_port_power(model,z),d.extra_port_power(model,rotated),rtol=2e-12,atol=2e-12)
    _,e=model.energy_and_gradient(model.interface_to_state(z))
    n=build_variant(source('cpu'),'N');hn,gn=n.energy_and_gradient(n.interface_to_state(z))
    h,_=model.energy_and_gradient(model.interface_to_state(z))
    torch.testing.assert_close(h-hn,.2*z[:,30:].pow(4).sum(-1),rtol=2e-11,atol=2e-11)
    torch.testing.assert_close(e[:,30:]-gn[:,30:],.8*z[:,30:].pow(3),rtol=2e-11,atol=2e-11)
    values(model,.2,.3)
    positive=d.extra_port_power(model,z)
    values(model,.2,-.3)
    torch.testing.assert_close(d.extra_port_power(model,z),-positive,rtol=0,atol=0)
