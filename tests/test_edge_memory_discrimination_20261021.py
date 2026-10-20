"""Single preregistered CUDA QA; synthetic timing arrays, no formal val."""
import copy
import types
import sys
from pathlib import Path
import numpy as np
import pytest
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'experiments'),str(ROOT/'src')]
from edge_memory_discrimination_20261021 import (
    SIN_TERM,COS_TERM,canonical_terms,generic_field,timing_observations,
    snapshot,load_snapshot,equal_tree,qualify_fixture)
from closure_causal_recovery_20261019 import runtime_model
from closure_discovery_benchmarks import toy_source
from closure_causal_history_residual_20261019 import CausalHistoryResidual,make_optimizer,loss,prediction


def model(term):
    torch.set_num_threads(1)
    assert torch.cuda.is_available(), 'registered CUDA QA cannot skip'
    base=runtime_model(toy_source(),[term],5101,'cuda')
    # QA alone uses nonunit inertia, keeping both interface and skeleton
    # buffers consistent. Formal source/contract construction is untouched.
    with torch.no_grad():
        inertia=torch.linspace(.7,1.6,10,device='cuda',dtype=torch.float64)
        base.mass.copy_(inertia)
        base.structured.mass.copy_(inertia)
    tr,_,sc=timing_observations('cuda',5101)
    obs=torch.as_tensor(tr,device='cuda')
    m=CausalHistoryResidual.from_train(base,obs[:,:11])
    with torch.no_grad():
        m.base.coefficients.fill_(.17)
        m.history_coefficients.copy_(torch.linspace(-.003,.003,6,device='cuda'))
    return m,obs,torch.as_tensor(sc,device='cuda')


def oracle(m):
    ref=copy.deepcopy(m)
    ref.base.field=types.MethodType(generic_field,ref.base)
    return ref


def close(a,b,tol):torch.testing.assert_close(a,b,rtol=tol,atol=tol)


def test_canonical_identity_counts():
    def count(a):return 1+sum(count(x) for x in a.get('args',[]))
    terms=canonical_terms()
    assert [count(t['expression']) for t in terms]==[7,5]
    assert terms[0]!=terms[1]


@pytest.mark.parametrize('bad',['coarse','fine','scales','difference'])
def test_fixture_qualification_rejects_nonfinite(bad):
    values={'coarse':np.ones((2,3)),'fine':np.ones((2,3)),
            'scales':np.ones(3),'difference':0.}
    if bad=='difference':values[bad]=float('nan')
    else:values[bad].flat[0]=float('nan')
    with pytest.raises(FloatingPointError):qualify_fixture(**values)


@pytest.mark.parametrize('term',[SIN_TERM,COS_TERM])
def test_full_field_generic_ast_and_nonunit_mass(term):
    m,obs,scale=model(term);ref=oracle(m)
    z=torch.cat((obs[:3,-1],m.encode(obs[:3,:11])),-1)
    assert not torch.allclose(m.mass,torch.ones_like(m.mass))
    torch.testing.assert_close(m.field(z),ref.field(z),rtol=2e-11,atol=1e-11)
    u=torch.full((3,10),.013,device='cuda',dtype=torch.float64)
    torch.testing.assert_close(m.field(z,u),ref.field(z,u),rtol=2e-11,atol=1e-11)


@pytest.mark.parametrize('term',[SIN_TERM,COS_TERM])
def test_complete_half_second_loss_all_gradients_clip_adam_and_restore(term,tmp_path):
    m,obs,scale=model(term);ref=oracle(m)
    opt=make_optimizer(m);rop=make_optimizer(ref)
    rng=np.random.default_rng(14)
    # Two steps exercise empty/nonempty Adam; replay after CPU deserialization.
    for step in (1,2):
        ix=torch.as_tensor(rng.permutation(16)[:12],device='cuda')
        opt.zero_grad(set_to_none=True);rop.zero_grad(set_to_none=True)
        a=loss(m,obs[ix],scale);b=loss(ref,obs[ix],scale)
        close(a,b,2e-10);a.backward();b.backward()
        left=dict(m.named_parameters());right=dict(ref.named_parameters())
        assert left.keys()==right.keys()
        for n,p in left.items():
            q=right[n];assert (p.grad is None)==(q.grad is None),n
            if p.grad is not None:close(p.grad,q.grad,2e-9)
        n1=torch.nn.utils.clip_grad_norm_(m.parameters(),100.)
        n2=torch.nn.utils.clip_grad_norm_(ref.parameters(),100.);close(n1,n2,2e-9)
        opt.step();rop.step()
        for n,p in left.items():close(p,right[n],2e-9)
        for pa,pb in zip(opt.state.values(),rop.state.values()):
            for key in pa:close(pa[key],pb[key],2e-9)
        if step==1:
            torch.save(snapshot(m,opt,rng),tmp_path/'snapshot.pt')
        else:
            final=snapshot(m,opt,rng)
            loaded=torch.load(tmp_path/'snapshot.pt',map_location='cpu',weights_only=False)
            torch.rand(7);torch.rand(7,device='cuda');rng.random(7)
            load_snapshot(loaded,m,opt,rng)
            actual_ix=torch.as_tensor(rng.permutation(16)[:12],device='cuda')
            assert torch.equal(actual_ix,ix)
            opt.zero_grad(set_to_none=True);v=loss(m,obs[actual_ix],scale);v.backward()
            torch.nn.utils.clip_grad_norm_(m.parameters(),100.);opt.step()
            equal_tree(final,snapshot(m,opt,rng))
    m.export(tmp_path/'model.json');restored=CausalHistoryResidual.from_export(tmp_path/'model.json').cuda()
    equal_tree(m.state_dict(),restored.state_dict())
    z=torch.cat((obs[:2,-1],m.encode(obs[:2,:11])),-1)
    torch.testing.assert_close(m.field(z),restored.field(z),rtol=2e-11,atol=1e-11)


@pytest.mark.parametrize('term',[SIN_TERM,COS_TERM])
def test_symmetry_cache_cause_and_parameter_identity(term):
    m,obs,scale=model(term)
    params=list(m.parameters());assert len({id(p) for p in params})==len(params)
    z=torch.cat((obs[:2,10],m.encode(obs[:2,:11])),-1)
    original=m.field(z)
    shifted=z.clone();shifted[:,:10]+=1.23
    torch.testing.assert_close(m.field(shifted),original,rtol=2e-11,atol=1e-11)
    reversed_model=copy.deepcopy(m)
    with torch.no_grad():
        seen=set()
        for name,buffer in reversed_model.named_buffers():
            if name.endswith('base_D') or name.endswith('phi') or name.endswith('.D'):
                if buffer.data_ptr() not in seen:seen.add(buffer.data_ptr());buffer.neg_()
    torch.testing.assert_close(reversed_model.field(z),original,rtol=2e-11,atol=1e-11)
    m.begin_graph_cache()
    try:
        torch.testing.assert_close(m.field(z),original,rtol=2e-11,atol=1e-11)
        with pytest.raises(RuntimeError):m.begin_graph_cache()
    finally:m.end_graph_cache()
    calls=[];old=m.encode
    def counted(history):calls.append(history.shape);return old(history)
    m.encode=counted
    prediction(m,obs,deadline=None);assert calls==[torch.Size([16,11,30])]
    with pytest.raises(ValueError):m.encode(obs[:,:12])
    future=obs.clone();future[:,11:]+=100
    close(old(obs[:,:11]),old(future[:,:11]),0.)
