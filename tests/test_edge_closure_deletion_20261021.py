"""Unique registered empty-closure QA. No formal synthetic val or project data."""
import copy
import sys
import types
from pathlib import Path
import numpy as np
import pytest
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'experiments'),str(ROOT/'src')]
import edge_closure_deletion_20261021 as study
from edge_memory_discrimination_20261021 import generic_field,timing_observations,snapshot,load_snapshot,equal_tree
from closure_discovery_benchmarks import toy_source
from closure_causal_recovery_20261019 import runtime_model
from closure_causal_history_residual_20261019 import CausalHistoryResidual,make_optimizer,loss,prediction


def model():
    torch.set_num_threads(1)
    assert torch.cuda.is_available(), 'registered CUDA QA cannot skip'
    tr,_,sc=timing_observations('cuda',5101)
    obs=torch.as_tensor(tr,device='cuda')
    base=runtime_model(toy_source(),[],5101,'cuda')
    with torch.no_grad():
        inertia=torch.linspace(.7,1.6,10,device='cuda',dtype=torch.float64)
        base.mass.copy_(inertia);base.structured.mass.copy_(inertia)
    m=CausalHistoryResidual.from_train(base,obs[:,:11])
    with torch.no_grad():m.history_coefficients.copy_(torch.linspace(-.003,.003,6,device='cuda'))
    study.assert_empty(m)
    return m,obs,torch.as_tensor(sc,device='cuda')


def close(a,b,tol): torch.testing.assert_close(a,b,rtol=tol,atol=tol)


def test_empty_full_field_generic_nonunit_and_export(tmp_path):
    m,obs,sc=model();ref=copy.deepcopy(m)
    ref.base.field=types.MethodType(generic_field,ref.base)
    assert not torch.allclose(m.mass,torch.ones_like(m.mass))
    z=torch.cat((obs[:3,10],m.encode(obs[:3,:11])),-1)
    torch.testing.assert_close(m.field(z),ref.field(z),rtol=2e-11,atol=1e-11)
    u=torch.full((3,10),.013,device='cuda',dtype=torch.float64)
    torch.testing.assert_close(m.field(z,u),ref.field(z,u),rtol=2e-11,atol=1e-11)
    m.export(tmp_path/'model.json')
    restored=CausalHistoryResidual.from_export(tmp_path/'model.json').cuda()
    study.assert_empty(restored);equal_tree(m.state_dict(),restored.state_dict())
    torch.testing.assert_close(m.field(z),restored.field(z),rtol=2e-11,atol=1e-11)


def test_half_second_all_gradients_clip_adam_and_nonempty_restore(tmp_path):
    m,obs,sc=model();ref=copy.deepcopy(m)
    ref.base.field=types.MethodType(generic_field,ref.base)
    opt=make_optimizer(m);other=make_optimizer(ref);rng=np.random.default_rng(14)
    assert m.base.coefficients.numel()==0
    for step in (1,2):
        ix=torch.as_tensor(rng.permutation(16)[:12],device='cuda')
        opt.zero_grad(set_to_none=True);other.zero_grad(set_to_none=True)
        a=loss(m,obs[ix],sc);b=loss(ref,obs[ix],sc);close(a,b,2e-10)
        a.backward();b.backward()
        left=dict(m.named_parameters());right=dict(ref.named_parameters())
        assert left.keys()==right.keys()
        for name,p in left.items():
            q=right[name];assert (p.grad is None)==(q.grad is None),name
            if p.grad is not None:close(p.grad,q.grad,2e-9)
        close(torch.nn.utils.clip_grad_norm_(m.parameters(),100.),
              torch.nn.utils.clip_grad_norm_(ref.parameters(),100.),2e-9)
        opt.step();other.step()
        for name,p in left.items():close(p,right[name],2e-9)
        for astate,bstate in zip(opt.state.values(),other.state.values()):
            assert astate.keys()==bstate.keys()
            for key in astate:close(astate[key],bstate[key],2e-9)
        if step==1:
            assert opt.state
            torch.save(snapshot(m,opt,rng),tmp_path/'snapshot.pt')
        else:
            final=snapshot(m,opt,rng)
            state=torch.load(tmp_path/'snapshot.pt',map_location='cpu',weights_only=False)
            torch.rand(7);torch.rand(7,device='cuda');rng.random(7)
            load_snapshot(state,m,opt,rng)
            actual=torch.as_tensor(rng.permutation(16)[:12],device='cuda');assert torch.equal(actual,ix)
            opt.zero_grad(set_to_none=True);loss(m,obs[actual],sc).backward()
            torch.nn.utils.clip_grad_norm_(m.parameters(),100.);opt.step()
            equal_tree(final,snapshot(m,opt,rng))
    m.export(tmp_path/'trained.json');loaded=CausalHistoryResidual.from_export(tmp_path/'trained.json').cuda()
    study.assert_empty(loaded);equal_tree(m.state_dict(),loaded.state_dict())


def test_once_initialization_and_future_rejection():
    m,obs,sc=model();calls=[];original=m.encode
    def counted(history):calls.append(history.shape);return original(history)
    m.encode=counted
    prediction(m,obs);assert calls==[torch.Size([16,11,30])]
    with pytest.raises(ValueError):m.encode(obs[:,:12])
    future=obs.clone();future[:,11:]+=100
    close(original(obs[:,:11]),original(future[:,:11]),0.)


@pytest.mark.parametrize('bad',['seed','count','source','term'])
def test_endpoint_rejects_wrong_identity(bad):
    seed=5101;commit='1'*40
    directory=ROOT/'private_data/closure_mechanism_20261019/P1/edge_closure_deletion01/models/5101_empty/empty_5101/checkpoint_00512'
    state={'completed_updates':512,'trace':[0.]*512,'identity':{'seed':seed,'source_commit':commit}}
    payload={'base':{'terms':[],'state_dict':{'coefficients':[]}}}
    study.validate_endpoint(directory,seed,commit,state,payload)
    if bad=='seed':state['identity']['seed']=5102
    if bad=='count':state['completed_updates']=511
    if bad=='source':state['identity']['source_commit']='2'*40
    if bad=='term':payload['base']['terms']=[{'channel':'edge'}]
    with pytest.raises(ValueError):study.validate_endpoint(directory,seed,commit,state,payload)


def test_freeze_rejects_incomplete_barriers():
    commit='1'*40
    for frozen in ({},{'all_three_frozen':True,'source_commit':commit,'selection':'train_only','rows':[]},
                   {'all_three_frozen':True,'source_commit':commit,'selection':'train_only',
                    'rows':[{'seed':s,'files':{}} for s in study.SEEDS]}):
        with pytest.raises(ValueError):study.validate_frozen(frozen,commit)
    with pytest.raises(ValueError):study.barrier({'endpoints':{'5101':'missing'}},commit)
