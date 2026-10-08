"""Synthetic 0103 runner checks; never read TRAIN/TEST or take optimizer steps."""
import copy
import json
from pathlib import Path
import sys
import numpy as np
import pytest
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'experiments'))
import train_cross_dissipation_0103 as r


@pytest.fixture(autouse=True)
def forbid_data_and_updates(monkeypatch):
    def forbidden(*args,**kwargs):raise AssertionError('data/monitor/updates forbidden in synthetic tests')
    monkeypatch.setattr(r,'train_inputs',forbidden)
    monkeypatch.setattr(r,'run_monitor',forbidden)
    monkeypatch.setattr(torch.optim.Adam,'step',forbidden)


class Toy(torch.nn.Module):
    dim=40
    def __init__(self):
        super().__init__()
        self.structured=torch.nn.Linear(1,1,bias=False,dtype=torch.float64)
        self.local_head=torch.nn.Linear(1,1,bias=False,dtype=torch.float64)
        self.port_raw=torch.nn.Parameter(torch.ones(1,dtype=torch.float64))
        self._template_scalars=torch.nn.ParameterList([torch.nn.Parameter(torch.tensor(1.,dtype=torch.float64)) for _ in range(7)])
        self._template_scalars[6].requires_grad_(False)
        self.register_buffer('template_mask',torch.tensor([1,1,1,1,1,1,0,1],dtype=torch.float64))
    def begin_graph_cache(self):pass
    def end_graph_cache(self):pass


def test_exact_source_rejected_before_file_read(tmp_path):
    with pytest.raises(ValueError,match='step00650'):r.verify_source(tmp_path)


def test_source_restoration_guard_and_rng_coverage(monkeypatch):
    source_model=Toy();source_hash=r._model_hash(source_model)
    saved={'coverage':{'cover_order':list(range(346)),'cursor':0},
           'torch':torch.get_rng_state(),'numpy_global':np.random.get_state(),
           'numpy':np.random.default_rng(5).bit_generator.state}
    meta={'model_state_sha256':source_hash,'fixed_buffers_sha256':r.buffer_hash(source_model),
          'requires_grad':{n:p.requires_grad for n,p in source_model.named_parameters()},
          'parameter_numel':{n:p.numel() for n,p in source_model.named_parameters()},'coverage':saved['coverage']}
    monkeypatch.setattr(r,'verify_source',lambda source:meta)
    monkeypatch.setattr(r,'load_model',lambda path:copy.deepcopy(source_model))
    monkeypatch.setattr(r.torch,'load',lambda path,**kwargs:(source_model.state_dict() if str(path).endswith('model.pt') else saved))
    monkeypatch.setattr(r.prior,'validate_coverage',lambda coverage,n:None)
    control,rng,cov,_=r.restore_arm('S-C')
    opened,_,cov2,_=r.restore_arm('S-R')
    assert r._model_hash(control)==source_hash
    assert cov==cov2==saved['coverage']
    assert r.prior.rng_digest(r.prior.capture_rng(rng))==r.prior.rng_digest(saved)
    assert tuple(opened.template_mask.tolist())==(1.,)*8
    assert opened._template_scalars[6].requires_grad
    assert torch.equal(opened._template_scalars[6],control._template_scalars[6])


def test_only_cross_scalar_added_to_closure_adam():
    control=Toy();opened=copy.deepcopy(control);r.activate(opened,'S-R')
    copt=r.optimizer_for(control,r.RATES);ropt=r.optimizer_for(opened,r.RATES)
    c_names=[n for g in r.parameter_layout(control,copt) for n in g['parameter_names']]
    r_names=[n for g in r.parameter_layout(opened,ropt) for n in g['parameter_names']]
    assert set(r_names)-set(c_names)=={'_template_scalars.6'}
    assert not set(c_names)-set(r_names)
    assert not copt.state and not ropt.state
    assert {g['block']:g['lr'] for g in ropt.param_groups}==r.RATES
    assert r.parameter_block('_template_scalars.6')=='closure'


def test_no_update_backward_preserves_model_adam_and_rng(monkeypatch):
    model=Toy();r.activate(model,'S-R');opt=r.optimizer_for(model,r.RATES)
    before=r._model_hash(model);state=torch.get_rng_state().clone()
    def objective(m,*args):
        val=sum(p.square().sum() for p in m.parameters() if p.requires_grad)
        return val,{'complexity':m._template_scalars[6].square()}
    monkeypatch.setattr(r.prior,'objective',objective)
    monkeypatch.setattr(r.prior,'histories_for',lambda samples:torch.zeros(1,11,30))
    row=r.compute_gradients(model,[('synthetic',10)],1.,1.,opt)
    assert row['cross_raw_gradient'] is not None and row['gradient_block_l2']['closure']>0
    assert before==r._model_hash(model) and not opt.state and torch.equal(state,torch.get_rng_state())


def make_plan(tmp_path,monkeypatch,n=600):
    card=tmp_path/'card.md';card.write_text('synthetic card')
    cost=tmp_path/'costs.json';cost.write_text('{}')
    monkeypatch.setattr(r,'CARD',card)
    monkeypatch.setattr(r,'snapshot',lambda:{'synthetic':'fixed'})
    snap=r.snapshot()
    return {'loop':'0103','registered':True,'arm':'S-C','updates':n,
            'horizons':[r.horizon(r.START_STEP+i) for i in range(1,n+1)],
            'monitor_steps':[0,n//2,n],'rates':r.RATES,'clip_norm':100.,
            'cap_seconds':4700.,'monitor_seconds':60.,'cost_evidence':str(cost),
            'cost_evidence_sha256':r.digest(cost),'card_sha256':r.digest(card),
            'source_model_sha256':r.EXPECTED['model.pt'],'snapshot':snap,
            'snapshot_sha256':r.prior._json_digest(snap),'allow_refine':True}


@pytest.mark.parametrize('mutation',[{'registered':False},{'arm':'F-O'},
    {'updates':650},{'monitor_steps':[0,200,600]},{'cap_seconds':3000.},
    {'source_model_sha256':'wrong'},{'clip_norm':99.},{'cost_evidence_sha256':'wrong'}])
def test_plan_fail_closed(tmp_path,monkeypatch,mutation):
    plan=make_plan(tmp_path,monkeypatch);assert r.validate_plan(plan)==plan
    plan.update(mutation)
    with pytest.raises(ValueError):r.validate_plan(plan)


def test_plan_rejects_horizon_drift_and_unfrozen_snapshot(tmp_path,monkeypatch):
    plan=make_plan(tmp_path,monkeypatch,840);assert r.validate_plan(plan)==plan
    bad=copy.deepcopy(plan);bad['horizons'][1]=5.
    with pytest.raises(ValueError):r.validate_plan(bad)
    bad=copy.deepcopy(plan);bad['snapshot']['synthetic']='changed'
    with pytest.raises(ValueError):r.validate_plan(bad)


def test_opened_arm_uses_registered_compute_cap(tmp_path,monkeypatch):
    plan=make_plan(tmp_path,monkeypatch)
    plan['arm']='S-R'
    plan['cap_seconds']=4700.
    assert r.validate_plan(plan)==plan
    plan['cap_seconds']=2500.
    with pytest.raises(ValueError):r.validate_plan(plan)


def test_checkpoint_snapshot_rejected_before_deserialization(tmp_path,monkeypatch):
    meta={'format':'cross_dissipation_0103','valid_training_state':True,
          'arm':'S-R','source_model_sha256':r.EXPECTED['model.pt'],
          'source_snapshot':{'stale':True}}
    for name in ('model.pt','optimizer.pt','rng.pt','structured_numeric_export.json'):
        (tmp_path/name).write_text('unreadable synthetic placeholder')
    (tmp_path/'checkpoint.json').write_text(json.dumps(meta))
    hashes={p.name:r.digest(p) for p in tmp_path.iterdir()}
    (tmp_path/'artifact_hashes.json').write_text(json.dumps(hashes))
    monkeypatch.setattr(r,'snapshot',lambda:{'fresh':True})
    monkeypatch.setattr(r,'load_model',lambda path:pytest.fail('must reject before model load'))
    with pytest.raises(ValueError,match='identity mismatch'):r.restore_checkpoint(tmp_path,'S-R')
