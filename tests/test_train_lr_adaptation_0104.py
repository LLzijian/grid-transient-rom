"""Synthetic 0104 checks: no dataset access, monitoring or optimizer updates."""
import copy
import json
from pathlib import Path
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'experiments'))
import train_lr_adaptation_0104 as r


@pytest.fixture(autouse=True)
def forbid_updates(monkeypatch):
    monkeypatch.setattr(torch.optim.Adam,'step',lambda *a,**k:pytest.fail('optimizer step forbidden'))


class Toy(torch.nn.Module):
    dim=40
    def __init__(self,arm):
        super().__init__()
        self.structured=torch.nn.Linear(1,1,bias=False,dtype=torch.float64)
        self.local_gru=torch.nn.Linear(1,1,bias=False,dtype=torch.float64)
        self.local_head=torch.nn.Linear(1,1,bias=False,dtype=torch.float64)
        self.port_raw=torch.nn.Parameter(torch.ones(1,dtype=torch.float64))
        self._template_scalars=torch.nn.ParameterList(
            [torch.nn.Parameter(torch.tensor(1.,dtype=torch.float64)) for _ in range(7)])
        self._template_scalars[6].requires_grad_(arm=='S-R')
        self.register_buffer('template_mask',torch.tensor([1,1,1,1,1,1,int(arm=='S-R'),1],dtype=torch.float64))


def source_fixture(monkeypatch,arm):
    model=Toy(arm)
    opt=r.prior.optimizer_for(model,r.prior.RATES)
    for group in opt.param_groups:
        for p in group['params']:
            opt.state[p]={'step':torch.tensor(300.),'exp_avg':torch.ones_like(p),
                          'exp_avg_sq':torch.full_like(p,2.)}
    rng=np.random.default_rng(104)
    coverage={'cover_order':list(range(346)),'cursor':3}
    meta={'step':3117,'completed_updates':300,'optimizer_reset':True,
          'model_state_sha256':r.prior._model_hash(model),'dimension':40,
          'requires_grad':{n:p.requires_grad for n,p in model.named_parameters()}}
    monkeypatch.setattr(r.prior,'restore_checkpoint',lambda directory,a:(model,opt,rng,coverage,meta))
    monkeypatch.setattr(r.prior,'digest',lambda path:r.EXPECTED[arm][Path(path).name]
                        if Path(path).name in r.EXPECTED[arm] else 'synthetic')
    return model,opt,rng,coverage


@pytest.mark.parametrize('arm',['S-C','S-R'])
def test_restore_preserves_adam_rng_coverage_and_all_trainable_blocks(monkeypatch,arm):
    model,opt,rng,coverage=source_fixture(monkeypatch,arm)
    state_before=copy.deepcopy(opt.state_dict())
    rng_before=copy.deepcopy(rng.bit_generator.state)
    mask_before=model.template_mask.clone()
    restored,adapted,actual_rng,actual_coverage,_=r.restore_source(arm)
    assert restored is model and adapted is opt and actual_rng is rng and actual_coverage is coverage
    assert rng.bit_generator.state==rng_before
    assert torch.equal(model.template_mask,mask_before)
    assert {g['block']:g['lr'] for g in adapted.param_groups}==r.RATES
    assert {g['block']:g['lr'] for g in state_before['param_groups']}==r.prior.RATES
    for index,entry in state_before['state'].items():
        assert torch.equal(adapted.state_dict()['state'][index]['exp_avg'],entry['exp_avg'])
        assert torch.equal(adapted.state_dict()['state'][index]['exp_avg_sq'],entry['exp_avg_sq'])
        assert int(adapted.state_dict()['state'][index]['step'])==300
    names=[n for g in r.prior.parameter_layout(model,adapted) for n in g['parameter_names']]
    assert 'local_gru.weight' in names and 'local_head.weight' in names
    assert ('_template_scalars.6' in names)==(arm=='S-R')


def test_source_hash_rejected_before_deserialization(monkeypatch,tmp_path):
    monkeypatch.setitem(r.SOURCE,'S-C',tmp_path)
    monkeypatch.setattr(r.prior,'digest',lambda path:'wrong')
    monkeypatch.setattr(r.prior,'restore_checkpoint',lambda *a:pytest.fail('must reject first'))
    with pytest.raises(ValueError,match='artifact mismatch'):
        r.restore_source('S-C')


def make_plan(tmp_path,monkeypatch,n=210):
    card=tmp_path/'card.md';card.write_text('frozen synthetic card')
    costs=tmp_path/'costs.json';costs.write_text('{}')
    monkeypatch.setattr(r,'CARD',card)
    monkeypatch.setattr(r,'source_snapshot',lambda:{'synthetic':'fixed'})
    snap=r.source_snapshot()
    return {'loop':'0104','registered':True,'arm':'S-C','updates':n,
            'horizons':[r.prior.horizon(r.START_STEP+i) for i in range(1,n+1)],
            'monitor_steps':[0,n//2,n],'rates':r.RATES,'clip_norm':100.,
            'cap_seconds':2000.,'monitor_seconds':80.,'cost_evidence':str(costs),
            'cost_evidence_sha256':r.prior.digest(costs),'card_sha256':r.prior.digest(card),
            'source_model_sha256':r.EXPECTED['S-C']['model.pt'],'snapshot':snap,
            'snapshot_sha256':r.prior.prior._json_digest(snap),'allow_refine':True}


@pytest.mark.parametrize('n',[210,180])
def test_plan_accepts_only_registered_common_schedules(tmp_path,monkeypatch,n):
    plan=make_plan(tmp_path,monkeypatch,n)
    assert r.validate_plan(plan)==plan
    bad=copy.deepcopy(plan);bad['horizons'][0]=5.
    with pytest.raises(ValueError):r.validate_plan(bad)
    bad=copy.deepcopy(plan);bad['monitor_steps'][1]+=1
    with pytest.raises(ValueError):r.validate_plan(bad)


@pytest.mark.parametrize('mutation',[{'updates':181},{'rates':{'skeleton':.01,'closure':.0002,'encoder':.00006}},
                                     {'cap_seconds':2100.},{'allow_refine':False},{'card_sha256':'drift'},
                                     {'snapshot_sha256':'drift'},{'registered':False},
                                     {'cost_evidence_sha256':'drift'}])
def test_plan_rejects_drift(tmp_path,monkeypatch,mutation):
    plan=make_plan(tmp_path,monkeypatch)
    plan.update(mutation)
    with pytest.raises(ValueError):r.validate_plan(plan)


def test_no_update_timing_preserves_adam_and_rng(monkeypatch,tmp_path):
    model,opt,rng,coverage=source_fixture(monkeypatch,'S-R')
    monkeypatch.setattr(r,'source_snapshot',lambda:{'fixed':'hash'})
    monkeypatch.setattr(r.prior,'train_inputs',lambda:('synthetic-events',1.))
    monkeypatch.setattr(r.prior,'next_samples',lambda events,g,c:([0],{'sample':0}))
    monkeypatch.setattr(r.prior,'compute_gradients',lambda *a,**k:
                        {'seconds':1.,'cross_raw_gradient':1.})
    monkeypatch.setattr(r.prior,'build_manifest',lambda events:{'synthetic':True})
    monkeypatch.setattr(r.prior,'run_monitor',lambda *a,**k:
                        {'complete':True,'numerical_pass':True,'seconds':2.})
    monkeypatch.setattr(r.prior,'guard',lambda *a,**k:None)
    monkeypatch.setattr(r.prior,'PROCESS_STARTED',r.time.perf_counter())
    before=r.prior._model_hash(model)
    row=r.time_no_update('S-R',tmp_path/'timing',cap_seconds=100.)
    assert row['complete'] and row['parameter_updates'] is False
    assert row['optimizer_sha256_before']==row['optimizer_sha256_after']
    assert row['model_state_sha256_before']==row['model_state_sha256_after']==before
    assert [x['horizon'] for x in row['batches']]==[1,2,5]
    assert row['cross_raw_gradient_seen'] and row['monitor_numerical_pass']
    assert json.loads((tmp_path/'timing/costs.json').read_text())['complete']


def test_cost_check_requires_matching_no_update_evidence(tmp_path,monkeypatch):
    model,opt,_,_=source_fixture(monkeypatch,'S-C')
    r.restore_source('S-C')
    snap={'card0104':'after','fixed':'same'}
    costs={'complete':True,'parameter_updates':False,'arm':'S-C','checkpoint_step':3117,
           'source_model_sha256':r.EXPECTED['S-C']['model.pt'],'rates':r.RATES,
           'clip_norm':100.,'model_state_sha256_before':r.prior._model_hash(model),
           'model_state_sha256_after':r.prior._model_hash(model),
           'optimizer_sha256_before':r.optimizer_fingerprint(opt),
           'optimizer_sha256_after':r.optimizer_fingerprint(opt),'rng_and_model_unchanged':True,
           'monitor_complete':True,'monitor_numerical_pass':True,'monitor_seconds':80.,
           'source_before':{'card0104':'before','fixed':'same'},
           'source_after':{'card0104':'before','fixed':'same'},
           'step_seconds':{'1':1.,'2':2.,'5':3.}}
    plan={'arm':'S-C','monitor_seconds':80.,'horizons':[1.,2.,5.]*70,'cap_seconds':2000.}
    assert r.check_costs(plan,costs,snap,model,opt)>0
    costs['optimizer_sha256_after']='drift'
    with pytest.raises(ValueError,match='timing evidence'):
        r.check_costs(plan,costs,snap,model,opt)
