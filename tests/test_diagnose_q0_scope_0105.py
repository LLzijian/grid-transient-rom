"""Synthetic 0105 semantics; these tests never read private trajectories or run a fit."""
from __future__ import annotations

import copy
import json
from pathlib import Path
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'experiments'))
import diagnose_q0_scope_0105 as r
from train_structured_rom import _event_rollout_errors


class Toy(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight=torch.nn.Parameter(torch.tensor(.7,dtype=torch.float64))
        self.register_buffer('mass',torch.linspace(1.,2.,10,dtype=torch.float64))
        self.register_buffer('p_scale',torch.ones(10,dtype=torch.float64))

    def encode(self,history):
        return history[:,-1,:10]*self.weight

    def begin_graph_cache(self): pass
    def end_graph_cache(self): pass

    def rk4(self,state,dt):
        # Exact integration for the constant local q-driven vector field.
        return torch.cat((state[:,:30]+dt*torch.cat((state[:,30:40],
             torch.zeros_like(state[:,:20])),dim=1),state[:,30:]),dim=1)


def events():
    out=[]
    for j in range(12):
        n=75 if j==0 else (165 if j==1 else 520)
        x=np.zeros((n,30),dtype=np.float64)
        x[:,0]=.01*j+.02*np.arange(n)
        out.append({'id':f'TRAIN{j:02d}','root':'ieee39_postfault',
                    'group':str(j),'label':'unstable' if j<3 else 'stable','x':x})
    return out


def test_genuine_prefixes_match_existing_event_rollout_and_keep_model_fixed():
    model=Toy(); panel=events()
    history=torch.as_tensor(np.stack([e['x'][:11] for e in panel]),dtype=torch.float64)
    physical=history[:,-1]
    q=model.encode(history).detach().clone().requires_grad_(True)
    before=copy.deepcopy(model.state_dict())
    values=r.rollout(model,panel,physical,q,dt=.01)
    for prefix in r.PREFIXES:
        reference=_event_rollout_errors(model,[(e,10) for e in panel],history,float(prefix))
        torch.testing.assert_close(values[prefix],reference,rtol=0,atol=1e-12)
    assert r.metrics(panel,values)['5']['events'][0]['available_steps']==64
    assert r.metrics(panel,values)['1']['events'][1]['available_steps']==100
    r.risk(values[5]).backward()
    assert q.grad is not None and torch.isfinite(q.grad).all()
    assert model.weight.grad is None
    assert all(torch.equal(before[k],v) for k,v in model.state_dict().items())


def test_two_grid_node_and_finite_selection_require_all_twelve_events():
    model=Toy();panel=events()
    history=torch.as_tensor(np.stack([e['x'][:11] for e in panel]),dtype=torch.float64)
    node=r.evaluate_node(model,panel,history[:,-1],model.encode(history).detach(),0)
    assert node['numeric_qualified']
    assert not node['unresolved_events']
    assert node['grids']['0.005']['5']['events'][0]['available_seconds']==pytest.approx(.64)
    with pytest.raises(FloatingPointError,match='12 finite'):
        r.risk(torch.ones(11,dtype=torch.float64))


def test_strict_source_identity_precedes_export_deserialization(monkeypatch):
    monkeypatch.setattr(r,'verify_source',lambda path: (_ for _ in ()).throw(ValueError('source hash mismatch')))
    monkeypatch.setattr(r,'load_model',lambda *a:pytest.fail('must not deserialize'))
    with pytest.raises(ValueError,match='source hash mismatch'):
        r.restore_fixed()


def test_panel_requires_train_roots_and_registered_mix(monkeypatch):
    panel=events()
    monkeypatch.setattr(r,'read_train',lambda:(list(panel)*28+[panel[0]]*10,[None]*23))
    monkeypatch.setattr(r,'panel_events',lambda events:panel)
    assert r.train_panel()==panel
    bad=copy.deepcopy(panel);bad[0]['root']='test'
    monkeypatch.setattr(r,'panel_events',lambda events:bad)
    with pytest.raises(ValueError,match='TRAIN panel drift'):
        r.train_panel()


def plan_fixture(tmp_path,monkeypatch,iterations=50):
    card=tmp_path/'card.md';card.write_text('registered synthetic card',encoding='utf-8')
    monkeypatch.setattr(r,'CARD',card)
    monkeypatch.setattr(r,'snapshot',lambda:{'source':'locked'})
    snap=r.snapshot_digest(r.snapshot())
    timing=tmp_path/'timing.json'
    timing.write_text(json.dumps({'complete':True,'within_cap':True,
        'cap_seconds':r.TIMING_CAP_SECONDS,'parameter_updates':False,
        'source_snapshot_sha256':snap,'source_model_sha256':r.SOURCE_MODEL_SHA,
        'panel_sha256':'selected TRAIN payload','model_state_sha256_before':'fixed',
        'model_state_sha256_after':'fixed','source_unchanged':True,
        'backward':{'seconds':9.},'two_grid_seconds':20.}),encoding='utf-8')
    return {'loop':'0105','registered':True,'source_model_sha256':r.SOURCE_MODEL_SHA,
        'card_sha256':r.digest(card),'source_snapshot_sha256':snap,
        'panel_sha256':'selected TRAIN payload','timing_evidence':str(timing),
        'timing_sha256':r.digest(timing),'iterations':iterations,
        'nodes':[0,iterations//2,iterations],'lr':r.LR,'clip_norm':r.CLIP,
        'cap_seconds':r.CAP_SECONDS,'grids':list(r.GRID),
        'objective':'5s genuine TRAIN risk: 0.8 mean + 0.2 top10%',
        'selection':'lowest numeric-qualified node'}


def test_plan_requires_timing_hash_and_exact_schedule(tmp_path,monkeypatch):
    plan=plan_fixture(tmp_path,monkeypatch)
    _,estimate=r.validate_plan(plan)
    assert estimate==pytest.approx(1.2*(50*9+3*20))
    short=copy.deepcopy(plan);short['iterations']=30;short['nodes']=[0,15,30]
    r.validate_plan(short)
    for key,value in [('nodes',[0,15,50]),('lr',.02),('card_sha256','drift'),
                      ('source_snapshot_sha256','drift'),('timing_sha256','drift'),
                      ('source_model_sha256','other'),('grids',[.01]),
                      ('iterations',40)]:
        bad=copy.deepcopy(plan);bad[key]=value
        with pytest.raises(ValueError):r.validate_plan(bad)
    timing=Path(plan['timing_evidence']);timing.write_text('{}',encoding='utf-8')
    with pytest.raises(ValueError,match='timing evidence hash drift'):
        r.validate_plan(plan)
