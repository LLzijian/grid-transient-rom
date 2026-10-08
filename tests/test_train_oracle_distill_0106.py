"""Synthetic 0106 protocol checks; no TRAIN reads, monitor, or optimizer updates."""
import json
from pathlib import Path
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'experiments'))
import train_oracle_distill_0106 as r


@pytest.fixture(autouse=True)
def forbid_live_work(monkeypatch):
    def forbidden(*args,**kwargs):raise AssertionError('live data/monitor/update forbidden')
    monkeypatch.setattr(r.prior,'train_inputs',forbidden)
    monkeypatch.setattr(r,'run_monitor',forbidden)
    monkeypatch.setattr(torch.optim.Adam,'step',forbidden)


def test_registered_auxiliary_phase_and_horizon():
    assert [r.auxiliary_weight(i,60) for i in (1,30,31,60)]==[10.,10.,0.,0.]
    assert [r.auxiliary_weight(i,45) for i in (1,22,23,45)]==[10.,10.,0.,0.]
    assert [r.prior.horizon(r.START_STEP+i) for i in range(1,7)]==[1.,2.,5.,1.,2.,5.]
    with pytest.raises(ValueError):r.auxiliary_weight(46,45)


def test_target_identity_and_payload_guard(tmp_path,monkeypatch):
    path=tmp_path/'result.json'
    node={'step':50,'numeric_qualified':True,'q0':np.zeros((12,10)).tolist()}
    result={'status':'completed','failure':None,'source_model_sha256':r.source.EXPECTED['model.pt'],
            'selected_node':50,'nodes':{'50':node},'panel_sha256':r.PANEL_SHA256,
            'panel_ids':list(r.PANEL_IDS),'test_read':False,'validation_read':False}
    path.write_text(json.dumps(result),encoding='utf-8')
    monkeypatch.setattr(r,'TARGET',path)
    monkeypatch.setattr(r,'TARGET_SHA256',r.digest(path))
    events=[{'id':name,'x':np.zeros((12,30))} for name in r.PANEL_IDS]
    monkeypatch.setattr(r,'panel_events',lambda rows:rows)
    import diagnose_q0_scope_0105 as diagnostic
    monkeypatch.setattr(diagnostic,'panel_fingerprint',lambda panel:r.PANEL_SHA256)
    histories,q=r.verify_targets(events)
    assert histories.shape==(12,11,30) and q.shape==(12,10)
    result['nodes']['50']['numeric_qualified']=False
    path.write_text(json.dumps(result),encoding='utf-8')
    monkeypatch.setattr(r,'TARGET_SHA256',r.digest(path))
    with pytest.raises(ValueError,match='qualification'):
        r.verify_targets(events)


def test_plan_fits_only_measured_60_or_45_and_rejects_drift(tmp_path,monkeypatch):
    locked={'code':{'path':'synthetic','sha256':'fixed'}}
    monkeypatch.setattr(r,'snapshot',lambda:locked)
    monkeypatch.setattr(r,'CARD',tmp_path/'card.md')
    r.CARD.write_text('registered',encoding='utf-8')
    costs={'complete':True,'parameter_updates':False,'source_snapshot':locked,
           'target_sha256':r.TARGET_SHA256,'source_model_sha256':r.source.EXPECTED['model.pt'],
           'model_state_sha256_before':'same','model_state_sha256_after':'same',
           'rng_unchanged':True,'coverage_unchanged':True,'monitor_complete':True,
           'monitor_numerical_pass':True,
           'monitor_seconds':40.,
           'batches':[{'horizon':h,'seconds':4.} for h in (1.,2.,5.)]}
    path=tmp_path/'costs.json';path.write_text(json.dumps(costs),encoding='utf-8')
    plan=r.make_plan(path)
    assert plan['updates']==60 and plan['auxiliary_steps']==30
    assert plan['projected_seconds_including_margin_and_reserve']<700
    assert r.validate_plan(plan)==plan
    bad=dict(plan,auxiliary_steps=29)
    with pytest.raises(ValueError,match='registered'):
        r.validate_plan(bad)
    costs['batches']=[{'horizon':h,'seconds':8.} for h in (1.,2.,5.)]
    path.write_text(json.dumps(costs),encoding='utf-8')
    fallback=r.make_plan(path)
    assert fallback['updates']==45 and fallback['auxiliary_steps']==22
    with pytest.raises(ValueError,match='registered'):
        r.validate_plan(plan)
    costs['batches']=[{'horizon':h,'seconds':11.} for h in (1.,2.,5.)]
    path.write_text(json.dumps(costs),encoding='utf-8')
    with pytest.raises(TimeoutError):r.make_plan(path)
