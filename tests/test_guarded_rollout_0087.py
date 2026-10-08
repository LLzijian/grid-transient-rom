"""Synthetic contracts for Loop0087; no learned checkpoint or event reader is used."""
import importlib.util
import sys
from pathlib import Path
import numpy as np
import pytest
import torch

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/"experiments"))
spec=importlib.util.spec_from_file_location("rollout0087",ROOT/"experiments"/"train_guarded_rollout_0087.py")
mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)


class _Structured(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.storage_gamma=torch.nn.ParameterList([torch.nn.Parameter(torch.tensor(float(i),dtype=torch.float64)) for i in range(8)])
        self.port_coefficient=torch.nn.ParameterList([torch.nn.Parameter(torch.arange(8,dtype=torch.float64)),torch.nn.Parameter(torch.arange(8,16,dtype=torch.float64))])


class _Model(torch.nn.Module):
    def __init__(self):
        super().__init__();self.structured=_Structured();self.frozen=torch.nn.Parameter(torch.tensor(1.),requires_grad=False)


def test_original_24_theta_pack_set_order_and_frozen_parameter_enforcement():
    model=_Model();params=mod.theta_parameters(model);before={key:value.detach().clone() for key,value in model.state_dict().items()};theta=mod._theta(model);optimizer=torch.optim.Adam(params,lr=1e-3)
    assert theta.shape==(24,) and sum(param.numel() for param in params)==24 and not model.frozen.requires_grad
    sum(param.sum() for param in params).backward();optimizer.step();moments={id(param):{key:(value.clone() if torch.is_tensor(value) else value) for key,value in optimizer.state[param].items()} for param in params};ids=[id(param) for param in params]
    changed=mod._theta(model)+np.linspace(.01,.24,24);mod.set_theta(model,changed)
    assert np.array_equal(mod._theta(model),changed) and torch.equal(model.state_dict()["frozen"],before["frozen"])
    assert ids==[id(param) for param in params]
    for param in params:
        for key,value in moments[id(param)].items():
            assert torch.equal(optimizer.state[param][key],value) if torch.is_tensor(value) else optimizer.state[param][key]==value


def test_free_projection_preserves_adam_state_and_binding_projection_is_feasible():
    parameter=torch.nn.Parameter(torch.tensor(.0));optimizer=torch.optim.Adam([parameter],lr=1e-3);parameter.grad=torch.tensor(1.);optimizer.step();moment=optimizer.state[parameter]["exp_avg"].clone()
    L=np.zeros((24,32,32));free=mod._project(np.linspace(-.1,.1,24),np.zeros(24),L,float("inf"))
    assert free["projection_pass"] and np.allclose(free["beta"],np.linspace(-.1,.1,24),atol=1e-8)
    binding=L.copy();binding[0]=np.eye(32);out=mod._project(np.r_[1.,np.zeros(23)],np.zeros(24),binding,float("inf"))
    assert out["projection_pass"] and out["beta"][0]==pytest.approx(.49,abs=2e-6)
    assert torch.equal(optimizer.state[parameter]["exp_avg"],moment)


def test_resource_screen_and_failure_shapes_are_strict():
    assert mod.estimated_seconds([1.,2.,3.])==750 and mod.estimated_seconds([8.,8.,8.])>2400
    with pytest.raises(ValueError):mod.estimated_seconds([1.,2.])
    with pytest.raises(ValueError):mod.estimated_seconds([1.,np.nan,2.])
    assert mod.allow_final_forecast(300,1.,2.) and not mod.allow_final_forecast(299,1.,2.) and not mod.allow_final_forecast(300,3.,2.)


def test_snapshot_strictly_detects_frozen_state_mutation_without_preflight_update():
    model=_Model();mod.theta_parameters(model);source=mod._snapshot(model);allowed={name for name,param in model.named_parameters() if param.requires_grad}
    assert mod._frozen_equal(model,source,allowed)
    with torch.no_grad():model.frozen.add_(1.)
    assert not mod._frozen_equal(model,source,allowed)


def test_preflight_does_not_update_any_synthetic_parameter(monkeypatch,tmp_path):
    model=_Model();before=mod._snapshot(model)
    model.begin_graph_cache=lambda:None
    model.end_graph_cache=lambda:None
    monkeypatch.setattr(mod,"hashes",lambda _: {**mod.ANCHORS,"baseline_model":mod.BASELINE_SHA})
    monkeypatch.setattr(mod,"load_model",lambda _:model)
    monkeypatch.setattr(mod,"_events",lambda:([{"id":"synthetic"}],{},{}))
    monkeypatch.setattr(mod,"sample_batch",lambda *_: ([( {"id":"synthetic"},10)]*32,torch.zeros((32,11,30))))
    def toy_objective(model,*args,**kwargs):
        loss=sum(p.square().sum() for p in model.parameters() if p.requires_grad)
        return loss,{"risk":loss}
    monkeypatch.setattr(mod,"objective",toy_objective)
    result=mod.preflight(tmp_path,tmp_path/"out")
    assert result["completed"] and [r["horizon"] for r in result["traces"]]==[1.,2.,5.]
    assert all(torch.equal(v,model.state_dict()[k]) for k,v in before.items())
