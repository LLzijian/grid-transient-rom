"""Synthetic contracts for E1 fitting; no project trajectory is opened."""
from pathlib import Path
import sys
from types import SimpleNamespace
import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'experiments'), str(ROOT/'src')]
import train_local_memory_0098 as trainer


def fixture_model(arm='L10-S'):
    from local_memory_rom_0098 import LocalMemoryROM
    D = np.zeros((10,10))
    for i in range(10):
        D[i,i] = 1; D[i,(i+1)%10] = -1
    base = SimpleNamespace(mass=np.linspace(1.,2.,10), incidence=D.T,
                           coupling=np.ones(10)*.2, delta_star=np.linspace(-.1,.1,10),
                           damping=np.ones(10)*.1, rate=np.linspace(.1,.3,10),
                           gain=np.linspace(.2,.4,10))
    return LocalMemoryROM(base, np.ones(20), np.ones(10), arm=arm)


def fixture_events():
    result=[]
    for label in ('unstable','stable'):
        for group in range(12):
            x=np.random.default_rng(group).normal(scale=.01,size=(14+group,30))
            result.append({'id': f'{label}_{group:02d}', 'group': str(group),
                           'label': label, 'x': x, 'torque': np.zeros((len(x),10))})
    return result


def test_panel_is_deterministic_group_diverse_and_does_not_mutate_pool():
    events=fixture_events()
    first=trainer.panel_events(events)
    second=trainer.panel_events(list(reversed(events)))
    assert [e['id'] for e in first]==[e['id'] for e in second]
    assert len(first)==12 and sum(e['label']=='unstable' for e in first)==3
    assert len({e['group'] for e in first if e['label']=='unstable'})==3
    assert len(events)==24


@pytest.mark.parametrize('arm',trainer.ARMS)
def test_objective_dynamic_memory_shape_and_real_short_prefix_gradients(arm):
    torch.set_num_threads(1)
    model=fixture_model(arm)
    events=fixture_events()[:3]
    samples=[(e,10) for e in events]
    with trainer.graph(model):
        loss,terms=trainer.objective(model,samples,trainer.histories_for(samples),1.,1.)
        loss.backward()
    assert torch.isfinite(loss)
    assert model.rank in (3,10,11)
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
    assert torch.isfinite(terms['torque'])


def test_autonomous_objective_encodes_first_window_only():
    model=fixture_model()
    events=fixture_events()[:2]
    seen=[]
    original=model.encode
    def tracked(history):
        seen.append(history.detach().clone())
        return original(history)
    model.encode=tracked
    with torch.no_grad(),trainer.graph(model):
        trainer.autonomous(model,events,1.)
    assert len(seen)==1
    np.testing.assert_array_equal(seen[0].numpy(),np.stack([e['x'][:11] for e in events]))


def test_growth_is_bounded_and_uses_autonomous_gradients():
    torch.set_num_threads(1)
    model=fixture_model()
    prior=model.template_mask.clone()
    rows=trainer.grow(model,fixture_events()[:3])
    assert sum(row['activated'] for row in rows)<=3
    assert all(np.isfinite(row['score']) for row in rows)
    assert bool(model.template_mask[3]) and bool(model.template_mask[5])
    assert int((model.template_mask-prior).sum())<=3
    with trainer.graph(model):
        trainer.risk(trainer.autonomous(model,fixture_events()[:3],.2)).backward()
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())


def test_horizons_follow_registered_curriculum():
    assert [trainer.horizon(i) for i in (1,100,101,102,299,300,301,302,303)]==[1,1,1,2,1,2,1,2,5]


def test_failed_growth_restores_candidate_probe_state(monkeypatch):
    model=fixture_model()
    before={k:v.detach().clone() for k,v in model.state_dict().items()}
    def fail(*args):
        raise FloatingPointError('synthetic candidate failure')
    monkeypatch.setattr(trainer,'autonomous',fail)
    with pytest.raises(FloatingPointError):
        trainer.grow(model,fixture_events()[:3])
    assert all(torch.equal(v,model.state_dict()[k]) for k,v in before.items())
    assert not model.template_mask.requires_grad


def test_prune_cannot_spend_prior_improvement_on_later_harm(monkeypatch):
    model=fixture_model()
    model.set_active(0,True); model.set_active(1,True)
    def synthetic_loss(*args):
        a,b=bool(model.template_mask[0]),bool(model.template_mask[1])
        value=100. if (a and b) else (90. if b else 99.)
        return torch.tensor([value],dtype=torch.float64)
    monkeypatch.setattr(trainer,'autonomous',synthetic_loss)
    rows=trainer.prune(model,fixture_events()[:3],trainer.optimizer_for(model))
    assert rows[0]['removed']
    assert rows[1]['baseline_loss']==90.
    assert not rows[1]['removed']
