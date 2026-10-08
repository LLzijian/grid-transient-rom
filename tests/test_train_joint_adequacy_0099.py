import json
import sys
from pathlib import Path

import numpy as np
import torch
import pytest

ROOT = Path(__file__).parents[1]
sys.path[:0] = [str(ROOT / "experiments"), str(ROOT / "src")]
import train_joint_adequacy_0099 as runner
from train_joint_campaign import sample_batch


def _events(n=346):
    rows = []
    for i in range(n):
        rows.append({"id": f"e{i:03d}", "group": str(i % 24),
                     "label": "unstable" if i < 35 else "stable",
                     "x": np.zeros((20, 30)), "torque": np.zeros((20, 10))})
    return rows


def test_replay_recovers_old_trace_cursor_and_rng(tmp_path):
    events = _events()
    rng = np.random.default_rng(98)
    order = rng.permutation(len(events)); cursor = 0; rows = []
    for step in range(1, 4):
        samples, _ = sample_batch(events, rng, 10)
        ids = [e["id"] for e, _ in samples]; endpoints = [int(i) for _, i in samples]
        for _ in range(2):
            if cursor >= len(events):
                order = rng.permutation(len(events)); cursor = 0
            event = events[int(order[cursor])]; cursor += 1
            ids.append(event["id"]); endpoints.append(int(rng.integers(10, len(event["x"]) - 1)))
        rows.append({"step": step, "sample_ids": ids, "endpoints": endpoints})
    source = tmp_path / "source"; source.mkdir()
    (source / "checkpoint.json").write_text(json.dumps({"step": 3, "arm": "G3-S"}))
    torch.save({"numpy": rng.bit_generator.state, "torch": torch.get_rng_state()}, source / "rng.pt")
    (source / "trace.jsonl").write_text("\n".join(json.dumps(row) for row in rows))
    got = runner.replay_sampler(events, source)
    assert got["cursor"] == cursor
    assert got["cover_order"] == order.tolist()
    assert got["replayed_trace_digest"] == runner._json_digest(rows)


def test_monitor_manifest_is_fixed_and_covers_events():
    manifest = runner.build_monitor_manifest(_events())
    assert manifest["seed"] == 99
    assert len(manifest["batches"]) == 173
    assert set(manifest["event_ids"]) == {f"e{i:03d}" for i in range(346)}
    assert all(len(batch) == 12 for batch in manifest["batches"])


def test_optional_800_requires_complete_strictly_decreasing_fine_j():
    def row(j, *s):
        return {"complete": True, "numerical_pass": True, "J_fine": j,
                "S_fine": {"1": s[0], "2": s[1], "5": s[2]}}
    good = {str(k): row(v, 1., 1., 1.) for k, v in ((0, 4.), (200, 3.), (400, 2.), (600, 1.))}
    assert runner._eligible_800(good)
    bad = dict(good); bad["400"] = row(3.5, 1., 1., 1.)
    assert not runner._eligible_800(bad)
    incomplete = dict(good); incomplete["600"] = {"complete": False, "J_fine": None}
    assert not runner._eligible_800(incomplete)


def test_800_does_not_require_valid_initial_monitor():
    rows={str(s): {'complete':True,'numerical_pass':True,'J_fine':j,
                   'S_fine':{'1':1.,'2':1.,'5':1.}} for s,j in [(200,3.),(400,2.),(600,1.)]}
    rows['0']={'complete':True,'numerical_pass':False,'J_fine':None}
    assert runner._eligible_800(rows)
    rows['400']['S_fine']['2']=None
    assert not runner._eligible_800(rows)


def test_trace_is_required_and_parent_layout_supported(tmp_path):
    checkpoint=tmp_path/'step';checkpoint.mkdir()
    with pytest.raises(FileNotFoundError):runner._trace_rows(checkpoint)
    (tmp_path/'trace.jsonl').write_text(json.dumps({'step':1})+'\n')
    assert runner._trace_rows(checkpoint)==[{'step':1}]
    (checkpoint/'checkpoint.json').write_text(json.dumps({'step':2}))
    with pytest.raises(ValueError,match='every completed step'):
        runner.replay_sampler(_events(),checkpoint)


def test_best_within_float_tolerance_keeps_earliest_and_invalid_excluded():
    def row(j):return {'complete':True,'numerical_pass':True,'J_fine':j}
    rows={'0':row(1.),'200':row(1.-1e-10),'400':{'complete':True,'numerical_pass':False,'J_fine':None}}
    assert runner._select_best(rows)=='0'
    rows['600']=row(.99)
    assert runner._select_best(rows)=='600'


def test_unresolved_severe_uses_raw_without_none_error():
    baseline={h:1. for h in ('1','2','5')};counts=dict.fromkeys(baseline,0)
    row={'S_fine':dict.fromkeys(baseline,None),'raw_S_fine':{'1':11.,'2':1.,'5':1.}}
    assert runner.severe_transition(row,baseline,counts) is None
    assert runner.severe_transition(row,baseline,counts)=='numerical_risk_10x_twice'
    counts=dict.fromkeys(baseline,0)
    assert runner.severe_transition(row,baseline,counts) is None
    row['S_fine']={'1':11.,'2':1.,'5':1.}
    assert runner.severe_transition(row,baseline,counts)=='numerical_risk_10x_twice'


def test_finite_unresolved_monitor_does_not_stop_basic_joint_run(tmp_path,monkeypatch):
    from contextlib import nullcontext
    class Toy(torch.nn.Module):
        def __init__(self):
            super().__init__();self.weight=torch.nn.Parameter(torch.tensor(.2,dtype=torch.float64))
            self.register_buffer('template_mask',torch.tensor([1.,1.,1.,1.,1.,1.,0.,1.]))
            self.dim=33
        def config(self):return {}
    model=Toy();opt=torch.optim.Adam(model.parameters(),lr=.001)
    events=_events();rng=np.random.default_rng(98)
    meta={'replay':{'cover_order':list(range(346)),'cursor':102}}
    monkeypatch.setattr(runner,'load_resume',lambda *a:(model,opt,rng,meta))
    monkeypatch.setattr(runner,'_all_sources',lambda *a:{})
    monkeypatch.setattr(runner,'check_structure',lambda *a:None)
    monkeypatch.setattr(runner,'graph',lambda *a:nullcontext())
    monkeypatch.setattr(runner,'process_memory_bytes',lambda:100)
    monkeypatch.setattr(runner,'objective',lambda m,*a:(m.weight.square(),{'toy':m.weight.square()}))
    norm=tmp_path/'norm.npz';np.savez(norm,force_scale=1.);monkeypatch.setattr(runner,'NORM',norm)
    def monitor(m,e,manifest,path,**kwargs):
        step=int(path.name[-4:]);valid=step>=400
        return {'complete':True,'numerical_pass':valid,'J_fine':1./max(step,1) if valid else None,
                'S_fine':{'1':1.,'2':1.,'5':1.},'raw_S_fine':{'1':1.,'2':1.,'5':1.},'seconds':.001}
    monkeypatch.setattr(runner,'_monitor_callable',lambda:monitor)
    result=runner.run('G3-S',tmp_path/'source',tmp_path/'run',events=events,new_steps=800)
    assert result['new_steps']==600 and result['stop_reason']=='optional_800_gate'
    assert result['best_monitor_step']=='600' and len(result['traces'])==600
    assert float(model.weight.detach())<.2
