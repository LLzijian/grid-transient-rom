"""Synthetic recovery/sampler metadata tests: no physical integration or real data."""
import copy
import json
import sys
from pathlib import Path
import numpy as np
import pytest
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'experiments'))
import train_joint_numerical_pair_0100 as r


class Toy(torch.nn.Module):
    dim = 40
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor([1.,2.], dtype=torch.float64))
        self.bias = torch.nn.Parameter(torch.tensor([3.], dtype=torch.float64))
        self.register_buffer('template_mask', torch.tensor(r.MASK))
    def config(self):
        return {'synthetic': True}


def adam(model, step=1167):
    opt = torch.optim.Adam([{'params':[model.weight], 'lr':.01},
                            {'params':[model.bias], 'lr':.0003}])
    for p in model.parameters():
        opt.state[p] = {'step':torch.tensor(float(step)),
                        'exp_avg':torch.zeros_like(p), 'exp_avg_sq':torch.ones_like(p)}
    return opt


def meta(model, opt):
    return {'step':1167, 'arm':'L10-S', 'requires_grad_parameters':[n for n,p in model.named_parameters()],
            'optimizer_parameter_names':r.parameter_layout(model,opt),
            'coverage':{'cover_order':[2,0,1], 'cursor':2},
            'model_state_sha256':r._model_hash(model)}


def test_optimizer_names_counters_and_all_parameters():
    model=Toy(); opt=adam(model); m=meta(model,opt)
    r.validate_optimizer(model,opt,m)
    bad=copy.deepcopy(m); bad['optimizer_parameter_names'][0]['parameter_names']=['bias']
    with pytest.raises(ValueError,match='names/order'):
        r.validate_optimizer(model,opt,bad)
    opt.state[model.weight]['step']=torch.tensor(1.)
    with pytest.raises(ValueError,match='counter'):
        r.validate_optimizer(model,opt,m)


def test_restore_exact_model_adam_rng_coverage(tmp_path, monkeypatch):
    model=Toy(); opt=adam(model); m=meta(model,opt)
    torch.save(model.state_dict(),tmp_path/'model.pt')
    torch.save(opt.state_dict(),tmp_path/'optimizer.pt')
    np_rng=np.random.default_rng(33); expected_np=copy.deepcopy(np_rng.bit_generator.state)
    torch_state=torch.get_rng_state().clone()
    torch.save({'numpy':expected_np,'torch':torch_state},tmp_path/'rng.pt')
    (tmp_path/'checkpoint.json').write_text(json.dumps(m))
    monkeypatch.setattr(r,'SOURCE',tmp_path)
    monkeypatch.setattr(r,'EXPECTED',{n:r.digest(tmp_path/n) for n in r.EXPECTED})
    def loader(path):
        new=Toy(); new.load_state_dict(torch.load(path/'model.pt',weights_only=True)); return new
    monkeypatch.setattr(r,'load_model',loader)
    monkeypatch.setattr(r,'optimizer_for',adam)
    got,opt2,rng,coverage,_=r.restore(tmp_path,3)
    assert r._model_hash(got)==r._model_hash(model)
    assert rng.bit_generator.state==expected_np and torch.equal(torch.get_rng_state(),torch_state)
    assert coverage==m['coverage']
    assert torch.equal(opt2.state[got.weight]['exp_avg_sq'],torch.ones(2))
    (tmp_path/'model.pt').write_bytes(b'altered')
    with pytest.raises(ValueError,match='hash mismatch'):
        r.restore(tmp_path,3)


def events():
    return [{'id':str(i),'label':'unstable' if i%2 else 'stable','group':str(i%3),
             'x':np.zeros((15+i,30))} for i in range(12)]


def test_pair_sampler_exact_and_wrap_preserves_endpoints():
    es=events(); a=np.random.default_rng(7); b=np.random.default_rng(7)
    ca={'cover_order':list(range(12)),'cursor':11}; cb=copy.deepcopy(ca)
    for step in range(1168,1172):
        sa,ta=r.next_samples(es,a,ca); sb,tb=r.next_samples(es,b,cb)
        assert ta==tb and ca==cb and len(sa)==12
        assert all(10<=end<len(e['x'])-1 for e,end in sa)
        assert r.horizon(step) in (1,2,5)
    assert len(ta['coverage_before']['cover_order'])==12


@pytest.mark.parametrize('coverage',[{'cover_order':[0,0,2],'cursor':0},
                                    {'cover_order':[0,1,2],'cursor':4}])
def test_invalid_coverage(coverage):
    with pytest.raises(ValueError): r.validate_coverage(coverage,3)


def test_checkpoint_valid_and_failed_state_distinct(tmp_path):
    model=Toy(); opt=adam(model); rng=np.random.default_rng(98)
    args=(model,opt,rng,{'cover_order':[0],'cursor':1},1167,.005,{'sha256':'x'},{},r.time.perf_counter())
    good=Path(r.save_checkpoint(tmp_path,'step0000',*args))
    fail=Path(r.save_checkpoint(tmp_path,'failed_state',*args,valid=False,attempted_step=1168))
    g=json.loads((good/'checkpoint.json').read_text()); f=json.loads((fail/'checkpoint.json').read_text())
    assert g['step']==1167 and g['training_dt']==.005 and g['new_steps']==0
    assert f['step'] is None and f['new_steps'] is None and f['valid_training_state'] is False
    assert f['last_verified_step']==1167 and f['attempted_step']==1168
    payload=torch.load(good/'rng.pt',weights_only=False)
    assert payload['coverage']==g['coverage'] and 'numpy_global' in payload
    assert set(json.loads((good/'artifact_hashes.json').read_text()))==set(r.EXPECTED)


def test_cost_counts_registered_horizons_and_reserve():
    costs={'1':1.,'2':2.,'5':5.}
    expected=sum(costs[str(int(r.horizon(1167+i)))] for i in range(1,101))
    assert r.estimated_cost(100,costs,200)==expected+250+367.5
    assert r.reserve_seconds(200,[1.,5.])==367.5
    assert r.estimated_cost(200,costs,200)>r.estimated_cost(100,costs,200)


def test_cache_identity_keeps_unresolved_j(tmp_path,monkeypatch):
    model=Toy(); h=r._model_hash(model); monkeypatch.setattr(r,'sources',lambda:{'fake':'source'})
    row={'complete':True,'J_fine':None,'model_state_sha256_before':h,'model_state_sha256_after':h,
         'manifest_sha256':'m','monitor_code_sha256':r.digest(r.ROOT/'experiments/monitor_joint_adequacy_0099.py'),
         'source_before':{'fake':'source'},'source_after':{'fake':'source'},
         'sources_unchanged':True,'rng_unchanged':True,'seconds':200.}
    p=tmp_path/'evidence.json'; p.write_text(json.dumps(row))
    cached,audit=r.cached_initial_monitor(tmp_path,model,{'sha256':'m'})
    assert cached['J_fine'] is None and audit['accepted']
    row['rng_unchanged']=False; p.write_text(json.dumps(row))
    cached,audit=r.cached_initial_monitor(tmp_path,model,{'sha256':'m'})
    assert cached is None and not audit['accepted']


def test_cli_guards_before_data_access(tmp_path):
    for kwargs in ({'dt':.0025},{'new_steps':300},{'cap_seconds':3501}):
        args={'dt':.01,'source':tmp_path,'output':tmp_path/'new',**kwargs}
        with pytest.raises(ValueError): r.run(**args)


def test_cost_evidence_requires_matching_source_and_no_updates(tmp_path):
    p=tmp_path/'cost.json'
    costs={'training_dt':.005,'checkpoint_step':1167,'complete':True,'parameter_updates':False,
           'source_model_sha256':r.EXPECTED['model.pt'],
           'objective_code_sha256':r.digest(r.ROOT/'experiments/joint_training_grid_0100.py'),
           'step_seconds':{'1':1.,'2':2.,'5':5.}}
    p.write_text(json.dumps(costs)); assert r.read_cost_evidence(p,.005)==costs
    with pytest.raises(ValueError): r.read_cost_evidence(p,.01)
    for key,value in [('parameter_updates',True),('source_model_sha256','wrong'),
                      ('objective_code_sha256','wrong'),('checkpoint_step',1166)]:
        p.write_text(json.dumps(dict(costs,**{key:value})))
        with pytest.raises(ValueError): r.read_cost_evidence(p,.005)
