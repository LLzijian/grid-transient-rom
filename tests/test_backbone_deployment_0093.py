"""Synthetic contracts for the single fixed Loop0093 deployment."""
import importlib.util,sys
from pathlib import Path
import numpy as np
import pytest
ROOT=Path(__file__).resolve().parents[1];spec=importlib.util.spec_from_file_location('deploy0093',ROOT/'experiments'/'audit_backbone_deployment_0093.py');m=importlib.util.module_from_spec(spec);sys.modules[spec.name]=m;spec.loader.exec_module(m)

def test_inverse_softplus_positive_and_expected_blocks():
    x=np.array([1e-5,.2,3.]);assert np.allclose(np.log1p(np.exp(m.inverse_softplus(x))),x,atol=1e-12,rtol=0)
    D=np.zeros((26,10));D[0,:2]=[1,-1];out=m.expected_delta_A(np.ones(10),D,np.zeros(26),np.r_[2.,np.zeros(25)],np.arange(10.))
    assert out[10,0]==pytest.approx(-2) and out[10,1]==pytest.approx(2) and out[10,10]==0 and out[19,19]==pytest.approx(-9)
    with pytest.raises(ValueError):m.inverse_softplus([0])

def test_gate_branching_keeps_modal_failure_for_forecast():
    assert m.decision(True,True,0)[0]
    assert not m.decision(False,True,13)[0] and not m.decision(True,False,13)[0]

def test_numerical_sensitivity_and_local_alpha_are_distinct():
    # A sensitivity failure is numerical-inconclusive, not a local rejection.
    assert m.decision(False, True, 13)==(False,'numerical_inconclusive')
    assert m.decision(True, False, 13)==(False,'local_candidate_rejected')

def test_risk_severe_metrics_and_prefix_shape_fail_closed():
    labels=np.array(['stable']*311+['unstable']*35);base=np.ones((3,346));cand=.8*base
    _,_,ratio,passed=m.trajectory_gates({'event_error':cand,'labels':labels},{'event_error':base,'labels':labels})
    assert passed and ratio['average']==pytest.approx(.8) and ratio['severe_worst4']==pytest.approx(.8)
    with pytest.raises(ValueError):m.trajectory_summary(np.ones((3,345)),labels)

def test_synthetic_write_reload_masks_and_frozen_state(tmp_path):
    import torch
    class Structured(torch.nn.Module):
        def __init__(self, mask=True):
            super().__init__();self.coupling=torch.nn.Parameter(torch.zeros(26,dtype=torch.float64));self.damping=torch.nn.Parameter(torch.zeros(10,dtype=torch.float64));self.frozen=torch.nn.Parameter(torch.tensor([7.],dtype=torch.float64),requires_grad=False)
            self.register_buffer('coupling_mask',torch.ones(26,dtype=torch.float64) if mask else torch.zeros(26,dtype=torch.float64));self.register_buffer('damping_mask',torch.ones(10,dtype=torch.float64))
    class Model(torch.nn.Module):
        def __init__(self, mask=True):super().__init__();self.structured=Structured(mask)
    model=Model();before={k:v.detach().clone() for k,v in model.state_dict().items()}
    m._apply_amplitudes(model.structured,np.linspace(.1,1.,26),np.linspace(.2,1.1,10))
    changed=m._freeze_check(before,model.state_dict());assert set(changed)=={'structured.coupling','structured.damping'}
    path=tmp_path/'model.pt';torch.save(model.state_dict(),path);loaded=Model();loaded.load_state_dict(torch.load(path,weights_only=True));assert m._freeze_check(before,loaded.state_dict())==changed
    with pytest.raises(AssertionError):m._apply_amplitudes(Model(mask=False).structured,np.ones(26),np.ones(10))

def test_cache_copy_never_reuses_prior_candidate_save_keys(tmp_path):
    raw=tmp_path/'old.npz'
    np.savez_compressed(raw,event_ids=np.arange(346),labels=np.array(['stable']*346),lengths=np.ones(346),truth=np.zeros((346,2,30)),qenc=np.zeros((346,3)),mass=np.ones(10),p_scale=np.ones(10),fine_baseline=np.zeros((346,3,33)),coarse_baseline=np.zeros((346,2,33)),fine_candidate=np.ones((346,3,33)),coarse_candidate=np.ones((346,2,33)))
    with np.load(raw,allow_pickle=False) as data: cache=m._cache_mapping(data)
    assert 'fine_candidate' not in cache and 'coarse_candidate' not in cache
    target=tmp_path/'new.npz';np.savez_compressed(target,**cache,fine_candidate=np.zeros((346,3,33)),coarse_candidate=np.zeros((346,2,33)))
    with np.load(target,allow_pickle=False) as saved: assert saved['fine_candidate'].shape==(346,3,33)

def test_missing_nonfinite_sources_and_frozen_mutation_fail_closed(tmp_path):
    import torch
    with pytest.raises(FileNotFoundError): m.sha(tmp_path/'missing-source.json')
    with pytest.raises(ValueError): m.inverse_softplus([np.nan])
    before={'structured.coupling':torch.zeros(1), 'structured.damping':torch.zeros(1),
            'encoder':torch.ones(1)}
    after={key:value+1 for key,value in before.items()}
    with pytest.raises(AssertionError): m._freeze_check(before,after)

def test_baseline_row_reproduction_rejects_changed_identity_or_value():
    rows=[{'event_id':str(i),'label':'stable','available_steps':10,'horizon':1,
           'baseline':1.} for i in range(1038)]
    copy=[dict(row) for row in rows]
    m._same_rows(rows,copy)
    copy[1]['baseline']=1.001
    with pytest.raises(AssertionError): m._same_rows(rows,copy)
