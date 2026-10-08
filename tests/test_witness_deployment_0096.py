"""Synthetic witness partition/reload guards for Loop0096."""
import importlib.util,sys
from pathlib import Path
import numpy as np
import pytest
ROOT=Path(__file__).resolve().parents[1];spec=importlib.util.spec_from_file_location('witness0096',ROOT/'experiments'/'audit_witness_deployment_0096.py');m=importlib.util.module_from_spec(spec);sys.modules[spec.name]=m;spec.loader.exec_module(m)

def test_witness_partition_shape_sign_and_near_floor_softplus():
    beta=np.r_[np.linspace(-.1,.1,26),np.full(10,-.99999999e-8)];k0=np.ones(26);d0=np.full(10,1e-8)
    k,d=m.partition_witness(beta,k0,d0)
    assert k.shape==(26,) and d.shape==(10,) and np.all(k>0) and np.all(d>0)
    sys.path.insert(0,str(ROOT/'experiments'));import audit_backbone_deployment_0093 as d93
    raw=d93.inverse_softplus(d);assert np.max(np.abs(np.log1p(np.exp(raw))-d))<1e-15
    assert len(m.C95_SHA)==64

def test_invalid_witnesses_fail_closed():
    with pytest.raises(ValueError):m.partition_witness(np.zeros(35),np.ones(26),np.ones(10))
    with pytest.raises(ValueError):m.partition_witness(np.r_[np.zeros(26),-np.ones(10)],np.ones(26),np.zeros(10))
    with pytest.raises(ValueError):m.partition_witness(np.full(36,np.nan),np.ones(26),np.ones(10))


def test_actual_near_floor_parameter_reload(tmp_path):
    import torch
    sys.path.insert(0,str(ROOT/'experiments'));import audit_backbone_deployment_0093 as d93
    class Pair(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.coupling=torch.nn.Parameter(torch.zeros(26,dtype=torch.float64))
            self.damping=torch.nn.Parameter(torch.zeros(10,dtype=torch.float64))
            self.register_buffer('coupling_mask',torch.ones(26,dtype=torch.float64))
            self.register_buffer('damping_mask',torch.ones(10,dtype=torch.float64))
    model=Pair(); k=np.ones(26); d=np.linspace(1e-8,1e-7,10)
    d93._apply_amplitudes(model,k,d)
    path=tmp_path/'synthetic.pt';torch.save(model.state_dict(),path)
    loaded=Pair();loaded.load_state_dict(torch.load(path,weights_only=True))
    np.testing.assert_allclose(torch.nn.functional.softplus(loaded.damping).detach().numpy(),d,rtol=1e-13,atol=0)


def test_missing_witness_file_and_vector_fail_closed(tmp_path,monkeypatch):
    path=tmp_path/'synthetic.npz';monkeypatch.setattr(m,'A95',path)
    with pytest.raises(FileNotFoundError):m._witness({})
    np.savez(path,beta=np.zeros(36),k0=np.ones(26),d0=np.ones(10))
    with pytest.raises(KeyError):m._witness({})
