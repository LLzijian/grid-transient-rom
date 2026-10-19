import sys,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'experiments'))
import numpy as np
import torch
import pytest
from closure_oracle_q0_20261019 import reconstruct,oracle_prediction
from closure_causal_recovery_20261019 import fixture,truths,toy_source,copy_runtime

def test_reconstruction_and_prediction_at_forecast_origin():
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    source,clean,noisy,scale,meta=fixture('saturation',5101,0.,device,time.perf_counter()+60)
    qs,digests=reconstruct('saturation',5101,device,time.perf_counter()+60)
    assert digests==meta['clean_sha256']
    term,coefficient=truths()['saturation'];truth=copy_runtime(source,[term],device)
    with torch.no_grad():truth.coefficients.fill_(coefficient)
    truth.encode=lambda *args: (_ for _ in ()).throw(AssertionError('oracle must never call encoder'))
    obs=torch.tensor(noisy[0],dtype=torch.float64,device=device)
    with torch.no_grad():prediction=oracle_prediction(truth,obs,qs[0],50,.01,time.perf_counter()+60)
    np.testing.assert_allclose(prediction.cpu().numpy(),clean[0][:,11:],atol=2e-13,rtol=2e-13)
    assert not qs[0].requires_grad

def test_expired_reconstruction_cannot_proceed():
    with pytest.raises(TimeoutError):reconstruct('edge',5101,torch.device('cpu'),time.perf_counter()-1)
