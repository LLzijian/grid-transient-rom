"""Information isolation and deterministic causal synthetic-run checks."""
import sys
from pathlib import Path
import numpy as np
import pytest
import torch

sys.path[:0] = [str(Path(__file__).resolve().parents[1]/'experiments'), str(Path(__file__).resolve().parents[1]/'src')]
import closure_causal_recovery_20261019 as study


def test_truth_copy_preserves_field_and_trainable_parameters():
    torch.set_num_threads(1)
    base = study.toy_source()
    term, _ = study.truths()['node']
    reference = base.with_terms([term])
    model = study.copy_runtime(base, [term], 'cpu')
    z = torch.randn(2, 40, dtype=torch.float64)*.1
    torch.testing.assert_close(reference.field(z), model.field(z), rtol=0, atol=1e-13)
    assert {n for n,p in reference.named_parameters() if p.requires_grad} == {n for n,p in model.named_parameters() if p.requires_grad}


def test_observation_only_fixture_noise_both_inputs_and_targets():
    torch.set_num_threads(1)
    _, clean, noisy, scales, metadata = study.fixture('saturation', 5101, .01, 'cpu', count=4, train_steps=13, heldout_steps=14)
    _, clean2, noisy2, scales2, _ = study.fixture('saturation', 5101, .01, 'cpu', count=4, train_steps=13, heldout_steps=14)
    assert clean[0].shape == (4,14,30)
    assert clean[1].shape == (4,15,30)
    assert not np.array_equal(noisy[0][:,:11],clean[0][:,:11])
    assert not np.array_equal(noisy[0][:,11:],clean[0][:,11:])
    assert not np.array_equal(clean[0],clean[1][:,:14])
    np.testing.assert_array_equal(clean[0],clean2[0])
    np.testing.assert_array_equal(noisy[0],noisy2[0])
    np.testing.assert_array_equal(scales,scales2)
    assert not metadata['q_supervision'] and not metadata['rhs_supervision']


def test_prediction_future_not_input_and_joint_gradients():
    base = study.toy_source()
    term, _ = study.truths()['edge']
    model = study.runtime_model(base,[term],5101,'cpu')
    obs = torch.randn(3,14,30,dtype=torch.float64)*.1
    perturbed = obs.clone()
    perturbed[:,11:] += 10
    predicted = study.predict(model,obs,3)
    torch.testing.assert_close(predicted,study.predict(model,perturbed,3),rtol=0,atol=0)
    loss = study.trajectory_loss(model,obs,torch.ones(30,dtype=torch.float64))
    loss.backward()
    assert model.coefficients.grad.abs().max() > 0
    assert any(p.grad is not None and p.grad.abs().max()>0 for n,p in model.named_parameters() if 'gru' in n)
    assert any(p.grad is not None and p.grad.abs().max()>0 for n,p in model.named_parameters() if 'structured' in n)
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)


def test_exactly_one_encoder_call_per_autonomous_prediction(monkeypatch):
    model = study.runtime_model(study.toy_source(),[study.truths()['saturation'][0]],5101,'cpu')
    calls = []
    original = model.encode
    def encode(history):
        calls.append(history.shape)
        return original(history)
    monkeypatch.setattr(model,'encode',encode)
    study.predict(model,torch.randn(2,16,30,dtype=torch.float64)*.1,5)
    assert calls == [torch.Size([2,11,30])]


def test_checkpoint_retains_full_state_and_atomic_pointer(tmp_path):
    model = study.runtime_model(study.toy_source(),[study.truths()['saturation'][0]],5101,'cpu')
    opt = study.optimizer(model)
    study.save_checkpoint(tmp_path,model,opt,np.random.default_rng(7),0,[])
    import json
    pointer = json.loads((tmp_path/'latest.json').read_text())
    assert pointer['updates']==0
    state = torch.load(tmp_path/pointer['checkpoint']/'training.pt',weights_only=False)
    assert 'optimizer' in state and 'torch_rng' in state and 'fit_numpy_rng' in state
    from symbolic_closure_rom import SymbolicClosureROM
    loaded = SymbolicClosureROM.from_export(tmp_path/pointer['checkpoint']/'model.json')
    z = torch.randn(2,40,dtype=torch.float64)*.1
    torch.testing.assert_close(model.field(z),loaded.field(z),rtol=0,atol=1e-13)
    with pytest.raises(OSError):
        study.save_checkpoint(tmp_path,model,opt,np.random.default_rng(7),0,[])
    assert (tmp_path/'checkpoint_00000'/'model.json').exists()


def test_expired_fit_keeps_failure_without_complete_endpoint(tmp_path):
    import time
    import json
    folder = tmp_path/'expired'
    with pytest.raises(TimeoutError):
        study.fit(study.toy_source(),[study.truths()['node'][0]],5101,'cpu',
                  torch.zeros(2,13,30,dtype=torch.float64),torch.ones(30,dtype=torch.float64),
                  1,folder,time.perf_counter()-1)
    failure = json.loads((folder/'failure.json').read_text())
    assert failure['completed_updates'] == 0
    assert not (folder/'result.json').exists()


@pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA unavailable')
def test_actual_cuda_joint_update_matches_cpu():
    torch.set_num_threads(1)
    term = study.truths()['node'][0]
    cpu = study.runtime_model(study.toy_source(),[term],5101,'cpu')
    gpu = study.copy_runtime(cpu,[term],'cuda')
    observations = torch.randn(3,13,30,dtype=torch.float64)*.1
    scales = torch.ones(30,dtype=torch.float64)
    a = study.trajectory_loss(cpu,observations,scales)
    b = study.trajectory_loss(gpu,observations.cuda(),scales.cuda())
    a.backward(); b.backward()
    torch.testing.assert_close(a,b.cpu(),rtol=1e-9,atol=1e-12)
    for p,q in zip(cpu.parameters(),gpu.parameters()):
        if p.grad is not None:
            torch.testing.assert_close(p.grad,q.grad.cpu(),rtol=1e-8,atol=1e-11)
    before = gpu.coefficients.detach().clone()
    study.optimizer(gpu).step()
    assert not torch.equal(before,gpu.coefficients)
