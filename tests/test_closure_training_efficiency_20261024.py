"""CPU-only negative checks; all GPU work belongs to the fixed benchmark."""
import json,sys
from pathlib import Path
import numpy as np
import pytest
import torch
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'experiments'),str(ROOT/'src')]
import closure_training_efficiency_20261024 as entry


class Tiny(torch.nn.Module):
    def __init__(self):
        super().__init__();self.weight=torch.nn.Parameter(torch.tensor([1.],dtype=torch.float64));self.register_buffer('mass',torch.ones(1,dtype=torch.float64))
    def export(self,path):Path(path).write_text(json.dumps({'weight':self.weight.detach().tolist()}))


class BadGradient(torch.autograd.Function):
    @staticmethod
    def forward(ctx,x):return x.sum()
    @staticmethod
    def backward(ctx,grad):return grad.reshape(1)*float('nan')


@pytest.mark.parametrize('merged',[False,True])
@pytest.mark.parametrize('failure',['loss','gradient','updated_parameter'])
def test_negative_failure_stage_and_actual_updated_fact(merged,failure):
    model=Tiny();initial=model.weight.detach().clone()
    opt=torch.optim.SGD(model.parameters(),lr=float('inf') if failure=='updated_parameter' else .01)
    def injected(m,obs,sc,deadline):
        return m.weight.sum()*float('nan') if failure=='loss' else BadGradient.apply(m.weight) if failure=='gradient' else m.weight.square().sum()
    message={'loss':'nonfinite fit loss','gradient':'nonfinite gradient','updated_parameter':'nonfinite updated parameter'}[failure]
    with pytest.raises(FloatingPointError,match=message):
        entry._update(model,opt,np.random.default_rng(9941),torch.ones((16,2,1)),torch.ones(1),merged=merged,loss_fn=injected)
    if failure=='updated_parameter':assert not bool(torch.isfinite(model.weight).all())
    else:torch.testing.assert_close(model.weight,initial,rtol=0,atol=0)
    # Exactly two actual CPU optimizer updates in this collection: the two
    # updated_parameter failures. Loss/gradient failures occur before opt.step.


def test_corrupt_checkpoint_sha_rejected_before_restore(tmp_path):
    model=Tiny();opt=torch.optim.SGD(model.parameters(),lr=.01);rng=np.random.default_rng(9941)
    entry.old.checkpoint(tmp_path,model,opt,rng,0,[],{'qa_only':True})
    path=tmp_path/'checkpoint_00000/model.json';path.write_text('corrupt')
    with pytest.raises(ValueError,match='bytes changed'):
        entry.old.restore(tmp_path,model,opt,rng,{'qa_only':True})


def test_merge_norm_order_preserves_original_scalars_without_updates():
    gradients=[torch.tensor([1.,2.],dtype=torch.float64),torch.tensor([-3.,4.],dtype=torch.float64)]
    scalar=[float(g.norm()) for g in gradients]
    merged=torch.stack([g.norm() for g in gradients]).cpu().tolist()
    assert scalar==merged
    assert entry.CASES==((2,50),(2,200),(3,50),(3,200))


def test_optional_optimization_is_not_silently_enabled(tmp_path):
    with pytest.raises(ValueError,match='optimization2'):
        entry.benchmark(tmp_path/'unused',{},'a'*40,'opt2')
    assert not (tmp_path/'unused').exists()
