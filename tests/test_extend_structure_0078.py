"""Synthetic contracts for the 0078 extension runner; no project data."""
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest
torch = pytest.importorskip('torch')

ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT/'experiments'))
spec=importlib.util.spec_from_file_location('extension',ROOT/'experiments'/'extend_structure_0078.py')
extension=importlib.util.module_from_spec(spec); spec.loader.exec_module(extension)


def test_extension_schedule_is_the_registered_three_horizon_cycle():
    assert [extension.horizon_for_extension(s) for s in (1201,1202,1203,3000)] == [1.,2.,5.,5.]
    with pytest.raises(ValueError): extension.horizon_for_extension(1200)


def test_optimizer_restore_preserves_named_groups_and_moments(tmp_path):
    class M(torch.nn.Module):
        def __init__(self): super().__init__(); self.a=torch.nn.Parameter(torch.tensor([1.],dtype=torch.float64)); self.b=torch.nn.Parameter(torch.tensor([2.],dtype=torch.float64))
    model=M(); original=torch.optim.Adam([{'params':[model.a],'lr':1e-3},{'params':[model.b],'lr':1e-2}])
    (model.a.square()+model.b.square()).backward(); original.step()
    torch.save(original.state_dict(),tmp_path/'optimizer.pt')
    (tmp_path/'checkpoint.json').write_text(json.dumps({'optimizer_parameter_names':[['a'],['b']]}))
    restored=extension.restore_optimizer(model,tmp_path)
    assert [g['lr'] for g in restored.param_groups] == [1e-3,1e-2]
    assert torch.equal(restored.state[model.a]['exp_avg'],original.state[model.a]['exp_avg'])
    assert torch.equal(restored.state[model.b]['exp_avg_sq'],original.state[model.b]['exp_avg_sq'])


def test_rng_restore_continues_exact_fit_and_selection_sequences(tmp_path):
    fit=np.random.default_rng(78); selection=np.random.default_rng(780); _=fit.integers(100); _=selection.integers(100)
    expected_fit=copy_fit=np.random.default_rng(); copy_fit.bit_generator.state=fit.bit_generator.state; expected_fit=copy_fit.integers(2**31,size=5)
    copy_select=np.random.default_rng(); copy_select.bit_generator.state=selection.bit_generator.state; expected_select=copy_select.integers(2**31,size=5)
    torch.save({'fit_numpy':fit.bit_generator.state,'selection_numpy':selection.bit_generator.state,'torch':torch.get_rng_state()},tmp_path/'rng_state.pt')
    got_fit,got_select=extension.restore_rng(tmp_path)
    assert np.array_equal(got_fit.integers(2**31,size=5),expected_fit)
    assert np.array_equal(got_select.integers(2**31,size=5),expected_select)


def test_frozen_state_detects_support_or_buffer_mutation():
    class M(torch.nn.Module):
        def __init__(self): super().__init__(); self.p=torch.nn.Parameter(torch.tensor(1.)); self.register_buffer('support',torch.ones(2))
    model=M(); frozen=extension.frozen_state(model); extension.assert_frozen(model,frozen)
    model.support[0]=0.
    with pytest.raises(AssertionError): extension.assert_frozen(model,frozen)


def test_terminal_failure_makes_only_that_arm_ineligible(tmp_path):
    def completed(arm):
        return {'arm':arm,'smoke':False,'steps':1200,'monitors':{0:{'risk':100.},800:{'risk':90.},1200:{'risk':80.}}}
    paths={}
    for arm in ('C-free','G-auto'):
        path=tmp_path/f'{arm}.json'; path.write_text(json.dumps(completed(arm))); paths[arm]=path
    failed=tmp_path/'G-fixed.json'; failed.write_text(json.dumps({'arm':'G-fixed','type':'FloatingPointError'})); paths['G-fixed']=failed
    evidence,chosen=extension.select_extension('C-free',paths)
    assert evidence['arm']=='C-free' and chosen==['C-free','G-auto']
    with pytest.raises(ValueError): extension.select_extension('G-fixed',paths)


def test_selected_checkpoint_must_equal_evidence_final_checkpoint(tmp_path):
    chosen=tmp_path/'step1200'; chosen.mkdir(); other=tmp_path/'other'; other.mkdir()
    hashes={'source_model':'abc','extension_code':'xyz'}
    evidence={'final_checkpoint':str(other),'hashes':{'source_model':'abc'}}
    with pytest.raises(ValueError): extension._match_training_provenance(evidence,chosen,hashes)
