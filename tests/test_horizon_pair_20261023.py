"""One collection: two independent CUDA32-to34 checks, no formal data."""
import copy
import inspect
import json
from pathlib import Path
import sys
import numpy as np
import pytest
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'experiments'),str(ROOT/'src'),str(ROOT/'tests')]
import horizon_pair_20261023 as entry
from test_finite_grammar_discovery_20261021 import capture,exact
from test_finite_grammar_execution_20261022 import tiny_checkpoint


def test_shapes_slices_no_hidden_variables_and_registered_workload():
    observations=np.arange(16*511*30,dtype=float).reshape(16,511,30)
    assert entry.fit_slice(observations,50).shape==(16,61,30)
    assert entry.fit_slice(observations,200).shape==(16,211,30)
    np.testing.assert_array_equal(entry.fit_slice(observations,200),observations[:,:211])
    with pytest.raises(ValueError):entry.fit_slice(np.zeros((16,511,40)),200)
    with pytest.raises(ValueError):entry.fit_slice(observations,100)
    assert len(entry.order())==12 and len(set(entry.order()))==12
    assert entry.counts()=={'models':12,'updates_per_model':2048,'formal_updates':24576,
        'checkpoints':36,'restores':12,'dual_step_model_evaluations':24,
        'optimization_rk4_steps_short':614400,'optimization_rk4_steps_long':2457600}
    source=inspect.getsource(entry)
    assert 'fixture(' not in source and 'evaluate_study' not in source


def test_four_model_batch_flows_are_paired():
    for seed in entry.SEEDS:
        streams=[np.random.default_rng(np.random.SeedSequence([seed,7])) for _ in range(4)]
        for _ in range(5):
            batches=[rng.permutation(16)[:12] for rng in streams]
            assert all(np.array_equal(batches[0],b) for b in batches[1:])


def test_real_small_state_cannot_claim2048_or_corrupt_adam_rng_trace(tmp_path):
    _,state,identity=tiny_checkpoint(tmp_path) # two actual CPU updates only
    assert entry.checks.validate_state(state,identity,2)['updates']==2
    with pytest.raises(ValueError):entry.checks.validate_state(state,identity,2048)
    for field,value in [('trace',[]),('identity',{}),('torch_rng',torch.ones(2))]:
        bad=copy.deepcopy(state);bad[field]=value
        with pytest.raises(ValueError):entry.checks.validate_state(bad,identity,2)
    bad=copy.deepcopy(state);bad['optimizer']['state']={}
    with pytest.raises(ValueError):entry.checks.validate_state(bad,identity,2)
    # The final-state validator has no shared-restore<=512 restriction.
    from closure_causal_history_residual_20261019 import restore
    assert 'range(0, 513, 32)' in inspect.getsource(restore)
    assert '513' not in inspect.getsource(entry.checks.validate_checkpoint)


def test_metric_failure_retained_and_pairing_is_unresolved():
    rows=[{'seed':s,'candidate':c,'future':w,'metrics':{'512':{'error':'numerical_failure'}}}
          for s,c,w in entry.order()]
    paired=entry.paired_profile(rows)
    assert len(paired)==12 and all(not r['valid'] and r['decision']=='UNRESOLVED' for r in paired)


def test_pair_sign_and_combined_numerical_bounds():
    rows=[{'seed':s,'candidate':c,'future':w,'metrics':{str(t):{'primary':{
        'valid':True,'mean':1. if c==2 else 2.,'B':.1}} for t in (512,2048)}} for s,c,w in entry.order()]
    paired=entry.paired_profile(rows)
    assert all(r['competitor_minus_teacher_mse']==1. and r['numerical_error_bound']==.2
               and r['decision']=='TEACHER' for r in paired)
    changes=entry.paired_changes(paired)
    assert changes['long2048_teacher_separated_all_three']
    assert len(changes['contrasts'])==12
    assert all(r['relative_teacher_gain_change']==0. and r['numerical_error_bound']==.4 and r['direction']=='UNSEPARATED'
               for r in changes['contrasts'])


def test_missing_model_and_unresolved_endpoint_barrier():
    rows=[{'seed':s,'candidate':c,'future':w,'status':'FROZEN'} for s,c,w in entry.order()]
    entry.require_model_rows(rows)
    with pytest.raises(ValueError,match='all12'):entry.require_model_rows(rows[:-1])
    bad=copy.deepcopy(rows);bad[-1]['status']='UNRESOLVED'
    with pytest.raises(ValueError,match='actual endpoints'):entry.require_model_rows(bad)


@pytest.mark.parametrize('future,seed',[(50,9931),(200,9932)])
def test_real_window32_to34_exact_and_single_causal_initialization(tmp_path,future,seed):
    from closure_causal_history_residual_20261019 import checkpoint,restore,train_update,prediction,loss
    torch.set_num_threads(1);assert torch.cuda.is_available(),'registered CUDA QA cannot skip'
    # Independent numerical arrays, not a generated teacher trajectory.
    raw=np.random.default_rng(seed).normal(0,.01,(16,511,30))
    pool=json.loads((ROOT/'private_data/closure_mechanism_20261019/P1/bounded_edge_grammar01/frozen_pool/pool.json').read_text(encoding='utf-8'))
    terms=[pool['selected'][candidate] for candidate in entry.CANDIDATES]
    model,opt=entry.science.make(terms[0],seed,raw)
    other,_=entry.science.make(terms[1],seed,raw)
    assert torch.count_nonzero(model.base.coefficients)==0
    for name,value in model.state_dict().items():torch.testing.assert_close(value,other.state_dict()[name],rtol=0,atol=0)
    del other
    observed=torch.as_tensor(entry.fit_slice(raw,future),device='cuda');scales=torch.ones(30,device='cuda',dtype=torch.float64)
    calls=[];original=model.encode
    def counted(history):
        calls.append(history.detach().cpu().clone());return original(history)
    model.encode=counted
    with torch.no_grad():predicted=prediction(model,observed)
    assert len(calls)==1 and calls[0].shape==(16,11,30)
    torch.testing.assert_close(calls[0],observed[:,:11].cpu(),rtol=0,atol=0)
    assert predicted.shape==(16,future,30)
    expected_loss=((predicted-observed[:,11:])/scales).square().mean()
    torch.testing.assert_close(loss(model,observed,scales),expected_loss,rtol=0,atol=0)
    model.encode=original
    rng=np.random.default_rng(np.random.SeedSequence([seed,7]));trace=[]
    identity={'qa_only':True,'seed':seed,'future_steps':future}
    for _ in range(32):value,_=train_update(model,opt,rng,observed,scales,None);trace.append(value)
    checkpoint(tmp_path,model,opt,rng,32,trace,identity)
    for _ in range(2):value,_=train_update(model,opt,rng,observed,scales,None);trace.append(value)
    expected=capture(model,opt,rng)
    with torch.no_grad():expected_output=prediction(model,observed).clone()
    torch.rand(7);torch.rand(7,device='cuda');rng.random(7)
    step,restored=restore(tmp_path,model,opt,rng,identity);assert step==32
    for _ in range(2):value,_=train_update(model,opt,rng,observed,scales,None);restored.append(value)
    exact(expected,capture(model,opt,rng));assert trace==restored
    with torch.no_grad():actual=prediction(model,observed)
    torch.testing.assert_close(actual,expected_output,rtol=0,atol=0)
