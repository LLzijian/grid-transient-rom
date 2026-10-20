"""Focused QA with independent arrays; never generate formal extended teacher data."""
import copy
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
import pytest
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'experiments'),str(ROOT/'src')]
import long_horizon_discrimination_20261021 as study


def arrays():return np.zeros((16,500,30),dtype=np.float64)


def test_raw_pairs_retain_observed_arrays_and_refuse_overwrite(tmp_path):
    a=np.arange(180,dtype=float).reshape(2,3,30);b=a+.125
    identity=study.save_raw_pair(tmp_path,'actual',a,b)
    with np.load(identity['path']) as saved:
        np.testing.assert_array_equal(saved['coarse'],a)
        np.testing.assert_array_equal(saved['fine'],b)
        assert set(saved.files)=={'coarse','fine'}
    assert identity['sha256']==hashlib.sha256(Path(identity['path']).read_bytes()).hexdigest()
    with pytest.raises(ValueError):study.save_raw_pair(tmp_path,'actual',a,b)
    with pytest.raises(ValueError):study.save_raw_pair(tmp_path,'hidden',np.zeros((2,3,40)),np.zeros((2,3,40)))


def test_integer_windows_and_complete_denominators():
    zero=arrays();model=np.broadcast_to(np.arange(1,501,dtype=float)[None,:,None],zero.shape).copy()
    result=study.score_model(model,model,zero,zero,np.ones(30))
    for name,(start,end) in study.WINDOWS.items():
        row=result[name]
        assert row['time_points']==end-start and row['count']==16 and row['eligible_count']==16
        assert row['mean']==pytest.approx(np.arange(1,501,dtype=float)[start:end].dot(np.arange(1,501,dtype=float)[start:end])/(end-start))
        assert len(row['per_event_mse'])==16
    with pytest.raises(ValueError):study.summarize([0.]*15,[True]*15)
    with pytest.raises(ValueError):study.score_model(model[:15],model[:15],zero[:15],zero[:15],np.ones(30))


def test_discrete_perturbation_bound_and_pair_direction():
    rng=np.random.default_rng(2831);tc=rng.normal(size=(16,500,30));mc=tc+rng.normal(size=tc.shape)*.01
    tf=tc+rng.normal(size=tc.shape)*1e-6;mf=mc+rng.normal(size=tc.shape)*1e-6
    sc=np.linspace(.7,1.4,30);score=study.score_model(mc,mf,tc,tf,sc)
    for name,(start,end) in study.WINDOWS.items():
        fine_mse=np.mean(((mf[:,start:end]-tf[:,start:end])/sc)**2,axis=(1,2))
        assert np.all(np.abs(fine_mse-np.array(score[name]['per_event_mse']))<=np.array(score[name]['per_event_B'])+1e-16)
    z=arrays();sin=study.score_model(z,z,z,z,np.ones(30))
    other=study.score_model(z+.1,z+.1,z,z,np.ones(30))
    pairs=study.pairwise({'sin':sin,'cos':other,'empty':other})
    assert all(p['primary_2_5']['sin_advantage_rule'] for p in pairs.values())
    assert all(p['primary_2_5']['events_D_gt_B']==16 for p in pairs.values())


@pytest.mark.parametrize('failure',['nonfinite','step'])
def test_full_prefix_qualification_retains_two_second_secondary(failure):
    z=arrays();mc=z.copy();mf=z.copy()
    if failure=='nonfinite':mc[3,400,0]=np.nan
    else:mf[3,400,0]=.002
    result=study.score_model(mc,mf,z,z,np.ones(30))
    assert result['primary_2_5']['status']=='UNRESOLVED'
    assert result['primary_2_5']['mean'] is None
    assert result['primary_2_5']['count']==16
    assert result['primary_2_5']['eligible_count']==15
    assert result['secondary_1_2']['status']=='QUALIFIED'
    assert result['cumulative_0_2']['status']=='QUALIFIED'
    json.dumps(result,allow_nan=False)
    # A reconverged tail cannot erase earlier numerical failure.
    mf=z.copy();mf[3,50,0]=.002
    assert study.score_model(z,mf,z,z,np.ones(30))['primary_2_5']['mean'] is None


def manifest(tmp_path):
    rows=[]
    for seed in study.SEEDS:
        models={}
        for name in study.NAMES:
            a=tmp_path/f'{seed}_{name}_model.json'
            terms={'sin':[study.prior.canonical_terms()[0]],'cos':[study.prior.canonical_terms()[1]],'empty':[]}[name]
            a.write_text(json.dumps({'format':'causal_history_residual_v1','base':{'terms':terms}}),encoding='utf-8')
            b=tmp_path/f'{seed}_{name}_training.pt';b.write_bytes(b'frozen-not-deserialized')
            models[name]={'path':str(a.resolve()),'training_path':str(b.resolve()),
                          'model_sha256':hashlib.sha256(a.read_bytes()).hexdigest(),
                          'training_sha256':hashlib.sha256(b.read_bytes()).hexdigest()}
        rows.append({'seed':seed,'scales':[1.]*30,'old_val_clean_sha256':'0'*64,
                     'teacher_source_model_sha256':'1'*64,'models':models})
    return {'seeds':list(study.SEEDS),'rows':rows}


def test_nine_model_byte_barrier_and_seed_structure_rejections(tmp_path):
    value=manifest(tmp_path);study.validate_manifest(value)
    bad=copy.deepcopy(value);bad['rows'][0]['seed']=5102
    with pytest.raises(ValueError):study.validate_manifest(bad)
    bad=copy.deepcopy(value);del bad['rows'][2]['models']['empty']
    with pytest.raises(ValueError):study.validate_manifest(bad)
    Path(value['rows'][2]['models']['empty']['training_path']).write_bytes(b'changed')
    with pytest.raises(ValueError):study.validate_manifest(value)
    with pytest.raises(ValueError):study.read_manifest({})


@pytest.mark.parametrize('name',study.NAMES)
def test_each_step_single_initialization_and_future_guard(name):
    from closure_causal_recovery_20261019 import runtime_model
    from closure_discovery_benchmarks import toy_source
    from closure_causal_history_residual_20261019 import CausalHistoryResidual
    torch.set_num_threads(1)
    assert torch.cuda.is_available(), 'registered CUDA QA cannot skip'
    arr,_,_=study.prior.timing_observations('cuda',5101)
    terms={'sin':[study.prior.SIN_TERM],'cos':[study.prior.COS_TERM],'empty':[]}[name]
    history=arr[:,:11].copy()
    m=CausalHistoryResidual.from_train(runtime_model(toy_source(),terms,5101,'cuda'),torch.as_tensor(history,device='cuda'))
    calls=[];original=m.encode
    def counted(h):calls.append(tuple(h.shape));return original(h)
    m.encode=counted
    for dt in (.01,.005):
        pred=study.predict(m,history,dt,None,steps=2)
        assert pred.shape==(16,2,30)
    assert calls==[(16,11,30),(16,11,30)]
    with pytest.raises(ValueError):study.predict(m,arr[:,:12],.01,None,steps=2)
    with pytest.raises(ValueError):m.encode(torch.as_tensor(arr[:,:12],device='cuda'))
    future=arr.copy();future[:,11:]+=100
    torch.testing.assert_close(original(torch.as_tensor(arr[:,:11],device='cuda')),
                               original(torch.as_tensor(future[:,:11],device='cuda')),rtol=0,atol=0)
