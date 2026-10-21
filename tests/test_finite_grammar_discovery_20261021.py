"""Finite-study QA uses independent inputs, never formal teacher train or val."""
import copy,inspect,json,sys
from pathlib import Path
import numpy as np
import pytest
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'experiments'),str(ROOT/'src')]
import finite_grammar_discovery_20261021 as study


def test_budget_counts_and_selector_isolation():
    assert study.counts()=={'updates_per_seed':768,'total_updates':2304,'checkpoints_per_seed':15,
                            'total_checkpoints':45,'total_restores':9,'total_rankings':27,
                            'total_constructions':27,'final_evaluation_loads':3}
    names=set(inspect.signature(study.search_seed).parameters)
    assert not names&{'teacher','teacher_identity','val','validation'}
    assert {'terms','coarse','fine','sc'}<=names


def test_pool_rule_and_signature_uniqueness():
    from closure_discovery_support import term_key
    from closure_expression_identity import mechanism_signature
    import hashlib
    p=study.generate_pool();terms=p['eligible_pool']
    assert len(terms)>=6 and p['selected']==terms[:6]
    assert p['keys']==[term_key(t) for t in terms[:6]]
    assert len(set(mechanism_signature(t) for t in terms))==len(terms)
    assert terms==sorted(terms,key=lambda t:(hashlib.sha256(term_key(t).encode('utf-8')).hexdigest(),term_key(t)))
    assert all(t['channel']=='edge' and study.variables(t['expression'])&{'qi','qj'} and study.variables(t['expression'])&{'s','c'} for t in terms)


def test_interval_chain_is_transitive_and_invalid_never_revives():
    def row(key,mean,B,valid=True):return {'key':key,'score':{'mean':mean,'B':B,'valid':valid}}
    # a overlaps b, b overlaps c, a does not overlap c: one connected group.
    rows=[row('c',0.,1.),row('b',1.5,1.),row('a',3.,1.),row('invalid',-100.,0.,False)]
    selected,ambiguous=study.choose(rows,2)
    assert [r['key'] for r in selected]==['a','b'] and ambiguous
    assert selected[0]['overlap_group']==['a','b','c']
    assert study.choose([row('a',0.,0.),row('bad',None,None,False)],2)==([],False)


def test_ranking_integer_window_denominator_and_all_event_qualification():
    tc=np.zeros((16,500,30));mc=np.broadcast_to(np.arange(1,501)[None,:,None]/1000,tc.shape).copy()
    score=study.score(mc,mc,tc,tc,np.ones(30))
    assert score['count']==16 and score['indices']==[200,500]
    assert score['mean']==pytest.approx(np.mean((np.arange(201,501)/1000)**2))
    fine=mc.copy();fine[0,1,0]+=.002
    failed=study.score(mc,fine,tc,tc,np.ones(30))
    assert not failed['valid'] and failed['mean'] is None and failed['count']==16
    assert sum(failed['eligible'])==15
    with pytest.raises(ValueError):study.score(mc[:15],mc[:15],tc[:15],tc[:15],np.ones(30))


def test_teacher_qualification_checks_history_and_all16_events():
    coarse=np.zeros((16,511,30));fine=coarse.copy()
    assert study.qualify_teacher(coarse,fine,np.ones(30))['valid']
    fine[15,0,0]=.002
    failed=study.qualify_teacher(coarse,fine,np.ones(30))
    assert not failed['valid'] and failed['count']==16 and sum(failed['eligible'])==15
    fine=coarse.copy();fine[7,510,0]=np.nan
    assert not study.qualify_teacher(coarse,fine,np.ones(30))['valid']
    with pytest.raises(ValueError):study.qualify_teacher(coarse[:15],fine[:15],np.ones(30))


def timing_observations():
    rng=np.random.default_rng(9902)
    return rng.normal(0,.01,(16,61,30)),np.ones(30)
def capture(m,opt,rng):
    return copy.deepcopy({'model':m.state_dict(),'optimizer':opt.state_dict(),'numpy':rng.bit_generator.state,
                          'cpu':torch.get_rng_state(),'cuda':torch.cuda.get_rng_state_all()})
def exact(a,b):
    if torch.is_tensor(a):assert torch.equal(a.cpu(),b.cpu()) and a.dtype==b.dtype
    elif isinstance(a,np.ndarray):np.testing.assert_array_equal(a,b)
    elif isinstance(a,dict):
        assert a.keys()==b.keys()
        for k in a:exact(a[k],b[k])
    elif isinstance(a,(list,tuple)):
        assert len(a)==len(b)
        for x,y in zip(a,b):exact(x,y)
    else:assert a==b


def test_real_32_checkpoint_resume_34_exact_params_adam_rng_output(tmp_path):
    from closure_causal_history_residual_20261019 import checkpoint,restore,train_update,prediction
    torch.set_num_threads(1);assert torch.cuda.is_available(),'registered CUDA QA cannot skip'
    obs,sc=timing_observations();term=study.generate_pool()['selected'][0]
    m,opt=study.make(term,9902,obs);rng=np.random.default_rng(np.random.SeedSequence([9902,7]))
    observed=torch.as_tensor(obs,device='cuda');scales=torch.as_tensor(sc,device='cuda')
    trace=[];identity={'qa_only':True,'source':'independent timing observations'}
    for _ in range(32):v,_=train_update(m,opt,rng,observed,scales);trace.append(v)
    checkpoint(tmp_path,m,opt,rng,32,trace,identity)
    for _ in range(2):v,_=train_update(m,opt,rng,observed,scales);trace.append(v)
    expected=capture(m,opt,rng)
    with torch.no_grad():output=prediction(m,observed).clone()
    torch.rand(7);torch.rand(7,device='cuda');rng.random(7)
    step,restored_trace=restore(tmp_path,m,opt,rng,identity)
    assert step==32 and len(restored_trace)==32
    for _ in range(2):v,_=train_update(m,opt,rng,observed,scales);restored_trace.append(v)
    exact(expected,capture(m,opt,rng));assert trace==restored_trace
    with torch.no_grad():actual=prediction(m,observed)
    torch.testing.assert_close(output,actual,rtol=0,atol=0)


def test_full_ranking_once_history_initialization_and_bound():
    torch.set_num_threads(1);assert torch.cuda.is_available()
    obs,_=timing_observations();m,_=study.make(study.generate_pool()['selected'][0],9902,obs)
    calls=[];original=m.encode
    def counted(h):calls.append(tuple(h.shape));return original(h)
    m.encode=counted
    coarse=study.predict(m,obs,.01,None);fine=study.predict(m,obs,.005,None)
    assert calls==[(16,11,30),(16,11,30)] and coarse.shape==(16,500,30) and fine.shape==coarse.shape
    tc=np.zeros_like(coarse);tf=tc.copy();sc=np.ones(30)
    row=study.score(coarse,fine,tc,tf,sc)
    fine_mse=np.mean((fine[:,200:500]-tf[:,200:500])**2,(1,2))
    mse=np.array(row['per_event_mse']);B=np.array(row['per_event_B'])
    assert np.all(np.abs(fine_mse-mse)<=B+1e-14)
    with pytest.raises(ValueError):m.encode(torch.as_tensor(obs[:,:12],device='cuda'))
