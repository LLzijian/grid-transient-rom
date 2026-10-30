"""Artificial fixtures only. No real arrays, coefficients or predictions are read."""
import copy
import importlib.util
from pathlib import Path
import numpy as np
import pytest

spec=importlib.util.spec_from_file_location('history_eq',Path(__file__).parents[1]/'experiments/p_history_transfer_20261031.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)

def event(index=0,length=17,label='unstable'):
    t=np.arange(length)*.01;p=t*t+(.1+.01*index)*t+1+.03*index
    w=-(2*t+.1+.01*index+.3*p)/.7
    x=np.zeros((length,30));x[:,10:20]=w[:,None];x[:,20:30]=p[:,None]
    return {'id':str(index),'label':label,'x':x}

def events():return [event(i) for i in range(297)]

def test_H_cannot_see_last20_or_any_future():
    e=events();a,meta=m.target(e,0,'H',1.)
    e[0]['x'][9:,10:30]=np.nan
    b,_=m.target(e,0,'H',1.)
    assert a is not None and b is not None
    assert m.old.digest(a)==m.old.digest(b)
    assert meta['row_count']==594
    short=copy.deepcopy(e)
    for row in short:row['x']=row['x'][:9]
    c,_=m.target(short,0,'H',1.)
    assert m.old.digest(a)==m.old.digest(c)

def test_A_stable8_endpoint_inclusive_but_never_later():
    e=event(length=900,label='stable')
    a=m.collection([e],0,'all','A',4,1.)[0]
    e['x'][811:,10:]=np.nan
    b=m.collection([e],0,'all','A',4,1.)[0]
    assert m.old.digest(a)==m.old.digest(b)
    assert a['windows']==202 and a['tail_steps']==2
    e['x'][810,20]=np.nan
    assert m.collection([e],0,'all','A',4,1.)[0]['available'] is False

def test_A_phase_and_equal_event_window_weights():
    e=events();e[1]=event(1,length=18)
    t,meta=m.target(e,0,'A',1.)
    assert meta['event_weight_sums']==[1/297]*297
    assert np.sum(t['weights'])==pytest.approx(1.)
    c=m.source.coverage(18,'unstable')['A']['4']
    assert c['counts']=={'history':2,'cross_t0':1,'future':1} and c['tail_steps']==1

@pytest.mark.parametrize('damage',['missing','nonfinite'])
def test_target_does_not_reweight_missing_original_denominator(damage):
    e=events()
    if damage=='missing':e.pop()
    else:e[0]['x'][4,20]=np.nan
    t,meta=m.target(e,0,'H',1.)
    assert t is None and meta['denominator']==297

def test_known_shared_coefficients_and_independent_QR():
    t,_=m.target(events(),0,'H',1.)
    p=m.old.primary_svd(t);q=m.old.qr_check(t,p)
    np.testing.assert_allclose(p['beta40'],[.7,.3],rtol=1e-9,atol=1e-9)
    assert p['rank']==2 and p['svd_numerical_checks_passed'] and q['status']=='verified'

def test_rank_deficient_projection_is_never_response_coefficient():
    p={'rank':1,'beta40':[1.,2.]}
    assert m.coefficient(p,{'numeric_optimality_passed':True}) is None
    p['rank']=2
    assert m.coefficient(p,{'numeric_optimality_passed':False}) is None

def test_parameter_sealed_before_QR_and_no_reestimate(tmp_path,monkeypatch):
    t,c=m.target(events(),0,'H',1.);folder=tmp_path/'slot'
    real=m.old.qr_check;calls=[]
    def qr(target,p):
        assert (folder/'parameters.json').exists()
        calls.append('QR');return real(target,p)
    monkeypatch.setattr(m.old,'qr_check',qr)
    m.estimate_slot(folder,'H',0,t,c,{'hash':'frozen'})
    monkeypatch.setattr(m.old,'primary_svd',lambda *a:pytest.fail('reestimate'))
    m.estimate_slot(folder,'H',0,t,c,{'hash':'frozen'})
    assert calls==['QR']
    with pytest.raises(ValueError):m.estimate_slot(folder,'H',0,t,c,{'hash':'different'})

def test_unknown_started_slot_is_not_replayed(tmp_path):
    folder=tmp_path/'slot';folder.mkdir();(folder/'started.json').write_text('{}')
    with pytest.raises(RuntimeError):m.estimate_slot(folder,'H',0,None,{}, {})

def test_future_response_never_resets_or_reads_future_p_as_state():
    truth=np.column_stack([np.linspace(.2,.8,17),np.full(17,.3)])
    a,b=m.response_paths(truth,[.7,.3],1.)
    truth[1:,1]=np.nan
    c,d=m.response_paths(truth,[.7,.3],1.)
    np.testing.assert_array_equal(a,c);np.testing.assert_array_equal(b,d)
    assert np.unique(a[1:]).size>1

def test_equal_event_statistics_use_each_cell_own_proxy_and_require_all():
    small=m.old.equal_event_stats([np.array([1.])],[np.array([1.])],1,kind='weak')
    large=m.old.equal_event_stats([np.array([1.])],[np.array([0.])],1,kind='weak')
    assert small['discernible'] is True and large['compatible'] is True
    unknown=m.old.equal_event_stats([np.array([1e308])],[np.array([1e308])],1,kind='weak')
    assert unknown['status']=='unknown' and unknown['Q_S'] is None
    assert m.old.equal_event_stats([np.ones(1)],[np.ones(1)],297)['denominator']==297

def test_response_epsilon_cannot_borrow_another_coefficients_check_error():
    a=m.old.equal_event_stats([np.array([.1])],[np.array([.1])],1,check_error=0.)
    b=m.old.equal_event_stats([np.array([.1])],[np.array([.1])],1,check_error=.1)
    assert a['discernible'] is True and b['compatible'] is True

def cell(mode='compatible',size=1.):
    return {'status':'computed','compatible':mode=='compatible','discernible':mode=='resolved','Q_S':size,'Q_T':size,'error_L':size,'error_R':size}

def cells():
    result={}
    for kind in m.KINDS:
        result[kind]={}
        for scope in ('history_weak','history_local','A80','history_full_local'):
            result[kind][scope]={g:{str(s):cell() for s in (2,4,8)} for g in ('all',*m.PRIMARY)}
        for scope in ('history_full20','history_persistent','future20_weak','future20_local','future_persistent','A40'):
            result[kind][scope]={g:cell() for g in ('all',*m.PRIMARY)}
    return result

def qual():return {k:{'numeric_optimality':True,'identifiable':True,'signs_legal':True,'rank':2} for k in ('H','A')}

def test_C_requires_both_main_groups_full_history20_and_beta0_discriminability():
    c=cells();q=qual()
    assert m.branch_decision(q,c)['C_H'] is False
    c['beta0']['future20_weak']['stable']=cell('resolved')
    assert m.branch_decision(q,c)['C_H'] is True
    c['H']['history_full20']['unstable']=cell('unknown')
    assert m.branch_decision(q,c)['C_H'] is False

def test_transfer_failure_requires_joint_same_group_instant_and_persistent():
    c=cells();q=qual()
    for scope in ('future20_weak','future20_local','future_persistent'):c['H'][scope]['unstable']=cell('resolved')
    assert m.branch_decision(q,c)['transfer_failure_full_target_compatible'] is True
    c['H']['future20_weak']['unstable']=cell()
    d=m.branch_decision(q,c)
    assert d['transfer_failure_full_target_compatible'] is False
    assert d['accumulation_or_mixed_groups']['H']==['unstable']

def test_F_can_coexist_H_unidentifiable_and_illegal_A_reference():
    c=cells();q=qual();q['H']['identifiable']=False;q['A']['signs_legal']=False
    c['A']['A40']['all']=cell('resolved',1.)
    for s in (2,4,8):c['A']['A80']['unstable'][str(s)]=cell('resolved',1.)
    for scope in ('future20_weak','future20_local','future_persistent'):c['A'][scope]['unstable']=cell('resolved',1.)
    d=m.branch_decision(q,c)
    assert d['F_A'] and d['H_unidentifiable_A_identifiable'] and d['A_unconstrained_reference_only']
    assert d['C_H'] is False
    c['A']['A80']['unstable']['8']=cell('unknown')
    assert m.branch_decision(q,c)['F_A'] is False

def test_F_history_location_needs_all_three_weak_and_local_scales():
    c=cells();q=qual();c['A']['A40']['all']=cell('resolved')
    for scope in ('A80','history_weak','history_local'):
        for s in (2,4,8):c['A'][scope]['stable'][str(s)]=cell('resolved')
    c['A']['history_persistent']['stable']=cell('resolved')
    assert m.branch_decision(q,c)['F_A']
    c['A']['history_local']['stable']['8']=cell()
    assert not m.branch_decision(q,c)['F_A']

def test_stage_release_freezes_adoption_source_and_exact_budget():
    expected={'source_manifest_sha256':'S','source_adoption_sha256':'adopt','maximum_main_parameter_estimates':20}
    r={'released':True,'allowed_stage':'estimate','scalar_filter_calls_upper_bound':0,**expected}
    m.verify_release('estimate',r,expected)
    for update in ({'allowed_stage':'analyze'},{'source_manifest_sha256':'new'},{'source_adoption_sha256':'new'},{'maximum_main_parameter_estimates':21}):
        with pytest.raises(PermissionError):m.verify_release('estimate',{**r,**update},expected)

def test_analysis_helper_has_no_estimator_calls(monkeypatch):
    monkeypatch.setattr(m.old,'primary_svd',lambda *a:pytest.fail('Q must not estimate'))
    monkeypatch.setattr(m.old,'qr_check',lambda *a:pytest.fail('Q must not QR'))
    e=[event()];coeff={k:np.array([.7,.3]) for k in m.KINDS};checks={k:{'status':'PASS','normalized_error':0.} for k in m.KINDS}
    row,private=m.analyze_machine(e,0,coeff,1.,checks,qual())
    assert row['cells']['H']['future_persistent']['all']['status']=='unknown'
    assert row['cells']['H']['future_persistent']['all']['denominator']==297
    assert row['decision']['C_H'] is False

@pytest.mark.parametrize('a',[0.,-0.2,1.e-10,1.])
def test_fixed_positive_negative_time_analytic_checks(a):
    r=m.old.coefficient_check(.7,a,1.)
    assert r['status']=='PASS' and r['normalized_error']>=0
    assert r['signs_legal']==(a>=0)

def test_fixed_artificial_suite_schema():
    r=m.old.filters.artificial_qualification()
    assert r['status']=='PASS' and r['cases']==16

def test_ten_machine_combined_RMS_cannot_drop_unknown_machine():
    rows=[{'cells':{'beta0':{'cell':{**cell(), 'denominator':297}}}} for i in range(10)]
    out=m.combined_rms(rows)['beta0']['cell']
    assert out['status']=='computed' and out['error_L']==1.
    rows[4]['cells']['beta0']['cell']['status']='unknown'
    out=m.combined_rms(rows)['beta0']['cell']
    assert out['status']=='unknown' and out['error_L'] is None and out['machine_denominator']==10

def test_future_truth_p_only_scores_frozen_paths():
    truth=np.column_stack([np.linspace(-.2,.3,15),np.linspace(.1,.5,15)])
    before=m.response_paths(truth,[.7,.3],1.)[0]
    truth[1:,1]+=100
    after=m.response_paths(truth,[.7,.3],1.)[0]
    np.testing.assert_array_equal(before,after)
    assert np.mean((before[1:]-truth[1:,1])**2)>1000

def test_release_addendum_and_actual_source_closure_are_required():
    expected=m.release_expected('analyze')
    r={'released':True,'allowed_stage':'analyze','fixed_coefficient_analysis_only':True,**expected}
    m.verify_release('analyze',r,expected)
    assert expected['maximum_main_parameter_estimates']==0
    assert expected['fixed_coefficient_numerical_check_slots_upper_bound']==30
    for key in ('source_closure_sha256','numerical_reference_addendum_sha256','source_adoption_sha256'):
        with pytest.raises(PermissionError):m.verify_release('analyze',{**r,key:'different'},expected)
