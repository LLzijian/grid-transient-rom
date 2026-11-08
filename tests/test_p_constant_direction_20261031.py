"""Engineering and fixed small scalar projection fixtures; no real data or MP slot."""
import builtins
import importlib.util
import math
from pathlib import Path
import pytest

spec=importlib.util.spec_from_file_location('constant_eq_engineering',Path(__file__).parents[1]/'experiments/p_constant_direction_20261031.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)

def test_release_is_exact_and_Q_has_no_estimate_or_MP_budget():
    expected={'source_sha256':'frozen','eta_estimates_upper_bound':0,'MP100_checks_upper_bound':0}
    r={'released':True,'allowed_stage':'analyze',**expected};m.verify_release('analyze',r,expected)
    for update in ({'released':'true'},{'allowed_stage':'estimate'},{'eta_estimates_upper_bound':1},{'MP100_checks_upper_bound':1},{'source_sha256':'different'}):
        with pytest.raises(PermissionError):m.verify_release('analyze',{**r,**update},expected)

def test_read_only_release_method_never_imports_scientific_packages(monkeypatch):
    original=builtins.__import__
    def guarded(name,*a,**kw):
        if name.split('.')[0] in ('numpy','scipy','mpmath','andes'):pytest.fail('read-only engineering path imported science package')
        return original(name,*a,**kw)
    monkeypatch.setattr(builtins,'__import__',guarded)
    assert m.release_expected('estimate')['eta_estimates_upper_bound']==10
    assert m.release_expected('estimate')['MP100_checks_upper_bound']==10
    assert m.release_expected('analyze')['eta_estimates_upper_bound']==0

def test_no_unknown_completed_stage_replay(tmp_path):
    directory=tmp_path/'estimate';m.claim_stage(directory,'estimate',{'frozen':'release'})
    with pytest.raises(FileExistsError):m.claim_stage(directory,'estimate',{})
    m.write_once(directory/'complete.json',{'status':'COMPLETED'})
    with pytest.raises(FileExistsError):m.claim_stage(directory,'estimate',{})
    with pytest.raises(FileExistsError):m.write_once(directory/'started.json',{})

def test_all_ten_main_files_must_seal_before_reference_phase(tmp_path):
    # Opaque file controls only: no scalar estimates or MP evaluations here.
    for i in range(9):m.write_once(tmp_path/f'machine_{i:02d}'/'eta.json',{'opaque_slot':i})
    with pytest.raises(RuntimeError):m.seal_all_eta(tmp_path)
    assert not (tmp_path/'all_eta_sealed.json').exists()
    m.write_once(tmp_path/'machine_09'/'eta.json',{'opaque_slot':9})
    seal=m.seal_all_eta(tmp_path);m.verify_eta_seal(tmp_path,seal)
    assert len(seal['eta_sha256'])==10
    (tmp_path/'machine_02'/'eta.json').write_text('changed')
    with pytest.raises(ValueError):m.verify_eta_seal(tmp_path,seal)

def test_fixed_small_projection_uses_event_then_window_weights():
    events=[[1.,3.],[5.]]
    eta=m.weighted_mean(events,2)
    assert eta==3.5 and eta!=sum([1.,3.,5.])/3
    result=m.binary_projection_diagnostics(events,eta,2)
    assert result['corrected_mean']==0. and result['J0']==15.
    assert result['J_eta_direct']==2.75 and result['J_eta_plus_eta_squared']==15.
    assert result['identity_passed'] and result['corrected_mean_zero']

@pytest.mark.parametrize('events,N',[([[1.]],2),([[]],1),([[float('nan')]],1),([[float('inf')]],1)])
def test_missing_or_nonfinite_scalar_input_is_not_zero_or_reweighted(events,N):
    with pytest.raises(m.NumericUnknown):m.weighted_mean(events,N)

def test_same_constant_subtraction_keeps_difference_and_frozen_nu():
    Q,L,eta=2.,4.,3.
    assert (Q-eta)-(L-eta)==Q-L
    before=m.corrected_condition(4.,16.,1.,2.)
    after=m.corrected_condition(1.,1.,1.,2.)
    assert before['nu']==after['nu']==2.
    assert after['compatible'] is True

def test_empty_stage_is_zero_square_contribution_but_unknown_gate():
    result=m.corrected_condition(0.,0.,0.,None)
    assert result['reason']=='no_window' and result['R_Q'] is None and result['nu'] is None
    assert result['compatible'] is None and result['resolved'] is None
    with pytest.raises(m.NumericUnknown):m.corrected_condition(1.,0.,0.,None)
    with pytest.raises(m.NumericUnknown):m.corrected_condition(float('nan'),0.,0.,None)

def test_corrected_stage_retains_original_whole_event_weight(monkeypatch):
    monkeypatch.setattr(m,'DENOM',{'stable':2})
    class Matrix:
        def __init__(self,values):self.values=values
        def __getitem__(self,index):return self.values[index[0]]
    arrays={'Q8':Matrix([-2.5,-.5,1.5]),'L8':Matrix([-2.5,-.5,1.5])}
    rows=[{'id':'a','label':'stable','windows':{'8':{'offset':0,'count':2,'classes':['history','cross_t0']}}},
      {'id':'b','label':'stable','windows':{'8':{'offset':2,'count':1,'classes':['future']}}}]
    whole,_=m.corrected_C(arrays,rows,0,8,'stable')
    future,detail=m.corrected_C(arrays,rows,0,8,'stable','future')
    assert whole['C_Q']==2.75
    assert future['C_Q']==1.125 and future['N_available']==1
    assert detail[1]['original_weight']==.5

def test_Q_not_released_without_all_E_numeric_qualifications(tmp_path,monkeypatch):
    monkeypatch.setattr(m,'PRIVATE',tmp_path)
    with pytest.raises(PermissionError):m.checked_release('analyze')
    E=tmp_path/'estimate'
    m.write_once(E/'manifest.json',{'qualification':'unknown','slots':[{'numeric_passed':False} for _ in range(10)]})
    m.write_once(E/'complete.json',{});m.write_once(E/'all_eta_sealed.json',{})
    with pytest.raises(PermissionError):m.checked_release('analyze')

def test_whole_RMS_uses_squared_group_mixture_not_RMS_average():
    C=267/297*4.+30/297*9.
    result=m.corrected_condition(C,C,1.,1e-10,overall=True)
    assert result['R_Q']==math.sqrt(C)
    assert result['R_Q']!=267/297*2.+30/297*3.
