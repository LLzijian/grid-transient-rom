"""Engineering and fixed interval endpoint logic only. No fit, scoring, W or data."""
import builtins
import importlib.util
import math
import os
from pathlib import Path
import pytest

spec=importlib.util.spec_from_file_location('dissipative_eq_engineering',Path(__file__).parents[1]/'experiments/p_dissipative_identifiability_20261031.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)

def test_exact_independent_releases_and_zero_Q_estimation():
    expected={'source_sha256':'frozen','main_parameter_estimates_upper_bound':0}
    r={'released':True,'allowed_stage':'analyze',**expected}
    m.verify_release('analyze',r,expected)
    for update in ({'released':'true'},{'allowed_stage':'estimate'},{'source_sha256':'other'},
                   {'main_parameter_estimates_upper_bound':1},{'main_parameter_estimates_upper_bound':False}):
        with pytest.raises(PermissionError):m.verify_release('analyze',{**r,**update},expected)

def test_release_expected_does_not_decode_or_import_numerical_libraries(monkeypatch):
    original=builtins.__import__
    def guarded(name,*a,**kw):
        if name.split('.')[0] in ('numpy','scipy','mpmath','andes'):pytest.fail('engineering control imported scientific package')
        return original(name,*a,**kw)
    monkeypatch.setattr(builtins,'__import__',guarded)
    e=m.release_expected('estimate');q=m.release_expected('analyze')
    assert e['main_parameter_estimates_upper_bound']==10 and e['same_target_qr_checks_upper_bound']==10
    assert q['main_parameter_estimates_upper_bound']==0 and q['same_target_qr_checks_upper_bound']==0
    assert e['prepare_adoption_sha256']==q['prepare_adoption_sha256']
    assert q['operator_qualification_cells_upper_bound']==0

def test_Q_cannot_release_before_completed_E_hashes_exist(tmp_path,monkeypatch):
    monkeypatch.setattr(m,'PRIVATE',tmp_path)
    expected=m.release_expected('analyze')
    assert expected['estimate_manifest_sha256'] is None
    with pytest.raises(PermissionError):m.checked_release('analyze')

def test_unknown_started_and_completed_stage_cannot_replay(tmp_path):
    directory=tmp_path/'estimate';m.claim_stage(directory,'estimate',{'frozen':'control'})
    saved=m.read(directory/'started.json')
    assert saved['native_interpreter']['pid']==os.getpid()
    with pytest.raises(FileExistsError):m.claim_stage(directory,'estimate',{})
    m.write_once(directory/'complete.json',{'status':'COMPLETED'})
    with pytest.raises(FileExistsError):m.claim_stage(directory,'estimate',{})
    with pytest.raises(FileExistsError):m.write_once(directory/'started.json',{})

def test_interval_division_and_signed_multiplication_round_both_endpoints_outward():
    div=m.interval_binary([1.,2.],[3.,4.],'div')
    assert div[0]<.25 and div[1]>2/3
    product=m.interval_binary([-2.,-1.],[-3.,4.],'mul')
    assert product[0]<-8 and product[1]>6
    sub=m.interval_binary([1.,2.],[3.,4.],'sub')
    assert sub[0]<-3 and sub[1]>-1

@pytest.mark.parametrize('denominator',[[-1.,1.],[0.,2.],[-2.,0.]])
def test_interval_division_cannot_use_point_value_to_bypass_zero(denominator):
    assert m.interval_binary([1.,2.],denominator,'div') is None

def test_initial_halfwidth_division_and_endpoints_both_round_outward():
    box=m.theta_box([1.,2.,3.,4.],.1,[3.,3.,3.,3.])
    for center,interval,width in zip([1.,2.,3.,4.],box['theta'],box['halfwidths']):
        assert width[0]<.1/3<width[1]
        assert interval[0]<center-.1/3 and interval[1]>center+.1/3

@pytest.mark.parametrize('radius,scale',[(float('nan'),1.),(-1.,1.),(1.,0.),(1.,float('inf'))])
def test_invalid_initial_box_is_unknown_not_clipped(radius,scale):
    assert m.theta_box([1.,2.,3.,4.],radius,[scale]*4) is None

def test_overflowing_interval_endpoint_is_unknown():
    assert m.interval_binary([1e308,1e308],[2.,2.],'mul') is None
    assert m.theta_box([1e308]*4,1e308,[1.]*4) is None

def test_joint_box_sign_logic_and_unknown_g_zero():
    r=m.map_box({'theta':[[2.,2.],[.5,.5],[1.,1.],[1.,1.]]})
    j=m.map_box({'theta':[[2.,2.],[2.,2.],[1.,1.],[1.,1.]]})
    assert r['R'] is True and r['J'] is False
    assert j['J'] is True and j['R'] is False
    unknown=m.map_box({'theta':[[2.,2.],[.5,.5],[0.,1.],[1.,1.]]})
    assert unknown['status']=='unknown' and unknown['R'] is None and unknown['J'] is None

def test_nonfinite_and_reversed_boxes_do_not_pass_signs():
    for box in (None,{}, {'theta':[[2.,1.]]*4},{'theta':[[float('nan'),1.]]*4}):
        out=m.map_box(box)
        assert out['status']=='unknown' and out['R'] is None and out['J'] is None

def test_file_sealing_is_exclusive_and_nonfinite_serializes_null(tmp_path):
    path=tmp_path/'sealed.json';m.write_once(path,{'status':'unknown','value':float('inf')})
    assert m.read(path)['value'] is None
    before=m.sha(path)
    with pytest.raises(FileExistsError):m.write_once(path,{'value':0.})
    assert m.sha(path)==before

def test_B1_initial_unknown_has_explicit_null_steps_and_no_positive_sign_gate():
    result=m.sensitivity_unknown('not reached','original_qualification',{})
    assert result['failure_stage']=='original_qualification'
    assert all(value is None for value in result['intermediate'].values())
    assert all(value is None for value in result['relative'].values())
    assert result['mapping']['R'] is None and result['mapping']['J'] is None

def test_B1_premise_failure_keeps_completed_values_and_available_ratios(tmp_path):
    # Bookkeeping fixture only: these are supplied scalars, not evaluated matrix
    # norms, a fitted target, or an actual sensitivity-radius calculation.
    state={'D':[1.,2.,3.,4.],'sigma_min':1.,'delta_A':2.,'delta_v':.5,'v_norm':2.,
      't':-1.,'r':[.1,.2],'r_N':[.01,.02,.03,.04],'r_norm':.25,'r_N_norm':.05}
    result=m.sensitivity_unknown('premise not passed','radius_premise',state)
    assert result['intermediate']['t']==-1.
    assert result['intermediate']['r']==state['r'] and result['intermediate']['r_N']==state['r_N']
    assert result['intermediate']['delta_A']==2. and result['intermediate']['sigma_min']==1.
    assert result['relative']=={'delta_A_over_sigma_min':2.,'delta_v_over_v':.25,'radius_over_gamma':None}
    assert result['intermediate']['radius'] is None
    assert result['mapping']['R'] is None and result['mapping']['J'] is None
    path=tmp_path/'private_progress.json';m.write_once(path,result)
    assert m.read(path)==result

def test_B1_partial_nonfinite_step_does_not_erase_prior_finite_steps():
    result=m.sensitivity_unknown('radius unavailable','radius',{
      'sigma_min':2.,'delta_A':1.,'delta_v':.5,'v_norm':2.,'gamma_norm':3.,'t':1.,
      'radius_term1':.75,'radius_term2':float('inf'),'radius':float('inf'),
      'r':[1.,float('nan'),2.]})
    assert result['intermediate']['radius_term1']==.75
    assert result['intermediate']['radius_term2'] is None and result['intermediate']['radius'] is None
    assert result['intermediate']['r']==[1.,None,2.]
    assert result['relative']['delta_A_over_sigma_min']==.5
    assert result['relative']['delta_v_over_v']==.25
    assert result['relative']['radius_over_gamma'] is None
    assert result['mapping']['R'] is None and result['mapping']['J'] is None

def test_B1_spectral_failure_can_keep_independent_rhs_progress():
    result=m.sensitivity_unknown('spectral norm unresolved','spectral_norm',
      {'sigma_min':2.,'delta_v':.2,'v_norm':.5,'r':[.1],'r_N':[.2]})
    assert result['relative']['delta_A_over_sigma_min'] is None
    assert result['relative']['delta_v_over_v']==.4
    assert result['intermediate']['delta_A'] is None and result['intermediate']['t'] is None
    assert result['intermediate']['r']==[.1]
