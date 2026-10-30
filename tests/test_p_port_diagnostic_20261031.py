"""Analytic/synthetic only: no event payload, model inference, optimizer or output job."""
import importlib.util
import math
from pathlib import Path
import sys

import numpy as np
import pytest
from scipy.linalg import expm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))
import p_port_diagnostic_20261031 as p


def test_constant_omega_matches_closed_solution():
    t = np.arange(51)*.01
    omega = np.full((len(t), 2), [1.2, -.8])
    g, a, p0 = np.array([.4,.7]), np.array([.6,1.1]), np.array([.2,.3])
    actual = p.linear_filter(omega, p0, g, a, .01)
    expected = np.exp(-t[:,None]*a)*p0 - g*omega*(1-np.exp(-t[:,None]*a))/a
    np.testing.assert_allclose(actual, expected, atol=1e-12, rtol=1e-12)


def test_linear_omega_matches_independent_augmented_matrix_exponential():
    t = np.arange(51)*.01
    w0, slope, a, g, p0 = .2, -.3, .8, .7, .4
    matrix = np.array([[-a,-g,0],[0,0,1],[0,0,0]],dtype=float)
    expected = np.array([(expm(matrix*x) @ [p0,w0,slope])[0] for x in t])
    actual = p.linear_filter((w0+slope*t)[:,None],[p0],[g],[a],.01)[:,0]
    np.testing.assert_allclose(actual, expected, atol=1e-12, rtol=1e-12)


def test_zero_a_is_exact_trapezoid_linear_forcing():
    omega=np.array([[1.,-2.],[2.,0.],[4.,3.]])
    result=p.linear_filter(omega,[.3,.8],[.2,.5],[0.,0.],.01)
    expected=np.vstack(([.3,.8],np.array([.3,.8])-np.cumsum(.005*(omega[:-1]+omega[1:])*[.2,.5],axis=0)))
    np.testing.assert_allclose(result,expected,atol=1e-15,rtol=1e-15)


@pytest.mark.parametrize("a",[0.,1e-15,1e-10,1e-5,.8])
def test_forward_reverse_reproduce_initial_path(a):
    t=np.arange(51)*.01
    omega=np.column_stack((np.sin(t),.1+.2*t))
    forward=p.linear_filter(omega,[.2,-.1],[.3,.7],[a,a],.01)
    reverse=p.linear_filter(omega[::-1],forward[-1],[.3,.7],[a,a],-.01)[::-1]
    np.testing.assert_allclose(forward,reverse,atol=1e-12,rtol=1e-12)


def test_reverse_matrix_matches_independent_exponential_and_filter():
    g,a=np.array([.2,.7]),np.array([0.,.6])
    expected=np.zeros((2,6))
    full=np.block([[-np.diag(a),-np.diag(g),np.zeros((2,2))],[np.zeros((2,2)),np.zeros((2,2)),np.eye(2)],[np.zeros((2,6))]])
    np.testing.assert_allclose(p.reverse_matrix(g,a),expm(-.01*full)[:2],atol=1e-12,rtol=1e-12)


def test_squared_terms_and_all_signed_cross_terms_exactly_reconstruct():
    rng=np.random.default_rng(19)
    truth=rng.normal(size=(30,3));a,b,c=rng.normal(size=(3,30,3))
    pred=truth+a+b+c;tf=truth+c;mf=tf+b
    result=p.squared_decomposition(pred,mf,tf,truth,np.array([.2,.4,.8]))
    values=result['per_machine_mse_pu2']
    rebuilt=sum(value for key,value in values.items() if key!='total')
    np.testing.assert_allclose(rebuilt,values['total'],atol=1e-12,rtol=1e-12)
    assert len(values)==7
    assert result['identity_max_abs_pu2']<1e-12


def test_cross_terms_allow_exact_cancellation():
    z=np.zeros((3,1));one=np.ones((3,1))
    result=p.squared_decomposition(z,z+one,z+2*one,z,[1.])
    assert result['per_machine_mse_pu2']['total'][0]==0
    assert sum(v[0] for k,v in result['per_machine_mse_pu2'].items() if k!='total')==0
    assert any(v[0]<0 for k,v in result['per_machine_mse_pu2'].items() if '__' in k)


def test_simpson_windows_and_partial_window_are_fixed():
    t=np.arange(6)*.01
    result,delta,dropped=p.window_integrals(np.column_stack((t**2,t)))
    np.testing.assert_allclose(result[:,0],[(.02**3)/3,(.04**3-.02**3)/3],atol=1e-15,rtol=1e-12)
    assert dropped==1 and len(delta)==2


def test_event_equal_weights_do_not_give_long_event_more_weight():
    design1=np.array([[1.,0.],[0.,1.]])
    design2=np.repeat(design1,5,axis=0)
    fit=p.event_equal_svd([design1,design2],[design1@[1.,2.],design2@[3.,4.]])
    np.testing.assert_allclose(fit['correction_projection_coefficients'],[2.,3.],atol=1e-12)
    np.testing.assert_allclose(fit['event_weight_sums'],[.5,.5],atol=1e-15)
    assert fit['rank']==2
    np.testing.assert_allclose(fit['weighted_projection_orthogonal_inner_product'],0,atol=1e-12)


def test_rank_deficiency_and_zero_columns_are_reported_without_ridge():
    design=np.column_stack((np.arange(1.,5.),np.arange(1.,5.)))
    fit=p.event_equal_svd([design],[design@[2.,3.]])
    assert fit['rank']==1 and fit['condition_scaled'] is None
    assert fit['weighted_residual_rms_pu']<1e-12
    zero=p.event_equal_svd([np.zeros((4,2))],[np.ones(4)])
    assert zero['rank']==0
    assert zero['weighted_residual_rms_pu']==pytest.approx(1.)
    assert p.event_equal_svd([np.empty((0,2))],[np.empty(0)])['status']=='unknown'


def test_trapezoid_check_reuses_fixed_simpson_coefficients(monkeypatch):
    x=np.array([[1.,2.],[2.,1.]])
    fit=p.event_equal_svd([x],[x@[2.,3.]])
    monkeypatch.setattr(np.linalg,'svd',lambda *a,**k:pytest.fail('sensitivity must not refit'))
    assert p.fixed_coefficient_residual([x],[x@[2.,3.]],fit['correction_projection_coefficients'])<1e-12


def test_import_does_not_read_frozen_payloads(monkeypatch):
    monkeypatch.setattr(Path,'read_text',lambda *a,**k:pytest.fail('import must be inert'))
    spec=importlib.util.spec_from_file_location('p_port_inert',p.__file__)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    assert module.MODELS==('A96','step000016')


def test_exact_frozen_scalar_qualification_including_stiff_a100():
    result=p.artificial_qualification()
    assert result['status']=='PASS' and result['cases']==16
    assert result['max_abs_error']<1e-12


def test_self_closure_requires_both_absolute_and_sensitivity_gates():
    assert p.self_closure_gate(.001,.0005)
    assert not p.self_closure_gate(.0010001,.1)
    assert not p.self_closure_gate(.0001,0.)
    assert p.self_closure_gate(1e-10,0.)


def test_right_endpoint_reference_is_exact_constant_forcing_and_reverse():
    omega=np.array([[1.],[2.],[3.]])
    result=p.right_endpoint_filter(omega,[.4],[.7],[0.],.01)
    np.testing.assert_allclose(result[:,0],[.4,.386,.365],atol=1e-15)
    reverse=p.right_endpoint_filter(omega[::-1],result[-1],[.7],[0.],-.01)[::-1]
    np.testing.assert_allclose(reverse,result,atol=1e-15)


def test_missing_event_windows_do_not_redistribute_its_weight():
    fit=p.event_equal_svd([np.eye(2),np.empty((0,2))],[np.ones(2),np.empty(0)])
    assert fit['status']=='unknown'
    assert fit['reason']=='incomplete_event_windows_no_reweighting'


def test_eta_uses_primary_scaling_without_another_parameter_fit(monkeypatch):
    x=np.array([[1.,2.],[2.,1.],[1.,0.]])
    fit=p.event_equal_svd([x],[x@[.2,.4]])
    monkeypatch.setattr(np.linalg,'svd',lambda *a,**k:pytest.fail('no sensitivity SVD fit'))
    result=p.svd_integration_sensitivity([x],[x.copy()],fit)
    assert result['eta']==0 and result['condition_gate'] and result['eta_gate']


def test_shared_absolute_beta_is_exactly_ten_fits_not_per_scope_or_model(monkeypatch):
    events=[]
    for i in range(297):
        x=np.zeros((15,30));t=np.arange(5)*.01
        x[10:,10:20]=t[:,None]+np.arange(1,11)[None,:]*.1
        x[10:,20:30]=1+t[:,None]**2+np.arange(10)[None,:]*.2
        events.append({'id':str(i),'label':'stable' if i<267 else 'unstable','x':x})
    def scopes(e,s):
        if e['label']=='stable' and s in ('stable5','stable8'):return 2 if s=='stable5' else 4
        if e['label']=='unstable' and s=='unstable_actual15':return 4
    monkeypatch.setattr(p,'scope_steps',scopes)
    parameters={label:{'p_scale':np.ones(10),'g':np.full(10,.7),'a':np.full(10,.5)} for label in p.MODELS}
    rows={label:[{'event_id':e['id'],'scope':scope,'qualified':True,'self_closure_passed':True,'tau_e':1e-8,'fixed_error_normalized_rms':1.}
                 for e in events for scope in p.SCOPES if scopes(e,scope) is not None] for label in p.MODELS}
    calls=[];original=np.linalg.svd
    def counted(*args,**kwargs):
        calls.append(args[0].shape)
        return original(*args,**kwargs)
    monkeypatch.setattr(np.linalg,'svd',counted)
    result,arrays=p.shared_constant_analysis(events,parameters,rows,{'coordinate_alignment':'matched','equilibrium_reference_alignment':'matched_at_1e12','mechanical_reverse_blocks_qualified':True})
    assert len(calls)==10 and result['estimated_parameter_sets']==10
    assert all(shape==(594,2) for shape in calls)
    assert len(arrays)>0
    assert len(result['machines'][0]['scopes']['stable5']['event_ids'])==267
    assert len(result['machines'][0]['scopes']['unstable_actual15']['event_ids'])==30


def test_unknown_coordinates_block_cross_truth_estimation():
    result,arrays=p.shared_constant_analysis([],{}, {},{'coordinate_alignment':'unknown'})
    assert result['status']=='unknown' and arrays=={}
