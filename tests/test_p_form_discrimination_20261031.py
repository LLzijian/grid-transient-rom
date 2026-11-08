"""Artificial-only: no private source, events, weights, fitted beta or predictions."""
import ast
import copy
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"experiments"))
import p_form_discrimination_20261031 as p


def row(name,x,y):
    return {'event_id':name,'available':True,'windows':len(x),'X_S':np.asarray(x),'X_T':np.asarray(x).copy(),'y':np.asarray(y)}


def test_known_coefficients_unique_svd_and_independent_same_target_QR(monkeypatch):
    x=np.array([[1.,0.],[0.,1.],[.2,.7]])
    beta=np.array([.7,.3]);rows=[row('one',x,x@beta),row('two',x*.4,x*.4@beta)]
    target,coverage=p.weighted_target(rows,2)
    calls=[];original=np.linalg.svd
    def counted(*args,**kwargs):calls.append(1);return original(*args,**kwargs)
    monkeypatch.setattr(np.linalg,'svd',counted)
    primary=p.primary_svd(target);check=p.qr_check(target,primary)
    np.testing.assert_allclose(primary['beta40'],beta,atol=1e-12,rtol=1e-12)
    assert calls==[1] and check['status']=='verified'
    assert primary['svd_numerical_checks_passed'] and primary['independent_identifiability']
    assert primary['normalized_normal_equation_residual']<=1e-12


def test_event_equal_and_event_internal_equal_weights_ignore_record_length():
    short=np.eye(2);long=np.tile(short,(50,1))
    target,_=p.weighted_target([row('short',short,short@[1.,3.]),row('long',long,long@[3.,5.])],2)
    result=p.primary_svd(target)
    np.testing.assert_allclose(result['beta40'],[2.,4.],atol=1e-12)
    assert target['weights'][:2].sum()==pytest.approx(.5)
    assert target['weights'][2:].sum()==pytest.approx(.5)


def test_rank_deficiency_keeps_broader_projection_without_identifying_parameters():
    x=np.column_stack((np.arange(1.,6.),2*np.arange(1.,6.)))
    target,_=p.weighted_target([row('one',x,x@[.7,.2])],1)
    result=p.primary_svd(target);check=p.qr_check(target,result)
    assert result['rank']==1 and not result['independent_identifiability']
    assert result['condition'] is None and result['eta'] is None
    assert check['fit_vector_equal_1e12'] and check['beta_equal_1e12'] is None


def test_zero_column_and_missing_event_never_get_ridge_or_reweighted():
    x=np.column_stack((np.arange(1.,5.),np.zeros(4)))
    target,_=p.weighted_target([row('one',x,np.arange(1.,5.))],1)
    result=p.primary_svd(target)
    assert result['rank']==1 and result['column_weighted_L2_norms'][1]==0
    target,coverage=p.weighted_target([row('one',x,np.arange(1.,5.)),{'event_id':'missing','available':False}],2)
    assert target is None and coverage['denominator']==2 and coverage['available_events']==1


def test_multiscale_composite_simpson_tail_and_common80_coverage():
    t=np.arange(20)*.01;truth=np.column_stack((t**2,t))
    full={n:p.window_design(truth,n,1.) for n in p.LENGTHS}
    assert [(full[n]['windows'],full[n]['tail_steps']) for n in p.LENGTHS]==[(9,1),(4,3),(2,3)]
    for n in p.LENGTHS:
        common=p.window_design(truth,n,1.,common80=True)
        assert common['coverage_steps']==16 and common['tail_steps']==3
        assert common['windows']==16//n
        # Mean integral of t^2 on the first original-sample window.
        assert full[n]['X_S'][0,0]==pytest.approx((n*.01)**2/3,abs=1e-15)
    short=p.window_design(truth[:8],8,1.,common80=True)
    assert not short['available'] and short['reason']=='no_complete_window'


def test_nonfinite_tail_is_retained_as_unknown_instead_of_dropped():
    truth=np.zeros((20,2));truth[-1,1]=np.nan
    result=p.window_design(truth,4,1.)
    assert not result['available'] and result['windows']==4
    stats=p.equal_event_stats([np.ones(2),None],[np.ones(2),None],2,kind='weak')
    assert stats['status']=='unknown' and stats['denominator']==2 and stats['Q_S'] is None


def test_nu_epsilon_are_specific_to_each_coefficient_and_event_equal():
    old=p.equal_event_stats([np.full(2,1e-4)],[np.full(2,1e-4)],1,kind='weak')
    new=p.equal_event_stats([np.full(2,1e-6)],[np.full(2,-1e-6)],1,kind='weak')
    assert old['nu']==1e-10 and old['discernible']
    assert new['nu']==pytest.approx(2e-6) and new['compatible']
    stats=p.equal_event_stats([np.ones(2),np.full(100,3.)],[np.ones(2),np.full(100,3.)],2)
    assert stats['error_L']==pytest.approx(np.sqrt(5.))
    assert p.relative_change({'error_L':0.},{'error_L':1.},'error_L')['ratio'] is None


@pytest.mark.parametrize('kind',['weak','response'])
def test_finite_extreme_residuals_cannot_overflow_into_compatible_null_stats(kind):
    result=p.equal_event_stats([np.array([1e200])],[np.array([-1e200])],1,kind=kind)
    assert result['status']=='unknown' and result['reason']=='nonfinite_derived_statistics'
    assert result['denominator']==1 and result['available_events']==1
    assert result['compatible'] is None and result['discernible'] is None
    assert result['error_L'] is None and result['epsilon'] is None
    encoded=p.safe(result)
    assert encoded['compatible'] is None
    assert 'error_L' in result['nonfinite_fields']


@pytest.mark.parametrize('check_error',[float('inf'),float('nan'),1e308])
def test_response_analytic_error_and_comparison_threshold_must_be_finite(check_error):
    result=p.equal_event_stats([np.ones(2)],[np.ones(2)],1,check_error=check_error)
    assert result['status']=='unknown' and result['available_events']==1
    assert result['compatible'] is None and result['epsilon'] is None
    assert result['reason'] in ('nonfinite_analytic_check_error','nonfinite_derived_statistics')


@pytest.mark.parametrize('g,a',[(.7,0.),(.7,1e-10),(-.2,-.4),(.7,.5)])
def test_oracle_scalar_positive_negative_zero_a_and_signed_h(g,a):
    check=p.coefficient_check(g,a,.2)
    assert check['status']=='PASS' and check['normalized_error']<1e-12
    assert check['signs_legal']==(g>0 and a>=0)


def test_local_oracle_matches_known_constant_omega_solution():
    n=16;w=np.ones((n+1,1));g,a=.7,.5
    truth_p=p.filters.linear_filter(w,[.2],[g],[a],.01)[:,0]
    truth=np.column_stack((w[:,0],truth_p))
    left,right=p.local_response(truth,4,np.array([g,a]),1.)
    np.testing.assert_allclose(left,0,atol=1e-12)
    np.testing.assert_allclose(right,0,atol=1e-12)


def test_sustained_oracle_keeps_full_tail_and_does_not_reset_true_p():
    n=19;w=np.ones((n+1,10));beta0=np.column_stack((np.full(10,.7),np.full(10,.5)))
    truth=p.filters.linear_filter(w,np.full(10,.2),beta0[:,0],beta0[:,1],.01)
    x=np.zeros((n+11,30));x[10:,10:20]=w;x[10:,20:30]=truth
    checks={kind:[{'status':'PASS','normalized_error':0.} for _ in range(10)] for kind in ('beta0','beta40')}
    data,paths,counts=p.persistent_oracles([{'id':'synthetic','label':'unstable','x':x}],beta0,list(beta0),np.ones(10),checks)
    assert paths['synthetic']['L'].shape[0]==n+1
    errors=data[('beta0',0,'unstable_actual')][0][1]
    assert len(errors)==19 and np.max(np.abs(errors))<1e-12
    assert counts['oracle_future_steps_L']==20*n


def branch_group(mode):
    return {'weak':{str(n):{'old_discernible':True,'new_compatible':mode=='C','remaining':mode=='F'} for n in p.LENGTHS},
            'local':{str(n):{'beta40':{'compatible':mode=='C','discernible':mode=='F'}} for n in p.LENGTHS},
            'persistent':{'beta40':{'compatible':mode=='C','discernible':mode=='F'}}}


def test_C_requires_both_groups_all_three_lengths_and_legal_signs():
    groups={g:branch_group('C') for g in p.PRIMARY};full={'beta40':{'compatible':True,'discernible':False}}
    assert p.branch_decision(True,True,True,full,groups)['C_compatible']
    assert not p.branch_decision(True,True,False,full,groups)['C_compatible']
    groups['unstable_actual']['local']['8']['beta40']['compatible']=None
    assert p.branch_decision(True,True,True,full,groups)['status']=='mixed_or_unresolved'


def test_C_old_residual_must_be_same_group_all_three_lengths():
    groups={g:branch_group('C') for g in p.PRIMARY};full={'beta40':{'compatible':True,'discernible':False}}
    groups['stable8']['weak']['2']['old_discernible']=False
    groups['unstable_actual']['weak']['4']['old_discernible']=False
    assert not p.branch_decision(True,True,True,full,groups)['C_compatible']


def test_F_requires_full40_optimal_and_same_group_all_weak_local_sustained_not_legal_sign():
    groups={'stable8':branch_group('F'),'unstable_actual':branch_group('C')};full={'beta40':{'compatible':False,'discernible':True}}
    result=p.branch_decision(True,True,False,full,groups)
    assert result['F_remaining_evidence'] and result['unconstrained_reference_only']
    assert result['F_supporting_groups']==['stable8'] and not result['model_adopted']
    assert not p.branch_decision(False,True,False,full,groups)['F_remaining_evidence']
    assert not p.branch_decision(True,False,False,full,groups)['F_remaining_evidence']
    full['beta40']={'compatible':True,'discernible':False}
    assert not p.branch_decision(True,True,False,full,groups)['F_remaining_evidence']


def test_frozen_source_binding_and_exact_shared_parameters(tmp_path):
    file=tmp_path/'asset.txt';file.write_text('synthetic')
    binding={'parent_commit':p.PARENT_COMMIT,'asset':{'path':'asset.txt','sha256':p.sha(file)}}
    p.verify_source_bindings(binding,tmp_path)
    file.write_text('different')
    with pytest.raises(ValueError,match='source binding hash'):p.verify_source_bindings(binding,tmp_path)
    values={name:{'g':np.ones(10),'a':np.zeros(10),'p_scale':np.ones(10)} for name in p.MODELS}
    p.shared_parameters(values)
    values[p.MODELS[1]]['a'][0]=-0.
    with pytest.raises(ValueError,match='shared-constant premise'):p.shared_parameters(values)


def test_unknown_started_is_refused_without_replay(tmp_path):
    started=tmp_path/'started.json';started.write_text('{}')
    with pytest.raises(RuntimeError,match='no automatic retry'):p.reject_unknown(started,tmp_path/'complete.json')


def test_stage_release_checks_exact_hashes_stage_and_zero_or_ten_budget(tmp_path,monkeypatch):
    monkeypatch.setattr(p,'ROOT',tmp_path);monkeypatch.setattr(p.utility,'ROOT',tmp_path)
    tests=tmp_path/'tests/test_p_form_discrimination_20261031.py';tests.parent.mkdir();tests.write_text('synthetic test identity')
    protocol=tmp_path/'protocol';protocol.write_text('synthetic protocol')
    auth=tmp_path/'auth';auth.write_text(json.dumps({'scientific_computation_released':False}))
    bindings=tmp_path/'bindings';bindings.write_text('{}');monkeypatch.setattr(p,'BINDINGS',bindings)
    release={'scientific_computation_released':True,'allowed_stage':'estimate','source_sha256':p.sha(Path(p.__file__)),'tests_sha256':p.sha(tests),'protocol_sha256':p.sha(protocol),'authorization_sha256':p.sha(auth),'source_bindings_sha256':p.sha(bindings),
        'maximum_main_parameter_estimates':10,'same_objective_independent_qr_checks_upper_bound':10,'optimizer_upper_bound':0,'new_autonomous_rom_rollouts_upper_bound':0,'model_writes_upper_bound':0}
    path=tmp_path/'release';path.write_text(json.dumps(release))
    p.verify_stage_release('estimate',path,protocol,auth)
    with pytest.raises(PermissionError):p.verify_stage_release('analyze',path,protocol,auth)
    release['maximum_main_parameter_estimates']=11;path.write_text(json.dumps(release))
    with pytest.raises(PermissionError):p.verify_stage_release('estimate',path,protocol,auth)


def test_estimates_are_saved_before_QR_and_completed_estimates_never_repeat(tmp_path,monkeypatch):
    monkeypatch.setattr(p,'ROOT',tmp_path);monkeypatch.setattr(p.utility,'ROOT',tmp_path);monkeypatch.setattr(p,'PRIVATE',tmp_path/'tool')
    p.write_once(p.PRIVATE/'manifest.json',{'synthetic':True})
    manifest={'parameters':{'p_scale':[1.]*10},'protocol':{'path':'protocol'},'authorization':{'path':'auth'}}
    monkeypatch.setattr(p,'checked_manifest',lambda:manifest);monkeypatch.setattr(p,'verify_stage_release',lambda *args:{'synthetic':True});monkeypatch.setattr(p,'checked_events',lambda *args:['synthetic'])
    x=np.eye(2);rows=[row(str(i),x,x@[.7,.2]) for i in range(297)]
    monkeypatch.setattr(p,'collection',lambda *args,**kwargs:rows)
    calls=[];original=p.qr_check
    def verify_after_persist(target,primary):
        machine=len(calls);file=p.PRIVATE/'estimates'/f'machine_{machine:02d}'/'parameters.json'
        assert file.is_file()
        np.testing.assert_allclose(p.read(file)['beta40'],[.7,.2],atol=1e-12)
        calls.append(machine)
        return original(target,primary)
    monkeypatch.setattr(p,'qr_check',verify_after_persist)
    receipt=p.estimate()
    assert receipt['main_SVD_estimates']==receipt['same_target_QR_checks']==10
    assert len(calls)==10
    hashes=[p.sha(p.PRIVATE/'estimates'/f'machine_{i:02d}'/'parameters.json') for i in range(10)]
    monkeypatch.setattr(p,'primary_svd',lambda *args:pytest.fail('completed parameters must not be re-estimated'))
    p.estimate()
    assert hashes==[p.sha(p.PRIVATE/'estimates'/f'machine_{i:02d}'/'parameters.json') for i in range(10)]


def test_analyze_contains_no_parameter_estimation_call_and_cli_is_last():
    tree=ast.parse(Path(p.__file__).read_text(encoding='utf-8'))
    for node in tree.body:
        if isinstance(node,ast.FunctionDef) and node.name in ('analyze','analyze_tables'):
            names={n.func.id for n in ast.walk(node) if isinstance(n,ast.Call) and isinstance(n.func,ast.Name)}
            assert not names&{'primary_svd','qr_check','estimate'}
    assert isinstance(tree.body[-1],ast.If) and ast.unparse(tree.body[-1].test)=="__name__ == '__main__'"
    result=subprocess.run([sys.executable,p.__file__,'--help'],capture_output=True,text=True,timeout=20)
    assert result.returncode==0,result.stderr
    assert 'estimate' in result.stdout and '--release' in result.stdout
