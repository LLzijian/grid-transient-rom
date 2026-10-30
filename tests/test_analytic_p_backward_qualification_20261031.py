"""Artificial fixtures only: no private payload, new full evaluation or optimizer."""
import ast
import copy
import inspect
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/"experiments"),str(ROOT/"tests")]
import analytic_p_backward_qualification_20261031 as a
from state_gated_memory_model_20261028 import StateGatedMemoryROM
from memory_transport_model_20261029 import MemoryTransportROM,helmert_basis
from test_memory_transport_model_20261029 import config,history


@pytest.fixture
def synthetic_overlay(tmp_path,monkeypatch):
    monkeypatch.setattr(a,'ROOT',tmp_path)
    monkeypatch.setattr(a.reverse,'ROOT',tmp_path)
    monkeypatch.setattr(a,'PRIVATE',tmp_path/'qualification')
    monkeypatch.setattr(a,'source_closure',lambda:{'synthetic_fixture':'fixed'})
    protocol=tmp_path/'protocol.md';protocol.write_text('artificial test protocol')
    paths={}
    def install(label,base):
        parent=tmp_path/f'{label}_parent.json';base.export(parent)
        paths[label]=parent
        monkeypatch.setitem(a.PARENT_SHA,label,a.sha(parent))
        monkeypatch.setattr(a.previous,'model_spec',lambda name:{'export':paths[name],'loader':'state_gated' if name=='A96' else 'transport'})
        return base,protocol,tmp_path/'qualification'/('A96_analytic' if label=='A96' else 'A16_analytic')
    return install


@pytest.mark.parametrize('label',['A96','step000016'])
def test_explicit_overlay_roundtrip_and_unchanged_strict_old_loader(label,synthetic_overlay):
    if label=='A96':base=StateGatedMemoryROM(config(),'A').double().eval()
    else:
        plane=helmert_basis();base=MemoryTransportROM(config(),'A',u=plane[:,0],v=plane[:,1]).double().eval()
    base,protocol,folder=synthetic_overlay(label,base)
    before=a.state_identity(base)
    candidate,receipt=a.create_overlay('A96_analytic' if label=='A96' else 'A16_analytic',base,protocol,folder)
    assert a.state_identity(base)==before
    assert a.assert_only_buffer(base,candidate)['flags_unchanged']
    assert receipt['roundtrip_bitwise_equal']
    overlay=folder/'candidate_overlay.json'
    assert a.read(overlay)['format']=='analytic_p_backward_overlay_v1'
    with pytest.raises(ValueError):type(base).from_export(overlay)
    candidate.export(folder/'third_overlay.json')
    assert a.state_identity(a.load_overlay(folder/'third_overlay.json'))==a.state_identity(candidate)
    assert a.state_identity(copy.deepcopy(candidate))==a.state_identity(candidate)
    old=type(base).from_export(protocol.parent/f'{label}_parent.json')
    assert a.reverse.tensor_hash(old.p_backward)==before['p_backward']


def test_only_B_change_is_enforced():
    base=StateGatedMemoryROM(config(),'A').double().eval();candidate=copy.deepcopy(base)
    with torch.no_grad():candidate.kappa_raw.add_(1.)
    with pytest.raises(ValueError,match='beyond p_backward'):a.assert_only_buffer(base,candidate)


def test_actual_constructed_buffer_is_analytic80_not_expm_selection(monkeypatch):
    import mpmath as mp
    seen=[];original=a.reverse.high_precision_reference
    def record(q,dps):
        seen.append(dps)
        return original(q,dps)
    monkeypatch.setattr(a.reverse,'high_precision_reference',record)
    q=a.reverse.generator_q([.7],[.6],[1.])
    matrix,receipt=a.analytic_reference(q)
    assert receipt['qualified'] and seen==[80,120]
    _,rows80=original(q,80)
    expected=np.array([[float(v) for v in rows80[0]]])
    np.testing.assert_array_equal(matrix,expected)
    assert 'analytic80 rounded once' in receipt['construction']


def test_causal_entry_uses_patch_ignores_early_p_and_survives_hot_cache():
    base=StateGatedMemoryROM(config(),'A').double().eval()
    q=a.reverse.generator_q(base.jpi_p.numpy(),base.rpp.numpy(),base.storage_base.numpy())
    matrix,proof=a.analytic_reference(q)
    candidate=a.reverse.replace_copy(base,matrix)
    h=history().numpy()[None]
    result=a.causal_entry(candidate,h)
    assert result['passed']
    assert all(all(cell['gates'].values()) for cell in result['history_steps'].values())


def test_score_uses_its_own_perturbation_and_null_is_unresolved():
    assert a.scalar_score_gate(1.000001,1.,1e-6,1e-6)['passed']
    assert not a.scalar_score_gate(1.1,1.,1e-6,1e-6)['passed']
    assert not a.scalar_score_gate(None,1.,1.,1.)['passed']


def test_equal_classification_counts_cannot_swap_event_identity():
    old=[{'event_id':'u1','label':'unstable','numerically_admissible':True,'predicted_los':True,'category':'predicted_los'},
         {'event_id':'u2','label':'unstable','numerically_admissible':True,'predicted_los':False,'category':'window_no_los_unsettled'}]
    new=copy.deepcopy(old)
    for row in new:
        row['predicted_los']=not row['predicted_los'];row['category']='predicted_los' if row['predicted_los'] else 'window_no_los_unsettled'
    assert len(a.category_sets(old)['detected_unstable'])==len(a.category_sets(new)['detected_unstable'])
    assert a.category_sets(old)!=a.category_sets(new)


def trow(truth=.2,fine=.3,coarse=.31):
    return {'numerically_qualified':True,'truth':{'time_s':truth},'fine':{'time_s':fine},'coarse':{'time_s':coarse}}


def test_fixed_T_base_never_shrinks_or_substitutes_score_uncertainty():
    old={'u1':trow(),'u2':trow()};new={'u1':trow()}
    gate=a.fixed_timing_gate(old,new)
    assert not gate['passed'] and gate['missing_ids']==['u2']
    assert gate['T_base']==['u1','u2']
    assert not a.fixed_timing_gate({}, {})['passed']
    assert not a.fixed_timing_gate(old,{'u1':trow(fine=.5,coarse=.5),'u2':trow()})['passed']
    assert not a.fixed_timing_gate(old,{'u1':trow(truth=.21),'u2':trow()})['passed']


def test_same_grid_extra_p_gate_catches_unchanged_delta_omega():
    event={'id':'fixture','group':'g','label':'stable','x':np.zeros((811,30)),
           'classification_meta':{'equilibrium_delta':np.zeros(10),'inertia':np.ones(10),'omega_s':2*np.pi*60}}
    old=np.zeros((1501,40));new=old.copy();new[:,20:30]=.002
    row=a.cross_interval(old,new,event,np.ones(10),np.ones(10),1,800)
    assert row['coi_normalized_rmse']==0 and row['common_normalized_rmse']==0
    assert row['p_normalized_rms']==pytest.approx(.002)
    assert not row['qualified']


def test_fine_substep_failure_is_not_hidden_by_output_decimation():
    event={'id':'fixture','group':'g','label':'stable','x':np.zeros((811,30)),
           'classification_meta':{'equilibrium_delta':np.zeros(10),'inertia':np.ones(10),'omega_s':2*np.pi*60}}
    old=np.zeros((3001,40));new=old.copy();new[1,0]=np.nan
    row=a.cross_interval(old,new,event,np.ones(10),np.ones(10),2,800)
    assert not row['qualified'] and not row['finite_full_interval']


def test_cli_help_and_main_guard_after_every_definition():
    result=subprocess.run([sys.executable,a.__file__,'--help'],capture_output=True,text=True,timeout=20)
    assert result.returncode==0,result.stderr
    assert 'prepare' in result.stdout and '--candidate' in result.stdout
    tree=ast.parse(Path(a.__file__).read_text(encoding='utf-8'));guard=tree.body[-1]
    assert isinstance(guard,ast.If) and ast.unparse(guard.test)=="__name__ == '__main__'"
    assert all(node.lineno<guard.lineno for node in tree.body if isinstance(node,ast.FunctionDef))


def test_parent_archive_uses_real_json_path_and_matching_verified_commits(monkeypatch):
    paths=[]
    def fake(path):
        paths.append(path)
        return {'ordinary_push_verified':True,'local_commit':a.PARENT_COMMIT,'remote_commit':a.PARENT_COMMIT}
    monkeypatch.setattr(a,'read',fake)
    path,proof=a.verify_parent_archive()
    assert path.name=='archival_verified.json' and paths==[path]
    monkeypatch.setattr(a,'read',lambda path:{'ordinary_push_verified':True,'local_commit':'parent','remote_commit':'different'})
    with pytest.raises(ValueError,match='matching local/remote'):a.verify_parent_archive()


@pytest.mark.parametrize('proof',[
    {'ordinary_push_verified':True,'local_commit':'another','remote_commit':'another'},
    {'ordinary_push_verified':1,'local_commit':a.PARENT_COMMIT,'remote_commit':a.PARENT_COMMIT},
    {'ordinary_push_verified':False,'local_commit':a.PARENT_COMMIT,'remote_commit':a.PARENT_COMMIT},
])
def test_parent_archive_rejects_wrong_frozen_parent_and_nonboolean_flags(monkeypatch,proof):
    monkeypatch.setattr(a,'read',lambda path:proof)
    with pytest.raises(ValueError,match='frozen parent'):a.verify_parent_archive()


def unresolved_fixture():
    events=[{'id':f'fixture_{i}','group':'g','label':'stable' if i<267 else 'unstable','x':np.zeros((12,30)),'classification_meta':{}} for i in range(297)]
    rows=[{'event_id':e['id'],'label':e['label'],'numerically_admissible':False,'predicted_los':False,'category':'numerical_unresolved','window_settled':False} for e in events]
    detail={'classification':{scope:{'rows':copy.deepcopy(rows)} for scope in ('8s','actual15')}}
    timing={e['id']:{'numerically_qualified':False,'truth':{'time_s':None},'fine':{'time_s':None},'coarse':{'time_s':None}} for e in events}
    horizons={h:{'value':None,'coarse_value':None} for h in ('1','2','5','8')}
    profile={'horizon_risks':{h:horizons[h] for h in ('5','8')},'unstable_actual15_risk':None,'unstable_actual15_coarse_risk':None,'common_frequency_rmse_8s':None,'normalized_p_rmse_8s':None,'all_8s_rows_qualified':False,'timing_actual15':timing}
    summary={'decision_profile':profile,'prediction_files':[{'path':'synthetic','sha256':'synthetic'}],'official_horizon_risks':horizons,'original_inference_cost':{'seconds':None}}
    return events,detail,summary


def mock_unresolved_paths(monkeypatch,events):
    ids=[e['id'] for e in events]
    histories=np.zeros((297,11,30));paths=np.zeros((297,1,40))
    monkeypatch.setattr(a,'prediction_blocks',lambda files:iter([(ids,histories,paths,paths)]))
    monkeypatch.setattr(a,'path_score_rows',lambda *args:{h:None for h in ('1','2','5','8','actual15')})
    import evaluate_structured_rom
    monkeypatch.setattr(evaluate_structured_rom,'_classification',lambda *args,**kwargs:{'category':'numerical_unresolved','predicted_los':False,'window_settled':False,'trigger_index':None})


def test_unresolved_candidate_keeps_297_and_null_official_scores_old_baseline_stays_strict(monkeypatch):
    events,detail,summary=unresolved_fixture();mock_unresolved_paths(monkeypatch,events)
    model=StateGatedMemoryROM(config(),'A').double().eval()
    with pytest.raises(ValueError,match='baseline original metric reproduction'):
        a.baseline_description(model,summary,detail,events)
    result=a.baseline_description(model,summary,detail,events,strict_baseline=False)
    assert (result['event_count'],result['stable_denominator'],result['unstable_denominator'])==(297,267,30)
    assert all(v is None for grid in result['scores'].values() for v in grid.values())
    assert len(result['sets']['actual15']['numerical_unresolved'])==297


def test_coarse_category_and_window_changes_are_reported_without_new_hard_gate(monkeypatch):
    events,detail,summary=unresolved_fixture();mock_unresolved_paths(monkeypatch,events)
    model=StateGatedMemoryROM(config(),'A').double().eval()
    old=a.baseline_description(model,summary,detail,events,strict_baseline=False)
    new=copy.deepcopy(old)
    # Isolate the category gate from numeric/timing gates in this artificial case.
    for description in (old,new):
        description['scores']={grid:{key:1. for key in ('5','8','actual15','common_frequency_rmse_8s','normalized_p_rmse_8s')} for grid in ('fine','coarse')}
    new['classification']['8s']['coarse'][0]['category']='window_settled'
    new['classification']['8s']['coarse'][0]['window_settled']=True
    monkeypatch.setattr(a,'cross_interval',lambda *args,**kwargs:{'qualified':True})
    monkeypatch.setattr(a,'fixed_timing_gate',lambda *args:{'passed':True,'T_base':['synthetic']})
    result=a.compare_candidate(model,events,old,new)
    assert result['passed']
    assert result['classification_gates']['8s']=={'frozen_sets_equal':True}
    assert result['classification_changes']['8s']['coarse']['category_changed_ids']==['fixture_0']
    assert result['classification_changes']['8s']['coarse']['window_settled_changed_ids']==['fixture_0']


def test_actual_helper_source_closure_is_bound():
    source=a.source_closure()
    for name in ('memory_transport_study_20261029.py','checkpoint_selection_decision_20261030.py','p_port_diagnostic_20261031.py','closure_probe_20261030.py','evaluate_l1_latent.py'):
        assert source[name]==a.sha(ROOT/'experiments'/name)


def test_null_candidate_completes_failed_receipt_through_comparison_science_and_summary(tmp_path,monkeypatch):
    import state_gated_memory_study_20261028 as sgm
    events,detail,fixture_summary=unresolved_fixture();mock_unresolved_paths(monkeypatch,events)
    monkeypatch.setattr(a,'ROOT',tmp_path);monkeypatch.setattr(a.reverse,'ROOT',tmp_path)
    private=tmp_path/'qualification';public=tmp_path/'public'
    monkeypatch.setattr(a,'PRIVATE',private);monkeypatch.setattr(a,'PUBLIC',public)
    # Only one artificial independent case is needed to exercise summary's
    # full completion path. Production retains both fixed candidate IDs.
    monkeypatch.setattr(a,'CANDIDATES',{'A96_analytic':'A96'})
    a.write_once(private/'manifest.json',{'fixture':True})
    a.write_once(tmp_path/'prepare.json',{})
    metadata=[{k:v for k,v in e.items() if k!='x'} for e in events]
    np.savez(tmp_path/'cache.npz',metadata_json=np.asarray(json.dumps(metadata)),**{f'event_{i:04d}_x':e['x'] for i,e in enumerate(events)})
    manifest={'data':{'fit_cache':{'path':'cache.npz'},'prepare':{'path':'prepare.json'}}}
    monkeypatch.setattr(a,'verify_prepared',lambda:manifest)
    model=StateGatedMemoryROM(config(),'A').double().eval()
    monkeypatch.setattr(a,'load_overlay',lambda path:model)
    monkeypatch.setattr(sgm,'valid',lambda profile:profile.get('all_required_rows_qualified',False))
    monkeypatch.setattr(a.previous,'classification_detail',lambda extras,profile,key:detail['classification']['8s' if key=='classification_8s' else 'actual15'])
    folder=private/'A96_analytic'
    for name in ('candidate_overlay.json','candidate_roundtrip.json'):a.write_once(folder/name,{'fixture':True})
    a.write_once(private/'math.json',{'fixture':True})
    entry={'causal_entry':{'passed':True},'buffer_and_load_seconds':0.,'candidate_identity':{'overlay':a.ident(folder/'candidate_overlay.json'),'math_receipt':a.ident(private/'math.json')}}
    a.write_once(folder/'entry.json',entry)
    old=a.baseline_description(model,fixture_summary,detail,events,strict_baseline=False)
    old['scores']={grid:{h:1. for h in ('1','2','5','8','actual15','common_frequency_rmse_8s','normalized_p_rmse_8s')} for grid in ('fine','coarse')}
    anchor_id='fixture_267'
    reference_profile={'all_required_rows_qualified':True,'timing_actual15':{anchor_id:trow()}}
    old['timing_actual15']=reference_profile['timing_actual15'];old['profile']=reference_profile
    old['original_inference_cost']={'seconds':1.}
    a.write_once(private/'baselines.json',{'A96':old,'R_N':{'profile':reference_profile}})
    a.write_once(private/'science_anchors.json',{'detection_union':{'8s':[anchor_id],'actual15':[anchor_id]},'T_anchor':[anchor_id],'truth_times':{anchor_id:.2}})
    calls=[]
    def failed_numerical_evaluation(candidate,fit,output,prepared):
        calls.append(len(fit));output.mkdir()
        for name in ('evidence.json','state_gated_profile.json','complete.json'):a.write_once(output/name,{'fixture':True})
        path=output/'predictions_0000.npz';path.write_bytes(b'artificial unresolved cache')
        return {'decision_profile':fixture_summary['decision_profile'],'extras':[],
                'prediction_files':[{'path':str(path),'sha256':a.sha(path)}],
                'profiles':{'fit':{'horizon_risks':fixture_summary['official_horizon_risks']}},'seconds':0.,'timing':{}}
    monkeypatch.setattr(sgm,'evaluate',failed_numerical_evaluation)
    monkeypatch.setattr(a,'cross_interval',lambda *args,**kwargs:{'qualified':False})
    result=a.evaluate('A96_analytic')
    assert calls==[297]
    assert result['status']=='COMPLETED' and not result['implementation_qualified']
    assert not result['science_entry']['science_entry_eligible']
    assert result['denominators']=={'event_count':297,'stable':267,'unstable':30}
    assert all(value is None for grid in result['stable_1_2_5_8_scores'].values() for value in grid.values())
    assert (folder/'complete.json').is_file()
    assert a.checked_candidate_complete(folder)['status']=='COMPLETED'
    public_result=a.summary()['candidates']['A96_analytic']
    assert not public_result['implementation_qualified'] and not public_result['science_entry']['science_entry_eligible']
    assert public_result['denominators']['event_count']==297
    assert (public/'public_qualification.json').is_file()
