"""Selective-file and finite aggregation fixtures only; no parent data or fitting."""
import copy
import importlib.util
import json
import math
from pathlib import Path
import pytest

spec=importlib.util.spec_from_file_location('residual_stage_engineering',Path(__file__).parents[1]/'experiments/p_residual_stage_20261031.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)

def container(records=None):
    records=records if records is not None else [{'id':'e0','Q_residual':[1.],'L_residual':[1.]}]
    value={'machine':0,'sensitivity':{'forbidden':'DO_NOT_DECODE'},
      'fixed_residuals':{'all':{'8':records,'4':records},'stable':{'forbidden':'DO_NOT_DECODE'},'unstable':{'forbidden':'DO_NOT_DECODE'}},
      'statistics':{'forbidden':'DO_NOT_DECODE'},'parameters_sha256':'DO_NOT_DECODE'}
    return (json.dumps(value,indent=2)+'\n').encode()

@pytest.mark.parametrize('eol',[b'\n',b'\r\n'])
def test_only_machine_and_all_subtree_reach_JSON_decoder(eol):
    seen=[]
    class Spy:
        def raw_decode(self,text):
            assert 'DO_NOT_DECODE' not in text
            seen.append(text)
            return json.JSONDecoder(object_pairs_hook=m.unique_object,parse_constant=m.reject_constant).raw_decode(text)
    data=container().replace(b'\n',eol)
    before=m.hashlib.sha256(data).hexdigest()
    machine,selected,size=m.selected_residuals(data,Spy())
    assert machine==0 and set(selected)=={'8','4'} and size>0
    assert len(seen)==2 and seen[0]=='0'
    assert m.hashlib.sha256(data).hexdigest()==before

@pytest.mark.parametrize('change',['CRLF_with_LF_inside','LF_with_CRLF_inside','broken_CR'])
def test_mixed_or_broken_producer_line_boundaries_are_rejected(change):
    if change=='CRLF_with_LF_inside':
        data=container().replace(b'\n',b'\r\n').replace(b'  "fixed_residuals": {\r\n',b'  "fixed_residuals": {\n',1)
    elif change=='LF_with_CRLF_inside':
        data=container().replace(b'  "fixed_residuals": {\n',b'  "fixed_residuals": {\r\n',1)
    else:
        data=container().replace(b'\n',b'\r\n').replace(b'    "all": {\r\n',b'    "all": {\r',1)
    with pytest.raises(m.QualificationUnknown):m.selected_residuals(data)

def test_invalid_forbidden_payload_is_never_decoded():
    # The allowed subtree is valid even when a forbidden sibling contains an
    # intentionally invalid JSON sentinel. Whole-object decoding would fail.
    data=container().replace(b'"DO_NOT_DECODE"',b'FORBIDDEN_NOT_JSON')
    assert m.selected_residuals(data)[0]==0

@pytest.mark.parametrize('change',['root_duplicate','group_duplicate','selected_duplicate','extra_field','boundary'])
def test_nonunique_keys_anchors_extra_fields_and_bad_boundary_stop(change):
    data=container()
    if change=='root_duplicate':data=data.replace(b'  "machine": 0,',b'  "machine": 0,\n  "machine": 0,',1)
    if change=='group_duplicate':data=data.replace(b'    "stable":',b'    "all": {},\n    "stable":',1)
    if change=='selected_duplicate':data=data.replace(b'"id": "e0",',b'"id": "e0", "id": "e0",',1)
    if change=='extra_field':data=container([{'id':'e0','Q_residual':[1.],'L_residual':[1.],'unexpected':0}])
    if change=='boundary':data=data.replace(b'    "stable":',b'     "stable":',1)
    with pytest.raises(m.QualificationUnknown):m.selected_residuals(data)

def alignment_fixture():
    rows=[{'id':f'e{i:03d}','windows':{'8':{'count':1},'4':{'count':1}}} for i in range(297)]
    records=[{'id':r['id'],'Q_residual':[1.],'L_residual':[1.]} for r in rows]
    return rows,{'8':copy.deepcopy(records),'4':copy.deepcopy(records)}

@pytest.mark.parametrize('change',['order','duplicate','length','nonfinite','nested','bool','machine'])
def test_alignment_uses_exact_keys_order_shapes_and_finite_values(change):
    rows,value=alignment_fixture();machine=0
    m.align_residuals(machine,value,0,rows)
    if change=='order':value['8'][0],value['8'][1]=value['8'][1],value['8'][0]
    if change=='duplicate':value['8'][1]['id']=value['8'][0]['id']
    if change=='length':value['4'][0]['Q_residual'].append(1.)
    if change=='nonfinite':value['8'][0]['Q_residual']=[float('nan')]
    if change=='nested':value['8'][0]['Q_residual']=[[1.]]
    if change=='bool':value['8'][0]['Q_residual']=[True]
    if change=='machine':machine=1
    with pytest.raises(m.QualificationUnknown):m.align_residuals(machine,value,0,rows)

def test_small_partition_keeps_original_event_window_weights(monkeypatch):
    monkeypatch.setattr(m,'DENOM',{'stable':2,'unstable':1,'all':3})
    rows=[{'id':'a','label':'stable','windows':{'8':{'count':1,'classes':['history']}}},
      {'id':'b','label':'stable','windows':{'8':{'count':3,'classes':['history','cross_t0','future']}}}]
    records=[{'id':'a','Q_residual':[2.],'L_residual':[2.]},{'id':'b','Q_residual':[1.,3.,5.],'L_residual':[1.,3.,5.]}]
    direct,_=m.moments(records,rows,'stable',8)
    parts=[m.moments(records,rows,'stable',8,p)[0] for p in m.PHASES]
    assert direct['C_Q']==pytest.approx(47/6)
    assert [v['mu'] for v in parts]==pytest.approx([2/3,1/6,1/6])
    assert [v['C_Q'] for v in parts]==pytest.approx([13/6,9/6,25/6])
    assert parts[2]['N_available']==1 and parts[2]['N_original']==2
    assert m.conditional(parts[2])['R_Q']==pytest.approx(5.)
    assert parts[2]['C_Q']/direct['C_Q']==pytest.approx(25/47)
    assert m.partition_closure(parts,direct)['status']=='verified'
    parent={'status':'computed','denominator':2,'available_events':2,'Q_Q':math.sqrt(47/6),'Q_L':math.sqrt(47/6),'nu':1e-10,'compatible':False,'resolved':True}
    assert m.parent_closure(direct,parent)['status']=='verified'
    parent['resolved']=False
    with pytest.raises(m.QualificationUnknown):m.parent_closure(direct,parent)

def test_empty_stage_is_zero_contribution_but_unknown_conditional(monkeypatch):
    monkeypatch.setattr(m,'DENOM',{'stable':1})
    rows=[{'id':'a','label':'stable','windows':{'8':{'count':1,'classes':['history']}}}]
    records=[{'id':'a','Q_residual':[2.],'L_residual':[2.]}]
    moment,_=m.moments(records,rows,'stable',8,'future')
    assert [moment[k] for k in ('mu','C_Q','C_L','C_diff')]==[0.,0.,0.,0.]
    result=m.conditional(moment)
    assert result['reason']=='no_window' and result['R_Q'] is None and result['nu'] is None
    assert result['compatible'] is None and result['resolved'] is None

def test_all297_mixes_squared_moments_not_RMS():
    stable={'mu':1.,'C_Q':4.,'C_L':4.,'C_diff':0.,'N_available':267,'windows':10}
    unstable={'mu':1.,'C_Q':9.,'C_L':9.,'C_diff':0.,'N_available':30,'windows':2}
    mixed=m.mix_groups(stable,unstable)
    result=m.conditional(mixed)
    assert result['R_Q']==pytest.approx(math.sqrt(267/297*4+30/297*9))
    assert result['R_Q']!=pytest.approx(267/297*2+30/297*3)

def test_same_cell_nu_and_exact_atol_rtol_comparison():
    moment={'mu':1.,'C_Q':1.,'C_L':1.,'C_diff':0.,'N_available':1,'windows':1}
    assert m.conditional(moment)['resolved'] is True
    assert m.conditional({**moment,'C_diff':1.})['compatible'] is True
    assert m.close(1.+1.5e-12,1.)
    assert not m.close(1.+3e-12,1.)

def test_release_and_no_unknown_completed_replay(tmp_path):
    expected={'source_sha256':'frozen','parameter_estimates_upper_bound':0}
    release={'released':True,'allowed_stage':'aggregate',**expected}
    m.verify_release(release,expected)
    for update in ({'released':'true'},{'allowed_stage':'estimate'},{'source_sha256':'changed'},{'parameter_estimates_upper_bound':1}):
        with pytest.raises(PermissionError):m.verify_release({**release,**update},expected)
    directory=tmp_path/'private';m.claim_private(directory,{'frozen':'control'})
    with pytest.raises(FileExistsError):m.claim_private(directory,{})
    m.write_once(directory/'complete.json',{'status':'COMPLETED'})
    with pytest.raises(FileExistsError):m.claim_private(directory,{})

def test_pair_keeps_both_future_scales_and_early_pattern():
    rows=[{'phase':'future','steps':8,'classification':'resolved','resolved':True},
      {'phase':'future','steps':4,'classification':'compatible','resolved':False},
      {'phase':'history','steps':8,'classification':'resolved','resolved':True}]
    out=m.pair_classification(rows)
    assert out['classification']=='mixed_future_scales' and out['any_future_resolved'] is True
    rows[0].update(classification='compatible',resolved=False)
    assert m.pair_classification(rows)['classification']=='both_future_compatible_early_resolved'
