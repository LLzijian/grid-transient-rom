"""Artificial S fixtures only; never read the project dataset or checkpoints."""
import importlib.util
from pathlib import Path
import numpy as np
import pytest

spec=importlib.util.spec_from_file_location('history_source',Path(__file__).parents[1]/'experiments/p_history_transfer_20261031_source.py')
s=importlib.util.module_from_spec(spec);spec.loader.exec_module(s)

def test_prefix_keeps_real_history_and_endpoint_without_model_fill():
    x=np.arange(20*30,dtype=np.float64).reshape(20,30)
    p=s.member_prefix(x)
    assert np.array_equal(p[:11,10:],x[:11,20:30])
    assert np.array_equal(p[10,10:],x[10,20:30])
    assert np.array_equal(p[12,10:],x[12,20:30])
    p[:]=0
    assert x[0,20]==20

@pytest.mark.parametrize('kind',['dtype','short','shape','nan'])
def test_bad_prefix(kind):
    x=np.ones((13,30),dtype=np.float64)
    if kind=='dtype': x=x.astype(np.float32)
    if kind=='short': x=x[:12]
    if kind=='shape': x=x[:,:29]
    if kind=='nan': x[1,22]=np.nan
    with pytest.raises(ValueError): s.member_prefix(x)

@pytest.mark.parametrize('raw,expected',[(None,None),('',None),('0',0.),('.025',.025)])
def test_delay_original_metadata_not_cache_fallback(raw,expected):
    r=s.time_evidence({'record_start_delay_s':0},{'clearing_time_s':'1.2','record_start_delay_s':raw},verified_grid=True,verified_postclear=True)
    assert r['record_start_delay_s']==expected
    assert r['status']=='verified'
    assert r['deployment_since_clearing_s']==(None if expected is None else expected+.1)

def test_missing_domain_stops_even_with_perfect_time_grid():
    r=s.time_evidence({'t':np.arange(13)*.01},None,verified_postclear=False)
    assert r['status']=='unknown' and r['postclear_domain']=='unknown'

@pytest.mark.parametrize('kind',['interval','nonfinite','start','null'])
def test_explicit_time_contradiction_cannot_use_indirect_grid(kind):
    t=np.arange(13)*.01
    if kind=='interval':t[4]+=.001
    if kind=='nonfinite':t[5]=np.nan
    if kind=='start':t+=.002
    if kind=='null':t=None
    r=s.time_evidence({'t':t},None,verified_grid=True,verified_postclear=True)
    assert r['status']=='unknown'
    assert r['grid_evidence']=='explicit_time_conflict'
    assert r['source_contract_conflict'] is True

def test_absent_time_can_use_bound_indirect_grid():
    r=s.time_evidence({},None,verified_grid=True,verified_postclear=True)
    assert r['status']=='verified' and r['grid_evidence']=='bound_producer_check'
    assert r['source_contract_conflict'] is False

def test_negative_original_delay_is_not_postclear():
    assert s.time_evidence({}, {'record_start_delay_s':'-.01'},verified_grid=True,verified_postclear=True)['status']=='unknown'

def test_exact_id_filter_ignores_other_split_fields():
    text='event_id,split,record_start_delay_s\na,test,0.1\na_extra,train,9\nb,val,3\n'
    assert s.metadata_rows(text,['a'])=={'a':{'event_id':'a','clearing_time_s':None,'record_start_delay_s':'0.1','valid_length':None,'trajectory_end_s':None}}

def test_duplicate_selected_metadata_rejected():
    with pytest.raises(ValueError): s.metadata_rows('event_id\na\na\n',['a'])

def test_H_and_A_window_classification_tail_and_weights():
    c=s.coverage(18,'unstable') # n=7, last=17
    assert c['H40']['windows']==2 and c['H40']['tail_steps']==2
    assert c['H80']['windows']==1 and c['H20']['windows']==5
    a=c['A']['4']
    assert a['counts']=={'history':2,'cross_t0':1,'future':1}
    assert a['tail_steps']==1 and a['phase']==0 and a['last_included_index']==16
    assert sum(a['class_row_weight_sums'].values())==pytest.approx(1/297)
    assert c['future20_indices']==[10,12]

@pytest.mark.parametrize('length,label,okay',[(811,'stable',True),(810,'stable',False),(13,'unstable',True),(12,'unstable',False),(2000,'unstable',True)])
def test_coverage_never_borrows_history_for_future(length,label,okay):
    c=s.coverage(length,label)
    assert (c['status']=='verified')==okay
    if length==2000: assert c['n_future']==1500

def test_hash_mismatch_stops_and_no_unbound_discovery(tmp_path):
    f=tmp_path/'a';f.write_text('source')
    b={'one':{'path':'a','sha256':s.sha(f)}}
    assert list(s.verify_bindings(b,tmp_path))==['a']
    f.write_text('changed')
    with pytest.raises(ValueError):s.verify_bindings(b,tmp_path)

def test_hdf_and_outside_bound_paths_rejected(tmp_path):
    with pytest.raises(ValueError):s.bound_path(tmp_path,{'path':'../outside'})
    with pytest.raises(ValueError):s.bound_path(tmp_path,{'path':'raw.h5'})

def test_source_release_is_exact_and_zero_only():
    hashes={'source_sha256':'abc'}
    r={'released':True,'allowed_stage':'source',**hashes,**{k:0 for k in s.ZERO_LIMITS}}
    s.verify_release(r,hashes)
    for change in ({'released':'true'},{'allowed_stage':'estimate'},{'source_sha256':'changed'},{s.ZERO_LIMITS[0]:1},{s.ZERO_LIMITS[0]:False}):
        with pytest.raises(PermissionError):s.verify_release({**r,**change},hashes)

def test_private_cannot_overwrite_or_retry_unknown_started(tmp_path):
    p=tmp_path/'private';s.claim_private(p)
    s.write_once(p/'source_started.json',{'status':'STARTED'})
    with pytest.raises(FileExistsError):s.claim_private(p)
    with pytest.raises(FileExistsError):s.write_once(p/'source_started.json',{})

def test_current_sources_alone_do_not_prove_production_lineage():
    bindings={'producer_code':{},'generator_source':{'sha256':'x'}}
    r=s.source_proof(bindings,{'code':{}},{})
    assert r['status']=='unknown'

@pytest.mark.parametrize('cached,expected',[(0,None),(.015,.015),(-.01,-.01)])
def test_new_delay_decoder_nondefault_evidence(cached,expected):
    r=s.time_evidence({'record_start_delay_s':cached,'clearing_time_s':1.2},None,verified_grid=True,verified_postclear=True,bound_nondefault_decoder=True)
    assert r['record_start_delay_s']==expected
    assert r['status']==('unknown' if cached<0 else 'verified')

def test_parent_constants_require_both_original_hash_receipts():
    b=np.arange(20,dtype=float).reshape(10,2);scale=np.ones(10)
    h={k:s.array_hash(v) for k,v in {'g':b[:,0],'a':b[:,1],'p_scale':scale}.items()}
    p={'parameters':{'beta0':b.tolist(),'p_scale':scale.tolist()},'parent_qualification':{k:{'all_original_qualification_passed':True,'parameter_hashes':dict(h)} for k in ('A96_analytic','A16_analytic')}}
    assert s.parent_constants(p)==p['parameters']
    p['parent_qualification']['A16_analytic']['parameter_hashes']['g']='bad'
    with pytest.raises(ValueError):s.parent_constants(p)

def test_frozen_metadata_alone_without_actual_producer_binding_remains_unknown():
    m0={'asset_identity':'verified_frozen_model_cache','coordinate_alignment':'matched_source_mapping_derivation_and_fingerprint',
      'equilibrium_reference_alignment':'matched_at_1e12','fit_source_metadata_covered':297,'fit_source_metadata_denominator':297,
      'raw_units':'Pm=GENROU.tm','metadata_fragment':{'state_mapping':[['GENROU','tm']]},'historical_metadata_fingerprint':'f'}
    r=s.source_proof({'producer_code':{},'generator_source':{'sha256':'x'}},{'code':{}},{},m0)
    assert r['historical_export_declaration_verified'] is True
    assert r['status']=='unknown' and r['producer_code_matches'] is False
