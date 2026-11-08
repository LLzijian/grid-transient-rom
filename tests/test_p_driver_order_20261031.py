"""File/release engineering only. No case matrices, exponents or scientific runs."""
import importlib.util
from pathlib import Path
import pytest

spec=importlib.util.spec_from_file_location('driver_order_engineering',Path(__file__).parents[1]/'experiments/p_driver_order_20261031.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)

def test_only_scalar_leaf_key_matches_not_ancestor_or_event_lists():
    data={'environment':{'events':[{'case':'forbidden','event_id':'secret'}], 'unrelated':'do not inherit match',
      'python_executable':'C:/frozen/python.exe','nested':{'case_sha256':'sha'}},
      'attestation':[{'version':'never expand lists'}],'version':'2.0.0'}
    result=m.scalar_identities(data)
    assert result==[
      {'key_path':['environment','python_executable'],'value':'C:/frozen/python.exe'},
      {'key_path':['environment','nested','case_sha256'],'value':'sha'},
      {'key_path':['version'],'value':'2.0.0'}]

def test_leaf_nonfinite_is_not_an_identity():
    assert m.scalar_identities({'version':float('nan'),'python':None})==[{'key_path':['python'],'value':None}]

def test_bound_file_hash_size_static_type_and_root(tmp_path):
    p=tmp_path/'one.json';p.write_text('{"version":"fixed"}')
    e={'path':'one.json','bytes':p.stat().st_size,'sha256':m.sha(p)}
    assert m.bound_file(tmp_path,e)==p
    with pytest.raises(ValueError):m.bound_file(tmp_path,{**e,'sha256':'bad'})
    with pytest.raises(ValueError):m.bound_file(tmp_path,{**e,'bytes':0})
    with pytest.raises(ValueError):m.bound_file(tmp_path,{'path':'../outside.json','bytes':1,'sha256':'x'})
    for suffix in ('h5','hdf5','npz','npy','pt','xlsx'):
        with pytest.raises(ValueError):m.bound_file(tmp_path,{'path':'forbidden.'+suffix,'bytes':0,'sha256':'x'})

def test_oversized_static_file_rejected_before_read(tmp_path):
    p=tmp_path/'large.json';p.write_bytes(b' '*(1024*1024+1))
    with pytest.raises(ValueError):m.bound_file(tmp_path,{'path':p.name,'bytes':p.stat().st_size,'sha256':'not_needed'})

def test_independent_stage_release_exact_hash_limit_and_boolean():
    expected={'source_sha256':'frozen','matrix_cases_upper_bound':6}
    r={'released':True,'allowed_stage':'mathematics',**expected}
    m.verify_release('mathematics',r,expected)
    for update in ({'released':'true'},{'allowed_stage':'bootstrap'},{'source_sha256':'new'},{'matrix_cases_upper_bound':7}):
        with pytest.raises(PermissionError):m.verify_release('mathematics',{**r,**update},expected)

def test_unknown_started_and_completed_both_reject_replay(tmp_path):
    p=tmp_path/'stage';m.claim_stage(p,'bootstrap',{'hash':'h'})
    assert m.read(p/'started.json')['status']=='STARTED'
    with pytest.raises(FileExistsError):m.claim_stage(p,'bootstrap',{})
    m.write_once(p/'complete.json',{'status':'COMPLETED'})
    with pytest.raises(FileExistsError):m.claim_stage(p,'bootstrap',{})
    with pytest.raises(FileExistsError):m.write_once(p/'started.json',{})

def test_nonfinite_serialization_stays_explicit_null(tmp_path):
    p=tmp_path/'result.json';m.write_once(p,{'unknown':float('inf'),'finite':1.})
    assert m.read(p)=={'unknown':None,'finite':1.}

def test_release_expected_never_imports_scientific_packages(monkeypatch):
    import builtins
    original=builtins.__import__
    def guarded(name,*a,**kw):
        if name.split('.')[0] in ('andes','numpy','scipy','mpmath'):pytest.fail('engineering check imported scientific package')
        return original(name,*a,**kw)
    monkeypatch.setattr(builtins,'__import__',guarded)
    assert m.release_expected('bootstrap')['matrix_cases_upper_bound']==0
    assert m.release_expected('mathematics')['scipy_expm_calls_upper_bound']==6
