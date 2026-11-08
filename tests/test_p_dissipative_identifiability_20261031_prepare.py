"""Engineering checks only: never generate/evaluate W or decode real payloads."""
import builtins
import importlib.util
import json
from pathlib import Path
import pytest

spec=importlib.util.spec_from_file_location('dissipative_prepare_engineering',Path(__file__).parents[1]/'experiments/p_dissipative_identifiability_20261031_prepare.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)

def test_release_requires_frozen_identity_and_zero_estimation_limits():
    expected={'source_sha256':'frozen','operator_qualification_cells_upper_bound':16,'parameter_estimates_upper_bound':0}
    release={'released':True,'allowed_stage':'prepare',**expected}
    m.verify_release(release,expected)
    for update in ({'released':'true'},{'allowed_stage':'estimate'},{'source_sha256':'changed'},
                   {'operator_qualification_cells_upper_bound':17},{'parameter_estimates_upper_bound':1},{'parameter_estimates_upper_bound':False}):
        with pytest.raises(PermissionError):m.verify_release({**release,**update},expected)

def test_unknown_or_completed_directory_cannot_replay(tmp_path):
    directory=tmp_path/'prepare';m.claim_private(directory,{'hash':'frozen'})
    assert m.read(directory/'prepare_started.json')['status']=='STARTED'
    with pytest.raises(FileExistsError):m.claim_private(directory,{})
    m.write_once(directory/'prepare_complete.json',{'status':'COMPLETED'})
    with pytest.raises(FileExistsError):m.claim_private(directory,{})
    with pytest.raises(FileExistsError):m.write_once(directory/'prepare_started.json',{})

def fixture_bindings(tmp_path):
    files={}
    for index in range(15):
        p=tmp_path/f'file{index}.json';p.write_text('{}')
        files[str(index)]={'path':p.name,'bytes':p.stat().st_size,'sha256':m.sha(p)}
    return {'files':files}

def test_exact_fifteen_files_hash_size_and_no_pointer_following(tmp_path):
    bindings=fixture_bindings(tmp_path);bindings['unfollowed']={'path':'missing.npz','sha256':'do not follow'}
    assert len(m.bound_paths(bindings,tmp_path))==15
    bindings['files']['0']['sha256']='wrong'
    with pytest.raises(ValueError):m.bound_paths(bindings,tmp_path)

def test_bound_files_cannot_escape_root_or_open_HDF(tmp_path):
    bindings=fixture_bindings(tmp_path)
    bindings['files']['0']['path']='../outside.json'
    with pytest.raises(ValueError):m.bound_paths(bindings,tmp_path)
    bindings['files']['0']['path']='raw.h5'
    with pytest.raises(ValueError):m.bound_paths(bindings,tmp_path)

def test_wrong_count_or_duplicate_bound_file_rejected(tmp_path):
    bindings=fixture_bindings(tmp_path);bindings['files'].pop('0')
    with pytest.raises(ValueError):m.bound_paths(bindings,tmp_path)
    bindings=fixture_bindings(tmp_path);bindings['files']['0']=dict(bindings['files']['1'])
    with pytest.raises(ValueError):m.bound_paths(bindings,tmp_path)

def test_release_expected_does_not_import_numeric_or_old_scientific_modules(monkeypatch):
    original=builtins.__import__
    def guarded(name,*args,**kwargs):
        if name.split('.')[0] in ('numpy','scipy','mpmath','andes') or name.startswith('p_history_'):
            pytest.fail('engineering path imported scientific implementation')
        return original(name,*args,**kwargs)
    monkeypatch.setattr(builtins,'__import__',guarded)
    expected=m.release_expected()
    assert expected['real_prepare_upper_bound']==1 and expected['operator_qualification_cells_upper_bound']==16
    assert expected['qr_checks_upper_bound']==0

def test_constant_field_extraction_does_not_decode_other_records(tmp_path):
    p=tmp_path/'constant_control.json'
    # Deliberately invalid unrelated JSON: a whole-record decoder would fail.
    p.write_text('{"unread_previous_candidate":NOT_JSON,"p_scale":[1,2],"unread_tail":NOT_JSON}')
    assert m.p_scale_only(p)==[1,2]

def test_failed_W_gate_never_reaches_array_decoder(tmp_path,monkeypatch):
    private=tmp_path/'private';public=tmp_path/'public'
    monkeypatch.setattr(m,'PRIVATE',private);monkeypatch.setattr(m,'PUBLIC',public)
    monkeypatch.setattr(m,'checked_release',lambda:{'frozen':'release'})
    original_read=m.read
    def controlled_read(path):
        if Path(path)==m.ROOT/m.FROZEN['source_bindings'][0]:return {'control':'only'}
        return original_read(path)
    monkeypatch.setattr(m,'read',controlled_read)
    monkeypatch.setattr(m,'bound_paths',lambda b:{})
    monkeypatch.setattr(m,'source_proofs',lambda p,b:{'status':'verified'})
    monkeypatch.setattr(m,'qualification',lambda:({}, {'status':'unknown','cells':[],'cell_count':0}))
    monkeypatch.setattr(m,'read_events',lambda *args:pytest.fail('array decode before W pass'))
    result=m.prepare()
    assert result['status']=='unknown' and result['real_arrays_decode_started'] is False
    assert result['denominators']=={'all':297,'stable':267,'unstable':30}
    assert not (private/'targets.npz').exists()
    with pytest.raises(FileExistsError):m.prepare()
