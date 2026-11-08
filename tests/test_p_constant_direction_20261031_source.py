"""Pure control/file/field-selection fixtures; no real payload or parent import."""
import builtins
import importlib.util
import math
from pathlib import Path
from types import SimpleNamespace
import pytest

spec=importlib.util.spec_from_file_location('constant_source_engineering',Path(__file__).parents[1]/'experiments/p_constant_direction_20261031_source.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)

def test_release_requires_exact_frozen_source_and_zero_eta_limits():
    expected={'source_sha256':'frozen','eta_estimates_upper_bound':0}
    release={'released':True,'allowed_stage':'source',**expected}
    m.verify_release(release,expected)
    for update in ({'released':'true'},{'allowed_stage':'estimate'},{'source_sha256':'other'},
                   {'eta_estimates_upper_bound':1},{'eta_estimates_upper_bound':False}):
        with pytest.raises(PermissionError):m.verify_release({**release,**update},expected)

def test_release_expected_does_not_import_parent_or_scientific_packages(monkeypatch):
    original=builtins.__import__
    def guarded(name,*args,**kwargs):
        if name.split('.')[0] in ('numpy','scipy','mpmath','andes') or name.startswith('p_residual_stage'):
            pytest.fail('read-only control path imported forbidden implementation')
        return original(name,*args,**kwargs)
    monkeypatch.setattr(builtins,'__import__',guarded)
    expected=m.release_expected()
    assert expected['bound_assets_upper_bound']==28 and expected['static_content_files_upper_bound']==8
    assert expected['eta_estimates_upper_bound']==0 and expected['MP_checks_upper_bound']==0

def test_unknown_or_completed_scientific_directory_rejects_replay(tmp_path):
    directory=tmp_path/'private';m.claim_private(directory,{'hash':'control'})
    with pytest.raises(FileExistsError):m.claim_private(directory,{})
    m.write_once(directory/'complete.json',{'status':'COMPLETED'})
    with pytest.raises(FileExistsError):m.claim_private(directory,{})
    with pytest.raises(FileExistsError):m.write_once(directory/'started.json',{})

def fixture_assets(tmp_path):
    files={}
    for i in range(28):
        p=tmp_path/f'file{i}.json';p.write_text('{}')
        files[str(i)]={'path':p.name,'bytes':p.stat().st_size,'sha256':m.sha(p),'kind':'static_content' if i<8 else 'control'}
    return {'files':files}

def test_new28_guard_never_uses_parent24_or_follows_record_pointers(tmp_path):
    bindings=fixture_assets(tmp_path);bindings['unfollowed_pointer']={'path':'missing.npz'}
    counts={'assets_verified':0,'source_bytes_hashed':0}
    assert len(m.verify_assets(bindings,tmp_path,counts))==28 and counts['assets_verified']==28
    bindings['files'].pop('27')
    with pytest.raises(m.SourceUnknown):m.verify_assets(bindings,tmp_path,counts)

def test_asset_hash_and_root_guard(tmp_path):
    bindings=fixture_assets(tmp_path);counts={'assets_verified':0,'source_bytes_hashed':0}
    bindings['files']['0']['sha256']='changed'
    with pytest.raises(m.SourceUnknown):m.verify_assets(bindings,tmp_path,counts)
    bindings['files']['0']['path']='../outside.json'
    with pytest.raises(ValueError):m.verify_assets(bindings,tmp_path,counts)

def test_static_excerpt_is_bounded_and_missing_needle_remains_unknown():
    text='before\nalpha first\nafter\nalpha second\nbeta\nend\n'
    entry={'needles':['alpha','beta'],'context_lines_each_side':1,'max_matching_lines':1,'max_excerpt_lines':2}
    result=m.excerpt(text,entry)
    assert result['matching_lines_selected']==1 and len(result['lines'])==2 and result['excerpt_truncated'] is True
    assert result['status']=='matched_static_declaration'
    result=m.excerpt(text,{**entry,'needles':['alpha','absent_source']})
    assert result['status']=='unknown' and result['needle_matches']['absent_source']==0

def test_only_whitelisted_bound_helpers_exposed_and_bad_hash_never_imported(tmp_path):
    path=tmp_path/'fake_helpers.py'
    code='\n'.join(f'def {name}(*a, **kw): return "{name}"' for name in m.HELPERS)
    path.write_text(code+'\ndef aggregate(): raise RuntimeError("forbidden entry")\ndef checked_release(): raise RuntimeError("forbidden control")\n')
    with pytest.raises(m.SourceUnknown):m.load_helpers(path,'bad')
    helpers=m.load_helpers(path,m.sha(path))
    assert set(vars(helpers))==set(m.HELPERS)
    assert not hasattr(helpers,'aggregate') and not hasattr(helpers,'checked_release') and not hasattr(helpers,'verify_assets')

def test_stage_comparison_keeps_empty_unknown_and_rejects_boolean_change():
    h=SimpleNamespace(close=lambda a,b:isinstance(a,(float,int)) and isinstance(b,(float,int)) and abs(a-b)<=1e-12+1e-12*abs(b),digest=lambda value:'finite_control_hash')
    moment={'mu':0.,'C_Q':0.,'C_L':0.,'C_diff':0.,'N_original':30,'N_available':0,'windows':0}
    stats={'status':'unknown','reason':'no_window','classification':'unknown','R_Q':None,'R_L':None,'nu':None,'compatible':None,'resolved':None}
    ref={**moment,**stats}
    assert m.compare_stage(h,moment,stats,ref)['status']=='verified'
    with pytest.raises(m.SourceUnknown):m.compare_stage(h,moment,stats,{**ref,'compatible':False})
    with pytest.raises(m.SourceUnknown):m.compare_stage(h,moment,stats,{**ref,'mu':float('nan')})

def test_M_is_proof_reference_not_numeric_eta_or_physical_h():
    record=m.math_reference()
    assert record['status']=='adopted_independently_reviewed_conditional_proof'
    assert record['h0_implies_fixed_theta_residual_mean_zero'] is False
    assert record['h0_implies_individual_drives_zero'] is False
    assert record['normalized_sign']=='h_norm=-eta'
    assert 'p_scale not read' in record['physical_sign_formal_only']
