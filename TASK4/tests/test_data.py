import json
import h5py
import numpy as np
import pytest
from task4.data import load_events,metadata_preflight
from task4.paths import project_path
from task4.synthetic import fixture

def make_source(tmp_path):
    root=tmp_path/'TASK4';root.mkdir();k=tmp_path/'kiki';d=k/'private_data/set';d.mkdir(parents=True)
    es,p=fixture(events=3)
    fields={key:value for key,value in vars(p).items() if key!='omega_s'}
    np.savez(k/'base.npz',**fields)
    splits={'train':['synthetic_0','synthetic_1'],'validation':['synthetic_2'],'test':['poison_test']}
    (d/'splits.json').write_text(json.dumps(splits))
    (d/'dataset_index.csv').write_text('event_id,fault_bus,label,trajectory_end_s\n'+''.join(f'{e.id},{e.group},stable,{e.t[-1]}\n' for e in es)+'poison_test,99,unstable,1.0\n')
    with h5py.File(d/'dataset.h5','w') as f:
        f.attrs['omega_s']=p.omega_s;f['inertia']=p.mass*p.omega_s
        f['equilibrium/delta']=p.delta_star;f['equilibrium/Pm']=p.pm_star
        for e in es:
            f[f'events/{e.id}/t']=e.t;f[f'events/{e.id}/delta']=e.y[...,0]+p.delta_star
            f[f'events/{e.id}/omega']=e.y[...,1]/p.omega_s+1
            f[f'events/{e.id}/Pm']=e.y[...,2]+p.pm_star;f[f'events/{e.id}/te']=e.torque+p.pm_star
        # No TEST payload exists; a loader that visits it would fail.
    cfg={'kiki_root':'../kiki','baseline':'base.npz','datasets':['private_data/set'],'sample_dt':.01,'history_ms':100}
    return root,k,cfg,es

def test_readonly_train_no_test(tmp_path):
    root,k,cfg,es=make_source(tmp_path)
    before={str(p.relative_to(k)):p.read_bytes() for p in k.rglob('*') if p.is_file()}
    e,phys,facts=load_events(cfg,'train',root)
    assert [a.id for a in e]==['synthetic_0','synthetic_1']
    np.testing.assert_allclose(e[0].y,es[0].y,atol=1e-13)
    after={str(p.relative_to(k)):p.read_bytes() for p in k.rglob('*') if p.is_file()}
    assert before==after and facts['test_payload_read'] is False

def test_test_denied(tmp_path):
    root,k,cfg,_=make_source(tmp_path)
    with pytest.raises(PermissionError):load_events(cfg,'test',root)

def test_overlap_refused(tmp_path):
    root,k,cfg,_=make_source(tmp_path);p=k/'private_data/set/splits.json';s=json.loads(p.read_text());s['test'].append(s['train'][0]);p.write_text(json.dumps(s))
    with pytest.raises(ValueError,match='overlap'):load_events(cfg,'train',root)

def test_expected_counts_fail_closed(tmp_path):
    root,k,cfg,_=make_source(tmp_path);cfg['expected']={'train_eligible':200}
    with pytest.raises(ValueError,match='changed'):load_events(cfg,'train',root)

def test_output_cannot_escape(tmp_path):
    root=tmp_path/'TASK4';root.mkdir()
    with pytest.raises(PermissionError):project_path('../kiki/out.npz',root)

def test_metadata_only_preflight(tmp_path):
    root,k,cfg,_=make_source(tmp_path);f=metadata_preflight(cfg,root)
    assert f['counts_metadata_only']['test']==1 and f['test_payload_read'] is False
