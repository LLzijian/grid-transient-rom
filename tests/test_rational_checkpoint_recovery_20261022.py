"""CPU checks with the actual shared checkpoint serializer, never science updates."""
import copy
from pathlib import Path
import sys
import numpy as np
import pytest
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'experiments'),str(ROOT/'src')]
import rational_checkpoint_recovery_20261022 as engine


class Tiny(torch.nn.Module):
    def __init__(self):
        super().__init__();self.weight=torch.nn.Parameter(torch.tensor(.5,dtype=torch.float64))
        self.register_buffer('mass',torch.ones(1,dtype=torch.float64))
    def export(self,path):engine.atomic(path,{'weight':float(self.weight.detach()),'mass':self.mass.tolist()})
    @classmethod
    def from_export(cls,path):
        value=engine.read_json(path);model=cls()
        with torch.no_grad():model.weight.copy_(torch.tensor(value['weight'],dtype=torch.float64));model.mass.copy_(torch.tensor(value['mass'],dtype=torch.float64))
        return model


def model_optimizer():
    model=Tiny();return model,torch.optim.Adam(model.parameters(),lr=.001)


@pytest.fixture
def saved(tmp_path):
    folder=tmp_path/'new';folder.mkdir();torch.manual_seed(9001)
    m,opt=model_optimizer();rng=np.random.default_rng(123)
    trace=[];gradient={'total_raw_norm':[],'clip_factor':[]};summary={'first':{},'last':{},'max':{}}
    data={name:np.zeros((16,61,30),dtype=np.float64) for name in ('coarse','fine','observed')};data['scales']=np.ones(30)
    engine.study.save_data(folder/'train.npz',data)
    data_identity={name+'_sha256':engine.study.array_sha(data[key]) for name,key in (
        ('clean','coarse'),('fine','fine'),('observed','observed'),('scales','scales'))}
    data_identity.update(seed=6301,noise=0,split=0)
    identity={'source_commit':'a'*40,'key':'0_rational_6301','data_identity':data_identity,
        'train_raw':engine.file_ref(folder/'train.npz'),'form':'rational'}
    engine.publish_checkpoint(folder,m,opt,rng,0,trace,gradient,summary,identity)
    for step in range(32):
        opt.zero_grad();loss=m.weight.square();loss.backward();norm=float(m.weight.grad.abs());opt.step()
        rng.normal();torch.rand(1)
        trace.append(float(loss.detach()));gradient['total_raw_norm'].append(norm);gradient['clip_factor'].append(1.)
        if not summary['first']:summary['first']={'weight':norm}
        summary['last']={'weight':norm};summary['max']={'weight':max(summary['max'].get('weight',0),norm)}
    engine.publish_checkpoint(folder,m,opt,rng,32,trace,gradient,summary,identity)
    snapshot=engine.capture_state(m,opt,rng,trace,gradient,summary)
    return folder,identity,snapshot


@pytest.fixture
def original(saved):
    folder,identity,snapshot=saved
    for step in (0,32):(folder/f'diagnostics_{step:05d}.json').rename(folder/f'checkpoint_{step:05d}.diagnostics.json')
    pointer=engine.read_json(folder/'resume.json');pointer['diagnostics']=engine.study.fileid(folder/'checkpoint_00032.diagnostics.json');engine.atomic(folder/'resume.json',pointer)
    engine.atomic(folder/'failed_trace.json',{'loss':snapshot['trace'],'gradient':snapshot['gradient'],'parameter_summary':snapshot['summary']})
    row={'key':'0_rational_6301','seed':6301,'folder':str(folder.resolve()),'source_commit':'a'*40,
        'files':[engine.file_ref(path) for path in sorted(folder.rglob('*')) if path.is_file()],
        'directories':[str(path.resolve()) for path in sorted(folder.rglob('*')) if path.is_dir()],
        'identity':identity}
    return row,snapshot


def test_new_diagnostic_names_allow_real_shared_restore(saved):
    folder,identity,expected=saved
    assert sorted(path.name for path in folder.glob('checkpoint_*'))==['checkpoint_00000','checkpoint_00032']
    m,opt=model_optimizer();rng=np.random.default_rng(99)
    step,trace,gradient,summary=engine.restore_complete(folder,m,opt,rng,identity)
    assert step==32 and engine.state_equal(engine.capture_state(m,opt,rng,trace,gradient,summary),expected)


def test_original_collision_copy_and_real_restore_preserve_bytes(original,tmp_path):
    row,expected=original;folder=Path(row['folder']);m,opt=model_optimizer();rng=np.random.default_rng(99)
    _,restore=engine.checkpoint_api()
    with pytest.raises(ValueError,match='uncommitted checkpoint'):restore(folder,m,opt,rng,row['identity'])
    target=engine.prepare_copy(row,tmp_path/'copy')
    assert (target/'diagnostics_00032.json').is_file() and not (target/'checkpoint_00032.diagnostics.json').exists()
    step,trace,gradient,summary=engine.restore_complete(target,m,opt,rng,row['identity'])
    assert step==32 and engine.state_equal(engine.capture_state(m,opt,rng,trace,gradient,summary),expected)
    assert all(engine.file_ref(ref['path'])==ref for ref in row['files'])


@pytest.mark.parametrize('entry',['pending_saved','checkpoint_00064','unknown_empty'])
def test_original_unknown_or_future_directory_not_omitted(original,tmp_path,entry):
    row,_=original;folder=Path(row['folder']);(folder/entry).mkdir()
    # Even a completely listed anomaly is rejected by the explicit original0/32 scope.
    row['directories'].append(str((folder/entry).resolve()))
    with pytest.raises(ValueError,match='only original0/32'):engine.prepare_copy(row,tmp_path/'copy')
    assert not (tmp_path/'copy').exists()


def test_original_changed_bytes_rejected_before_copy(original,tmp_path):
    row,_=original;path=Path(row['folder'])/'checkpoint_00032/model.json';path.write_bytes(path.read_bytes()+b' ')
    with pytest.raises(ValueError,match='manifest'):engine.prepare_copy(row,tmp_path/'copy')
    assert not (tmp_path/'copy').exists()


def test_missing_diagnostic_rejected(original,tmp_path):
    row,_=original;(Path(row['folder'])/'checkpoint_00032.diagnostics.json').unlink()
    with pytest.raises(ValueError,match='manifest'):engine.prepare_copy(row,tmp_path/'copy')


def test_manifest_cannot_omit_unknown_file_or_empty_directory(original,tmp_path,monkeypatch):
    row,_=original;folder=Path(row['folder']);(folder/'unlisted.json').write_text('{}',encoding='utf-8');(folder/'unlisted_empty').mkdir()
    monkeypatch.setattr(torch,'load',lambda *a,**k:pytest.fail('validation must not load training state'))
    with pytest.raises(ValueError,match='manifest'):engine.prepare_copy(row,tmp_path/'copy')
    assert not (tmp_path/'copy').exists()


def test_migration_loads_old_once_and_publishes_new_identity(original,tmp_path,monkeypatch):
    row,expected=original;calls=[];actual_load=torch.load
    def load(*args,**kwargs):calls.append(str(args[0]));return actual_load(*args,**kwargs)
    monkeypatch.setattr(torch,'load',load)
    monkeypatch.setattr(engine.study,'make',lambda seed,observed,form:model_optimizer())
    state=engine.migrate(row,tmp_path/'copy',tmp_path/'science','b'*40)
    assert len(calls)==1 and state['step']==32
    assert state['identity']['source_commit']=='b'*40 and state['identity']['origin']==row
    assert engine.read_json(tmp_path/'copy/resume.json')['identity']==row['identity']
    assert engine.read_json(tmp_path/'science/resume.json')['identity']==state['identity']
    engine.apply_global_rng(state['global_rng'])
    assert engine.state_equal(engine.capture_state(state['m'],state['opt'],state['rng'],state['trace'],state['gradient'],state['summary']),expected)
    assert engine.read_json(tmp_path/'science/migration.json')['publish_state_equal'] is True
    with pytest.raises(ValueError,match='already exists'):engine.migrate(row,tmp_path/'copy',tmp_path/'science','b'*40)
    assert len(calls)==1 and all(engine.file_ref(ref['path'])==ref for ref in row['files'])


def test_closed_old_attempt_cannot_be_resumed_or_restarted(tmp_path):
    engine.atomic(tmp_path/'complete.json',{'status':'COMPLETED','all_group_frozen':False})
    with pytest.raises(ValueError,match='completed attempts'):engine.train_group(tmp_path,'b'*40,'0_rational')
    with pytest.raises(TypeError):engine.train_group(tmp_path,'b'*40,'0_rational',resume=True)


def test_twelve_real_endpoint_barrier_remains():
    with pytest.raises(ValueError,match='all12'):engine.study.validate_selection({'all_twelve_frozen':False,'rows':[]},'b'*40)


def test_captured_state_is_independent_and_comparison_is_exact(saved):
    _,_,state=saved;other=copy.deepcopy(state)
    assert engine.state_equal(state,other)
    other['optimizer']['state'][0]['exp_avg'].add_(1e-15)
    assert not engine.state_equal(state,other)
    other=copy.deepcopy(state);other['fit_numpy_rng']['state']['state']+=1
    assert not engine.state_equal(state,other)
