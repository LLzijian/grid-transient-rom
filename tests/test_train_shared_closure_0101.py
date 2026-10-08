"""Synthetic metadata/guard tests; no private TRAIN updates or integration."""
import copy
import json
from pathlib import Path
import sys
import types
import numpy as np
import pytest
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'experiments'))
import train_shared_closure_0101 as r


class Toy(torch.nn.Module):
    dim=40
    def __init__(self,arm='S23'):
        super().__init__()
        self.arm=arm
        self.structured=torch.nn.Module()
        self.structured.coupling=torch.nn.Parameter(torch.tensor([1.],dtype=torch.float64))
        self.local_gru=torch.nn.Linear(2,2,dtype=torch.float64)
        self.local_head=torch.nn.Linear(2,1,bias=False,dtype=torch.float64)
        if arm=='S23':
            self._q_quadratic_raw=torch.nn.Parameter(torch.tensor(1.,dtype=torch.float64))
            self._template_scalars=torch.nn.ParameterList([torch.nn.Parameter(torch.tensor(1.,dtype=torch.float64)) for _ in range(7)])
            self._template_scalars[6].requires_grad_(False)
            self.port_raw=torch.nn.Parameter(torch.ones(6,dtype=torch.float64))
        else:
            self.closure_coefficients=torch.nn.Parameter(torch.ones(23,dtype=torch.float64))
        self.register_buffer('fixed_graph',torch.eye(2,dtype=torch.float64))
    def config(self):return {'arm':self.arm,'synthetic':True}
    def export(self,path):Path(path).write_text(json.dumps({'arm':self.arm,'synthetic':True}))
    def complexity(self):return sum(p.abs().sum() for p in self.parameters() if p.requires_grad)


def test_fresh_adam_all_parameters_and_registered_rates():
    for arm in r.ARMS:
        model=Toy(arm); opt=r.optimizer_for(model)
        layout=r.parameter_layout(model,opt)
        r.validate_optimizer(model,opt,layout,0)
        rates={n:g['lr'] for g in layout for n in g['parameter_names']}
        assert rates['structured.coupling']==.01
        assert rates['local_gru.weight']==.0003
        if arm=='S23':
            assert rates['_q_quadratic_raw']==.01 and rates['port_raw']==.001
        else:assert rates['closure_coefficients']==.001
        assert set(rates)=={n for n,p in model.named_parameters() if p.requires_grad}
        assert len(opt.state)==0


def test_unknown_parameter_rejected():
    model=Toy(); model.unregistered=torch.nn.Parameter(torch.ones(1))
    with pytest.raises(ValueError,match='unregistered'):r.optimizer_for(model)


def test_restore_mapping_does_not_consume_training_rng(monkeypatch):
    base=Toy(); rng=np.random.default_rng(98); cov={'cover_order':[0,1],'cursor':1}
    torch_before=torch.get_rng_state().clone(); np_before=copy.deepcopy(rng.bit_generator.state)
    meta={'step':1167}
    monkeypatch.setattr(r,'restore_ancestor',lambda *a:(base,object(),rng,cov,meta))
    def make_pair(model):
        return {'S23':Toy('S23'),'F23':Toy('F23')}
    monkeypatch.setitem(sys.modules,'shared_closure_rom_0101',types.SimpleNamespace(make_pair=make_pair))
    pair,got_rng,got_cov,got_meta=r.restore_model_or_pair()
    assert set(pair)==set(r.ARMS) and got_cov==cov
    assert torch.equal(torch_before,torch.get_rng_state()) and got_rng.bit_generator.state==np_before
    assert got_meta['optimizer_reset'] is True


def test_adam_relative_update_counter_and_finiteness():
    model=Toy('F23'); opt=r.optimizer_for(model); layout=r.parameter_layout(model,opt)
    for p in model.parameters():
        opt.state[p]={'step':torch.tensor(2.),'exp_avg':torch.zeros_like(p),'exp_avg_sq':torch.ones_like(p)}
    r.validate_optimizer(model,opt,layout,2)
    with pytest.raises(ValueError,match='fresh Adam'):r.validate_optimizer(model,opt,layout,0)
    with pytest.raises(ValueError,match='count'):r.validate_optimizer(model,opt,layout,1169)
    next(iter(opt.state.values()))['exp_avg'].fill_(float('nan'))
    with pytest.raises(FloatingPointError):r.validate_optimizer(model,opt,layout,2)


def test_fixed_buffer_hash_detects_drift():
    model=Toy(); before=r.buffer_hash(model)
    with torch.no_grad():model.fixed_graph[0,0]+=1
    assert r.buffer_hash(model)!=before


def test_cost_estimate_includes_correct_milestones_and_zero_monitor():
    costs={'1':1.,'2':2.,'5':5.}; monitor=200.
    for steps,count in [(500,3),(1000,4)]:
        train=sum(costs[str(int(r.horizon(1167+i)))] for i in range(1,steps+1))
        reserve=r.reserve_seconds(monitor,list(costs.values()))
        assert r.estimated_cost(steps,costs,monitor)==train+count*monitor+reserve
        assert r.estimated_cost(steps,costs,monitor,include_initial=False)==train+(count-1)*monitor+reserve


def test_cost_evidence_identity(tmp_path,monkeypatch):
    model=tmp_path/'model.py'; model.write_text('synthetic')
    objective=tmp_path/'objective.py'; objective.write_text('synthetic objective')
    monkeypatch.setattr(r,'MODEL_CODE',model);monkeypatch.setattr(r,'OBJECTIVE_CODE',objective)
    value={'arm':'F23','training_dt':.01,'checkpoint_step':1167,'complete':True,'parameter_updates':False,
           'source_model_sha256':r.EXPECTED['model.pt'],'model_code_sha256':r.digest(model),
           'objective_code_sha256':r.digest(objective),'step_seconds':{'1':1.,'2':2.,'5':5.},'monitor_seconds':207.2187496}
    path=tmp_path/'cost.json';path.write_text(json.dumps(value))
    assert r.read_cost_evidence(path,'F23')==value
    with pytest.raises(ValueError):r.read_cost_evidence(path,'S23')
    for key,change in [('parameter_updates',True),('model_code_sha256','wrong'),('training_dt',.005),('monitor_seconds',0)]:
        path.write_text(json.dumps(dict(value,**{key:change})))
        with pytest.raises(ValueError):r.read_cost_evidence(path,'F23')


def test_checkpoint_fresh_adam_rng_and_failure_metadata(tmp_path):
    model=Toy('F23');opt=r.optimizer_for(model);rng=np.random.default_rng(98)
    args=(model,opt,rng,{'cover_order':[0],'cursor':0},1167,'F23',None,{},r.time.perf_counter())
    good=Path(r.save(tmp_path,'step0000',*args))
    bad=Path(r.save(tmp_path,'failed_state',*args,valid=False,attempted_step=1168))
    m=json.loads((good/'checkpoint.json').read_text());b=json.loads((bad/'checkpoint.json').read_text())
    assert m['optimizer_reset'] and m['optimizer_updates']==0
    assert m['config']['arm']=='F23' and (good/'structured_numeric_export.json').exists()
    assert b['step'] is None and b['optimizer_updates'] is None and b['last_verified_step']==1167
    assert not (bad/'structured_numeric_export.json').exists()
    saved=torch.load(good/'rng.pt',weights_only=False)
    assert saved['coverage']==m['coverage'] and saved['numpy']==rng.bit_generator.state


def test_qa_only_never_reads_train_or_integrates(tmp_path,monkeypatch):
    model=Toy('S23');rng=np.random.default_rng(98)
    monkeypatch.setattr(r,'snapshot',lambda *a:{'synthetic':True})
    monkeypatch.setattr(r,'restore_model_or_pair',lambda *a:(model,rng,{'cover_order':[0],'cursor':0},{}))
    monkeypatch.setattr(r,'check_structure',lambda m:{'synthetic':True})
    monkeypatch.setattr(r,'process_memory_bytes',lambda:0)
    def forbidden(*args,**kwargs):raise AssertionError('QA-only performed integration or data access')
    monkeypatch.setattr(r,'read_train',forbidden);monkeypatch.setattr(r,'run_monitor',forbidden)
    result=r.run('S23',tmp_path/'qa',qa_only=True,process_start=r.time.perf_counter())
    assert result['status']=='qa_restore_structure_complete' and result['verified_completed_updates']==0
    assert result['monitors']=={} and result['checkpoint_selection'] is None


@pytest.mark.parametrize('kwargs',[{'arm':'other'},{'new_steps':200},{'cap_seconds':9001}])
def test_registration_guards_before_io(tmp_path,kwargs):
    args={'arm':'F23','output':tmp_path/'fit',**kwargs}
    with pytest.raises(ValueError):r.run(**args)


def test_real_checkpoint_export_roundtrip_without_integration(tmp_path):
    """Read only the pinned model asset; exercise actual evaluator-family loader."""
    from shared_closure_rom_0101 import load_model
    pair,rng,coverage,meta=r.restore_model_or_pair()
    source_torch=torch.get_rng_state().clone()
    probe=torch.linspace(-.02,.02,80,dtype=torch.float64).reshape(2,40)
    for arm,model in pair.items():
        optimizer=r.optimizer_for(model)
        r.validate_optimizer(model,optimizer,r.parameter_layout(model,optimizer),0)
        r.check_structure(model)
        directory=r.save(tmp_path,arm,model,optimizer,rng,coverage,1167,arm,None,{},r.time.perf_counter())
        recovered=load_model(directory)
        assert r._model_hash(recovered)==r._model_hash(model)
        assert [(n,p.requires_grad) for n,p in recovered.named_parameters()]==[(n,p.requires_grad) for n,p in model.named_parameters()]
        with torch.no_grad():
            assert torch.allclose(model.field(probe),recovered.field(probe),atol=1e-12,rtol=0)
            assert torch.allclose(model.complexity(),recovered.complexity(),atol=1e-12,rtol=0)
        r.check_structure(recovered)
        saved=torch.load(Path(directory)/'rng.pt',weights_only=False)
        assert torch.equal(saved['torch'],source_torch)
        torch.set_rng_state(source_torch)


@pytest.fixture
def cache_case(tmp_path,monkeypatch):
    directory=tmp_path/'cache';directory.mkdir()
    model=Toy('S23');manifest={'sha256':'fixed-manifest'}
    monkeypatch.setattr(r,'sources',lambda:{'synthetic':'source'})
    h=r._model_hash(model)
    complexity=float(model.complexity().detach())
    row={'loop':'0099','complete':True,'J_fine':None,'seconds':250.,
         'model_state_sha256_before':h,'model_state_sha256_after':h,
         'manifest_sha256':manifest['sha256'],
         'monitor_code_sha256':r.digest(r.ROOT/'experiments/monitor_joint_adequacy_0099.py'),
         'source_before':r.sources(),'source_after':r.sources(),'sources_unchanged':True,
         'rng_unchanged':True,'batch_terms':{'fine':{'term_names':['complexity'],
             'terms':[[[complexity] for _ in range(3)] for _ in range(173)]}}}
    context={'loop':'0101','objective':'C23','arm':'S23','model_code_sha256':r.digest(r.MODEL_CODE),
             'source_model_sha256':r.EXPECTED['model.pt'],
             'source_checkpoint_sha256':r.EXPECTED['checkpoint.json'],'model_state_sha256':h}
    def write(new_row=None,new_context=None,update_hash=True):
        rr=copy.deepcopy(row if new_row is None else new_row)
        cc=copy.deepcopy(context if new_context is None else new_context)
        (directory/'evidence.json').write_text(json.dumps(rr))
        if update_hash:cc['monitor_evidence_sha256']=r.digest(directory/'evidence.json')
        (directory/'context_0101.json').write_text(json.dumps(cc))
        (directory/'manifest.json').write_text(json.dumps(manifest))
    write()
    return directory,model,manifest,row,context,write


def test_cache_accepts_c23_unresolved_j_without_relabel_or_charge(cache_case):
    directory,model,manifest,*_=cache_case
    before=torch.get_rng_state().clone()
    row,audit=r.cached_initial_monitor(directory,'S23',model,manifest)
    assert row['J_fine'] is None and row['loop']=='0099'
    assert audit['accepted'] and audit['additional_monitor_seconds_charged']==0
    assert audit['cached_monitor_seconds']==250 and torch.equal(before,torch.get_rng_state())


def test_cache_rejects_missing_context_or_old_c7(cache_case):
    directory,model,manifest,row,context,write=cache_case
    (directory/'context_0101.json').unlink()
    with pytest.raises(FileNotFoundError):r.cached_initial_monitor(directory,'S23',model,manifest)
    for change in ({'objective':'C7'},{'loop':'0099'},{'arm':'F23'}):
        write(new_context=dict(context,**change))
        with pytest.raises(ValueError):r.cached_initial_monitor(directory,'S23',model,manifest)


def test_cache_rejects_any_model_identity_mismatch(cache_case):
    directory,model,manifest,row,context,write=cache_case
    for key in ('model_state_sha256_before','model_state_sha256_after'):
        write(new_row=dict(row,**{key:'wrong'}))
        with pytest.raises(ValueError):r.cached_initial_monitor(directory,'S23',model,manifest)
    write(new_context=dict(context,model_state_sha256='wrong'))
    with pytest.raises(ValueError):r.cached_initial_monitor(directory,'S23',model,manifest)


def test_cache_rejects_manifest_code_and_source_mismatches(cache_case):
    directory,model,manifest,row,context,write=cache_case
    for key in ('model_code_sha256','source_model_sha256','source_checkpoint_sha256'):
        write(new_context=dict(context,**{key:'wrong'}))
        with pytest.raises(ValueError):r.cached_initial_monitor(directory,'S23',model,manifest)
    for key,value in [('manifest_sha256','wrong'),('monitor_code_sha256','wrong'),
                      ('source_before',{}),('source_after',{})]:
        write(new_row=dict(row,**{key:value}))
        with pytest.raises(ValueError):r.cached_initial_monitor(directory,'S23',model,manifest)


def test_cache_rejects_incomplete_rng_source_integrity_or_bad_cost(cache_case):
    directory,model,manifest,row,context,write=cache_case
    for key,value in [('complete',False),('rng_unchanged',False),('sources_unchanged',False),
                      ('seconds',0),('seconds',float('nan'))]:
        write(new_row=dict(row,**{key:value}))
        with pytest.raises(ValueError):r.cached_initial_monitor(directory,'S23',model,manifest)


def test_cache_rejects_tampered_evidence_and_disguised_c7_terms(cache_case):
    directory,model,manifest,row,context,write=cache_case
    write(new_context=dict(context,monitor_evidence_sha256='wrong'),update_hash=False)
    with pytest.raises(ValueError,match='evidence_hash'):r.cached_initial_monitor(directory,'S23',model,manifest)
    changed=copy.deepcopy(row);changed['batch_terms']['fine']['terms'][0][0][0]+=1
    write(new_row=changed)
    with pytest.raises(ValueError,match='objective'):r.cached_initial_monitor(directory,'S23',model,manifest)


def test_cache_qa_cost_excludes_initial_and_never_integrates(cache_case,tmp_path,monkeypatch):
    directory,model,manifest,row,context,write=cache_case
    rng=np.random.default_rng(98)
    monkeypatch.setattr(r,'snapshot',lambda *a:{'synthetic':True})
    meta={'monitor_manifest_sha256':r._json_digest(manifest)}
    monkeypatch.setattr(r,'restore_model_or_pair',lambda *a:(model,rng,{'cover_order':[0],'cursor':0},meta))
    monkeypatch.setattr(r,'check_structure',lambda m:{'synthetic':True})
    monkeypatch.setattr(r,'process_memory_bytes',lambda:0)
    costs={'monitor_seconds':207.,'step_seconds':{'1':1.,'2':2.,'5':5.}}
    path=tmp_path/'cost.json';path.write_text(json.dumps(costs))
    monkeypatch.setattr(r,'read_cost_evidence',lambda *a:costs)
    def forbidden(*a,**kw):raise AssertionError('cached QA integrated or read TRAIN')
    monkeypatch.setattr(r,'read_train',forbidden);monkeypatch.setattr(r,'run_monitor',forbidden)
    result=r.run('S23',tmp_path/'qa',qa_only=True,cost_evidence=path,
                 initial_monitor=directory,process_start=r.time.perf_counter())
    assert result['monitors']['0']['loop']=='0099' and result['initial_monitor_cache']['accepted']
    for n in (500,1000):
        estimate=result['nominal_cost_estimates'][str(n)]
        assert estimate['remaining_seconds']==r.estimated_cost(n,costs['step_seconds'],250.,include_initial=False)
        assert estimate['monitor_seconds_used']==250. and not estimate['includes_initial_monitor']
    write(new_context=dict(context,objective='C7'))
    with pytest.raises(ValueError):r.run('S23',tmp_path/'rejected',qa_only=True,
        initial_monitor=directory,process_start=r.time.perf_counter())
    assert not (tmp_path/'rejected').exists()
