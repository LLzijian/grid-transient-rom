"""Synthetic 0102 infrastructure checks; no TRAIN/TEST reads or updates."""
import copy
import json
from pathlib import Path
import sys
import numpy as np
import pytest
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'experiments'))
import train_prediction_0102 as r

RATES={'skeleton':.01,'closure':.001,'encoder':.0003}


@pytest.fixture(autouse=True)
def forbid_data_and_updates(monkeypatch):
    def forbidden(*args,**kwargs):raise AssertionError('data/integration/update forbidden in infrastructure QA')
    monkeypatch.setattr(r,'read_train',forbidden)
    monkeypatch.setattr(r,'run_monitor',forbidden)
    monkeypatch.setattr(torch.optim.Adam,'step',forbidden)


class Toy(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.structured=torch.nn.Linear(1,1,bias=False,dtype=torch.float64)
        self.local_head=torch.nn.Linear(1,1,bias=False,dtype=torch.float64)
        self.closure_coefficients=torch.nn.Parameter(torch.ones(2,dtype=torch.float64))
        self.register_buffer('fixed',torch.ones(1))
    def begin_graph_cache(self):pass
    def end_graph_cache(self):pass
    def complexity(self):return self.closure_coefficients.abs().sum()


def test_fresh_optimizer_explicit_rates_and_names():
    model=Toy(); optimizer=r.optimizer_for(model,RATES)
    assert not optimizer.state
    assert {g['block']:g['lr'] for g in optimizer.param_groups}==RATES
    assert sum(len(g['params']) for g in optimizer.param_groups)==len(list(model.parameters()))
    model.unknown=torch.nn.Parameter(torch.ones(1))
    with pytest.raises(ValueError,match='unregistered'):r.optimizer_for(model,RATES)


def test_no_update_backward_uses_original_callable_and_preserves_model(monkeypatch):
    model=Toy(); optimizer=r.optimizer_for(model,RATES); state=r._model_hash(model)
    called=[]
    def loss_fn(m,samples,history,horizon,scale):
        called.append((history.shape,horizon,scale))
        loss=sum(p.square().sum() for p in m.parameters())
        return loss,{'complexity':m.complexity()}
    monkeypatch.setattr(r,'objective',loss_fn)
    samples=[({'x':np.zeros((12,30))},10)]
    row=r.compute_gradients(model,samples,2.,1.,optimizer,clip_norm=100.)
    assert called==[(torch.Size([1,11,30]),2.,1.)]
    assert row['gradient_norm']>0 and set(row['gradient_blocks'])==set(RATES)
    assert r._model_hash(model)==state and not optimizer.state


def test_rng_roundtrip_all_three_streams():
    rng=np.random.default_rng(123); state=r.capture_rng(rng)
    expected=(rng.random(),np.random.random(),torch.rand(1))
    r.restore_rng(state,rng)
    got=(rng.random(),np.random.random(),torch.rand(1))
    assert expected[:2]==got[:2] and torch.equal(expected[2],got[2])
    r.restore_rng(state,rng)
    assert r.rng_digest(state)==r.rng_digest(r.capture_rng(rng))


def plan(tmp_path,monkeypatch):
    card=tmp_path/'card.md'; card.write_text('registered synthetic fixture')
    monkeypatch.setattr(r,'CARD',card)
    return {'loop':'0102','registered':True,'card_sha256':r.digest(card),
        'source_model_sha256':r.EXPECTED['model.pt'],'arm':'S-O','phase':'autonomous',
        'updates':650,'horizons':[r.horizon(2167+i) for i in range(1,651)],
        'rates':RATES,'clip_norm':100.,'cap_seconds':6125.,
        'monitor_steps':[0,325,650],'monitor_seconds':120.,'cost_evidence':'unused','allow_refine':True}


@pytest.mark.parametrize('mutation',[{'registered':False},{'phase':'shooting'},
    {'source_model_sha256':'wrong'},{'monitor_steps':[0,2]},{'horizons':[1,2]},
    {'updates':700},{'cap_seconds':9000.},{'clip_norm':99.}])
def test_plan_fail_closed(tmp_path,monkeypatch,mutation):
    value=plan(tmp_path,monkeypatch); assert r.validate_plan(value)==value
    value.update(mutation)
    with pytest.raises(ValueError):r.validate_plan(value)


def test_reject_other_source_before_reading_files(tmp_path):
    with pytest.raises(ValueError,match='step2167'):r.verify_source(tmp_path)


def test_cached_monitor_checks_model_and_c23(tmp_path,monkeypatch):
    model=Toy(); manifest={'sha256':'fixed'}
    monkeypatch.setattr(r,'sources',lambda:{'synthetic':True})
    monkeypatch.setattr(r,'digest',lambda p:'hash')
    row={'complete':True,'model_state_sha256_before':r._model_hash(model),
        'model_state_sha256_after':r._model_hash(model),'manifest_sha256':'fixed',
        'monitor_code_sha256':'hash','source_before':{'synthetic':True},
        'source_after':{'synthetic':True},'sources_unchanged':True,'rng_unchanged':True,'seconds':10.,
        'batch_terms':{'fine':{'term_names':['complexity'],'terms':np.full((173,3,1),2.).tolist()}}}
    path=tmp_path/'evidence.json'; path.write_text(json.dumps(row))
    assert r.cached_initial_monitor(tmp_path,model,manifest)[1]['additional_monitor_seconds_charged']==0.
    for key,value in [('model_state_sha256_after','wrong'),('rng_unchanged',False)]:
        altered=copy.deepcopy(row); altered[key]=value; path.write_text(json.dumps(altered))
        with pytest.raises(ValueError):r.cached_initial_monitor(tmp_path,model,manifest)
    row['batch_terms']['fine']['terms'][0][0][0]=3.; path.write_text(json.dumps(row))
    with pytest.raises(ValueError,match='C23'):r.cached_initial_monitor(tmp_path,model,manifest)


def m_plan(tmp_path,monkeypatch):
    value=plan(tmp_path,monkeypatch)
    value.update(arm='S-M',phase='multiple_shooting_then_autonomous',model_optimizer_policy='continuous',
        shooting_cache='unused-cache',autonomous_cost_evidence='unused-O')
    phases=[]; offset=0
    for i,(count,budget) in enumerate(zip((32,32,68,518),(848.,848.,1802.,1802.))):
        phase={'kind':'shooting' if i<3 else 'autonomous','updates':count,
            'training_seconds':budget,'horizons':value['horizons'][offset:offset+count]}
        if i<3:phase.update(segment_seconds=(.5,1.,2.)[i],q_lr=.01,q_clip_norm=100.,
            aux_optimizer_policy='reset_all_each_shooting_phase',continuity_weight=(1.,10.,100.)[i])
        phases.append(phase);offset+=count
    value.update(phases=phases)
    return value


def test_m_plan_preserves_registered_course_and_final_budget(tmp_path,monkeypatch):
    value=m_plan(tmp_path,monkeypatch)
    assert r.validate_plan(value)==value
    bad=copy.deepcopy(value);bad['phases'][-1]['training_seconds']=25.
    with pytest.raises(ValueError,match='budget'):r.validate_plan(bad)
    bad=copy.deepcopy(value);bad['model_optimizer_policy']='reset'
    with pytest.raises(ValueError,match='continuous'):r.validate_plan(bad)
    bad=copy.deepcopy(value);bad['phase']='autonomous'
    with pytest.raises(ValueError,match='reset'):r.validate_plan(bad)
    bad=copy.deepcopy(value);bad['phases'][1]['continuity_weight']=1.
    with pytest.raises(ValueError,match='policy'):r.validate_plan(bad)


@pytest.mark.parametrize('saved',[None,{}, {'code':{'path':'fake','sha256':'old'}}])
def test_restore_rejects_snapshot_drift_before_loading_model(tmp_path,monkeypatch,saved):
    meta={'format':'prediction_optimization_0102','valid_training_state':True,
          'source_model_sha256':r.EXPECTED['model.pt'],'source_snapshot':saved}
    for name in ('model.pt','optimizer.pt','rng.pt','structured_numeric_export.json'):
        (tmp_path/name).write_text('synthetic placeholder; must never deserialize')
    (tmp_path/'checkpoint.json').write_text(json.dumps(meta))
    hashes={p.name:r.digest(p) for p in tmp_path.iterdir()}
    (tmp_path/'artifact_hashes.json').write_text(json.dumps(hashes))
    monkeypatch.setattr(r,'snapshot',lambda:{'code':{'path':'fake','sha256':'current'}})
    monkeypatch.setattr(r,'verify_source',lambda:pytest.fail('must reject drift before source recovery'))
    monkeypatch.setattr(r,'load_model',lambda *a:pytest.fail('must not deserialize stale model'))
    with pytest.raises(ValueError,match='source_snapshot'):r.restore_checkpoint(tmp_path,RATES)


@pytest.mark.parametrize('mode',['fit','phase_too_costly','total_too_costly'])
def test_m_cost_preflight_before_any_train_read(tmp_path,monkeypatch,mode):
    value=m_plan(tmp_path,monkeypatch);model=Toy();state_hash=r._model_hash(model)
    monkeypatch.setattr(r,'snapshot',lambda:{'synthetic':True})
    monkeypatch.setattr(r,'guard',lambda *a,**kw:None)
    monkeypatch.setattr(r,'restore_model_or_pair',lambda arm:(model,np.random.default_rng(1),{},{}))
    monkeypatch.setattr(r,'digest',lambda path:'hash')
    class ReadyForTrain(Exception):pass
    def stopped():raise ReadyForTrain('preflight complete; no TRAIN was read')
    monkeypatch.setattr(r,'train_inputs',stopped)
    common={'complete':True,'parameter_updates':False,'checkpoint_step':2167,
        'source_model_sha256':r.EXPECTED['model.pt'],'model_code_sha256':'hash',
        'objective_code_sha256':'hash','rates':RATES,'clip_norm':100.,
        'model_state_sha256_before':state_hash,'model_state_sha256_after':state_hash,
        'rng_and_model_unchanged':True,'step_seconds':{'1':1.,'2':1.,'5':1.}}
    ordinary=dict(common,arm='S-O')
    costs=dict(common,arm='S-M',shooting_code_sha256='hash',
        phase_step_seconds={str(s):{'1':1.,'2':1.,'5':1.} for s in (.5,1.,2.)},
        continuity_weights={'0.5':1.,'1.0':10.,'2.0':100.})
    if mode=='phase_too_costly':costs['phase_step_seconds']['0.5']={'1':100.,'2':100.,'5':100.}
    if mode=='total_too_costly':value['monitor_seconds']=200.
    for name,data in [('costs.json',costs),('ordinary.json',ordinary)]:
        (tmp_path/name).write_text(json.dumps(data))
    value['cost_evidence']=str(tmp_path/'costs.json');value['autonomous_cost_evidence']=str(tmp_path/'ordinary.json')
    plan_path=tmp_path/'plan.json';plan_path.write_text(json.dumps(value))
    error=ReadyForTrain if mode=='fit' else TimeoutError
    with pytest.raises(error):r.run_m_plan(plan_path,tmp_path/'output',value)


def test_boundary_layout_keeps_real_short_tail_and_q_transfer_resets_adam():
    model=Toy()
    events=[{'id':'long','x':np.zeros((532,30))},{'id':'short','x':np.zeros((86,30))}]
    layouts={str(s):r.shooting_boundaries(events,s) for s in (.5,1.,2.)}
    assert layouts['0.5']['short']==[10,60,85]
    assert layouts['2.0']['short']==[10,85]
    assert layouts['2.0']['long']==[10,210,410,510]
    keys={(e['id'],j) for e in events for layout in layouts.values() for j in layout[e['id']][1:-1]}
    cache={'layouts':layouts,'q_cache':{k:torch.full((10,),.25,dtype=torch.float64) for k in keys},'scales':torch.ones(40)}
    phase={'segment_seconds':.5,'q_lr':.01,'q_clip_norm':100.,'continuity_weight':1.,
        'aux_optimizer_policy':'reset_all_each_shooting_phase'}
    first,op1,_=r.shooting_problem(model,events,cache,phase)
    for record in first.segment_records:
        if record.start==210 and record.auxiliary_index is not None:
            with torch.no_grad():first.q_aux[record.auxiliary_index].fill_(.75)
    phase['segment_seconds']=2.
    second,op2,_=r.shooting_problem(model,events,cache,phase,first)
    assert not op1.state and not op2.state and op1 is not op2
    assert torch.equal(r.q_values_by_boundary(second)[('long',210)],torch.full((10,),.75,dtype=torch.float64))
    assert all(torch.equal(v,torch.full((10,),.25,dtype=torch.float64)) for v in cache['q_cache'].values())
    assert len(second.auxiliary_parameters)<len(first.auxiliary_parameters)


def test_non_scalar_metrics_are_compact_and_raw_is_opt_in():
    values=torch.arange(400,dtype=torch.float64).reshape(10,40)
    summary=r.summarize_term(values)
    assert summary['shape']==[10,40] and summary['count']==400 and 'values' not in summary
    assert set(summary['component_rms'])=={'delta','omega','p','q'}
    assert r.summarize_term(torch.zeros((0,40)))=={'shape':[0,40],'count':0}


def test_auxiliary_gradient_no_update_is_separate(monkeypatch):
    model=Toy();opt=r.optimizer_for(model,RATES)
    aux=torch.nn.Parameter(torch.ones(10,dtype=torch.float64));qopt=torch.optim.Adam([aux],lr=.01)
    def loss_fn(m,*args):
        loss=sum(p.square().sum() for p in m.parameters())+aux.square().sum()
        return loss,{'shooting_continuity_residuals':aux.repeat(4,4),'event_indices':torch.arange(4)}
    samples=[({'x':np.zeros((12,30))},10)]
    row=r.compute_gradients(model,samples,1.,1.,opt,clip_norm=100.,loss_fn=loss_fn,
        auxiliary_optimizer=qopt,auxiliary_parameters=[aux],auxiliary_clip_norm=100.)
    assert row['aux_gradient_scalars']==10 and 'raw_terms' not in row
    assert not opt.state and not qopt.state and torch.equal(aux,torch.ones(10,dtype=torch.float64))
