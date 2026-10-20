"""Synthetic production-consumer checks; no private data or source weights."""
import copy,json,sys
from pathlib import Path
import numpy as np,pytest,torch
sys.path[:0]=[str(Path(__file__).resolve().parents[1]/'experiments'),str(Path(__file__).resolve().parents[1]/'src')]
import temporal_objective_20261020 as r

def test_requests_and_phase_lock():
    for seed in r.SEEDS:
        for v in r.VARIANTS:
            r.validate_request(seed,v,512)
            assert [sum(r.horizon_for(v,i)==h for i in range(1,513)) for h in ((1,2,5) if v=='O' else (1,2,15))]==[171,171,170]
    for kwargs in ({'seed':0,'variant':'O','target':512},{'seed':6101,'variant':'B','target':512},{'seed':6101,'variant':'F','target':768},{'seed':6101,'variant':'T','target':512,'resume':True},{'seed':6101,'variant':'T','target':512,'resume':True,'external':True,'start':31}):
        with pytest.raises(ValueError):r.validate_request(**kwargs)
    r.validate_request(6101,'T',512,True,True,32)

def test_shared_sampling512_real_endpoints_and_no_horizon_rng():
    events=[{'id':f'synthetic_{i}','label':'stable' if i<30 else 'unstable','group':'g','x':np.zeros((30+i,30)),'torque':np.zeros((30+i,10))} for i in range(40)]
    rng=np.random.default_rng(8301);coverage={'cover_order':rng.permutation(40).tolist(),'cursor':13}
    rr=copy.deepcopy(rng);cc=copy.deepcopy(coverage);rows=r.schedule_rows(events,rng,coverage)
    assert rows==r.schedule_rows(events,rr,cc)
    assert len(rows)==512 and {x for row in rows for x in row['sampling']['sample_ids']}=={e['id'] for e in events}
    assert all(len(x['sampling']['sample_ids'])==12 for x in rows)
    assert rows[2]['horizons']=={'O':5.,'F':15.,'T':15.}
    assert rows[2]['long_counts']['O']==rows[2]['long_counts']['F'] # all short; no new actual truth
    with pytest.raises(ValueError):r.schedule_rows(events,rr,cc,768)

def test_actual_cost_document_roundtrip_and_full_nine_gate(tmp_path):
    timings={v:{str(h):1. for h in ((1,2,5) if v=='O' else (1,2,15))} for v in r.VARIANTS}
    admission=r.cost_admission(timings,1.,1.,100.)
    path=tmp_path/'admission.json';path.write_text(json.dumps(admission))
    assert r.load_admission(path)==admission and admission['allow512'] and not admission['allow768']
    assert admission['nine_models_seconds']==3*sum(admission['per_model_seconds'].values())
    timings['T']['15']=100.;assert not r.cost_admission(timings,1.,1.,100.)['allow512']
    path.write_text(json.dumps({'admission':admission}))
    with pytest.raises(ValueError):r.load_admission(path)

def test_fixed_sources_metadata_and_bytes_refuse_mutation(tmp_path,monkeypatch):
    meta={'step':512,'identity':{'variant':'B','seed':6101,'source_global_step':1280}}
    for n,value in (('model.json','synthetic export'),('training.pt','synthetic bytes'),('checkpoint.json',json.dumps(meta))):(tmp_path/n).write_text(value)
    monkeypatch.setattr(r,'source_folder',lambda seed:tmp_path)
    monkeypatch.setattr(r,'SOURCE_HASHES',{6101:tuple(r.sha(tmp_path/n) for n in ('model.json','training.pt','checkpoint.json'))})
    assert r.source_metadata(6101)==meta
    (tmp_path/'training.pt').write_text('changed')
    with pytest.raises(ValueError):r.source_metadata(6101)

def test_cost_panels_are_id_fixed_longest_and_short_unstable():
    events=[{'id':str(i),'label':'unstable' if i<13 else 'stable','x':np.zeros((20+i,30))} for i in range(30)]
    panels,cpu=r.cost_panels(list(reversed(events)))
    assert [e['id'] for e,_ in panels['long']]==[str(i) for i in range(29,17,-1)]
    assert [e['id'] for e,_ in panels['short_unstable']]==[str(i) for i in range(12)]
    assert len(cpu)==16 and all(i==10 for panel in panels.values() for _,i in panel)

def test_tail_metrics_consume_same_prediction_real_endpoint(monkeypatch):
    import train_structured_rom
    monkeypatch.setattr(train_structured_rom,'_point_error',lambda p,t,m:(p[:,0]-t[:,0]).square())
    event={'x':np.zeros((17,30))};pred=np.zeros((3001,40));pred[::2,0]=np.arange(1501)
    row=r.temporal_metrics(None,event,pred,None)
    assert row['actual_future_points']==6 and row['tail_points']==2 and row['endpoint_group']=='le5s'
    assert row['tail_point_error']==(25+36)/2 and row['mean_point_error']==sum(i*i for i in range(1,7))/6

def test_production_nonempty_adam_checkpoint_restore_next_update(tmp_path):
    from test_temporal_objective_runtime_20261020 import model_fixture,event_fixture
    from temporal_objective_runtime_20261020 import one_update
    from train_closure import make_optimizer
    from fixed_edge_pair_20261019 import save_safe
    from linear_cache_cost_gate_20261019 import load_runtime,_equal
    load_runtime();torch.set_num_threads(1);dev=torch.device('cpu');model=model_fixture(dev);opt=make_optimizer(model)
    for group in opt.param_groups:group['lr']=r.RATES[group['block']]
    _,store,samples=event_fixture(dev);rng=np.random.default_rng(1);coverage={'synthetic':True}
    row=one_update(model,opt,store,samples,1.,r.FORCE,tail_weighted=False)
    identity={'qa_only':True};folder=tmp_path/'checkpoints';folder.mkdir()
    save_safe(folder,model,opt,rng,coverage,1,identity,[{'stage_step':1,**row}])
    one_update(model,opt,store,samples,15.,r.FORCE,tail_weighted=True)
    restored,ro,rr,rc,meta,_=r.restore_stage(folder,dev,identity)
    one_update(restored,ro,store,samples,15.,r.FORCE,tail_weighted=True)
    _equal(model.state_dict(),restored.state_dict(),exact=False);_equal(opt.state_dict(),ro.state_dict(),exact=False)
    assert meta['step']==1 and rc==coverage and rr.bit_generator.state==rng.bit_generator.state


def test_production_queue_uses_one_prediction_pair_and_retains_tail_rows(tmp_path,monkeypatch):
    from test_temporal_objective_runtime_20261020 import model_fixture,event_fixture
    import evaluate_structured_rom as evaluation
    import summarize_train_local_memory_0098 as summary
    import fixed_edge_pair_20261019 as fixed
    import active_fom_pair_fit_20261020 as fit
    model=model_fixture(torch.device('cpu'));events,_,_=event_fixture(torch.device('cpu'))
    for e in events:e.update(label='stable',classification_meta={})
    calls=[]
    def rollouts(model,histories,steps,dt):
        calls.append((steps,dt));return np.zeros((len(histories),steps+1,40))
    monkeypatch.setattr(evaluation,'_rollouts',rollouts)
    monkeypatch.setattr(evaluation,'_pair_row',lambda *a,**k:{'trigger_index':None})
    monkeypatch.setattr(summary,'event_metrics',lambda *a:{k:[] for k in ('records','common_frequency','normalized_p','objective_point_error','numerical')})
    monkeypatch.setattr(summary,'build_summary',lambda rows:{})
    monkeypatch.setattr(fixed,'scientific_qualification',lambda *a:True)
    monkeypatch.setattr(fixed,'primary_risk',lambda *a:0.)
    monkeypatch.setattr(fixed,'physical_metrics',lambda *a:{})
    monkeypatch.setattr(fit,'interval15',lambda *a:{'numerically_admissible':True})
    monkeypatch.setattr(r,'cohort_sets',lambda es:{'original366':{e['id'] for e in es},'new19':set()})
    result=r.evaluate_queue(model,events,tmp_path/'queue',{'scope':'synthetic only'},lambda:None,export_analysis=False)
    assert calls==[(3000,.005),(1500,.01)] and result['event_count']==3
    assert len(result['extras'])==3 and all('temporal' in e for e in result['extras'])
    assert len(result['prediction_files'])==1 and result['cohorts']['original366']['event_count']==3


def test_nontrain_spec_and_outside_output_rejected(monkeypatch,tmp_path):
    monkeypatch.setattr(r,'read',lambda path:{'split':'val','old_train':True,'new_directories':['forbidden']})
    with pytest.raises(ValueError,match='registered385 train'):r.load_events()
    monkeypatch.setattr(r,'BASE',tmp_path/'study')
    with pytest.raises(ValueError,match='outside fixed'):r.inside(tmp_path/'other')

def test_cohort_membership_from_frozen_source_not_prefix(monkeypatch):
    old=[{'id':f'newtrain_old_{i}'} for i in range(366)];new=[{'id':f'new_{i}'} for i in range(19)]
    monkeypatch.setattr(r,'sha',lambda path:'0d2b3ed030cd742ae0809114008c31e3a99c35c56d0238d2d03c5473abcecbf5')
    monkeypatch.setattr(r,'read',lambda path:{'cohorts':{'original366':old,'new19':new}})
    sets=r.cohort_sets(old+new)
    assert sets['original366']=={e['id'] for e in old} and sets['new19']=={e['id'] for e in new}
    with pytest.raises(ValueError,match='cohort mismatch'):r.cohort_sets([{'id':'unknown'}])


def test_restore_witness_longest_long_panel_within_twelve_updates():
    for v in r.VARIANTS:
        cases=r.cost_cases(v)
        assert len(cases)==len(set(cases))==6
        assert cases[:2]==[(1,'long'),(5 if v=='O' else 15,'long')]
        actual=[(cases[0],1),(cases[1],1),(cases[1],2),(cases[0],2)]+[(c,rep) for c in cases[2:] for rep in (1,2)]
        assert len(actual)==12 and {c:sum(c==a for a,_ in actual) for c in cases}=={c:2 for c in cases}

def test_nine_zero_update_source_comparisons_restore_rng_on_success_and_failure(monkeypatch):
    from types import SimpleNamespace
    from test_temporal_objective_runtime_20261020 import model_fixture
    dev=torch.device('cpu');source=model_fixture(dev);calls=[]
    monkeypatch.setattr(r,'generic_source',lambda seed,device:copy.deepcopy(source))
    def new(seed,variant,device):
        calls.append((seed,variant));torch.rand(3)
        return copy.deepcopy(source),SimpleNamespace(state={}),None,None
    monkeypatch.setattr(r,'new_stage',new)
    before=torch.get_rng_state().clone();out=r.source_equivalence(dev,lambda:None)
    assert torch.equal(before,torch.get_rng_state()) and out['count']==9
    assert calls==[(seed,v) for seed in r.SEEDS for v in r.VARIANTS]
    assert all(c['updates']==0 for c in out['comparisons'])
    def broken(seed,variant,device):
        model,opt,rng,coverage=new(seed,variant,device)
        with torch.no_grad():next(model.parameters()).add_(1.)
        return model,opt,rng,coverage
    monkeypatch.setattr(r,'new_stage',broken)
    with pytest.raises(ValueError,match='source state differs'):r.source_equivalence(dev,lambda:None)
    assert torch.equal(before,torch.get_rng_state())
