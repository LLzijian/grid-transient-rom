"""Synthetic-only production schedule, identity, Adam/RNG and refusal tests."""
import copy
from pathlib import Path
import sys
import json
import numpy as np
import pytest
import torch
sys.path[:0]=[str(Path(__file__).resolve().parents[1]/'experiments'),str(Path(__file__).resolve().parents[1]/'src')]
import quadratic_memory_edge_20261020 as runner
from quadratic_memory_runtime_20261020 import build_variant,QuadraticMemoryRuntime
from test_quadratic_memory_runtime_20261020 import source

def test_exact_registered_identities_targets_and_extension_refusals():
    for seed in runner.SEEDS:
        for variant in runner.VARIANTS:runner.validate_request(seed,variant,512)
    for kwargs in ({'seed':9,'variant':'A','target':512},{'seed':6101,'variant':'other','target':512},
                   {'seed':6101,'variant':'C','target':768},{'seed':6101,'variant':'C','target':768,'resume':True,'start':512},
                   {'seed':6101,'variant':'C','target':512,'resume':True,'external':True,'start':31},
                   {'seed':6101,'variant':'C','target':512,'external':True}):
        with pytest.raises(ValueError):runner.validate_request(**kwargs)
    runner.validate_request(6101,'C',768,True,False,512,'explicit-lock')

def test_contiguous768_schedule_inherits_rng_coverage_and512_prefix():
    events=[{'id':f'synthetic_{i:03d}','label':'stable' if i<24 else 'unstable','group':str(i%4),
             'x':np.zeros((531,30)),'torque':np.zeros((531,10))} for i in range(32)]
    rng=np.random.default_rng(730);coverage={'cover_order':rng.permutation(32).tolist(),'cursor':11}
    state=copy.deepcopy(rng.bit_generator.state);initial=copy.deepcopy(coverage)
    rows=runner.schedule_rows(events,rng,coverage)
    rr=np.random.default_rng();rr.bit_generator.state=state;cc=copy.deepcopy(initial)
    first=runner.schedule_rows(events,rr,cc,512)
    assert rows[:512]==first
    assert rows[512]['stage_step']==513 and rows[512]['horizon_s']==5.
    assert rows[0]['numpy_rng_before']==state and rows[0]['sampling']['coverage_before']==initial
    assert [sum(r['horizon_s']==h for r in rows[:512]) for h in (1,2,5)]==[171,171,170]
    assert all(len(r['sampling']['sample_ids'])==12 for r in rows)
    assert {x for r in rows for x in r['sampling']['sample_ids']}=={e['id'] for e in events}

def test_registered_trend_segments_and_cost_all_nine():
    trace=[{'loss':9.} for _ in range(512)]
    for i in range(384,448):trace[i]['loss']=10.
    for i in range(448,512):trace[i]['loss']=9.6
    assert runner.extension_trend(trace)
    trace[-1]['loss']=float('nan');assert not runner.extension_trend(trace)
    with pytest.raises(ValueError):runner.extension_trend(trace[:511])
    d=runner.cost_admission({'A':{'1':1.,'2':1.,'5':1.},'C':{'1':2.,'2':2.,'5':2.}},1.,1.,100.)
    assert d['allow512'] and d['allow768'] and d['eighteen_evaluations_seconds']==2250.
    assert d['est512']['nine_models_seconds']==9*d['est512']['per_model_seconds']
    with pytest.raises(ValueError):runner.cost_admission({'A':{}},1,1,1)
    assert not runner.cost_admission({'A':{'1':20.,'2':20.,'5':20.},'C':{'1':30.,'2':30.,'5':30.}},1,1,100)['allow512']

@pytest.mark.parametrize('device',['cpu']+(['cuda'] if torch.cuda.is_available() else []))
def test_production_stage32_restore_preserves_new_adam_rng_and_runtime(tmp_path,device):
    from train_closure import make_optimizer
    from fixed_edge_pair_20261019 import save_safe
    from linear_cache_cost_gate_20261019 import load_runtime,_equal
    load_runtime();torch.set_num_threads(1);dev=torch.device('cuda',torch.cuda.current_device()) if device=='cuda' else torch.device(device)
    model=build_variant(source(device),'C');opt=make_optimizer(model)
    for group in opt.param_groups:group['lr']=runner.RATES[group['block']]
    assert not opt.state
    x=torch.randn(2,40,dtype=torch.float64,device=dev)*.02
    def update(m,o):
        o.zero_grad(set_to_none=True)
        loss=m.field(x).square().sum();loss.backward();torch.nn.utils.clip_grad_norm_(m.parameters(),100);o.step();m.end_graph_cache()
    for _ in range(32):update(model,opt)
    rng=np.random.default_rng(981);coverage={'cover_order':[2,0,1],'cursor':7}
    folder=tmp_path/'checkpoints';folder.mkdir();identity={'seed':6101,'variant':'C','source_global_step':1280,'qa_only':True}
    trace=[{'stage_step':i+1,'loss':1.} for i in range(32)]
    save_safe(folder,model,opt,rng,coverage,32,identity,trace)
    cpu=torch.get_rng_state().clone();cuda=torch.cuda.get_rng_state(dev).clone() if dev.type=='cuda' else None
    rm,ro,rr,rc,meta,changed=runner.restore_stage(folder,dev,identity)
    assert isinstance(rm,QuadraticMemoryRuntime) and not changed and meta['step']==32 and meta['trace']==trace
    _equal(model.state_dict(),rm.state_dict(),exact=True);_equal(opt.state_dict(),ro.state_dict(),exact=True)
    assert rr.bit_generator.state==rng.bit_generator.state and rc==coverage and torch.equal(cpu,torch.get_rng_state())
    if cuda is not None:assert torch.equal(cuda,torch.cuda.get_rng_state(dev))
    update(model,opt);update(rm,ro)
    _equal(model.state_dict(),rm.state_dict(),exact=False);_equal(opt.state_dict(),ro.state_dict(),exact=False)
    with pytest.raises(ValueError):runner.restore_stage(folder,dev,{**identity,'variant':'A'})

def test_path_source_and_binding_reject_unknown_or_nontrain(tmp_path,monkeypatch):
    monkeypatch.setattr(runner,'BASE',tmp_path)
    with pytest.raises(ValueError):runner.inside(tmp_path/'../old-stage/model.json')
    with pytest.raises(ValueError):runner.source_folder(9000)
    p=tmp_path/'binding.json';p.write_text(json.dumps({'files':{'private_data/mixed/dataset.h5':'0'*64}}))
    with pytest.raises(ValueError):runner.binding(p)
    p.write_text(json.dumps({'files':{}}))
    with pytest.raises(ValueError):runner.binding(p)

def test_cost_document_to_production_admission_consumer(tmp_path):
    admission=runner.cost_admission({'A':{'1':1.,'2':1.,'5':1.},'C':{'1':2.,'2':2.,'5':2.}},1.,1.,100.)
    path=tmp_path/'admission.json';path.write_text(json.dumps(admission))
    assert runner.load_admission(path)==admission
    path.write_text(json.dumps({'admission':admission}))
    with pytest.raises(ValueError):runner.load_admission(path)

def test_actual_evaluation_consumer_uses_c_runtime_without_linear_guard(tmp_path,monkeypatch):
    import evaluate_structured_rom as evaluate
    import summarize_train_local_memory_0098 as metrics
    import active_fom_pair_fit_20261020 as fit
    import fixed_edge_pair_20261019 as fixed
    import linear_cache_runtime_20261019 as linear
    model=build_variant(source(),'C')
    seen=[]
    def rollouts(m,h,steps,dt):
        assert isinstance(m,QuadraticMemoryRuntime)
        seen.append((steps,dt));return np.zeros((len(h),steps+1,40),dtype=np.float64)
    monkeypatch.setattr(evaluate,'_rollouts',rollouts)
    monkeypatch.setattr(linear,'enable_linear_cache',lambda _:(_ for _ in ()).throw(AssertionError('old linear-only guard invoked')))
    keys=('records','common_frequency','normalized_p','objective_point_error','numerical')
    monkeypatch.setattr(metrics,'event_metrics',lambda *a,**k:{n:[] for n in keys})
    monkeypatch.setattr(metrics,'build_summary',lambda records:{'synthetic_scoring':True})
    monkeypatch.setattr(evaluate,'_pair_row',lambda *a,**k:{'los_pair_pass':True,'finite_to_endpoint':True,'coi_normalized_rmse':0.,'common_normalized_rmse':0.})
    monkeypatch.setattr(fit,'interval15',lambda *a,**k:{'available':True,'numerically_admissible':True})
    monkeypatch.setattr(fixed,'scientific_qualification',lambda *a:True)
    monkeypatch.setattr(fixed,'primary_risk',lambda *a:{'value':None,'eligible':0})
    monkeypatch.setattr(fixed,'physical_metrics',lambda *a:{})
    e={'id':'synthetic_0','root':'synthetic','group':'0','label':'stable','x':np.zeros((12,30)),
       'torque':np.zeros((12,10)),'classification_meta':{}}
    result=runner.evaluate_queue(model,[e],tmp_path/'evaluation',{'synthetic_only':True},lambda:None,export_analysis=False)
    assert seen==[(3000,.005),(1500,.01)] and result['event_count']==1
    assert (tmp_path/'evaluation/complete.json').is_file() and result['prediction_origin']=='endpoint_autonomous_inference'

