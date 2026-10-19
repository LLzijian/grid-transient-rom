"""Synthetic finite QA. Run once under the root's registered QA envelope.

The real optimizer check performs three tiny GPU updates, not 256/1024-step
recovery proof. No historical event, FOM or val trajectory is loaded.
"""
from __future__ import annotations
import argparse
import copy
import json
import math
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'experiments'),str(ROOT/'src')]
import active_fom_pair_fit_20261020 as runner


def rejected(fn):
    try:
        fn()
    except (ValueError,FileExistsError,TypeError,KeyError):
        return
    raise AssertionError('expected fail-closed rejection')


def events_fixture(count=9):
    import numpy as np
    rng=np.random.default_rng(90210)
    return [{'id':f'synthetic-{i:03d}','group':str(i%3),
        'label':'unstable' if i%3==0 else 'stable',
        'x':rng.normal(0,.001,(14+i%4,30)),
        'torque':rng.normal(0,.001,(14+i%4,10))} for i in range(count)]


def check_schedule(output):
    import numpy as np
    import torch
    from train_joint_numerical_pair_0100 import next_samples
    events=events_fixture(390)
    before=torch.get_rng_state().clone()
    s=runner.freeze_schedules(events,output/'schedule')
    assert torch.equal(before,torch.get_rng_state())
    for seed in runner.SEEDS:
        plan=s['seeds'][str(seed)]; rows=plan['rows']
        assert plan['sampler_seed']==seed+1000
        assert [sum(r['horizon_s']==h for r in rows) for h in (1,2,5)]==[86,85,85]
        assert rows[0]['global_step']==1025 and rows[-1]['global_step']==1280
        assert len(plan['included_ids'])==390
        rng=np.random.default_rng(); rng.bit_generator.state=plan['initial_rng']
        cov=copy.deepcopy(plan['initial_coverage'])
        for i,row in enumerate(rows,1):
            chosen,sampling=next_samples(events,rng,cov)
            assert sampling==row['sampling']
            assert len(chosen)==12 and row['relative_step']==i
            assert row['long_counts']==[min(round(row['horizon_s']*100),len(e['x'])-11) for e,_ in chosen]
            assert row['local_counts']==[min(20,len(e['x'])-j-1) for e,j in chosen]
    assert runner.validate_resume_step(1024)==0 and runner.validate_resume_step(1280)==256
    for step in (0,1025,1055,1281):
        rejected(lambda step=step:runner.validate_resume_step(step))
    trace=[{'relative_step':i,'step':1024+i,'horizon_s':(1.,2.,5.)[(i-1)%3]} for i in range(1,33)]
    runner.validate_trace(trace,32)
    rejected(lambda:runner.validate_trace(trace[:-1],32))
    trace[4]['step']=999
    rejected(lambda:runner.validate_trace(trace,32))
    rejected(lambda:runner.freeze_schedules(events,output/'schedule'))


def check_metrics():
    import numpy as np
    import torch
    from types import SimpleNamespace
    assert math.isclose(runner.weighted_risk([1.,4.]),2.8)
    # Last 10% mass: 5% at 10 plus 5% at 2 => tail mean 6.
    assert math.isclose(runner.weighted_risk([0.,2.,10.],[.8,.15,.05]),1.84)
    rejected(lambda:runner.weighted_risk([1.,float('nan')]))
    model=SimpleNamespace(mass=torch.ones(10,dtype=torch.float64),p_scale=torch.ones(10,dtype=torch.float64))
    event={'x':np.zeros((1211,30))}
    fine=np.zeros((3001,40)); coarse=np.zeros((1501,40)); fine[:,20:30]=.02; coarse[:,20:30]=.02
    result=runner.interval15(model,event,fine,coarse)
    assert result['n_time_points']==1200 and result['last_record_time_s']==12.1
    assert result['numerically_admissible'] and math.isclose(result['power_rmse_MW'],2.)
    fine[2001,30]=np.nan  # failing hidden substep after eight seconds is retained
    failed=runner.interval15(model,event,fine,coarse)
    assert not failed['finite'] and not failed['numerically_admissible']
    # Classification receives only the actual interval, retaining its fine substeps.
    from evaluate_structured_rom import _pair_row
    short_event={'id':'short','group':'synthetic','label':'unstable','x':np.zeros((31,30)),
                 'classification_meta':{'equilibrium_delta':np.zeros(10),'inertia':np.ones(10),
                                        'omega_s':2*math.pi*60,'event_time_s':.3}}
    f=np.zeros((3001,40)); c=np.zeros((1501,40)); history=np.zeros((11,30))
    f[41:]=np.nan; c[21:]=np.nan
    sf,sc,n=runner.actual_interval_predictions(short_event,f,c)
    assert n==20 and sf.shape==(41,40) and sc.shape==(21,40)
    classified=_pair_row(sf,sc,history,short_event,short_event['classification_meta'],np.ones(10))
    assert classified['finite_to_endpoint'] and classified['los_pair_pass']
    assert runner.interval15(model,short_event,f,c)['numerically_admissible']
    f[39,30]=np.nan
    sf,sc,n=runner.actual_interval_predictions(short_event,f,c)
    classified_bad=_pair_row(sf,sc,history,short_event,short_event['classification_meta'],np.ones(10))
    assert not classified_bad['finite_to_endpoint']
    assert not runner.interval15(model,short_event,f,c)['numerically_admissible']
    evidence={}
    for seed in runner.SEEDS:
        for structure,value in [('full8',2.),('delete7',1.)]:
            evidence[f'{seed}_{structure}']={'event_count':3,'rows':{'objective_point_error':[
                {'event_id':'stable','horizon':'5','label':'stable','available':True,'finite':True,
                 'numerically_admissible':True,'mean_point_error':value}]},'extras':[
                {'event_id':'stable','exception':None,'scientific_numeric_8s':True,
                 'actual_interval15':{'numerically_admissible':True},
                 'classification_15s':{'los_pair_pass':True,'finite_to_endpoint':True,
                                      'coi_normalized_rmse':0.,'common_normalized_rmse':0.}}]}
    summary=runner.paired_summary(evidence)
    assert summary['all_three_same_direction'] and not summary['pairs'][0]['whole_24_resolved']
    # Exercise the complete 24-attempt gate, rather than its denominator short circuit.
    complete=copy.deepcopy(evidence)
    for value in complete.values():
        base_row=value['rows']['objective_point_error'][0]
        base_extra=value['extras'][0]
        value['event_count']=24
        value['rows']['objective_point_error']=[{**copy.deepcopy(base_row),'event_id':f'val-{i:02d}'} for i in range(24)]
        value['extras']=[{**copy.deepcopy(base_extra),'event_id':f'val-{i:02d}'} for i in range(24)]
    full_summary=runner.paired_summary(complete)
    assert all(p['whole_24_resolved'] for p in full_summary['pairs'])
    assert full_summary['delete7_improves_all_three']
    assert not full_summary['delete7_worsens_all_three']
    assert all(len(p['paired_events'])==24 for p in full_summary['pairs'])
    for field in ('coi_normalized_rmse','common_normalized_rmse'):
        bad=copy.deepcopy(complete)
        bad['6101_delete7']['extras'][0]['classification_15s'][field]=.00101
        assert not runner.paired_summary(bad)['pairs'][0]['whole_24_resolved']
    evidence['6101_delete7']['extras'][0]['scientific_numeric_8s']=False
    summary=runner.paired_summary(evidence)
    assert summary['pairs'][0]['delete7_minus_full8'] is None
    assert summary['pairs'][0]['models']['delete7']['unresolved_ids']==['stable']


def check_freeze(output):
    rows=[]
    for seed in runner.SEEDS:
        for structure in ('full8','delete7'):
            folder=output/f'{seed}_{structure}'; folder.mkdir()
            (folder/'model.json').write_text(json.dumps({'synthetic':True,'seed':seed,'structure':structure}))
            digest=runner.sha(folder/'model.json')
            runner.write(folder/'checkpoint.json',{'step':1280,'model_sha256':digest,
                'identity':{'seed':seed,'structure':structure}})
            rows.append({'seed':seed,'structure':structure,'path':str(folder/'model.json'),'sha256':digest})
    conditions=[{'id':f'new-val-{i}','bus_id':i+1,'line':f'synthetic-line-{i}',
                 'location':'from','duration_s':.125,'xf':.001,'rf':0.,
                 'fault_type':'3ph'} for i in range(24)]
    manifest=output/'conditions.json'; runner.write(manifest,{'val_conditions':conditions})
    runner.freeze_val(rows,conditions,manifest,output/'freeze.json')
    rejected(lambda:runner.freeze_val(rows[:-1],conditions,manifest,output/'missing.json'))
    Path(rows[0]['path']).write_text('changed')
    rejected(lambda:runner.freeze_val(rows,conditions,manifest,output/'changed.json'))
    rejected(lambda:runner.load_events({'split':'test'}))
    rejected(lambda:runner.load_events({'split':'val','old_train':True}))
    rejected(lambda:runner.load_events({'split':'train','new_directories':['private_data/closure_discovery_20261017/D/new_data/val']}))


def check_real_adam_rng_resume(output):
    import numpy as np
    import torch
    from train_closure import make_optimizer
    from closure_discovery_benchmarks import toy_source
    from closure_training_runtime import TrainingClosureROM,TensorEventStore
    from linear_cache_runtime_20261019 import enable_linear_cache
    from linear_cache_cost_gate_20261019 import one_update,restore_cached
    from fixed_edge_pair_20261019 import save_safe
    if not torch.cuda.is_available():
        raise RuntimeError('registered synthetic optimizer QA requires CUDA')
    device=torch.device('cuda',torch.cuda.current_device())
    from symbolic_closure_rom import _BASE_TERMS
    source=output/'synthetic_model.json'
    toy_source().with_terms([_BASE_TERMS[i] for i in (0,1,2,3,4,5,7,8)]).export(source)
    model=TrainingClosureROM.from_export(source).to(device); enable_linear_cache(model)
    opt=make_optimizer(model)
    for g in opt.param_groups:
        g['lr']=runner.RATES[g['block']]
    events=events_fixture(3); store=TensorEventStore(events,device)
    samples=[(e,10) for e in events]
    # Actual small update populates all Adam moments/counters before save.
    one_update(model,opt,store,samples,1.,.01)
    rng=np.random.default_rng(7101); coverage={'cover_order':[0,1,2],'cursor':0}
    identity={'seed':6101,'structure':'full8','fixture':'three tiny synthetic updates only'}
    checkpoints=output/'checkpoints'; checkpoints.mkdir()
    save_safe(checkpoints,model,opt,rng,coverage,1056,identity,[])
    before_cpu=torch.get_rng_state().clone(); before_cuda=torch.cuda.get_rng_state(device).clone()
    checkpoint_optimizer=copy.deepcopy(opt.state_dict())
    ref=one_update(model,opt,store,samples,2.,.01)
    expected={k:v.detach().clone() for k,v in model.state_dict().items()}
    final_optimizer=copy.deepcopy(opt.state_dict())
    rm,ro,rr,rc,meta,changed=restore_cached(checkpoints,device,identity)
    assert meta['step']==1056 and not changed
    assert torch.equal(before_cpu,torch.get_rng_state()) and torch.equal(before_cuda,torch.cuda.get_rng_state(device))
    assert rr.bit_generator.state==rng.bit_generator.state and rc==coverage
    for a,b in zip(checkpoint_optimizer['state'].values(),ro.state_dict()['state'].values()):
        for key,value in a.items():
            if torch.is_tensor(value):
                assert torch.equal(value.cpu(),b[key].cpu())
            else:
                assert value==b[key]
    resumed=one_update(rm,ro,store,samples,2.,.01)
    assert math.isclose(ref['loss'],resumed['loss'],rel_tol=2e-9,abs_tol=2e-9)
    for key,value in expected.items():
        torch.testing.assert_close(value,rm.state_dict()[key],atol=2e-9,rtol=2e-9)
    for a,b in zip(final_optimizer['state'].values(),ro.state_dict()['state'].values()):
        for key,value in a.items():
            if torch.is_tensor(value):
                torch.testing.assert_close(value.cpu(),b[key].cpu(),atol=2e-9,rtol=2e-9)
    rejected(lambda:restore_cached(checkpoints,device,{**identity,'fixture':'changed'}))
    return {'actual_updates':3,'safe_step_label':1056,
        'scope':'synthetic moments/RNG replay, not full 256-step or full-scale recovery proof'}


def run_all(output):
    from linear_cache_cost_gate_20261019 import load_runtime
    load_runtime()
    output=Path(output); output.mkdir(parents=True,exist_ok=False)
    runner.worker_identity(output); started=time.perf_counter(); cases=[]
    for name,fn in [('schedule',lambda:check_schedule(output)),('metrics',check_metrics),
                    ('freeze',lambda:check_freeze(output)),
                    ('real_adam_rng_resume',lambda:check_real_adam_rng_resume(output))]:
        result=fn(); cases.append({'name':name,'status':'PASS','details':result})
    final={'status':'PASS','cases':cases,'elapsed_seconds':time.perf_counter()-started,
           'historical_data_read':False,'scientific_training':False}
    runner.write(output/'final.json',final); return final


if __name__=='__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('--output',required=True,type=Path)
    print(json.dumps(run_all(parser.parse_args().output)))
