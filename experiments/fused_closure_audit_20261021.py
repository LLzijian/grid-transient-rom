"""Frozen synthetic-only N/D fused complete-update audit. No scientific data access."""
from __future__ import annotations
import argparse,copy,hashlib,json,math,os,statistics,sys,time,traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'private_data/closure_mechanism_20261019/P0/fused_runtime01'
FIXTURE=BASE/'prepare01'
FORCE=.01

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()
def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))
def write(path,value):
    with Path(path).open('x',encoding='utf-8') as f:json.dump(value,f,indent=2,allow_nan=False);f.write('\n')
def verify_binding(path):
    path=Path(path).resolve()
    if not path.is_relative_to(BASE):raise ValueError('current audit binding required')
    mapping=read(path)['files']
    if not mapping:raise ValueError('empty binding')
    for name,h in mapping.items():
        p=(ROOT/name).resolve()
        if not p.is_relative_to(ROOT) or p.suffix.lower() in ('.h5','.hdf5') or name.startswith('configs/') or ('private_data/' in name and not p.is_relative_to(BASE)):raise ValueError('no physical/private scientific assets allowed')
        if sha(p)!=h:raise ValueError('bound source/fixture changed:'+name)
    return sha(path)

def restore_rng(state,device):
    import torch,numpy as np
    torch.set_rng_state(state['cpu'])
    if str(device).startswith('cuda'):torch.cuda.set_rng_state(state['cuda'],device)
    np.random.set_state(state['numpy'])

def fixture_path():
    path=Path(os.environ.get('FUSED_CLOSURE_FIXTURE',str(FIXTURE))).resolve()
    if path!=FIXTURE.resolve():raise ValueError('exact frozen synthetic fixture required')
    return path

def verify_fixture():
    path=fixture_path();manifest=read(path/'manifest.json')
    if manifest.get('scope')!='SYNTHETIC_ONLY_FIXED_N_D' or manifest.get('seed')!=84101:raise ValueError('wrong fixture scope')
    if set(manifest['files'])!={'model_N.json','model_D.json','synthetic.npz','rng.pt'}:raise ValueError('exact four fixture assets required')
    for name,h in manifest['files'].items():
        if sha(path/name)!=h:raise ValueError('frozen fixture asset changed:'+name)
    return manifest

def load_fixture(device='cpu'):
    import torch,numpy as np
    from linear_cache_runtime_20261019 import enable_linear_cache
    from closure_training_runtime import TrainingClosureROM
    from cubic_memory_port_runtime_20261021 import DRuntime
    path=fixture_path();manifest=verify_fixture()
    models={'N':enable_linear_cache(TrainingClosureROM.from_export(path/'model_N.json')).to(device).double(),'D':DRuntime.from_export(path/'model_D.json').to(device).double()}
    with np.load(path/'synthetic.npz',allow_pickle=False) as arrays:
        if arrays['x'].shape!=(12,531,30) or arrays['torque'].shape!=(12,531,10):raise ValueError('fixed twelve531 arrays required')
        events=[{'id':str(arrays['event_ids'][i]),'group':str(arrays['groups'][i]),'label':'synthetic','x':arrays['x'][i].copy(),'torque':arrays['torque'][i].copy()} for i in range(12)]
    if data_identity(events,manifest['specification'])!=manifest['data_sha256']:raise ValueError('synthetic data identity changed')
    state=torch.load(path/'rng.pt',map_location='cpu',weights_only=False)
    return models,events,state,manifest

def data_identity(events,spec):
    import numpy as np
    h=hashlib.sha256()
    for event in events:
        h.update(event['id'].encode());h.update(np.ascontiguousarray(event['x'],dtype=np.float64).tobytes());h.update(np.ascontiguousarray(event['torque'],dtype=np.float64).tobytes())
    h.update(json.dumps(spec,sort_keys=True,separators=(',',':')).encode())
    return h.hexdigest()

def prepare(args,check):
    import torch,numpy as np
    from test_quartic_memory_runtime_20261020 import source
    from linear_cache_runtime_20261019 import enable_linear_cache
    from cubic_memory_port_runtime_20261021 import build_D,coefficient_indices
    if args.output!=FIXTURE.resolve():raise ValueError('single fixed prepare output')
    if not torch.cuda.is_available():raise RuntimeError('CUDA RNG required for frozen fixture')
    check();models={'N':enable_linear_cache(source('cpu')),'D':build_D(source('cpu'))}
    with torch.no_grad():
        indices=coefficient_indices(models['D']);models['D'].coefficients[indices['a']]=.07;models['D'].coefficients[indices['b']]=-.03
    for name,model in models.items():model.export(args.output/f'model_{name}.json')
    events,digest,spec=_synthetic_events()
    np.savez_compressed(args.output/'synthetic.npz',x=np.stack([e['x'] for e in events]),torque=np.stack([e['torque'] for e in events]),event_ids=np.asarray([e['id'] for e in events]),groups=np.asarray([e['group'] for e in events]))
    torch.manual_seed(84102);torch.cuda.manual_seed_all(84102);np.random.seed(84102)
    state={'cpu':torch.get_rng_state().clone(),'cuda':torch.cuda.get_rng_state().clone(),'numpy':copy.deepcopy(np.random.get_state())}
    torch.save(state,args.output/'rng.pt');check()
    manifest={'scope':'SYNTHETIC_ONLY_FIXED_N_D','seed':84101,'rng_seed':84102,'data_sha256':digest,'specification':spec,'files':{n:sha(args.output/n) for n in ('model_N.json','model_D.json','synthetic.npz','rng.pt')},'model_structure':{k:[{'name':n,'shape':list(p.shape),'requires_grad':p.requires_grad} for n,p in m.named_parameters()] for k,m in models.items()},'source_binding_sha256':sha(args.binding),'source_files':read(args.binding)['files'],'no_private_data':True}
    write(args.output/'manifest.json',manifest);return manifest

def audit_ratios(records):
    expected={(s,h,block,path,rep) for s in ('N','D') for h in (1,2,5) for block in ('AB','BA') for path in ('old','new') for rep in (0,1,2)}
    actual={(r['structure'],r['horizon'],r['block'],r['path'],r['rep']) for r in records}
    if len(records)!=72 or actual!=expected:raise ValueError('exact72 complete registered updates required')
    ratios={};totals={};medians={}
    for s in ('N','D'):
        medians[s]={}
        for h in (1,2,5):
            values={p:[r['seconds'] for r in records if (r['structure'],r['horizon'],r['path'])==(s,h,p) and r['rep']>0] for p in ('old','new')}
            if any(len(v)!=4 or any(not math.isfinite(x) or x<=0 for x in v) for v in values.values()):raise ValueError('four finite repeats per path required')
            medians[s][str(h)]={p:statistics.median(v) for p,v in values.items()};ratios[f'{s}_{h}']=medians[s][str(h)]['new']/medians[s][str(h)]['old']
        totals[s]=sum(v['new'] for v in medians[s].values())/sum(v['old'] for v in medians[s].values())
    if any(not math.isfinite(r['seconds']) or r['seconds']<=0 for r in records):raise ValueError('invalid first/repeated update')
    return {'horizon_ratios':ratios,'sum_median_ratios':totals,'medians':medians,'speed_gate_pass':all(v<=1.05 for v in ratios.values()) and all(v<=.90 for v in totals.values()),'total_updates':72,'scope':'fixed frozen synthetic N/D complete updates only'}

def benchmark(args,check):
    import torch
    from closure_training_runtime import TensorEventStore,synchronize
    from train_closure import make_optimizer
    from cubic_memory_port_runtime_20261021 import one_update as update_D
    from linear_cache_cost_gate_20261019 import one_update as update_N,load_runtime as load_old_runtime
    from fused_closure_runtime_20261021 import enable_fused_runtime,one_update as update_new
    load_old_runtime() # Initialize the unchanged old producer's module globals before any timing.
    models,events,rng,manifest=load_fixture('cpu')
    if not torch.cuda.is_available():raise RuntimeError('real CUDA benchmark required')
    torch.set_num_threads(1);device=torch.device('cuda');store=TensorEventStore(events,device);samples=[(e,10) for e in events];records=[];construct=[];save_costs=[]
    for structure in ('N','D'):
        for horizon in (1,2,5):
            for block,order in (('AB',('old','new')),('BA',('new','old'))):
                for path in order:
                    check();t=time.perf_counter();model=copy.deepcopy(models[structure]).to(device).double()
                    if path=='new':enable_fused_runtime(model)
                    opt=make_optimizer(model)
                    if opt.state:raise ValueError('each path starts empty Adam')
                    restore_rng(rng,device);synchronize(device);construct.append({'structure':structure,'horizon':horizon,'block':block,'path':path,'seconds':time.perf_counter()-t})
                    update=update_new if path=='new' else (update_D if structure=='D' else update_N)
                    torch.cuda.reset_peak_memory_stats()
                    for rep in range(3):
                        check();synchronize(device);t=time.perf_counter();row=update(model,opt,store,samples,float(horizon),FORCE);synchronize(device)
                        record={'structure':structure,'horizon':horizon,'block':block,'path':path,'rep':rep,'seconds':time.perf_counter()-t,'peak_cuda_bytes':torch.cuda.max_memory_allocated(),'update':row}
                        records.append(record)
                        (args.output/'partial.json').write_text(json.dumps({'records':records,'construction':construct},indent=2,allow_nan=False))
                    check();synchronize(device);t=time.perf_counter()
                    torch.save({'model':model.state_dict(),'optimizer':opt.state_dict()},args.output/f'trial_{structure}_{horizon}_{block}_{path}.pt')
                    synchronize(device);save_costs.append(time.perf_counter()-t)
                    model.end_graph_cache();del model,opt
    result={'records':records,'construction':construct,'save_seconds':save_costs,'fixture_manifest_sha256':sha(fixture_path()/'manifest.json'),'data_sha256':manifest['data_sha256'],'comparison':audit_ratios(records),'no_scientific_model':True}
    write(args.output/'benchmark.json',result);return result

def main(argv=None):
    p=argparse.ArgumentParser();p.add_argument('mode',choices=('prepare','benchmark'));p.add_argument('--output',type=Path,required=True);p.add_argument('--binding',type=Path,required=True);p.add_argument('--max-seconds',type=float,required=True);p.add_argument('--source-commit',required=True);a=p.parse_args(argv)
    a.output=a.output.resolve();a.binding=a.binding.resolve()
    expected=BASE/('prepare01' if a.mode=='prepare' else 'benchmark01')
    if a.output!=expected.resolve() or a.max_seconds!=({'prepare':40.,'benchmark':1740.}[a.mode]) or len(a.source_commit)!=40 or any(c not in '0123456789abcdef' for c in a.source_commit):raise ValueError('exact fixed current-stage invocation required')
    a.output.mkdir(parents=True,exist_ok=False)
    sys.path[:0]=[str(ROOT/'tests'),str(ROOT/'experiments'),str(ROOT/'src')]
    from closure_p3_disagreement_20261020 import own_creation_filetime_utc
    write(a.output/'worker.json',{'pid':os.getpid(),'creation_filetime_utc':own_creation_filetime_utc(),'executable':str(Path(sys.executable).resolve()),'argv':sys.argv})
    started=time.perf_counter()
    def check():
        if time.perf_counter()-started>=a.max_seconds:raise TimeoutError('registered audit self deadline')
    try:
        verify_binding(a.binding)
        result=prepare(a,check) if a.mode=='prepare' else benchmark(a,check)
        verify_binding(a.binding);write(a.output/'complete.json',{'status':'COMPLETED','result':result,'seconds':time.perf_counter()-started});return 0
    except BaseException as exc:
        write(a.output/'failure.json',{'error':repr(exc),'traceback':traceback.format_exc()});raise


def _synthetic_events():
    import numpy as np
    rng = np.random.default_rng(84101)
    events = []
    for index in range(12):
        x = np.concatenate((rng.normal(size=(531, 10)) * .02,
                            rng.normal(size=(531, 10)) * .002,
                            rng.normal(size=(531, 10)) * .02), axis=1)
        torque = rng.normal(size=(531, 10)) * .01
        events.append({"id": f"linear-cache-synthetic-{index:02d}",
                       "group": f"synthetic-{index % 3}", "label": "synthetic",
                       "x": x, "torque": torque})
    data_hash = hashlib.sha256()
    for event in events:
        data_hash.update(event["id"].encode())
        data_hash.update(np.ascontiguousarray(event["x"], dtype=np.float64).tobytes())
        data_hash.update(np.ascontiguousarray(event["torque"], dtype=np.float64).tobytes())
    specification = {"seed": 84101, "events": 12, "points_each": 531, "dt": .01,
        "x_scales": {"delta": .02, "omega": .002, "p": .02}, "torque_scale": .01,
        "distribution": "independent standard normal multiplied by fixed component scales",
        "generator": "numpy.default_rng", "endpoint": 10, "force_scale": .01,
        "generation_order_each_event": ["delta10", "omega10", "p10", "torque10"],
        "array_dtype": "float64", "flatten_order": "C"}
    data_hash.update(json.dumps(specification, sort_keys=True, separators=(",", ":")).encode())
    return events, data_hash.hexdigest(), specification

if __name__=="__main__":raise SystemExit(main())
