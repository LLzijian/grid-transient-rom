"""One fixed-directory adapter for the registered first physical execution."""
from __future__ import annotations
import argparse
from contextlib import contextmanager
import hashlib
import json
import math
from pathlib import Path
import shutil
import sys
import time
import traceback

ROOT=Path(__file__).resolve().parents[1]
OLD_BASE=ROOT/'private_data/closure_mechanism_20261019/P3/active_fom_pair01'
BASE=ROOT/'private_data/closure_mechanism_20261019/P3/active_fom_pair_first_physics01'
CAMPAIGN=BASE/'campaign01'
P2=ROOT/'private_data/closure_mechanism_20261019/P2/fixed_edge_pair01/campaign01'
ENV_QA=ROOT/'private_data/closure_mechanism_20261019/repair/fom_environment01/qa01/final.json'
DESIGN_SHA='c475a3bf11aaa17e2a3ca4642291a43d863541be14ec0fefdec5cd072de8c126'
VAL_SHA='9b8a44b7bb979d706ba8ff6cefababb8db3cf97027e99cca7ec3a5275013a224'
KEYS=tuple(f'{s}_{k}' for s in (6101,6102,6103) for k in ('full8','delete7'))

def read(p): return json.loads(Path(p).read_text(encoding='utf-8'))
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''): h.update(block)
    return h.hexdigest()
def write(p,obj):
    p=Path(p); p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('x',encoding='utf-8') as stream: json.dump(obj,stream,indent=2,allow_nan=False)

def inside(path,base=None):
    base=BASE if base is None else base
    p=Path(path).resolve()
    if p==base.resolve() or not p.is_relative_to(base.resolve()): raise ValueError('path outside fixed new stage: '+str(p))
    return p

def expected_environment(role):
    q=read(ENV_QA)
    if q.get('status')!='PASS': raise ValueError('actual accepted environment QA required')
    probe=q['probes'][role]
    if probe.get('exit_code')!=0 or probe.get('supervision_valid') is not True:
        raise ValueError('environment probe lacks qualified native receipt')
    return probe['result']['identity']

def early_worker(output):
    from experiments.fom_execution_environment_20261020 import _worker
    output=inside(output); output.mkdir(parents=True,exist_ok=False)
    write(output/'wrapper_worker.json',_worker())

def verify_source_files(source):
    if not isinstance(source,dict) or not source: raise ValueError('explicit nonempty scientific source identity required')
    for name,expected in source.items():
        p=(ROOT/name).resolve()
        if not p.is_relative_to(ROOT) or p.suffix.lower() in ('.h5','.hdf5','.pt','.npz'):
            raise ValueError('only repository source files accepted here')
        if sha(p)!=expected: raise ValueError('registered scientific source changed: '+name)

def verify_startup_history(design,record):
    attempts=record.get('attempts',[])
    if len(attempts)!=48 or record.get('qualified_count')!=0: raise ValueError('old failure denominator differs')
    if [r['condition'] for r in attempts]!=design['candidates']: raise ValueError('old failed condition order differs')
    for row in attempts:
        if set(row['steps'])!={'0.005','0.0025'}: raise ValueError('old failure pair incomplete')
        for step in row['steps'].values():
            if (step.get('status')!='failed' or step.get('error')!="ModuleNotFoundError: No module named 'andes'"
                    or step.get('raw_file') or step.get('diagnostics') not in ({},None)):
                raise ValueError('old attempt may contain physical information')
    return 96

def prepare_binding():
    old=OLD_BASE/'campaign01'
    if sha(old/'design.json')!=DESIGN_SHA or sha(old/'val_conditions.json')!=VAL_SHA:
        raise ValueError('old byte-locked design changed')
    design=read(old/'design.json'); history=read(old/'short/final.json')
    verify_startup_history(design,history)
    # Explicit old candidate/step raw directories only; never inspect old val data.
    for row in history['attempts']:
        for step in row['steps'].values():
            folder=Path(step['folder']).resolve()
            if not folder.is_relative_to((old/'short').resolve()): raise ValueError('old short path escape')
            if (folder/'raw').exists() and any((folder/'raw').iterdir()): raise ValueError('old attempt has raw physical output')
    legacy=read(old/'source_manifest.json'); verify_source_files(legacy['source'])
    # Reuse the exact registered metadata identities; do not deserialize model
    # weights or read train trajectories in this metadata-only preparation.
    if tuple(sorted(legacy['models']))!=tuple(sorted(KEYS)): raise ValueError('six P2 sources required')
    CAMPAIGN.mkdir(parents=True,exist_ok=False)
    for name in ('design.json','val_conditions.json'): shutil.copyfile(old/name,CAMPAIGN/name)
    legacy['design_artifacts']={str(CAMPAIGN/n):sha(CAMPAIGN/n) for n in ('design.json','val_conditions.json')}
    write(CAMPAIGN/'source_manifest.json',legacy)
    provenance={'old_campaign':str(old),'old_software_attempts':96,'unique_physical_candidates':48,
       'old_physical_solves':0,'old_short_final_sha256':sha(old/'short/final.json'),
       'old_source_manifest_sha256':sha(old/'source_manifest.json'),'design_sha256':DESIGN_SHA,
       'val_conditions_sha256':VAL_SHA,'environment_qa_sha256':sha(ENV_QA),
       'scope':'same physical conditions re-executed; first physical solution; historical failures unchanged'}
    write(CAMPAIGN/'provenance.json',provenance)
    return provenance

class BatchEnvironmentStop(BaseException): pass

def environment_failure(error):
    text=str(error)
    # Only known package/runtime/config identity failures. Numerical solver
    # failures and physical outcomes are deliberately outside this list.
    return text.startswith(('ModuleNotFoundError:','ImportError:')) or any(
        text.startswith(prefix) for prefix in ('ValueError: ANDES version differs from frozen config',
        'ValueError: runtime ANDES version differs from historical TRAIN',
        'ValueError: runtime ANDES case selection differs from historical TRAIN',
        'ValueError: runtime ANDES case-file SHA256 differs from historical TRAIN',
        'ValueError: run configuration SHA256 does not match the effective config',
        'ValueError: run attestation is bound to a different historical fingerprint'))

@contextmanager
def batch_guard(stage,guard,record):
    from experiments import closure_short_history_20261019 as short
    from experiments import closure_discovery_data as data
    original_short=short.run_short_history; original_full=data._run_bounded_simulation
    def wrap(fn):
        def call(*args,**kwargs):
            row={'condition':dict(args[0]),'step':kwargs.get('step',args[4] if len(args)>4 else None),'status':'started'}
            record.append(row)
            try: guard()
            except Exception as exc:
                row.update(status='environment_failure',error=repr(exc)); raise BatchEnvironmentStop(str(exc)) from exc
            try: result=fn(*args,**kwargs)
            except Exception as exc:
                if isinstance(exc,(ImportError,ModuleNotFoundError)) or environment_failure(exc):
                    row.update(status='environment_failure',error=repr(exc)); raise BatchEnvironmentStop(str(exc)) from exc
                row.update(status='condition_failure',error=repr(exc)); raise
            if isinstance(result,dict) and environment_failure(result.get('error','')):
                row.update(status='environment_failure',error=result['error']); raise BatchEnvironmentStop(result['error'])
            row['status']='returned'; return result
        return call
    short.run_short_history=wrap(original_short); data._run_bounded_simulation=wrap(original_full)
    try: yield
    finally: short.run_short_history=original_short; data._run_bounded_simulation=original_full

@contextmanager
def argv_for(argv):
    previous=sys.argv
    sys.argv=[str(Path(__file__)),*argv]
    try: yield
    finally: sys.argv=previous

def acquire(args,worker_dir):
    from experiments import active_fom_pair_20261020 as old
    from experiments.fom_execution_environment_20261020 import preflight
    if Path(args.campaign).resolve()!=CAMPAIGN.resolve(): raise ValueError('fixed campaign required')
    if Path(args.source_manifest).resolve()!=(CAMPAIGN/'source_manifest.json').resolve(): raise ValueError('fixed scientific manifest required')
    inside(args.output)
    role='gpu' if args.stage=='score' else 'fom'; expected=expected_environment(role)
    def guard():
        preflight(role,expected); verify_source_files(read(CAMPAIGN/'source_manifest.json')['source'])
    guard(); write(worker_dir/'preflight.json',{'role':role,'expected':expected,'environment_qa_sha256':sha(ENV_QA)})
    call=['--stage',args.stage,'--campaign',str(CAMPAIGN),'--output',str(args.output),
          '--source-manifest',str(CAMPAIGN/'source_manifest.json'),'--source-commit',args.source_commit,
          '--max-seconds',str(args.max_seconds)]
    if args.stage=='val':
        if args.freeze_manifest is None: raise ValueError('val freeze required')
        inside(args.freeze_manifest); old.verify_val_freeze(CAMPAIGN,args.freeze_manifest)
        call+=['--freeze-manifest',str(args.freeze_manifest)]
    attempts=[]
    try:
        with batch_guard(args.stage,guard,attempts),argv_for(call): return old.main()
    except BatchEnvironmentStop as exc:
        design=read(CAMPAIGN/'design.json')
        conditions=(design['candidates'] if args.stage=='short' else design['val_conditions'] if args.stage=='val'
                    else read(CAMPAIGN/'score/final.json')['selected_conditions'])
        attempted={r['condition']['id'] for r in attempts}
        write(worker_dir/'batch_stop.json',{'error':str(exc),'attempts':attempts,'denominator':len(conditions),
            'not_attempted':[{'condition':c,'status':'not_attempted_batch_environment_stop'} for c in conditions if c['id'] not in attempted]})
        raise

def check_model(path,*,allow_source=False):
    p=Path(path).resolve()
    sources={ (P2/k/'checkpoints/step001024/model.json').resolve():k for k in KEYS}
    if allow_source and p in sources: return 'source'
    inside(p); meta=read(p.parent/'checkpoint.json')
    if p.name!='model.json' or meta.get('step')!=1280 or meta.get('model_sha256')!=sha(p): raise ValueError('only complete current global1280 model allowed')
    key=f"{meta['identity']['seed']}_{meta['identity']['structure']}"
    if key not in KEYS or p!=(BASE/'models'/key/'checkpoints/step001280/model.json').resolve(): raise ValueError('unregistered model endpoint path')
    if sha(p.parent/'training.pt')!=meta['training_sha256']: raise ValueError('final optimizer/RNG bytes changed')
    return 'adapted'

def validate_inventory(path):
    if Path(path).resolve()!=(CAMPAIGN/'score/final.json').resolve(): raise ValueError('only actual pre-FOM frozen inventory allowed')
    inventory=read(path)
    if inventory.get('status')!='COMPLETED' or len(inventory.get('frozen_predictions',[]))!=144: raise ValueError('incomplete original prediction freeze')
    for row in inventory['frozen_predictions']:
        if row['model'] not in KEYS: raise ValueError('unknown frozen prediction source')
        if row.get('path'):
            p=Path(row['path']).resolve()
            if not p.is_relative_to((CAMPAIGN/'frozen_predictions').resolve()) or sha(p)!=row['sha256']:
                raise ValueError('frozen prediction path/hash differs')
    return inventory

def validate_events_spec(path):
    path=inside(path); spec=read(path)
    if type(spec.get('old_train',False)) is not bool or not isinstance(spec.get('new_directories',[]),list):
        raise ValueError('explicit boolean old_train and directory list required')
    if spec.get('split') not in ('train','val') or (spec['split']=='val' and spec.get('old_train')): raise ValueError('invalid events split')
    expected=(CAMPAIGN/spec['split']).resolve()
    for p in spec.get('new_directories',[]):
        if Path(p).resolve()!=expected: raise ValueError('only exact current split directory allowed')
    if spec['split']=='val':
        from experiments.active_fom_pair_20261020 import verify_val_freeze
        freeze=inside(spec['freeze_path']); verify_val_freeze(CAMPAIGN,freeze)
    if spec.get('frozen_source_inventory'): validate_inventory(spec['frozen_source_inventory'])
    return spec

def validate_fit(args):
    inside(args.output)
    for name in ('source_identity','stage_identity','schedule_path','frozen_predictions','manifest'):
        value=getattr(args,name,None)
        if value is not None: inside(value)
    spec=validate_events_spec(args.events_spec) if args.events_spec else None
    if args.mode in ('schedule','adapt','evaluate') and spec is None: raise ValueError('explicit events spec required')
    if args.mode in ('schedule','adapt') and spec['split']!='train': raise ValueError('training schedule/data must be train')
    if args.mode=='adapt':
        k=f'{args.seed}_{args.structure}'
        if k not in KEYS or Path(args.source_output).resolve()!=(P2/k/'checkpoints').resolve(): raise ValueError('only exact P2 registered checkpoint directory allowed')
        meta=read(P2/k/'checkpoints/step001024/checkpoint.json')
        if Path(args.output).resolve()!=(BASE/'models'/k).resolve(): raise ValueError('fixed new model output required')
        legacy=read(OLD_BASE/'campaign01/source_manifest.json')['models'][k]
        if (legacy['checkpoint.json']!=sha(P2/k/'checkpoints/step001024/checkpoint.json') or
                legacy['model.json']!=meta['model_sha256'] or legacy['training.pt']!=meta['training_sha256']):
            raise ValueError('P2 source differs from frozen historical identity')
        latest=read(P2/k/'checkpoints/latest.json')
        if latest.get('directory')!='step001024' or latest.get('checkpoint_sha256')!=sha(P2/k/'checkpoints/step001024/checkpoint.json'):
            raise ValueError('P2 latest must point to the registered1024 endpoint')
        for name,field in (('model.json','model_sha256'),('training.pt','training_sha256')):
            if sha(P2/k/'checkpoints/step001024'/name)!=meta[field]: raise ValueError('P2 source file bytes changed')
        if read(args.source_identity)!=meta['identity'] or meta['step']!=1024: raise ValueError('P2 source identity mismatch')
        if spec['split']!='train' or not spec.get('old_train') or spec.get('new_directories')!=[str(CAMPAIGN/'train')]:
            # Canonical path spelling may differ on Windows; resolve before comparison.
            if spec['split']!='train' or not spec.get('old_train') or len(spec.get('new_directories',[]))!=1: raise ValueError('adapt requires old366 plus current new train')
        stage=read(args.stage_identity)
        for n in ('model_sha256','training_sha256'):
            if stage.get('source_'+n)!=meta[n]: raise ValueError('stage must bind both source model/Adam hashes')
        if args.schedule_path is None: raise ValueError('explicit shared frozen schedule required')
    if args.mode=='evaluate':
        source=check_model(args.model,allow_source=True)=='source'
        if source and (spec.get('old_train') or not (args.frozen_predictions or spec.get('frozen_source_inventory'))): raise ValueError('source evaluation only from prefrozen new-train predictions')
        if source and spec['split']!='train': raise ValueError('source endpoints forbidden for new val')
        frozen=args.frozen_predictions or spec.get('frozen_source_inventory')
        if not source and frozen: raise ValueError('adapted model must use its own predictions')
        if frozen: validate_inventory(frozen)
        if source:
            expected=next(k for k in KEYS if Path(args.model).resolve()==(P2/k/'checkpoints/step001024/model.json').resolve())
            if (args.frozen_model_key or spec.get('modelkey'))!=expected: raise ValueError('frozen source model key mismatch')
    if args.mode=='freeze-val':
        from experiments.active_fom_pair_20261020 import key
        m=read(args.manifest)
        if len(m['models'])!=6 or {(r['seed'],r['structure']) for r in m['models']}!={(s,k) for s in (6101,6102,6103) for k in ('full8','delete7')}: raise ValueError('six endpoints required')
        for row in m['models']: check_model(row['path'])
        if Path(m['condition_manifest_path']).resolve()!=(CAMPAIGN/'val_conditions.json').resolve(): raise ValueError('fixed val condition file required')
        if [key(r) for r in m['conditions']]!=[key(r) for r in read(CAMPAIGN/'design.json')['val_conditions']]: raise ValueError('val conditions changed')
    if args.mode=='summary':
        for p in read(args.manifest)['evidence'].values(): inside(p)
    return spec

@contextmanager
def fit_campaign(fit):
    previous=fit.CAMPAIGN
    if Path(previous).resolve()!=OLD_BASE.resolve(): raise ValueError('old fit boundary unexpectedly changed')
    fit.CAMPAIGN=BASE.resolve()
    try: yield
    finally: fit.CAMPAIGN=previous

def fit_call(args,worker_dir,forward):
    from experiments import active_fom_pair_fit_20261020 as fit
    from experiments.fom_execution_environment_20261020 import preflight
    identity=preflight('gpu',expected_environment('gpu')); validate_fit(args)
    write(worker_dir/'preflight.json',{'role':'gpu','identity':identity})
    try:
        with fit_campaign(fit),argv_for([args.mode,*forward]):
            write(worker_dir/'fit_binding.json',{'original':str(OLD_BASE.resolve()),'effective':str(fit.CAMPAIGN)})
            return fit.main()
    finally:
        write(worker_dir/'fit_binding_exit.json',{'actual_restored':str(fit.CAMPAIGN),
              'restoration_pass':Path(fit.CAMPAIGN).resolve()==OLD_BASE.resolve()})

def parser():
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('command',choices=('prepare-binding','acquire','fit','qa'))
    p.add_argument('mode',nargs='?'); p.add_argument('--output',type=Path,required=True)
    p.add_argument('--wrapper-output',type=Path,required=True); p.add_argument('--max-seconds',type=float,required=True)
    p.add_argument('--source-commit',required=True); p.add_argument('--stage',choices=('short','score','train','val'))
    p.add_argument('--campaign',type=Path); p.add_argument('--source-manifest',type=Path); p.add_argument('--freeze-manifest',type=Path)
    for n in ('events-spec','source-output','source-identity','stage-identity','schedule-path','model','frozen-predictions','manifest'): p.add_argument('--'+n,type=Path)
    p.add_argument('--frozen-model-key'); p.add_argument('--seed',type=int); p.add_argument('--structure')
    p.add_argument('--recovery',action='store_true'); p.add_argument('--previous-seconds',type=float,default=0.)
    return p

def main(argv=None):
    argv=list(sys.argv[1:] if argv is None else argv); args=parser().parse_args(argv)
    if not math.isfinite(args.max_seconds) or args.max_seconds<=0: raise ValueError('finite cap required')
    if len(args.source_commit)!=40 or any(c not in '0123456789abcdef' for c in args.source_commit): raise ValueError('real archived source commit required')
    inside(args.output); early_worker(args.wrapper_output); started=time.perf_counter()
    if args.command in ('acquire','fit') and args.output.resolve()==args.wrapper_output.resolve():
        raise ValueError('early worker directory must be separate from exclusive scientific output')
    worker_dir=args.wrapper_output.resolve()
    try:
        if args.command=='prepare-binding':
            from experiments.fom_execution_environment_20261020 import preflight
            identity=preflight('gpu',expected_environment('gpu')); write(worker_dir/'preflight.json',identity)
            result=prepare_binding()
        elif args.command=='acquire': result={'old_exit':acquire(args,worker_dir)}
        elif args.command=='fit':
            forward=argv[2:]
            for name in ('--source-commit','--wrapper-output'):
                i=forward.index(name); del forward[i:i+2]
            result={'old_exit':fit_call(args,worker_dir,forward)}
        else:
            from experiments.fom_execution_environment_20261020 import preflight
            preflight('gpu',expected_environment('gpu'))
            sys.path.insert(0,str(ROOT/'tests'))
            from test_first_physics_pair_20261020 import run_qa
            result=run_qa()
            if result['status']!='PASS': raise ValueError('new wiring QA failed')
        write(worker_dir/'wrapper_complete.json',{'status':'COMPLETED','result':result,'seconds':time.perf_counter()-started})
        return 0
    except BaseException as exc:
        write(worker_dir/'wrapper_failure.json',{'error':repr(exc),'traceback':traceback.format_exc(),'seconds':time.perf_counter()-started})
        raise

if __name__=='__main__': raise SystemExit(main())
