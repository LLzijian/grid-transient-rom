"""Finite, source-preserving checkpoint migration and the original comparison."""
from __future__ import annotations
import copy
import hashlib
import json
import math
import shutil
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'experiments'),str(ROOT/'src')]
import rational_control_20261021 as study
from rational_execution_20261021 import atomic,new_or_equal

SAVE_STEPS=(0,32,128,512)


def read_json(path):return json.loads(Path(path).read_text(encoding='utf-8'))


def file_ref(path):
    path=Path(path).resolve()
    return {'path':str(path),'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'bytes':path.stat().st_size}


def checkpoint_api():
    from closure_causal_history_residual_20261019 import checkpoint,restore
    return checkpoint,restore


def _diagnostic(value,step,identity):
    if (set(value)!={'step','identity','loss','gradient','parameter_summary'} or value['step']!=step
            or value['identity']!=identity or len(value['loss'])!=step):raise ValueError('diagnostic fields/count/identity differ')
    if set(value['gradient'])!={'total_raw_norm','clip_factor'} or any(len(v)!=step for v in value['gradient'].values()):
        raise ValueError('gradient trajectory incomplete')
    if set(value['parameter_summary'])!={'first','last','max'}:raise ValueError('parameter gradient summary incomplete')
    if step and any(not value['parameter_summary'][key] for key in ('first','last','max')):
        raise ValueError('empty nonzero-step diagnostics')
    return value


def publish_checkpoint(folder,m,opt,rng,step,trace,gradient,summary,identity):
    folder=Path(folder);checkpoint,_=checkpoint_api();checkpoint(folder,m,opt,rng,step,trace,identity)
    sidecar=folder/f'diagnostics_{step:05d}.json'
    atomic(sidecar,{'step':step,'identity':identity,'loss':trace,'gradient':gradient,'parameter_summary':summary})
    directory=folder/f'checkpoint_{step:05d}'
    atomic(folder/'resume.json',{'step':step,'checkpoint':directory.name,'identity':identity,
        'files':{n:file_ref(directory/n)['sha256'] for n in ('model.json','training.pt')},'diagnostics':study.fileid(sidecar)})


def complete_sidecar(folder,identity):
    folder=Path(folder).resolve();pointer=read_json(folder/'resume.json');latest=read_json(folder/'latest.json');step=pointer.get('step')
    if type(step) is not int or step not in SAVE_STEPS or pointer.get('checkpoint')!=f'checkpoint_{step:05d}' or pointer.get('identity')!=identity:
        raise ValueError('invalid complete resume identity')
    if list(folder.glob('pending_*')):raise ValueError('pending checkpoint retained; no silent fallback')
    if latest.get('checkpoint')!=pointer['checkpoint'] or latest.get('updates')!=step or latest.get('sha256')!=pointer.get('files'):
        raise ValueError('latest checkpoint is ahead of complete diagnostics or differs')
    directory=folder/pointer['checkpoint']
    if set(pointer['files'])!={'model.json','training.pt'}:raise ValueError('incomplete model/state pair')
    if any(file_ref(directory/n)['sha256']!=digest for n,digest in pointer['files'].items()):raise ValueError('checkpoint changed')
    path=Path(pointer['diagnostics']['path']).resolve()
    if path!=folder/f'diagnostics_{step:05d}.json' or file_ref(path)['sha256']!=pointer['diagnostics']['sha256']:
        raise ValueError('diagnostic identity changed')
    return step,_diagnostic(read_json(path),step,identity)


def restore_complete(folder,m,opt,rng,identity):
    step,d=complete_sidecar(folder,identity);_,restore=checkpoint_api()
    actual,trace=restore(Path(folder),m,opt,rng,identity)
    if actual!=step or trace!=d['loss']:raise ValueError('model/RNG checkpoint and diagnostics disagree')
    return actual,trace,copy.deepcopy(d['gradient']),copy.deepcopy(d['parameter_summary'])


def _manifest(row):
    folder=Path(row['folder'])
    if not folder.is_absolute() or not folder.is_dir() or folder.is_symlink():raise ValueError('absolute original folder required')
    actual_files={};actual_dirs=set()
    for path in folder.rglob('*'):
        if path.is_symlink():raise ValueError('linked original content forbidden')
        if path.is_dir():actual_dirs.add(path.resolve())
        elif path.is_file():actual_files[path.resolve()]=file_ref(path)
        else:raise ValueError('unknown original filesystem entry')
    expected={}
    for ref in row.get('files',[]):
        path=Path(ref['path'])
        if not path.is_absolute() or path.resolve() in expected:raise ValueError('duplicate/nonabsolute original file')
        expected[path.resolve()]=ref
    directories={(folder/path).resolve() for path in row.get('directories',[])}
    if len(directories)!=len(row.get('directories',[])) or actual_dirs!=directories or actual_files!=expected:
        raise ValueError('original manifest omits/changes files or directories')
    return folder.resolve(),actual_files,actual_dirs


def validate_original(row):
    folder,files,directories=_manifest(row)
    allowed_dirs={folder/'checkpoint_00000',folder/'checkpoint_00032'}
    allowed_files={folder/name for name in ('checkpoint_00000.diagnostics.json','checkpoint_00032.diagnostics.json',
        'latest.json','resume.json','train.npz','failed_trace.json')}
    allowed_files.update(directory/name for directory in allowed_dirs for name in ('model.json','training.pt'))
    if directories!=allowed_dirs or set(files)!=allowed_files:
        raise ValueError('only original0/32 checkpoints and registered files allowed; unknown/pending/future evidence retained')
    identity=row['identity'];seed=row['seed'];commit=row['source_commit']
    if (type(seed) is not int or seed not in study.SEEDS or row.get('key')!=f'0_rational_{seed}'
            or len(commit)!=40 or any(c not in '0123456789abcdef' for c in commit)):
        raise ValueError('fixed original key/seed/source required')
    tr=study.load_data(folder/'train.npz')
    data_identity={'clean_sha256':study.array_sha(tr['coarse']),'fine_sha256':study.array_sha(tr['fine']),
        'observed_sha256':study.array_sha(tr['observed']),'scales_sha256':study.array_sha(tr['scales']),
        'seed':seed,'noise':0,'split':0}
    expected={'source_commit':commit,'key':row['key'],'data_identity':data_identity,
        'train_raw':file_ref(folder/'train.npz'),'form':'rational'}
    if identity!=expected:raise ValueError('original data/source identity differs')
    if not study.qualify(tr['coarse'],tr['fine'],tr['scales'])['valid']:raise ValueError('original teacher not numerically qualified')
    diagnostics={step:_diagnostic(read_json(folder/f'checkpoint_{step:05d}.diagnostics.json'),step,identity) for step in (0,32)}
    latest=read_json(folder/'latest.json');pointer=read_json(folder/'resume.json')
    hashes={name:files[folder/'checkpoint_00032'/name]['sha256'] for name in ('model.json','training.pt')}
    if latest!={'checkpoint':'checkpoint_00032','updates':32,'sha256':hashes}:raise ValueError('original latest count/hash differs')
    if pointer!={'step':32,'checkpoint':'checkpoint_00032','identity':identity,'files':hashes,
        'diagnostics':study.fileid(folder/'checkpoint_00032.diagnostics.json')}:raise ValueError('original complete pointer differs')
    failed=read_json(folder/'failed_trace.json')
    if failed!={key:diagnostics[32][key] for key in ('loss','gradient','parameter_summary')}:
        raise ValueError('original failure trajectory differs from committed32')
    return {'tr':tr,'identity':identity,'folder':folder,'row':copy.deepcopy(row),'diagnostics':diagnostics}


def prepare_copy(row,target):
    target=Path(target).resolve()
    if target.exists():raise ValueError('recovery copy already exists')
    original=validate_original(row);folder=original['folder'];target.mkdir(parents=True)
    copied=[]
    for ref in row['files']:
        source=Path(ref['path']);relative=source.relative_to(folder)
        if relative.name in ('checkpoint_00000.diagnostics.json','checkpoint_00032.diagnostics.json'):
            relative=relative.with_name('diagnostics_'+relative.name[11:16]+'.json')
        destination=target/relative;destination.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(source,destination)
        copied_ref=file_ref(destination)
        if (destination.read_bytes()!=source.read_bytes() or copied_ref['sha256']!=ref['sha256']
                or copied_ref['bytes']!=ref['bytes']):raise ValueError('recovery copy bytes differ')
        copied.append({'original':ref,'copy':file_ref(destination)})
    pointer=read_json(target/'resume.json');pointer['diagnostics']=study.fileid(target/'diagnostics_00032.json');atomic(target/'resume.json',pointer)
    for pair in copied:
        if Path(pair['copy']['path']).name=='resume.json':pair['copy']=file_ref(target/'resume.json')
    validate_original(row)
    atomic(target/'copy_verification.json',{'origin':copy.deepcopy(row),'copies':copied,
        'directories':[str((target/Path(d).relative_to(folder)).resolve()) if Path(d).is_absolute() else str((target/d).resolve()) for d in row['directories']]})
    _validate_copy(row,target)
    return target


def _validate_copy(row,target):
    record=read_json(target/'copy_verification.json')
    if record.get('origin')!=row:raise ValueError('recovery copy origin differs')
    pairs=record.get('copies',[])
    if len(pairs)!=len(row['files']) or {p['original']['path'] for p in pairs}!={r['path'] for r in row['files']}:
        raise ValueError('recovery copy incomplete')
    originals={ref['path']:ref for ref in row['files']};source_folder=Path(row['folder']).resolve()
    for pair in pairs:
        if pair['original']!=originals[pair['original']['path']]:raise ValueError('recovery original reference differs')
        relative=Path(pair['original']['path']).relative_to(source_folder)
        if relative.name in ('checkpoint_00000.diagnostics.json','checkpoint_00032.diagnostics.json'):
            relative=relative.with_name('diagnostics_'+relative.name[11:16]+'.json')
        if Path(pair['copy']['path']).resolve()!=target/relative:raise ValueError('recovery copy path differs')
        if relative.name!='resume.json' and any(pair['copy'][key]!=pair['original'][key] for key in ('sha256','bytes')):
            raise ValueError('recovery copy differs from original bytes')
    expected={Path(pair['copy']['path']).resolve():pair['copy'] for pair in pairs}
    actual={p.resolve():file_ref(p) for p in target.rglob('*') if p.is_file() and p!=target/'copy_verification.json'}
    if actual!=expected:raise ValueError('recovery copy changed/unknown files')
    expected_dirs={(target/Path(value).relative_to(source_folder)).resolve() if Path(value).is_absolute()
                   else (target/value).resolve() for value in row['directories']}
    recorded_dirs={Path(value).resolve() for value in record['directories']}
    if (recorded_dirs!=expected_dirs or len(recorded_dirs)!=len(record['directories'])
            or {p.resolve() for p in target.rglob('*') if p.is_dir()}!=expected_dirs):
        raise ValueError('recovery copy unknown directories')
    if any(p.is_symlink() for p in target.rglob('*')):raise ValueError('linked recovery copy forbidden')
    expected_pointer=read_json(source_folder/'resume.json');expected_pointer['diagnostics']=study.fileid(target/'diagnostics_00032.json')
    if read_json(target/'resume.json')!=expected_pointer:raise ValueError('recovery pointer changed beyond diagnostic path')
    complete_sidecar(target,row['identity'])
    return record


def _clone(value):
    import numpy as np,torch
    if torch.is_tensor(value):return value.detach().cpu().clone()
    if isinstance(value,np.ndarray):return value.copy()
    if isinstance(value,dict):return {key:_clone(item) for key,item in value.items()}
    if isinstance(value,list):return [_clone(item) for item in value]
    if isinstance(value,tuple):return tuple(_clone(item) for item in value)
    return copy.deepcopy(value)


def capture_state(m,opt,rng,trace,gradient,summary):
    import torch
    return _clone({'model':m.state_dict(),'optimizer':opt.state_dict(),'fit_numpy_rng':rng.bit_generator.state,
        'torch_rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state_all() if m.mass.is_cuda else [],
        'trace':trace,'gradient':gradient,'summary':summary})


def state_equal(a,b):
    import numpy as np,torch
    if torch.is_tensor(a) or torch.is_tensor(b):
        return torch.is_tensor(a) and torch.is_tensor(b) and a.dtype==b.dtype and a.shape==b.shape and bool(torch.equal(a.cpu(),b.cpu()))
    if isinstance(a,np.ndarray) or isinstance(b,np.ndarray):
        return isinstance(a,np.ndarray) and isinstance(b,np.ndarray) and a.dtype==b.dtype and np.array_equal(a,b)
    if type(a) is not type(b):return False
    if isinstance(a,dict):return a.keys()==b.keys() and all(state_equal(a[key],b[key]) for key in a)
    if isinstance(a,(list,tuple)):return len(a)==len(b) and all(state_equal(x,y) for x,y in zip(a,b))
    return a==b


def apply_global_rng(snapshot):
    import torch
    torch.set_rng_state(snapshot['torch_rng'].clone())
    if snapshot['cuda_rng']:
        if len(snapshot['cuda_rng'])!=torch.cuda.device_count():raise ValueError('CUDA identity changed')
        torch.cuda.set_rng_state_all([value.clone() for value in snapshot['cuda_rng']])


def migrate(row,copy_dir,science_folder,commit):
    import numpy as np,torch
    science_folder=Path(science_folder).resolve();copy_dir=Path(copy_dir).resolve()
    if science_folder.exists():raise ValueError('new migration destination already exists')
    if len(commit)!=40 or any(c not in '0123456789abcdef' for c in commit):raise ValueError('full migration source required')
    original=validate_original(row)
    if not copy_dir.exists():prepare_copy(row,copy_dir)
    _validate_copy(row,copy_dir)
    tr=original['tr'];m,opt=study.make(row['seed'],tr['observed'],'rational')
    rng=np.random.default_rng(np.random.SeedSequence([row['seed'],7]))
    step,trace,gradient,summary=restore_complete(copy_dir,m,opt,rng,original['identity'])
    if step!=32:raise ValueError('migration requires actual32 restore')
    before=capture_state(m,opt,rng,trace,gradient,summary)
    if not opt.state:raise ValueError('restored Adam state is empty')
    parameters=dict(m.named_parameters())
    for name in summary['max']:
        if name not in parameters or (parameters[name].numel() and not opt.state.get(parameters[name])):
            raise ValueError('restored gradient parameter missing Adam state')
    for group in opt.param_groups:
        for parameter in group['params']:
            state=opt.state.get(parameter)
            if state:
                if (not {'step','exp_avg','exp_avg_sq'}<=set(state) or float(state['step'])!=32
                        or any(state[key].shape!=parameter.shape or not bool(torch.isfinite(state[key]).all()) for key in ('exp_avg','exp_avg_sq'))):
                    raise ValueError('restored Adam state differs')
    validate_original(row);_validate_copy(row,copy_dir)
    science_folder.mkdir(parents=True);shutil.copyfile(original['folder']/'train.npz',science_folder/'train.npz')
    identity={'source_commit':commit,'key':row['key'],'data_identity':original['identity']['data_identity'],
        'train_raw':file_ref(science_folder/'train.npz'),'form':'rational','origin':copy.deepcopy(row)}
    publish_checkpoint(science_folder,m,opt,rng,32,trace,gradient,summary,identity)
    after=capture_state(m,opt,rng,trace,gradient,summary)
    if not state_equal(before,after):raise ValueError('migration publish changed computational state')
    complete_sidecar(science_folder,identity)
    atomic(science_folder/'migration.json',{'origin':copy.deepcopy(row),'old_identity':original['identity'],
        'new_identity':identity,'restored_step':32,'shared_restores':1,'publish_state_equal':True,
        'copy_verification':file_ref(copy_dir/'copy_verification.json'),
        'checkpoint_files':[file_ref(science_folder/'checkpoint_00032'/name) for name in ('model.json','training.pt')]})
    return {'m':m,'opt':opt,'rng':rng,'tr':tr,'trace':trace,'gradient':gradient,'summary':summary,
        'identity':identity,'folder':science_folder,'origin':copy.deepcopy(row),'global_rng':_clone({key:after[key] for key in ('torch_rng','cuda_rng')}),
        'computational_state':after,'step':32,'key':row['key'],'seed':row['seed']}


def train_group(out,commit,group,initial_states=None):
    import numpy as np,torch
    out=Path(out).resolve()
    if (out/'complete.json').exists() or (out/'selection.json').exists():raise ValueError('completed attempts cannot be resumed')
    if group not in study.GROUPS:raise ValueError('fixed scientific group required')
    expected={group+'_'+str(seed) for seed in study.SEEDS}
    if initial_states is None and any((out/key).exists() for key in expected):
        raise ValueError('model output already exists; no implicit resume')
    if group=='0_rational':
        if not isinstance(initial_states,dict) or set(initial_states)!=expected:raise ValueError('all three fixed migration states required')
        for key,state in initial_states.items():
            if (state['step']!=32 or state['identity']['source_commit']!=commit or state['identity']['key']!=key
                    or state['folder']!=out/key or complete_sidecar(state['folder'],state['identity'])[0]!=32):
                raise ValueError('migration live state source/folder/count differs')
            apply_global_rng(state['global_rng'])
            if not state_equal(capture_state(state['m'],state['opt'],state['rng'],state['trace'],state['gradient'],state['summary']),state['computational_state']):
                raise ValueError('migration live computational state changed')
    elif initial_states is not None:raise ValueError('only first rational group receives migration states')
    noise=int(group[0]);form=group[2:];rows=[]
    for seed in study.SEEDS:
        key=group+'_'+str(seed);folder=out/key;trace=[];gradient={'total_raw_norm':[],'clip_factor':[]};summary={'first':{},'last':{},'max':{}}
        try:
            if initial_states is not None:
                state=initial_states[key];m,opt,rng,tr=[state[name] for name in ('m','opt','rng','tr')]
                trace,gradient,summary,identity=[state[name] for name in ('trace','gradient','summary','identity')]
                apply_global_rng(state['global_rng']);step=32;train_raw=identity['train_raw']
            else:
                if folder.exists():raise ValueError('model output already exists; no implicit resume')
                folder.mkdir();tr=study.data(seed,0,noise,None);train_raw=study.save_data(folder/'train.npz',tr)
                if not tr['qualification']['valid']:
                    rows.append({'key':key,'status':'UNRESOLVED','qualification':tr['qualification'],'train_raw':train_raw});continue
                identity={'source_commit':commit,'key':key,'data_identity':tr['identity'],'train_raw':train_raw,'form':form}
                m,opt=study.make(seed,tr['observed'],form);rng=np.random.default_rng(np.random.SeedSequence([seed,7]));step=0
                publish_checkpoint(folder,m,opt,rng,0,trace,gradient,summary,identity)
            obs=torch.as_tensor(tr['observed'],device='cuda');scale=torch.as_tensor(tr['scales'],device='cuda')
            while step<512:
                atomic(out/'progress.json',{'key':key,'last_completed':step,'inflight':step+1,'source_commit':commit})
                study.advance(m,opt,rng,obs,scale,None,trace,gradient,summary);step+=1
                atomic(out/'progress.json',{'key':key,'last_completed':step,'inflight':None,'source_commit':commit})
                if step in (32,128,512):publish_checkpoint(folder,m,opt,rng,step,trace,gradient,summary,identity)
                if step==32:
                    m,opt=study.make(seed,tr['observed'],form);step,trace,gradient,summary=restore_complete(folder,m,opt,rng,identity)
            new_or_equal(folder/'trace.json',{'loss':trace,'gradient':gradient,'parameter_summary':summary})
            path=folder/'checkpoint_00512';row={'key':key,'seed':seed,'noise':noise,'form':form,'status':'FROZEN','path':str(path),
                'files':{name:file_ref(path/name)['sha256'] for name in ('model.json','training.pt')},'train_raw':train_raw,
                'trace':study.fileid(folder/'trace.json'),'identity':identity,'loss_first32':sum(trace[:32])/32,
                'loss_last32':sum(trace[-32:])/32,'resume_pointer':study.fileid(folder/'resume.json')}
            study.verify(row);rows.append(row)
        except (RuntimeError,FloatingPointError,ValueError,FileNotFoundError,KeyError) as error:
            if not folder.is_dir():raise
            atomic(folder/'failed_trace.json',{'loss':trace,'gradient':gradient,'parameter_summary':summary})
            rows.append({'key':key,'status':'UNRESOLVED','error':repr(error),'trace':study.fileid(folder/'failed_trace.json'),
                'raw_files':[study.fileid(path) for path in folder.glob('*.npz')]})
    result={'status':'COMPLETED','source_commit':commit,'group':group,'rows':rows,'all_group_frozen':all(row['status']=='FROZEN' for row in rows)}
    atomic(out/'selection.json',result)
    return result
