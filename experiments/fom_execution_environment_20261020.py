"""Registered environment guards and finite QA; no model loading or solving."""
from __future__ import annotations
import argparse
import hashlib
import importlib
import json
import math
import multiprocessing as mp
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

ROOT=Path(__file__).resolve().parents[1]
FOM_PY=Path('E:/03-nn_Transfer_Func/kron-reduce/.venv/Scripts/python.exe')
GPU_PY=ROOT/'.venv-gpu/Scripts/python.exe'
PRODUCTION=ROOT/'experiments/active_fom_pair_20261020.py'
ALLOWED={'short':'fom','train_fom':'fom','val_fom':'fom','prepare':'gpu','score':'gpu',
         'schedule':'gpu','adapt':'gpu','evaluate':'gpu','freeze':'gpu','summary':'gpu','qa':'gpu'}

def executable_for(category):
    if category not in ALLOWED: raise ValueError('unregistered execution category: '+str(category))
    return FOM_PY if ALLOWED[category]=='fom' else GPU_PY

def _sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''): h.update(block)
    return h.hexdigest()

def _load_dependency(name): return importlib.import_module(name)

def _identity(role):
    if role not in ('fom','gpu'): raise ValueError('unknown role')
    exe=(FOM_PY if role=='fom' else GPU_PY).resolve()
    if Path(sys.executable).resolve()!=exe: raise RuntimeError('wrong interpreter for '+role)
    prefix=Path(sys.prefix).resolve()
    if prefix!=exe.parent.parent: raise RuntimeError('wrong environment prefix for '+role)
    numpy=_load_dependency('numpy')
    return {'role':role,'executable':str(exe),'executable_sha256':_sha(exe),'prefix':str(prefix),
            'base_prefix':str(Path(sys.base_prefix).resolve()),'python':sys.version,
            'numpy':{'version':str(numpy.__version__),'path':str(Path(numpy.__file__).resolve())}}

def _fom_identity():
    identity=_identity('fom'); andes=_load_dependency('andes'); h5py=_load_dependency('h5py')
    if str(andes.__version__)!='2.0.0': raise RuntimeError('ANDES must be 2.0.0')
    for p in (ROOT,ROOT/'experiments',ROOT/'src'):
        if str(p) not in sys.path: sys.path.insert(0,str(p))
    from experiments import closure_discovery_data as data
    from experiments import active_fom_pair_20261020 as production
    config=data._safe_config()
    # This qualified adapter reads root attributes/equilibrium/inertia only.
    # It never enumerates event groups or loads checkpoints.
    reference=data._load_reference_fingerprint()
    data._assert_config_compatible(config,reference)
    attestation=data._runtime_system_attestation(config,reference)
    data._assert_attestation_compatible(attestation,config,reference)
    identity.update(andes={'version':str(andes.__version__),'path':str(Path(andes.__file__).resolve()),
                          'module_sha256':_sha(andes.__file__)},
        h5py={'version':str(h5py.__version__),'path':str(Path(h5py.__file__).resolve())},
        case={'configured':str(config['case']),'resolved_path':attestation['resolved_case_path'],
              'sha256':attestation['case_sha256']},
        train_physical_metadata_sha256=reference['fingerprint_sha256'],attestation=attestation)
    # Repository sources stay separate from external executable/module/case files.
    identity['source']={'production':{'path':str(PRODUCTION.resolve()),'sha256':_sha(PRODUCTION),'module':production.__name__},
        'preflight':{'path':str(Path(__file__).resolve()),'sha256':_sha(__file__)},
        'adapter':{'path':str(Path(data.__file__).resolve()),'sha256':_sha(data.__file__)}}
    return identity

def _gpu_identity():
    identity=_identity('gpu'); torch=_load_dependency('torch')
    available=bool(torch.cuda.is_available()); count=int(torch.cuda.device_count())
    if not available or count<1: raise RuntimeError('GPU role requires CUDA and at least one device')
    identity['torch']={'version':str(torch.__version__),'path':str(Path(torch.__file__).resolve()),
                      'cuda_version':torch.version.cuda,'cuda_available':available,
                      'devices':[{'index':i,'name':str(torch.cuda.get_device_name(i))} for i in range(count)]}
    return identity

def _text(value,label):
    if not isinstance(value,str) or not value: raise ValueError('nonempty string required: '+label)

def _digest(value,label):
    _text(value,label)
    if len(value)!=64 or any(c not in '0123456789abcdef' for c in value): raise ValueError('invalid SHA256: '+label)

def _shape(identity,role):
    if not isinstance(identity,dict) or identity.get('role')!=role: raise ValueError('missing or mismatched identity role')
    required={'role','executable','executable_sha256','prefix','base_prefix','python','numpy'}
    required|={'andes','h5py','case','train_physical_metadata_sha256','attestation','source'} if role=='fom' else {'torch'}
    if set(identity)!=required: raise ValueError('identity schema differs from registered role')
    for n in ('executable','prefix','base_prefix','python'): _text(identity[n],n)
    _digest(identity['executable_sha256'],'executable')
    expected=(FOM_PY if role=='fom' else GPU_PY).resolve()
    if Path(identity['executable']).resolve()!=expected or Path(identity['prefix']).resolve()!=expected.parent.parent:
        raise ValueError('identity executable/prefix role mismatch')
    for n in ('numpy',)+(('andes','h5py') if role=='fom' else ()):
        row=identity[n]
        if not isinstance(row,dict): raise ValueError('module identity must be mapping')
        for k in ('path','version'): _text(row.get(k),n+'.'+k)
    if role=='fom':
        if identity['andes']['version']!='2.0.0': raise ValueError('ANDES version mismatch')
        _digest(identity['andes'].get('module_sha256'),'ANDES module')
        _digest(identity['train_physical_metadata_sha256'],'physical metadata')
        if not isinstance(identity['case'],dict): raise ValueError('case identity must be mapping')
        for k in ('configured','resolved_path'): _text(identity['case'].get(k),'case.'+k)
        _digest(identity['case'].get('sha256'),'case')
        att=identity['attestation']
        if not isinstance(att,dict) or att.get('source')!='andes_runtime': raise ValueError('runtime attestation missing')
        for k in ('config_sha256','config_file_sha256','reference_fingerprint_sha256','case_sha256'): _digest(att.get(k),'attestation.'+k)
        if (att['case_sha256']!=identity['case']['sha256'] or att['reference_fingerprint_sha256']!=identity['train_physical_metadata_sha256']
                or att.get('andes_version')!='2.0.0' or att.get('case_config')!=identity['case']['configured']): raise ValueError('case/runtime/physical mismatch')
        if type(att.get('omega_s')) not in (int,float) or not math.isfinite(att['omega_s']) or att['omega_s']<=0: raise ValueError('invalid omega_s')
        if not isinstance(identity['source'],dict) or set(identity['source'])!={'production','preflight','adapter'}: raise ValueError('source identity missing')
        for row in identity['source'].values():
            if not isinstance(row,dict): raise ValueError('source entry must be mapping')
            _text(row.get('path'),'source path'); _digest(row.get('sha256'),'source')
    else:
        row=identity['torch']
        if not isinstance(row,dict) or row.get('cuda_available') is not True or not isinstance(row.get('devices'),list) or not row['devices']: raise ValueError('CUDA identity unavailable')
        for k in ('path','version','cuda_version'): _text(row.get(k),'torch.'+k)
        for i,device in enumerate(row['devices']):
            if not isinstance(device,dict) or type(device.get('index')) is not int or device['index']!=i: raise ValueError('invalid CUDA device index')
            _text(device.get('name'),'CUDA device name')

def validate_identity(current,expected,role):
    if role not in ('fom','gpu'): raise ValueError('unknown role')
    _shape(current,role); _shape(expected,role)
    if current!=expected: raise RuntimeError('environment/source/case identity changed')
    return current

def preflight(role,expected=None):
    if role not in ('fom','gpu'): raise ValueError('role must be fom or gpu')
    current=_fom_identity() if role=='fom' else _gpu_identity()
    _shape(current,role)
    if expected is not None: validate_identity(current,expected,role)
    return current

def guarded_call(category,callback,*,expected=None):
    executable_for(category)
    return callback(preflight(ALLOWED[category],expected))

def compare_child(parent,child): return validate_identity(child,parent,'fom')

def _worker():
    for p in (ROOT,ROOT/'experiments'):
        if str(p) not in sys.path: sys.path.insert(0,str(p))
    from experiments.closure_p3_disagreement_20261020 import own_creation_filetime_utc
    return {'pid':os.getpid(),'creation_filetime_utc':own_creation_filetime_utc(),'executable':str(Path(sys.executable).resolve())}

def _spawn_child(sender):
    try: sender.send({'ok':True,'worker':_worker(),'identity':preflight('fom')})
    except BaseException as exc: sender.send({'ok':False,'error':repr(exc),'worker':_worker()})
    finally: sender.close()

def _cleanup(child,deadline):
    if child.is_alive():
        child.terminate(); child.join(timeout=max(0.,min(1.,deadline-time.monotonic())))
    if child.is_alive():
        child.kill(); child.join(timeout=max(0.,min(1.,deadline-time.monotonic())))
    if child.is_alive(): raise RuntimeError('owned spawn cleanup unresolved')

def _probe(role,deadline):
    identity=preflight(role); result={'identity':identity}
    if role=='fom':
        if deadline-time.monotonic()<=4: raise TimeoutError('deadline before FOM spawn')
        ctx=mp.get_context('spawn'); receiver,sender=ctx.Pipe(duplex=False)
        child=ctx.Process(target=_spawn_child,args=(sender,)); started=time.monotonic()
        try:
            child.start(); sender.close()
            if not receiver.poll(max(0.,min(60.,deadline-time.monotonic()-4))): raise TimeoutError('FOM spawn identity deadline')
            payload=receiver.recv()
            child.join(timeout=max(0.,min(2.,deadline-time.monotonic()-2)))
            if not payload.get('ok'): raise RuntimeError('FOM child preflight failed: '+str(payload))
            if child.exitcode!=0: raise RuntimeError('FOM child nonzero or missing exit')
            compare_child(identity,payload['identity'])
            worker=payload['worker']
            if (type(worker.get('pid')) is not int or worker['pid']!=child.pid or
                    type(worker.get('creation_filetime_utc')) is not int or worker['creation_filetime_utc']<=0 or
                    Path(worker.get('executable','')).resolve()!=FOM_PY.resolve()):
                raise RuntimeError('actual FOM child process identity mismatch')
            result['spawn']={'pid':child.pid,'worker':payload['worker'],'exit_code':child.exitcode,
                             'identity_match':True,'identity':payload['identity'],'elapsed_seconds':time.monotonic()-started}
        finally:
            sender.close(); receiver.close()
            if child.pid is not None: _cleanup(child,deadline)
    return result

def _write(path,payload):
    with Path(path).open('x',encoding='utf-8') as stream: json.dump(payload,stream,indent=2,allow_nan=False)

def run_synthetic(real_fom,real_gpu):
    sys.path.insert(0,str(ROOT/'tests'))
    from test_fom_execution_environment_20261020 import run_qa
    return run_qa(real_fom,real_gpu)

def _run_probe(exe,role,out,deadline):
    remaining=deadline-time.monotonic()
    cap=min(100.,remaining-5)
    if cap<=25: raise TimeoutError('QA deadline before probe')
    cmd=[str(exe),str(Path(__file__).resolve()),'probe','--role',role,'--output',str(out),
         '--max-seconds',str(max(1,int(cap-15)))]
    command_path=out.parent/(out.name+'_command.json')
    _write(command_path,{'executable':cmd[0],'arguments':cmd[1:],
                         'arguments_line':subprocess.list2cmdline(cmd[1:])})
    native_dir=out.parent/(out.name+'_native')
    outer_dir=out.parent/(out.name+'_outer')
    worker_path=out/'worker.json'
    ps=Path(os.environ['SystemRoot'])/'System32/WindowsPowerShell/v1.0/powershell.exe'
    command=[str(ps),'-NoProfile','-NonInteractive','-ExecutionPolicy','Bypass','-File',
             str(ROOT/'experiments/native_process_receipt_20261020.ps1'),
             '-CommandJson',str(command_path),'-WorkingDirectory',str(ROOT),
             '-Output',str(native_dir),'-TimeoutSeconds',str(cap-10),'-MaxSeconds',str(cap),
             '-FixtureIdentityJson',str(worker_path)]
    for p in (ROOT/'experiments',ROOT/'src'):
        if str(p) not in sys.path: sys.path.insert(0,str(p))
    from fixed_edge_pair_20261019 import native_job
    started=time.monotonic()
    outer=native_job(command,outer_dir,cap,worker_json=worker_path)
    receipt=json.loads((native_dir/'receipt.json').read_text(encoding='utf-8-sig'))
    worker=json.loads(worker_path.read_text(encoding='utf-8'))
    supervision_valid=(outer.get('native_receipt_complete') is True
        and outer.get('timed_out') is False and receipt.get('native_receipt_complete') is True
        and receipt.get('status') in ('COMPLETED','FAILED_NONZERO')
        and outer.get('exit_code')==(0 if receipt.get('exit_code')==0 else 1)
        and receipt.get('cleanup_verified') is True and receipt.get('timeout') is False
        and receipt.get('fixture_identity_after')=='GONE' and receipt.get('exit_code_obtained') is True
        and worker.get('pid')==receipt.get('fixture_pid')==outer.get('actual_worker_pid')
        and worker.get('creation_filetime_utc')==receipt.get('fixture_creation_filetime_utc')
        and type(worker.get('pid')) is int and type(worker.get('creation_filetime_utc')) is int
        and Path(worker.get('executable','')).resolve()==Path(exe).resolve())
    rec={'command':cmd,'command_file':str(command_path),'command_sha256':_sha(command_path),
         'pid':worker['pid'],'worker':worker,'exit_code':receipt.get('exit_code'),
         'timeout':receipt.get('timeout'),'supervision_valid':bool(supervision_valid),
         'elapsed_seconds':time.monotonic()-started,'outer':outer,'native':receipt,
         'cost_scope':'subset of enclosing QA envelope; not an additional charge'}
    if (out/'worker.json').is_file(): rec['worker']=json.loads((out/'worker.json').read_text(encoding='utf-8'))
    if (out/'final.json').is_file(): rec['result']=json.loads((out/'final.json').read_text(encoding='utf-8'))
    return rec

def _qa(out,deadline):
    result={'status':'RUNNING','probes':{},'parent_gpu':preflight('gpu')}
    for name,exe,role in (('fom',FOM_PY,'fom'),('gpu',GPU_PY,'gpu'),('wrong_role',GPU_PY,'fom')):
        rec=_run_probe(exe,role,out/(name+'_probe'),deadline); result['probes'][name]=rec
        _write(out/(name+'_receipt.json'),rec)
        if not rec['supervision_valid'] or rec['timeout'] or rec['exit_code'] is None:
            raise RuntimeError('probe supervision/cleanup/missing exit: '+name)
        if name=='wrong_role':
            if rec['exit_code']==0 or rec.get('result',{}).get('status')!='FAIL': raise RuntimeError('wrong-role refusal failed')
        elif rec['exit_code']!=0 or rec.get('result',{}).get('status')!='PASS': raise RuntimeError('real environment failed: '+name)
    fom=result['probes']['fom']['result']['identity']; gpu=result['probes']['gpu']['result']['identity']
    validate_identity(gpu,result['parent_gpu'],'gpu')
    result['synthetic']=run_synthetic(fom,gpu)
    _write(out/'synthetic.json',result['synthetic'])
    if result['synthetic']['status']!='PASS': raise RuntimeError('synthetic production-guard QA failed')
    result['status']='PASS'
    return result

def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__); sub=parser.add_subparsers(dest='command',required=True)
    for name in ('probe','qa'):
        p=sub.add_parser(name); p.add_argument('--output',type=Path,required=True)
        p.add_argument('--max-seconds',type=int,default=90 if name=='probe' else 450)
        if name=='probe': p.add_argument('--role',choices=('fom','gpu'),required=True)
    args=parser.parse_args(argv)
    if not 1<=args.max_seconds<=500: raise ValueError('max-seconds must be 1..500')
    started=time.monotonic(); deadline=started+args.max_seconds
    args.output=args.output.resolve(); args.output.mkdir(parents=True,exist_ok=False)
    try:
        _write(args.output/'worker.json',_worker())
        result=_qa(args.output,deadline) if args.command=='qa' else {'status':'PASS',**_probe(args.role,deadline)}
    except BaseException as exc: result={'status':'FAIL','error':repr(exc),'traceback':traceback.format_exc()}
    result['elapsed_seconds']=time.monotonic()-started
    _write(args.output/'final.json',result)
    return 0 if result['status']=='PASS' else 1

if __name__=='__main__': raise SystemExit(main())
