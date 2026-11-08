"""Finite, observation-only comparison of two registered edge-memory terms.

The entry point deliberately imports the numerical stack only after its native
worker receipt and watchdog are established.  Scientific decisions remain
owned by the root task.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import uuid
import threading
import copy

PROCESS_START = time.perf_counter()
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'experiments'), str(ROOT / 'src')]
SEEDS = (5101, 5102, 5103)
SIN_TERM = {'channel': 'edge', 'expression': {'op': 'mul', 'args': [
    {'op': 'add', 'args': [{'op': 'var', 'name': 'qi'}, {'op': 'mul', 'args': [
        {'op': 'const', 'value': -1}, {'op': 'var', 'name': 'qj'}]}]},
    {'op': 'var', 'name': 's'}]}}
COS_TERM = {'channel': 'edge', 'expression': {'op': 'mul', 'args': [
    {'op': 'add', 'args': [{'op': 'var', 'name': 'qi'}, {'op': 'var', 'name': 'qj'}]},
    {'op': 'var', 'name': 'c'}]}}


def _atomic_json(path, value):
    path = Path(path)
    tmp = path.with_name(path.name + '.' + uuid.uuid4().hex + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False), encoding='utf-8')
    os.replace(tmp, path)


def _filetime():
    """Windows FILETIME integer for external liveness/ownership checks."""
    import ctypes
    class FT(ctypes.Structure):
        _fields_ = [('low', ctypes.c_uint32), ('high', ctypes.c_uint32)]
    kernel = ctypes.WinDLL('kernel32', use_last_error=True)
    kernel.GetCurrentProcess.restype = ctypes.c_void_p
    kernel.GetProcessTimes.argtypes = [ctypes.c_void_p] + [ctypes.POINTER(FT)] * 4
    values = [FT() for _ in range(4)]
    if not kernel.GetProcessTimes(kernel.GetCurrentProcess(), *[ctypes.byref(v) for v in values]):
        raise ctypes.WinError(ctypes.get_last_error())
    return (int(values[0].high) << 32) | int(values[0].low)


def _worker_receipt(out, phase, source_commit, binding):
    created=_filetime()
    receipt = {'pid': os.getpid(), 'created_filetime': created, 'creation_filetime':created,
               'creation_filetime_utc':created, 'phase': phase,
               'source_commit': source_commit, 'binding': str(binding.resolve()),
               'command': sys.argv, 'python': sys.version, 'executable': sys.executable}
    _atomic_json(out / 'worker.json', receipt)
    return receipt


def canonical_terms():
    from symbolic_closure_rom import canonicalize_expression, validate_expression
    result = []
    for term in (SIN_TERM, COS_TERM):
        result.append({'channel': term['channel'], 'expression':
                       validate_expression(term['expression'], term['channel'])})
    return result


def _read_binding(path):
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f'missing frozen source/contract binding: {p}')
    value = json.loads(p.read_text(encoding='utf-8'))
    old = value.get('old_contract_inputs')
    if not isinstance(old, dict) or not {'path', 'sha256', 'files'} <= set(old):
        raise ValueError('binding lacks root-frozen old_contract_inputs identity')
    table_path = (ROOT / old['path']).resolve()
    if not table_path.is_file():
        raise FileNotFoundError(f'missing frozen old-contract table: {table_path}')
    actual = hashlib.sha256(table_path.read_bytes()).hexdigest()
    if actual != old['sha256']:
        raise ValueError('old-contract table bytes differ from binding')
    table = json.loads(table_path.read_text(encoding='utf-8'))
    if table.get('files') != old['files']:
        raise ValueError('old-contract table file identities differ from binding')
    return value


def run_qa(out, deadline, binding):
    """Run only the single registered QA suite; no formal val fixture is made here."""
    result = {'status': 'UNRESOLVED', 'contract': 'pending-root-binding-validation',
              'terms': canonical_terms(), 'binding_sha256': hashlib.sha256(binding.read_bytes()).hexdigest(),
              'deadline_seconds': deadline - time.perf_counter()}
    # Require the root-built BASE contract verifier, if present; never silently
    # treat missing historical identity evidence as a pass.
    bound = json.loads(binding.read_text(encoding='utf-8'))
    result['contracts'] = verify_old_contract(bound, deadline)
    cmd = [sys.executable, '-m', 'pytest', '-q', str(ROOT / 'tests' / 'test_edge_memory_discrimination_20261021.py')]
    remaining = max(0.1, deadline - time.perf_counter())
    proc = subprocess.run(cmd, cwd=ROOT, timeout=remaining, capture_output=True, text=True)
    result.update(status='PASS' if proc.returncode == 0 else 'FAIL',
                  pytest_returncode=proc.returncode, stdout=proc.stdout, stderr=proc.stderr)
    (out / 'pytest_stdout.txt').write_text(proc.stdout, encoding='utf-8')
    (out / 'pytest_stderr.txt').write_text(proc.stderr, encoding='utf-8')
    _atomic_json(out / 'result.json', result)
    return result


def verify_old_contract(binding, deadline):
    """Validate the frozen shared stage-0/512 identities before any update."""
    import numpy as np
    import torch
    from closure_causal_recovery_20261019 import runtime_model, copy_runtime
    from closure_discovery_benchmarks import toy_source
    from closure_causal_history_residual_20261019 import CausalHistoryResidual
    torch.set_num_threads(1)
    table = json.loads((ROOT / binding['old_contract_inputs']['path']).read_text())
    files = table['files']
    for rel, expected in files.items():
        p = ROOT / rel
        if not p.is_file() or hashlib.sha256(p.read_bytes()).hexdigest() != expected:
            raise ValueError(f'old source evidence identity mismatch: {rel}')
    rows = []
    for seed in SEEDS:
        if time.perf_counter() >= deadline:
            raise TimeoutError('contract verification exceeded QA deadline')
        folder = ROOT / f'private_data/closure_mechanism_20261019/P1/history_residual/edge_run01/edge_{seed}_residual'
        fixture = json.loads((folder / 'fixture.json').read_text())
        if fixture.get('scales') is None or fixture.get('clean_sha256') is None:
            raise ValueError(f'old fixture identity incomplete for seed {seed}')
        source, arr, scales, meta = _train_fixture(seed, 'cuda', deadline)
        source_identity=hashlib.sha256(json.dumps(source._export_payload(),sort_keys=True,allow_nan=False).encode()).hexdigest()
        if source_identity != fixture.get('source_model_sha256'):
            raise ValueError('common unfitted toy source identity differs')
        obs = torch.tensor(arr, dtype=torch.float64, device='cuda')
        from closure_causal_history_residual_20261019 import make_optimizer
        models = []
        for term in (SIN_TERM, COS_TERM):
            model = CausalHistoryResidual.from_train(runtime_model(source, [term], seed, 'cuda'), obs[:, :11])
            opt = make_optimizer(model)
            rng = np.random.default_rng(np.random.SeedSequence([seed, 7]))
            captured = snapshot(model, opt, rng)
            old = CausalHistoryResidual.from_export(folder/'checkpoint_00000/model.json').cuda()
            state = torch.load(folder/'checkpoint_00000/training.pt', map_location='cpu', weights_only=False)
            equal_tree(model.state_dict(), old.state_dict())
            if [(n, p.requires_grad) for n,p in model.named_parameters()] != [(n,p.requires_grad) for n,p in old.named_parameters()]:
                raise ValueError('old parameter order/flags differ')
            equal_tree(captured['optimizer'], state['optimizer'])
            equal_tree(captured['numpy'], state['fit_numpy_rng'])
            equal_tree(captured['cpu'], state['torch_rng'])
            equal_tree(captured['cuda'], state['cuda_rng'])
            if state['completed_updates'] != 0 or state['trace'] or opt.state:
                raise ValueError('old stage0 is not empty Adam/trace')
            if bool(model.history_coefficients.any()) or bool(model.base.coefficients.any()):
                raise ValueError('nonzero initial coefficient')
            models.append(model)
            load_snapshot(captured,model,opt,rng)
        final = torch.load(folder/'checkpoint_00512/training.pt', map_location='cpu', weights_only=False)
        if final['completed_updates'] != 512 or len(final['trace']) != 512:
            raise ValueError('old endpoint incomplete')
        for step in (0,512):
            for name in ('model.json','training.pt'):
                rel = str((folder/f'checkpoint_{step:05d}'/name).relative_to(ROOT))
                if rel not in files and rel.replace('\\','/') not in files:
                    raise ValueError('old checkpoint absent from hash binding: '+rel)
        rows.append({'seed':seed,'stage0_exact':True,'rng_exact':True,'empty_adam':True,
                     'train_fixture':meta,'old_updates':512,'old_endpoint_sha256':sha_file(folder/'checkpoint_00512/model.json')})
    return rows


def sha_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def equal_tree(a, b):
    import torch
    import numpy as np
    if torch.is_tensor(a):
        if not torch.is_tensor(b) or a.dtype != b.dtype or not torch.equal(a.cpu(), b.cpu()):
            raise ValueError('tensor state differs')
    elif isinstance(a, np.ndarray):
        if not np.array_equal(a,b): raise ValueError('NumPy state differs')
    elif isinstance(a, dict):
        if set(a) != set(b): raise ValueError('state keys differ')
        for k in a: equal_tree(a[k], b[k])
    elif isinstance(a, (list,tuple)):
        if len(a) != len(b): raise ValueError('state lengths differ')
        for x,y in zip(a,b): equal_tree(x,y)
    elif a != b: raise ValueError('scalar state differs')


def snapshot(model,opt,rng):
    import torch
    return copy.deepcopy({'model':model.state_dict(),'optimizer':opt.state_dict(),
                          'numpy':rng.bit_generator.state,'cpu':torch.get_rng_state(),
                          'cuda':torch.cuda.get_rng_state_all()})


def load_snapshot(value,model,opt,rng):
    import torch
    model.load_state_dict(value['model']); opt.load_state_dict(value['optimizer'])
    rng.bit_generator.state=value['numpy']; torch.set_rng_state(value['cpu'].cpu())
    torch.cuda.set_rng_state_all([v.cpu() for v in value['cuda']])


def _bound_pass(binding, key):
    identity=binding.get(key)
    if not isinstance(identity,dict) or not {'path','sha256'} <= identity.keys():
        raise ValueError('missing bound receipt: '+key)
    p=ROOT/identity['path']
    if sha_file(p)!=identity['sha256']: raise ValueError('receipt identity changed')
    receipt=json.loads(p.read_text(encoding='utf-8'))
    result=receipt.get('result',receipt)
    if result.get('status')!='PASS': raise ValueError('receipt did not pass: '+key)
    return result


def timing_observations(device,seed):
    """Private nonscientific arrays; independent seed, no teacher or val stream."""
    import numpy as np
    rng=np.random.default_rng(np.random.SeedSequence([seed,20261021,998]))
    arrays=[]
    for count in (61,111):
        t=np.arange(count)[None,:,None]*.01
        initial=rng.normal(0,.02,(16,1,30))
        velocity=rng.normal(0,.01,(16,1,30))
        arrays.append(initial+velocity*t)
    scales=np.maximum(arrays[0].std((0,1)),1e-6)
    return arrays[0],arrays[1],scales


def find_old_train(summary,seed):
    """Read the old residual row's already frozen train metric, never re-evaluate."""
    def walk(value):
        if isinstance(value,dict):
            yield value
            for x in value.values(): yield from walk(x)
        elif isinstance(value,list):
            for x in value: yield from walk(x)
    candidates=[]
    for row in walk(summary):
        if row.get('seed')==seed and row.get('group')=='residual':
            for nested in walk(row):
                train=nested.get('train')
                if isinstance(train,dict) and 'mse' in train:candidates.append(float(train['mse']))
    if len(candidates)!=1:raise ValueError('old residual train metric missing/ambiguous')
    return candidates[0]


def generic_field(base,z,u=None):
    """QA oracle: parse AST values, autodifferentiate internal energy, apply J/R/B.

    Never calls the compiled closure derivative or TrainingClosureROM.field.
    The inherited physical skeleton is shared, while every added AST derivative
    is obtained independently from its scalar energy.
    """
    import torch
    with torch.enable_grad():
        x=base.interface_to_state(z)
        if not x.requires_grad:x=x.clone().requires_grad_(True)
        d,pi,p,q=x.split(10,-1)
        D=base.base_D; theta=base.phi+d@D.T
        atoms={'qi':q@D.clamp_min(0).T,'qj':q@(-D).clamp_min(0).T,
               's':theta.sin()-base.phi.sin(),'c':theta.cos()-base.phi.cos()}
        def value(ast):
            op=ast['op']
            if op=='var':return atoms[ast['name']]
            if op=='const':return ast['value']
            children=[value(a) for a in ast['args']]
            out=children[0]
            for child in children[1:]:out=out+child if op=='add' else out*child
            return out
        energy=base.structured.energy_and_gradient(x)[0]+.5*base.q_quadratic*q.square().sum(-1)
        weights=base.base_coupling/base.base_coupling.mean().clamp_min(1e-15)
        for a,term in zip(base.coefficients,base.terms):
            energy=energy+a*(weights*value(term['expression'])).sum(-1)
        grad=torch.autograd.grad(energy.sum(),x,create_graph=True)[0]
        J,R=base.matrices(); f=grad@(J-R).T
        pv=base.port_values(x) if u is None else torch.as_tensor(u,device=z.device,dtype=z.dtype)
        if pv.shape[-1]!=10:raise ValueError('external port width')
        return torch.cat((f[...,:10],(f[...,10:20]+pv)/base.mass,f[...,20:]),-1)


def run_phase(phase, output, max_seconds, source_commit, binding, seed=None, model=None):
    output = Path(output).resolve()
    category='P0' if phase in ('qa','cost') else 'P1'
    allowed=(ROOT/'private_data/closure_mechanism_20261019'/category/'edge_memory_discrimination01').resolve()
    if not output.is_relative_to(allowed) or output==allowed:
        raise ValueError('phase output outside registered exclusive stage directory')
    output.mkdir(parents=True, exist_ok=False)
    _worker_receipt(output, phase, source_commit, Path(binding))
    deadline = PROCESS_START + max_seconds
    def timeout():
        _atomic_json(output/'timeout.json', {'status':'TIMEOUT','max_seconds':max_seconds})
        os._exit(124)
    timer=threading.Timer(max(.001,deadline-time.perf_counter()),timeout);timer.daemon=True;timer.start()
    try:
        contract = _read_binding(binding)
        if phase == 'qa': result = run_qa(output, deadline, Path(binding))
        else: result = execute_science(phase, output, deadline, source_commit, contract, seed, model)
        result['source_commit']=source_commit
        result['worker_phase_elapsed_seconds']=time.perf_counter()-PROCESS_START
        _atomic_json(output / 'result.json',result)
        _atomic_json(output / 'complete.json', {'status': 'COMPLETED', 'result': result})
        return result
    except BaseException as error:
        import traceback
        _atomic_json(output/'failure.json', {'status':'UNRESOLVED','error':repr(error),'traceback':traceback.format_exc()})
        raise
    finally:timer.cancel()


def _train_fixture(seed, device, deadline, formal=False):
    """Generate exactly one stream; future stream is not materialized during fit/cost."""
    import numpy as np
    import torch
    from closure_causal_recovery_20261019 import copy_runtime, trajectory
    from closure_discovery_benchmarks import toy_source
    from closure_training_runtime import synchronize
    truth_term, coefficient = SIN_TERM, .15
    source = toy_source()
    truth = copy_runtime(source, [truth_term], device)
    with torch.no_grad(): truth.coefficients.fill_(coefficient)
    streams = np.random.SeedSequence([seed, 20261019]).spawn(4)
    index, steps = (1, 110) if formal else (0, 60)
    rng = np.random.default_rng(streams[index])
    strength = np.where(np.arange(16)[:, None] < 8, .08, .3)
    initial = rng.normal(size=(16, 40)) * strength
    initial[:, 30:] = rng.normal(size=(16, 10)) * np.where(np.arange(16)[:, None] < 8, .2, 1.2)
    z = torch.tensor(initial, dtype=torch.float64, device=device)
    with torch.no_grad():
        coarse = trajectory(truth, z, steps, deadline=deadline).cpu().numpy()
        fine = trajectory(truth, z, steps * 2, dt=.005, deadline=deadline).cpu().numpy()[:, ::2]
    old = json.loads((ROOT / f'private_data/closure_mechanism_20261019/P1/history_residual/edge_run01/edge_{seed}_residual/fixture.json').read_text())
    scales = np.asarray(old['scales']) if formal else np.maximum(coarse.std(axis=(0, 1)), 1e-6)
    meta = {'seed': seed, 'stream': index, 'steps': steps, 'scales': scales.tolist(),
            'clean_sha256': hashlib.sha256(coarse.tobytes()).hexdigest(),
            'dual_step_normalized_max_abs': float(np.max(np.abs(coarse-fine)/scales))}
    # The fixture identity is checked only after train-only data have been generated.
    expected_index = 0 if not formal else 1
    if meta['clean_sha256'] != old['clean_sha256'][expected_index] or meta['scales'] != old['scales']:
        raise ValueError('train/validation observations or scales differ from registered fixture')
    qualify_fixture(coarse,fine,scales,meta['dual_step_normalized_max_abs'])
    return source, coarse, scales, meta


def qualify_fixture(coarse,fine,scales,difference):
    import numpy as np
    if (not np.isfinite(coarse).all() or not np.isfinite(fine).all()
        or not np.isfinite(scales).all() or not (scales>=1e-6).all()
        or not np.isfinite(difference) or difference>1e-3):
        raise FloatingPointError('truth numerical qualification failed; no deletion')


def execute_science(phase, output, deadline, source_commit, binding, seed, model_path):
    import copy
    import numpy as np
    import torch
    from closure_causal_recovery_20261019 import runtime_model, optimizer
    from closure_causal_history_residual_20261019 import (CausalHistoryResidual, make_optimizer,
        train_update, checkpoint, restore, evaluate, sha, loss)
    from closure_training_runtime import TrainingClosureROM
    from closure_discovery_benchmarks import toy_source
    from closure_causal_recovery_20261019 import copy_runtime
    torch.set_num_threads(1)
    if phase == 'cost':
        qa = _bound_pass(binding,'qa_result')
        if len(qa.get('contracts',[]))!=3: raise ValueError('three actual contracts missing')
        rows=[]
        for s in SEEDS:
            if time.perf_counter() >= deadline: raise TimeoutError('cost cap')
            loaded=time.perf_counter()
            source, obs_np, scales_np, meta = _train_fixture(s, 'cuda', deadline)
            obs=torch.tensor(obs_np,dtype=torch.float64,device='cuda'); scales=torch.tensor(scales_np,dtype=torch.float64,device='cuda')
            m=CausalHistoryResidual.from_train(runtime_model(source,[COS_TERM],s,'cuda'),obs[:,:11]); opt=make_optimizer(m)
            rng=np.random.default_rng(np.random.SeedSequence([s,7])); times=[]; actual_trace=[]
            torch.cuda.synchronize(); load_seconds=time.perf_counter()-loaded
            torch.cuda.reset_peak_memory_stats()
            def update():
                t=time.perf_counter(); value,_=train_update(m,opt,rng,obs,scales,deadline)
                torch.cuda.synchronize(); times.append(time.perf_counter()-t); actual_trace.append(value)
            update()  # step 1
            snap=output/f'cost_{s}.pt'
            before=time.perf_counter(); torch.save(snapshot(m,opt,rng),snap)
            torch.cuda.synchronize(); snapshot_seconds=time.perf_counter()-before
            update()  # reference step 2
            ref=snapshot(m,opt,rng)
            before=time.perf_counter(); state=torch.load(snap,map_location='cpu',weights_only=False)
            torch.rand(7);torch.rand(7,device='cuda');rng.random(7)
            load_snapshot(state,m,opt,rng);torch.cuda.synchronize(); restore_seconds=time.perf_counter()-before
            update()  # restored step 2
            equal_tree(ref,snapshot(m,opt,rng))
            for state_row in opt.state.values():
                if 'step' in state_row and int(state_row['step']) != 2:
                    raise ValueError('cost-only optimizer step count failed restore')
            update()  # step 3
            save_folder=output/f'save_{s}';save_folder.mkdir()
            before=time.perf_counter()
            # Cost-only save: the same model export and optimizer/RNG payload
            # as a formal save, but never a counterfeit scientific checkpoint.
            m.export(save_folder/'model.json')
            torch.save({'optimizer':opt.state_dict(),'fit_numpy_rng':rng.bit_generator.state,
                        'torch_rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state_all(),
                        'completed_updates':3,'trace':[actual_trace[0],actual_trace[2],actual_trace[3]],
                        'identity':{'cost_only':True,'actual_updates':3},
                        # Serialized size allowance only; these bytes are not
                        # loss observations or completed optimization updates.
                        'trace_padding_bytes':b'0'*8192},save_folder/'cost_only_training.pt')
            _atomic_json(save_folder/'cost_only_save.json',{'cost_only':True,'actual_updates':3,
                         'executed_update_computations':4,'trace_padding_bytes':8192,
                         'files':{n:sha_file(save_folder/n) for n in ('model.json','cost_only_training.pt')}})
            torch.cuda.synchronize(); save_seconds=time.perf_counter()-before
            scores=[]
            for milestone in (128,256,512):
                before=time.perf_counter()
                with torch.no_grad(): measured=loss(m,obs,scales,deadline)
                if not bool(torch.isfinite(measured)): raise FloatingPointError('cost full objective nonfinite')
                torch.cuda.synchronize();scores.append(time.perf_counter()-before)
            # Independent timing-only observations, never formal stream1 or heldout fixture.
            before=time.perf_counter()
            tr,va,sc=timing_observations(m.mass.device,s)
            from closure_causal_recovery_20261019 import trajectory
            teacher=copy_runtime(toy_source(),[SIN_TERM],'cuda')
            with torch.no_grad():
                teacher.coefficients.fill_(.15)
                for arr,nsteps in ((tr,60),(va,110)):
                    initial=torch.cat((torch.as_tensor(arr[:,0],device='cuda'),torch.zeros((16,10),device='cuda',dtype=torch.float64)),-1)
                    trajectory(teacher,initial,nsteps,.01,deadline)
                    trajectory(teacher,initial,nsteps*2,.005,deadline)
            timed=evaluate(m,[tr,va],sc,deadline)
            _atomic_json(output/f'timing_evaluation_{s}.json',{'timing_only':True,'rows':timed})
            torch.cuda.synchronize(); eval_seconds=time.perf_counter()-before
            estimate_train=1.25*(512*max(times)+17*save_seconds+load_seconds+sum(scores)+20.)
            estimate_eval=1.25*(eval_seconds+20.)
            rows.append({'seed':s,'update_seconds':times,'max_update':max(times),'train_sha256':meta['clean_sha256'],
                         'restore_step':1,'restored_second_update_equal':True,'snapshot_bytes':snap.stat().st_size,
                         'snapshot_seconds':snapshot_seconds,'restore_seconds':restore_seconds,
                         'save_including_export_seconds':save_seconds,'load_seconds':load_seconds,
                         'export_extra_seconds':0.,'milestone_scores_seconds':scores,'eval_seconds':eval_seconds,
                         'train_estimate_seconds':estimate_train,'eval_estimate_seconds':estimate_eval,
                         'peak_cuda_bytes':torch.cuda.max_memory_allocated(),
                         'admitted':estimate_train<=640 and estimate_eval<=40})
        return {'status':'PASS' if all(r['admitted'] for r in rows) else 'REJECT',
                'seeds':rows,'cost_updates':12,'science_scores':None,'P1_complete_cap_seconds':3300,
                'supervision_reserve_seconds_per_job':20.,
                'supervision_reserve_kind':'preregistered conservative allowance; root must verify native overhead <=20',
                'envelope_total_seconds':3*700+3*100+100+800}
    if phase == 'train':
        admission=_bound_pass(binding,'admission')
        if admission.get('P1_complete_cap_seconds')!=3300: raise ValueError('P1 cap not registered')
        source, arr, scales_np, meta = _train_fixture(seed,'cuda',deadline)
        obs=torch.tensor(arr,dtype=torch.float64,device='cuda'); scales=torch.tensor(scales_np,dtype=torch.float64,device='cuda')
        model=CausalHistoryResidual.from_train(runtime_model(source,[COS_TERM],seed,'cuda'),obs[:,:11]); opt=make_optimizer(model)
        rng=np.random.default_rng(np.random.SeedSequence([seed,7])); folder=output/f'cos_{seed}'; folder.mkdir(exist_ok=False)
        identity={'source_commit':source_commit,'binding':binding,'seed':seed,'fixture':meta}
        trace=[]; milestones={}; checkpoint(folder,model,opt,rng,0,trace,identity)
        for step in range(1,513):
            if time.perf_counter() >= deadline: raise TimeoutError('training native cap')
            value,_=train_update(model,opt,rng,obs,scales,deadline); trace.append(value)
            if step in (128,256,512):
                with torch.no_grad(): milestones[str(step)]=float(loss(model,obs,scales,deadline))
            if step%32==0: checkpoint(folder,model,opt,rng,step,trace,identity)
        result={'status':'COMPLETED','seed':seed,'updates':512,'milestones':milestones,'fixture':meta,
                'model':str(folder/'checkpoint_00512/model.json'),'coefficient':float(model.base.coefficients[0]),
                'trace':trace,'source_commit':source_commit}
        _atomic_json(folder/'result.json',result)
        return result
    if phase=='freeze':
        endpoints=binding.get('endpoints',{})
        if set(endpoints)!=set(map(str,SEEDS)): raise ValueError('all three exact endpoints required')
        # Establish the complete byte barrier BEFORE any candidate train score.
        endpoint_hashes={}
        for s in SEEDS:
            directory=Path(endpoints[str(s)]).resolve()
            if directory.name!='checkpoint_00512' or directory.parent.name!=f'cos_{s}':
                raise ValueError('endpoint seed/count mismatch')
            state=torch.load(directory/'training.pt',map_location='cpu',weights_only=False)
            if state['completed_updates']!=512 or len(state['trace'])!=512 or state['identity']['seed']!=s or state['identity']['source_commit']!=source_commit:
                raise ValueError('all three endpoints must be complete and same-source before scoring')
            endpoint_hashes[s]={n:sha_file(directory/n) for n in ('model.json','training.pt')}
        rows=[]
        old_summary=json.loads((ROOT/'results/closure_mechanism_20261019/edge_causal_summary.json').read_text())
        for s in SEEDS:
            directory=Path(endpoints[str(s)]).resolve()
            if directory.name!='checkpoint_00512' or directory.parent.name!=f'cos_{s}':
                raise ValueError('endpoint seed/count mismatch')
            training=torch.load(directory/'training.pt',map_location='cpu',weights_only=False)
            if training['completed_updates']!=512 or len(training['trace'])!=512 or training['identity']['seed']!=s:
                raise ValueError('incomplete/wrong endpoint')
            if training['identity']['source_commit']!=source_commit: raise ValueError('endpoint source differs')
            _,arr,sc,meta=_train_fixture(s,'cuda',deadline)
            m=CausalHistoryResidual.from_export(directory/'model.json').cuda()
            if m.base.terms != [canonical_terms()[1]]: raise ValueError('not fixed cosine structure')
            with torch.no_grad(): score=float(loss(m,torch.as_tensor(arr,device='cuda'),torch.as_tensor(sc,device='cuda'),deadline))
            old=find_old_train(old_summary,s)
            candidates=[{'name':'sin','train_mse':old,'tree_nodes':7,'coefficients':1,'term':canonical_terms()[0]},
                        {'name':'cos','train_mse':score,'tree_nodes':5,'coefficients':1,'term':canonical_terms()[1]}]
            selected=min(candidates,key=lambda r:(r['train_mse'],r['tree_nodes'],r['coefficients'],json.dumps(r['term'],sort_keys=True)))
            if endpoint_hashes[s]!={n:sha_file(directory/n) for n in ('model.json','training.pt')}:
                raise ValueError('endpoint changed across train scoring')
            rows.append({'seed':s,'path':str(directory),'files':endpoint_hashes[s],
                         'fixture':meta,'candidates':candidates,'selected':selected['name']})
        result={'status':'COMPLETED','source_commit':source_commit,'all_three_frozen':True,'selection':'train_only',
                'old_summary_sha256':sha_file(ROOT/'results/closure_mechanism_20261019/edge_causal_summary.json'),
                'old_contract_inputs':binding['old_contract_inputs'],'rows':rows}
        _atomic_json(output/'frozen_selection.json',result)
        return result
    if phase == 'evaluate':
        # Formal val is generated only after the root supplies a frozen selection barrier.
        freeze=Path(binding.get('freeze_manifest',''))
        if not freeze.is_file() or sha_file(freeze)!=binding.get('freeze_manifest_sha256'):
            raise ValueError('missing/changed frozen train-selection manifest')
        frozen=json.loads(freeze.read_text())
        if frozen.get('all_three_frozen') is not True or frozen.get('source_commit')!=source_commit or frozen.get('selection')!='train_only':
            raise ValueError('not a complete same-source train freeze')
        if [r['seed'] for r in frozen['rows']]!=list(SEEDS): raise ValueError('freeze seeds differ')
        for row in frozen['rows']:
            for name,h in row['files'].items():
                if sha_file(Path(row['path'])/name)!=h: raise ValueError('frozen endpoint changed')
        endpoint=next(r for r in frozen['rows'] if r['seed']==seed)
        if Path(model_path).resolve()!=Path(endpoint['path'])/'model.json': raise ValueError('wrong evaluation endpoint')
        if seed not in SEEDS: return {'status':'REJECT','reason':'formal evaluation requires a registered seed'}
        _, train_arr, scales_np, train_meta = _train_fixture(seed,'cuda',deadline,formal=False)
        _, val_arr, _, meta = _train_fixture(seed,'cuda',deadline,formal=True)
        model=CausalHistoryResidual.from_export(model_path).to('cuda')
        evaluation=evaluate(model,[train_arr,val_arr],scales_np,deadline)
        evaluation['val']=evaluation.pop('heldout')
        return {'status':'COMPLETED','seed':seed,'evaluation':evaluation,'freeze_manifest_sha256':sha_file(freeze),
                'train_fixture':train_meta,'validation_fixture':meta}
    return {'status':'REJECT','reason':'unknown phase'}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=('qa', 'cost', 'train', 'freeze', 'evaluate'))
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--max-seconds', required=True, type=float)
    parser.add_argument('--source-commit', required=True)
    parser.add_argument('--binding', required=True, type=Path)
    parser.add_argument('--seed', type=int, choices=SEEDS)
    parser.add_argument('--model', type=Path)
    args = parser.parse_args(argv)
    caps={'qa':540.,'cost':540.,'train':640.,'freeze':40.,'evaluate':40.}
    if args.max_seconds != caps[args.phase] or len(args.source_commit) != 40 or any(c not in '0123456789abcdef' for c in args.source_commit):
        parser.error('positive --max-seconds and full 40-character --source-commit required')
    if args.phase == 'train' and args.seed not in SEEDS:
        parser.error('train requires --seed 5101, 5102, or 5103')
    if args.phase == 'evaluate' and args.model is None:
        parser.error('evaluate requires --model ABS')
    if args.phase == 'evaluate' and args.seed not in SEEDS:
        parser.error('evaluate requires --seed 5101, 5102, or 5103')
    result = run_phase(args.phase, args.output, args.max_seconds, args.source_commit,
                       args.binding, args.seed, args.model)
    return 0 if result.get('status') in ('PASS', 'COMPLETED') else 2


if __name__ == '__main__':
    raise SystemExit(main())
