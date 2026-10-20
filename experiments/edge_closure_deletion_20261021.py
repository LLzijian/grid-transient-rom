"""Registered empty-edge deletion/refit control; no project data readers.

Numerical imports occur only after the worker receipt and watchdog exist.
Historical implementations and all historical endpoints remain read-only.
"""
from __future__ import annotations
import argparse
import copy
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time

START = time.perf_counter()
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'experiments'), str(ROOT/'src')]
import edge_memory_discrimination_20261021 as prior
SEEDS = prior.SEEDS
CAPS = {'qa':210., 'cost':210., 'train':450., 'freeze':40., 'evaluate':35.}
EMPTY = {'terms':[], 'closure_coefficients':0}


def make_model(source, obs, seed):
    from closure_causal_recovery_20261019 import runtime_model
    from closure_causal_history_residual_20261019 import CausalHistoryResidual
    return CausalHistoryResidual.from_train(runtime_model(source, [], seed, 'cuda'), obs[:,:11])


def assert_empty(model):
    if model.base.terms != [] or model.base.coefficients.numel() != 0:
        raise ValueError('empty closure identity required')


def verify_contract(binding, deadline):
    """Exact common stage0 identities, allowing only the registered empty tensor."""
    import numpy as np
    import torch
    from closure_causal_history_residual_20261019 import CausalHistoryResidual, make_optimizer
    table = json.loads((ROOT/binding['old_contract_inputs']['path']).read_text(encoding='utf-8'))
    for path, digest in table['files'].items():
        if prior.sha_file(ROOT/path) != digest:
            raise ValueError('historical evidence changed: '+path)
    rows=[]
    for seed in SEEDS:
        source, arr, scales, meta=prior._train_fixture(seed,'cuda',deadline)
        folder=ROOT/f'private_data/closure_mechanism_20261019/P1/history_residual/edge_run01/edge_{seed}_residual'
        old_fixture=json.loads((folder/'fixture.json').read_text(encoding='utf-8'))
        import hashlib
        identity=hashlib.sha256(json.dumps(source._export_payload(),sort_keys=True,allow_nan=False).encode()).hexdigest()
        if identity!=old_fixture['source_model_sha256']: raise ValueError('unfitted source differs')
        obs=torch.as_tensor(arr,device='cuda',dtype=torch.float64)
        model=make_model(source,obs,seed); assert_empty(model)
        opt=make_optimizer(model); rng=np.random.default_rng(np.random.SeedSequence([seed,7]))
        captured=prior.snapshot(model,opt,rng)
        old=CausalHistoryResidual.from_export(folder/'checkpoint_00000/model.json').cuda()
        state=torch.load(folder/'checkpoint_00000/training.pt',map_location='cpu',weights_only=False)
        left=dict(model.state_dict()); right=dict(old.state_dict())
        if left.keys()!=right.keys(): raise ValueError('common state names differ')
        for name in left:
            if name!='base.coefficients': prior.equal_tree(left[name],right[name])
        if right['base.coefficients'].numel()!=1 or bool(right['base.coefficients'].any()):
            raise ValueError('old closure initial identity differs')
        if [(n,p.requires_grad) for n,p in model.named_parameters()]!=[(n,p.requires_grad) for n,p in old.named_parameters()]:
            raise ValueError('parameter order/flags differ')
        # The empty tensor occupies the same parameter/group slot. IDs and
        # group options therefore remain exact; its changed shape is explicit.
        prior.equal_tree(captured['optimizer'],state['optimizer'])
        for key, oldkey in [('numpy','fit_numpy_rng'),('cpu','torch_rng'),('cuda','cuda_rng')]:
            prior.equal_tree(captured[key],state[oldkey])
        if state['completed_updates']!=0 or state['trace'] or opt.state or bool(model.history_coefficients.any()):
            raise ValueError('not an unfitted empty-Adam common initialization')
        # Initial field and common raw gradients, before global clipping.
        from closure_causal_history_residual_20261019 import loss
        z=torch.cat((obs[:2,10],model.encode(obs[:2,:11])),-1)
        torch.testing.assert_close(model.field(z),old.field(z),rtol=2e-11,atol=1e-11)
        loss(model,obs[:2],torch.as_tensor(scales,device='cuda'),deadline).backward()
        loss(old,obs[:2],torch.as_tensor(scales,device='cuda'),deadline).backward()
        old_parameters=dict(old.named_parameters())
        for name,p in model.named_parameters():
            if name=='base.coefficients': continue
            q=old_parameters[name]
            if (p.grad is None)!=(q.grad is None): raise ValueError('common raw gradient presence differs')
            if p.grad is not None: torch.testing.assert_close(p.grad,q.grad,rtol=2e-9,atol=2e-9)
        rows.append({'seed':seed,'common_stage0_exact':True,'rng_exact':True,
                     'empty_adam':True,'parameter_group_mapping_exact':True,'fixture':meta,**EMPTY})
    return rows


def validate_endpoint(directory, seed, source_commit, state, payload):
    directory=Path(directory).resolve()
    allowed=(ROOT/'private_data/closure_mechanism_20261019/P1/edge_closure_deletion01').resolve()
    if (not directory.is_relative_to(allowed) or directory.name!='checkpoint_00512'
            or directory.parent.name!=f'empty_{seed}'):
        raise ValueError('wrong endpoint seed/count/location')
    if (state.get('completed_updates')!=512 or len(state.get('trace',[]))!=512
            or state.get('identity',{}).get('seed')!=seed
            or state.get('identity',{}).get('source_commit')!=source_commit):
        raise ValueError('incomplete/wrong-source endpoint')
    if payload.get('base',{}).get('terms')!=[] or payload['base']['state_dict'].get('coefficients')!=[]:
        raise ValueError('endpoint has a closure term/coefficient')


def barrier(binding, source_commit):
    import torch
    endpoints=binding.get('endpoints',{})
    if set(endpoints)!=set(map(str,SEEDS)): raise ValueError('all three endpoints required')
    rows=[]
    for seed in SEEDS:
        directory=Path(endpoints[str(seed)]).resolve()
        state=torch.load(directory/'training.pt',map_location='cpu',weights_only=False)
        payload=json.loads((directory/'model.json').read_text(encoding='utf-8'))
        validate_endpoint(directory,seed,source_commit,state,payload)
        rows.append({'seed':seed,'path':str(directory),
                     'files':{n:prior.sha_file(directory/n) for n in ('model.json','training.pt')}})
    return rows


def validate_frozen(frozen, source_commit):
    if (frozen.get('all_three_frozen') is not True or frozen.get('selection')!='train_only'
            or frozen.get('source_commit')!=source_commit
            or [r.get('seed') for r in frozen.get('rows',[])]!=list(SEEDS)):
        raise ValueError('missing complete same-source freeze barrier')
    for row in frozen['rows']:
        if set(row.get('files',{}))!={'model.json','training.pt'}: raise ValueError('missing endpoint bytes')
        for name,digest in row['files'].items():
            if prior.sha_file(Path(row['path'])/name)!=digest: raise ValueError('frozen endpoint changed')


def cost(output,deadline,binding,commit):
    import numpy as np
    import torch
    from closure_causal_history_residual_20261019 import make_optimizer,train_update,loss,evaluate,checkpoint
    from closure_causal_recovery_20261019 import copy_runtime,trajectory
    from closure_discovery_benchmarks import toy_source
    qa=prior._bound_pass(binding,'qa_result')
    if len(qa.get('contracts',[]))!=3: raise ValueError('missing actual contracts')
    rows=[]
    for seed in SEEDS:
        start=time.perf_counter()
        source,arr,sc,meta=prior._train_fixture(seed,'cuda',deadline)
        obs=torch.as_tensor(arr,device='cuda'); scales=torch.as_tensor(sc,device='cuda')
        m=make_model(source,obs,seed); assert_empty(m);opt=make_optimizer(m)
        rng=np.random.default_rng(np.random.SeedSequence([seed,7]))
        torch.cuda.synchronize();load_seconds=time.perf_counter()-start
        torch.cuda.reset_peak_memory_stats();times=[];trace=[]
        def update():
            t=time.perf_counter();v,_=train_update(m,opt,rng,obs,scales,deadline)
            torch.cuda.synchronize();times.append(time.perf_counter()-t);trace.append(v)
        update();start=time.perf_counter();snap=output/f'snapshot_{seed}.pt'
        torch.save(prior.snapshot(m,opt,rng),snap);torch.cuda.synchronize();snapshot_seconds=time.perf_counter()-start
        update();reference=prior.snapshot(m,opt,rng)
        start=time.perf_counter();state=torch.load(snap,map_location='cpu',weights_only=False)
        torch.rand(7);torch.rand(7,device='cuda');rng.random(7)
        prior.load_snapshot(state,m,opt,rng);torch.cuda.synchronize();restore_seconds=time.perf_counter()-start
        update();prior.equal_tree(reference,prior.snapshot(m,opt,rng));update()
        folder=output/f'save_{seed}';folder.mkdir()
        identity={'source_commit':commit,'binding':binding,'seed':seed,'fixture':meta,
                  **EMPTY,'cost_only':True,'actual_updates':3}
        nominal_identity_bytes=len(json.dumps(identity,sort_keys=True,allow_nan=False).encode('utf-8'))
        identity['cost_padding_bytes']=b'0'*65536
        start=time.perf_counter()
        checkpoint(folder,m,opt,rng,3,[trace[0],trace[2],trace[3]],identity)
        torch.cuda.synchronize();save_seconds=time.perf_counter()-start
        saved=folder/'checkpoint_00003'
        cost_save={'cost_only':True,'actual_updates':3,'executed_update_computations':4,
                   'identity_size_allowance_bytes':65536,
                   'cost_identity_nominal_json_bytes':nominal_identity_bytes,
                   'checkpoint':str(saved),
                   'files':{n:{'path':str(saved/n),'bytes':(saved/n).stat().st_size,
                               'sha256':prior.sha_file(saved/n)} for n in ('model.json','training.pt')}}
        prior._atomic_json(folder/'cost_only_save.json',cost_save)
        scores=[]
        for milestone in (128,256,512):
            start=time.perf_counter()
            with torch.no_grad(): value=loss(m,obs,scales,deadline)
            if not bool(torch.isfinite(value)): raise FloatingPointError('cost objective nonfinite')
            torch.cuda.synchronize();scores.append(time.perf_counter()-start)
        start=time.perf_counter();tr,va,scale=prior.timing_observations('cuda',seed)
        teacher=copy_runtime(toy_source(),[prior.SIN_TERM],'cuda')
        with torch.no_grad():
            teacher.coefficients.fill_(.15)
            for a,n in ((tr,60),(va,110)):
                initial=torch.cat((torch.as_tensor(a[:,0],device='cuda'),torch.zeros((16,10),device='cuda',dtype=torch.float64)),-1)
                trajectory(teacher,initial,n,.01,deadline);trajectory(teacher,initial,2*n,.005,deadline)
        timed=evaluate(m,[tr,va],scale,deadline)
        prior._atomic_json(output/f'timing_{seed}.json',{'timing_only':True,'rows':timed})
        torch.cuda.synchronize();eval_seconds=time.perf_counter()-start
        train_estimate=1.25*(512*max(times)+17*save_seconds+load_seconds+sum(scores)+20.)
        eval_estimate=1.25*(eval_seconds+20.)
        rows.append({'seed':seed,'update_seconds':times,'max_update':max(times),
                     'train_sha256':meta['clean_sha256'],'snapshot_seconds':snapshot_seconds,
                     'restore_seconds':restore_seconds,'restored_second_update_equal':True,
                     'save_including_export_seconds':save_seconds,'load_seconds':load_seconds,
                     'cost_checkpoint':cost_save,'identity_size_allowance_bytes':65536,
                     'cost_training_bytes':(saved/'training.pt').stat().st_size,
                     'cost_identity_nominal_json_bytes':nominal_identity_bytes,
                     'export_extra_seconds':0.,'milestone_scores_seconds':scores,'eval_seconds':eval_seconds,
                     'train_estimate_seconds':train_estimate,'eval_estimate_seconds':eval_estimate,
                     'peak_cuda_bytes':torch.cuda.max_memory_allocated(),
                     'admitted':train_estimate<=450 and eval_estimate<=35,**EMPTY})
    return {'status':'PASS' if all(r['admitted'] for r in rows) else 'REJECT','seeds':rows,
            'cost_updates':12,'science_scores':None,'P1_complete_cap_seconds':2450,
            'envelope_total_seconds':2450,'supervision_reserve_seconds_per_job':20.}


def execute(phase,output,deadline,commit,binding,seed,model_path):
    import numpy as np
    import torch
    from closure_causal_history_residual_20261019 import CausalHistoryResidual,make_optimizer,train_update,checkpoint,loss,evaluate
    torch.set_num_threads(1)
    if phase=='qa':
        contracts=verify_contract(binding,deadline)
        proc=subprocess.run([sys.executable,'-m','pytest','-q',str(ROOT/'tests/test_edge_closure_deletion_20261021.py')],
                            cwd=ROOT,capture_output=True,text=True,timeout=max(.1,deadline-time.perf_counter()))
        (output/'pytest_stdout.txt').write_text(proc.stdout,encoding='utf-8')
        (output/'pytest_stderr.txt').write_text(proc.stderr,encoding='utf-8')
        return {'status':'PASS' if proc.returncode==0 else 'FAIL','contracts':contracts,
                'pytest_returncode':proc.returncode,'stdout':proc.stdout,'stderr':proc.stderr,**EMPTY}
    if phase=='cost': return cost(output,deadline,binding,commit)
    if phase=='train':
        admission=prior._bound_pass(binding,'admission')
        if admission.get('P1_complete_cap_seconds')!=2450: raise ValueError('P1 admission cap differs')
        source,arr,sc,meta=prior._train_fixture(seed,'cuda',deadline)
        obs=torch.as_tensor(arr,device='cuda');scales=torch.as_tensor(sc,device='cuda')
        m=make_model(source,obs,seed);assert_empty(m);opt=make_optimizer(m)
        rng=np.random.default_rng(np.random.SeedSequence([seed,7]))
        folder=output/f'empty_{seed}';folder.mkdir()
        identity={'source_commit':commit,'binding':binding,'seed':seed,'fixture':meta,**EMPTY}
        trace=[];milestones={};checkpoint(folder,m,opt,rng,0,trace,identity)
        for step in range(1,513):
            if time.perf_counter()>=deadline: raise TimeoutError('training cap')
            value,_=train_update(m,opt,rng,obs,scales,deadline);trace.append(value)
            if step in (128,256,512):
                with torch.no_grad(): milestones[str(step)]=float(loss(m,obs,scales,deadline))
            if step%32==0: checkpoint(folder,m,opt,rng,step,trace,identity)
        result={'status':'COMPLETED','seed':seed,'updates':512,'milestones':milestones,
                'fixture':meta,'model':str(folder/'checkpoint_00512/model.json'),'trace':trace,**EMPTY}
        prior._atomic_json(folder/'result.json',result);return result
    if phase=='freeze':
        rows=barrier(binding,commit)  # byte barrier before any scoring
        oldref=binding.get('old_selection')
        if not isinstance(oldref,dict) or prior.sha_file(ROOT/oldref['path'])!=oldref['sha256']:
            raise ValueError('missing bound old two-structure train table')
        previous=json.loads((ROOT/oldref['path']).read_text(encoding='utf-8'))
        validate_frozen(previous,previous.get('source_commit'))
        for row in rows:
            s=row['seed'];_,arr,sc,meta=prior._train_fixture(s,'cuda',deadline)
            m=CausalHistoryResidual.from_export(Path(row['path'])/'model.json').cuda();assert_empty(m)
            with torch.no_grad(): score=float(loss(m,torch.as_tensor(arr,device='cuda'),torch.as_tensor(sc,device='cuda'),deadline))
            candidates=copy.deepcopy(next(r for r in previous['rows'] if r['seed']==s)['candidates'])
            if {r['name'] for r in candidates}!={'sin','cos'}: raise ValueError('old candidates differ')
            candidates.append({'name':'empty','train_mse':score,'tree_nodes':0,'coefficients':0,'term':None})
            row.update(candidates=candidates,fixture=meta,selected=min(candidates,key=lambda r:(r['train_mse'],r['tree_nodes'],r['coefficients'],r['name']))['name'])
        result={'status':'COMPLETED','source_commit':commit,'all_three_frozen':True,
                'selection':'train_only','rows':rows,'old_selection':oldref,**EMPTY}
        validate_frozen(result,commit);prior._atomic_json(output/'frozen_selection.json',result);return result
    if phase=='evaluate':
        manifest=Path(binding.get('freeze_manifest',''))
        if not manifest.is_file() or prior.sha_file(manifest)!=binding.get('freeze_manifest_sha256'):
            raise ValueError('missing frozen selection identity')
        frozen=json.loads(manifest.read_text(encoding='utf-8'));validate_frozen(frozen,commit)
        # Revalidate semantic identities as well as all three frozen byte pairs.
        barrier({'endpoints':{str(r['seed']):r['path'] for r in frozen['rows']}},commit)
        row=next(r for r in frozen['rows'] if r['seed']==seed)
        if Path(model_path).resolve()!=Path(row['path'])/'model.json': raise ValueError('wrong evaluation model')
        _,tr,sc,tmeta=prior._train_fixture(seed,'cuda',deadline)
        _,va,_,vmeta=prior._train_fixture(seed,'cuda',deadline,formal=True)
        m=CausalHistoryResidual.from_export(model_path).cuda();assert_empty(m)
        scores=evaluate(m,[tr,va],sc,deadline);scores['val']=scores.pop('heldout')
        return {'status':'COMPLETED','seed':seed,'evaluation':scores,'train_fixture':tmeta,
                'validation_fixture':vmeta,'freeze_manifest_sha256':prior.sha_file(manifest),**EMPTY}
    raise ValueError('unknown phase')


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase',choices=CAPS)
    for name in ('output','binding'): parser.add_argument('--'+name,required=True,type=Path)
    parser.add_argument('--source-commit',required=True)
    parser.add_argument('--max-seconds',required=True,type=float)
    parser.add_argument('--seed',choices=SEEDS,type=int)
    parser.add_argument('--model',type=Path)
    a=parser.parse_args(argv)
    if a.max_seconds!=CAPS[a.phase] or len(a.source_commit)!=40 or any(c not in '0123456789abcdef' for c in a.source_commit):
        parser.error('exact registered cap and full source commit required')
    if a.phase in ('train','evaluate') and a.seed not in SEEDS: parser.error('registered seed required')
    if a.phase=='evaluate' and a.model is None: parser.error('model required')
    output=a.output.resolve();category='P0' if a.phase in ('qa','cost') else 'P1'
    allowed=(ROOT/'private_data/closure_mechanism_20261019'/category/'edge_closure_deletion01').resolve()
    if not output.is_relative_to(allowed) or output==allowed: raise ValueError('output outside exclusive stage directory')
    output.mkdir(parents=True,exist_ok=False)
    prior._worker_receipt(output,a.phase,a.source_commit,a.binding)
    deadline=START+a.max_seconds
    def timeout():
        prior._atomic_json(output/'timeout.json',{'status':'TIMEOUT','max_seconds':a.max_seconds});os._exit(124)
    timer=threading.Timer(max(.001,deadline-time.perf_counter()),timeout);timer.daemon=True;timer.start()
    try:
        binding=prior._read_binding(a.binding)
        old=binding['old_contract_inputs']
        expected=ROOT/'private_data/closure_mechanism_20261019/P0/edge_memory_discrimination01/old_contract_inputs.json'
        if ((ROOT/old['path']).resolve()!=expected.resolve()
                or old['sha256']!='6af592c1374f0c1aed1e55b0c7c239ba2b4b0fc4b5356e092bf9269525314576'):
            raise ValueError('old-contract table is outside the frozen parent allowlist')
        result=execute(a.phase,output,deadline,a.source_commit,binding,a.seed,a.model)
        result.update(source_commit=a.source_commit,worker_phase_elapsed_seconds=time.perf_counter()-START)
        prior._atomic_json(output/'result.json',result)
        prior._atomic_json(output/'complete.json',{'status':'COMPLETED','result':result})
        return 0 if result['status'] in ('PASS','COMPLETED') else 2
    except BaseException as e:
        import traceback
        prior._atomic_json(output/'failure.json',{'status':'UNRESOLVED','error':repr(e),'traceback':traceback.format_exc()});raise
    finally: timer.cancel()


if __name__=='__main__': raise SystemExit(main())
