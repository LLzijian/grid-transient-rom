"""One bounded oracle-initialization diagnostic. Never a deployment interface."""
from __future__ import annotations
import time
START=time.perf_counter()
import os,sys,json,hashlib,argparse,traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'experiments'),str(ROOT/'src')]
import numpy as np
import torch
from closure_causal_recovery_20261019 import (fixture,truths,toy_source,copy_runtime,runtime_model,
    optimizer,trajectory,save_checkpoint,atomic_json,resolve_device,synchronize)

def reconstruct(case,seed,device,deadline):
    source=toy_source();term,coefficient=truths()[case]
    truth=copy_runtime(source,[term],device)
    with torch.no_grad():truth.coefficients.fill_(coefficient)
    qs=[];digests=[]
    streams=np.random.SeedSequence([seed,20261019]).spawn(4)
    for stream,steps in zip(streams[:2],(60,110)):
        rng=np.random.default_rng(stream)
        initial=rng.normal(size=(16,40))*np.where(np.arange(16)[:,None]<8,.08,.3)
        initial[:,30:]=rng.normal(size=(16,10))*np.where(np.arange(16)[:,None]<8,.2,1.2)
        state=torch.as_tensor(initial,dtype=torch.float64,device=device);obs=[state[:,:30]]
        truth.begin_graph_cache()
        try:
            with torch.no_grad():
                for k in range(1,steps+1):
                    if time.perf_counter()>=deadline:raise TimeoutError('reconstruction cap')
                    state=truth.rk4(state,.01);obs.append(state[:,:30])
                    if k==10:qs.append(state[:,30:].clone())
        finally:truth.end_graph_cache()
        raw=torch.stack(obs,1).cpu().numpy()
        digests.append(hashlib.sha256(raw.tobytes()).hexdigest())
    return qs,digests

def oracle_prediction(model,observations,q,steps,dt,deadline):
    initial=torch.cat((observations[:,10],q),-1)
    return trajectory(model,initial,steps,dt,deadline)[:,1:]

def one(case,seed,device,folder,deadline):
    folder.mkdir();source,clean,noisy,scales,meta=fixture(case,seed,0.,device,deadline)
    qs,digests=reconstruct(case,seed,device,deadline)
    if digests!=meta['clean_sha256']:raise ValueError('reconstructed fixture hash mismatch')
    original=ROOT/f'private_data/closure_mechanism_20261019/P1/known_controls/{case}_noise0_seed{seed}'
    attempt=original/json.loads((original/'latest.json').read_text())['attempt']
    original_fixture=json.loads((attempt/'fixture.json').read_text())
    if digests!=original_fixture['clean_sha256'] or meta['scales']!=original_fixture['scales']:
        raise ValueError('oracle observations/scales differ from completed control')
    if max(meta['dual_step_normalized_max_abs'])>1e-3:raise ValueError('unqualified truth')
    atomic_json(folder/'fixture.json',dict(meta,oracle_q_time_s=.1,diagnostic_only=True,reconstructed_sha256=digests,
        original_control=str(attempt),original_fixture_sha256=hashlib.sha256((attempt/'fixture.json').read_bytes()).hexdigest()))
    model=runtime_model(source,[truths()[case][0]],seed,device)
    for name,p in model.named_parameters():
        if 'gru' in name or 'head' in name:p.requires_grad_(False)
    encoder={n:p.detach().clone() for n,p in model.named_parameters() if 'gru' in n or 'head' in n}
    opt=optimizer(model);rng=np.random.default_rng(np.random.SeedSequence([seed,7]))
    train=torch.as_tensor(noisy[0],dtype=torch.float64,device=device)
    scale=torch.as_tensor(scales,dtype=torch.float64,device=device);trace=[]
    for step in range(1,513):
        if time.perf_counter()>=deadline:raise TimeoutError('oracle fit cap; no incomplete score')
        ix=torch.as_tensor(rng.permutation(16)[:12],device=device)
        opt.zero_grad(set_to_none=True)
        prediction=oracle_prediction(model,train[ix],qs[0][ix],50,.01,deadline)
        loss=((prediction-train[ix,11:])/scale).square().mean()
        if not bool(torch.isfinite(loss)):raise FloatingPointError('loss')
        loss.backward()
        pars=[p for p in model.parameters() if p.requires_grad]
        if any(p.grad is not None and not bool(torch.isfinite(p.grad).all()) for p in pars):raise FloatingPointError('gradient')
        torch.nn.utils.clip_grad_norm_(pars,100.);opt.step();synchronize(device)
        if any(not bool(torch.isfinite(p).all()) for p in pars):raise FloatingPointError('updated parameters')
        trace.append(float(loss.detach()))
        if step%32==0:save_checkpoint(folder,model,opt,rng,step,trace)
    errors=[];differences=[]
    for arr,q,horizon in zip(noisy,qs,(50,100)):
        obs=torch.as_tensor(arr,dtype=torch.float64,device=device)
        with torch.no_grad():
            coarse=oracle_prediction(model,obs,q,horizon,.01,deadline)
            fine=oracle_prediction(model,obs,q,horizon*2,.005,deadline)[:,1::2]
        if not bool(torch.isfinite(coarse).all() and torch.isfinite(fine).all()):raise FloatingPointError('evaluation')
        errors.append(((coarse-obs[:,11:])/scale).square().mean((1,2)).cpu().tolist())
        differences.append(float(((coarse-fine)/scale).abs().max()))
    for n,p in model.named_parameters():
        if n in encoder and not torch.equal(p,encoder[n]):raise ValueError('encoder changed')
    qualified=max(differences)<=1e-3
    result={'case':case,'seed':seed,'noise':0.,'updates':512,'status':'qualified' if qualified else 'numerically_unresolved',
        'train_event_mse':errors[0],'heldout_event_mse':errors[1],
        'train_mse':float(np.mean(errors[0])),'heldout_mse':float(np.mean(errors[1])),
        'model_dual_step_max_normalized':differences,'prediction_gate':qualified and np.mean(errors[1])<=.01,
        'diagnostic_only':True,'q_supplied_once_at_time_s':.1,'future_q_supervision':False,'encoder_updated':False}
    atomic_json(folder/'result.json',result);return result

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--max-seconds',type=float,default=4500.);p.add_argument('--device',default='cuda')
    a=p.parse_args()
    if not 0<a.max_seconds<=4500:p.error('finite cap<=4500 required')
    a.output.mkdir(parents=True,exist_ok=False);torch.set_num_threads(1);device=resolve_device(a.device)
    state={'status':'RUNNING','pid':os.getpid(),'rows':[],'max_seconds':a.max_seconds,'diagnostic_only':True,
        'code_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'command':sys.argv}
    state['dependencies']={name:hashlib.sha256((ROOT/'experiments'/name).read_bytes()).hexdigest() for name in
        ('closure_causal_recovery_20261019.py','closure_training_runtime.py','symbolic_closure_rom.py','closure_discovery_benchmarks.py')}
    atomic_json(a.output/'state.json',state);deadline=START+a.max_seconds
    try:
        for case in ('node','edge','saturation'):
            for seed in (5101,5102,5103):
                if time.perf_counter()>=deadline:raise TimeoutError('diagnostic batch cap')
                folder=a.output/f'{case}_{seed}';state['current']=folder.name;atomic_json(a.output/'state.json',state)
                try:row=one(case,seed,device,folder,deadline)
                except Exception as error:
                    row={'case':case,'seed':seed,'status':'unresolved','error':repr(error),'traceback':traceback.format_exc()}
                    folder.mkdir(exist_ok=True);atomic_json(folder/'failure.json',row)
                state['rows'].append(row);atomic_json(a.output/'state.json',state)
        state['status']='COMPLETED'
    except BaseException as error:state.update(status='INTERRUPTED',error=repr(error));raise
    finally:
        state['elapsed_seconds']=time.perf_counter()-START;atomic_json(a.output/'state.json',state)
        atomic_json(a.output/'receipt.json',{'stage':'P1','seconds':state['elapsed_seconds'],'status':state['status'],'includes_children':False})

if __name__=='__main__':main()
