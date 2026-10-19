"""One bounded oracle-initialization diagnostic. Never a deployment interface."""
from __future__ import annotations
import time
START=time.perf_counter()
import os,sys,json,hashlib,argparse,traceback,uuid,math
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'experiments'),str(ROOT/'src')]
import numpy as np
import torch
from closure_causal_recovery_20261019 import (fixture,truths,toy_source,copy_runtime,runtime_model,
    optimizer,trajectory,save_checkpoint,atomic_json,resolve_device,synchronize)

ORIGINAL_CODE_SHA='44cb9d96f28fb2f39a18565be978cf3d44645a171483e4d25bb7b34de9a81133'
CONFIG={'updates':512,'checkpoint_every':32,'noise':0.,'oracle_q_time_s':.1,
        'encoder_updated':False,'future_q_supervision':False,'diagnostic_only':True}

def restore_checkpoint(folder,model,opt,rng,device):
    """Restore a completed boundary; never guess from pending directories."""
    latest=json.loads((folder/'latest.json').read_text())
    step=latest['updates'];name=latest['checkpoint']
    if type(step) is not int or step not in range(32,513,32):raise ValueError('invalid checkpoint updates')
    if name!=f'checkpoint_{step:05d}':raise ValueError('checkpoint identity mismatch')
    checkpoint=folder/name
    if checkpoint.resolve().parent!=folder.resolve():raise ValueError('checkpoint must be immediate child')
    restored=type(model).from_export(checkpoint/'model.json').to(device)
    if restored.terms!=model.terms:raise ValueError('checkpoint terms changed')
    model.load_state_dict(restored.state_dict())
    state=torch.load(checkpoint/'training.pt',map_location='cpu',weights_only=False)
    if state['completed_updates']!=step or len(state['trace'])!=step:raise ValueError('training count mismatch')
    if not all(math.isfinite(v) for v in state['trace']):raise ValueError('nonfinite checkpoint trace')
    opt.load_state_dict(state['optimizer'])
    rng.bit_generator.state=state['fit_numpy_rng'];torch.set_rng_state(state['torch_rng'])
    cuda=state['cuda_rng']
    if torch.device(device).type=='cuda':
        if len(cuda)!=torch.cuda.device_count():raise ValueError('CUDA RNG device mismatch')
        torch.cuda.set_rng_state_all(cuda)
    elif cuda:raise ValueError('CUDA checkpoint requires CUDA continuation')
    return step,list(state['trace'])

def recovery_spend(output,previous,current_sha,dependencies,cap,extra):
    if previous['code_sha256'] not in (ORIGINAL_CODE_SHA,current_sha):raise ValueError('unknown code lineage')
    if previous['dependencies']!=dependencies:raise ValueError('dependency identity changed')
    if previous['max_seconds']!=cap:raise ValueError('cannot reset or change original cap')
    manifest_path=output/'recovery_checkpoint_manifest.json'
    if previous['code_sha256']==ORIGINAL_CODE_SHA:
        if not manifest_path.is_file():raise ValueError('original recovery requires checkpoint byte manifest')
        manifest=json.loads(manifest_path.read_text())
        if manifest['original_code_sha256']!=ORIGINAL_CODE_SHA:raise ValueError('manifest lineage mismatch')
        expected={f'{case}_{seed}' for case in ('node','edge','saturation') for seed in (5101,5102,5103)
                  if (output/f'{case}_{seed}'/'latest.json').exists()}
        declared=[row['folder'] for row in manifest['rows']]
        if set(declared)!=expected or len(declared)!=len(expected):raise ValueError('manifest checkpoint coverage mismatch')
        for row in manifest['rows']:
            if set(row['files'])!={'model.json','training.pt'}:raise ValueError('manifest must bind both checkpoint files')
            folder=output/row['folder']
            if folder.resolve().parent!=output.resolve():raise ValueError('manifest folder escaped output')
            pointer=json.loads((folder/'latest.json').read_text())
            if pointer['checkpoint']!=row['checkpoint'] or pointer['updates']!=row['updates']:
                raise ValueError('manifest checkpoint changed')
            checkpoint=folder/row['checkpoint']
            if checkpoint.resolve().parent!=folder.resolve():raise ValueError('manifest checkpoint escaped folder')
            for name,expected in row['files'].items():
                if name not in ('model.json','training.pt'):raise ValueError('unknown manifest file')
                if hashlib.sha256((checkpoint/name).read_bytes()).hexdigest()!=expected:
                    raise ValueError('checkpoint bytes changed')
    if not math.isfinite(extra) or extra<0:raise ValueError('invalid recovery cost')
    if previous['status']=='INTERRUPTED':prior=previous['elapsed_seconds']
    elif previous['status']=='RUNNING':
        interruption=json.loads((output/'runtime_interruption.json').read_text(encoding='utf-8-sig'))
        if interruption['pid']!=previous['pid'] or interruption['current']!=previous['current']:
            raise ValueError('unknown interruption identity')
        prior=previous.get('prior_elapsed_seconds',0.)+interruption['elapsed_seconds']
    else:raise ValueError('only interrupted runs can resume')
    if not math.isfinite(prior) or prior<=0 or prior+extra>=cap:raise ValueError('no verified remaining cap')
    return prior+extra

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

def one(case,seed,device,folder,deadline,resume=False):
    folder.mkdir(exist_ok=resume);source,clean,noisy,scales,meta=fixture(case,seed,0.,device,deadline)
    qs,digests=reconstruct(case,seed,device,deadline)
    if digests!=meta['clean_sha256']:raise ValueError('reconstructed fixture hash mismatch')
    original=ROOT/f'private_data/closure_mechanism_20261019/P1/known_controls/{case}_noise0_seed{seed}'
    attempt=original/json.loads((original/'latest.json').read_text())['attempt']
    original_fixture=json.loads((attempt/'fixture.json').read_text())
    if digests!=original_fixture['clean_sha256'] or meta['scales']!=original_fixture['scales']:
        raise ValueError('oracle observations/scales differ from completed control')
    if max(meta['dual_step_normalized_max_abs'])>1e-3:raise ValueError('unqualified truth')
    fixture_record=dict(meta,oracle_q_time_s=.1,diagnostic_only=True,reconstructed_sha256=digests,
        original_control=str(attempt),original_fixture_sha256=hashlib.sha256((attempt/'fixture.json').read_bytes()).hexdigest())
    if (folder/'fixture.json').exists():
        if json.loads((folder/'fixture.json').read_text())!=fixture_record:raise ValueError('fixture identity changed')
    else:atomic_json(folder/'fixture.json',fixture_record)
    identity={'case':case,'seed':seed,'config':CONFIG,
              'fixture_sha256':hashlib.sha256((folder/'fixture.json').read_bytes()).hexdigest()}
    if (folder/'result.json').exists():
        result=json.loads((folder/'result.json').read_text())
        if result.get('identity')!=identity or result.get('updates')!=512 or type(result.get('prediction_gate')) is not bool:
            raise ValueError('existing result identity invalid')
        return result
    model=runtime_model(source,[truths()[case][0]],seed,device)
    for name,p in model.named_parameters():
        if 'gru' in name or 'head' in name:p.requires_grad_(False)
    encoder={n:p.detach().clone() for n,p in model.named_parameters() if 'gru' in n or 'head' in n}
    opt=optimizer(model);rng=np.random.default_rng(np.random.SeedSequence([seed,7]))
    train=torch.as_tensor(noisy[0],dtype=torch.float64,device=device)
    scale=torch.as_tensor(scales,dtype=torch.float64,device=device);trace=[];completed=0
    if resume and (folder/'latest.json').exists():completed,trace=restore_checkpoint(folder,model,opt,rng,device)
    elif resume and any(folder.glob('checkpoint_*')):raise ValueError('checkpoint exists without authoritative latest pointer')
    for n,p in model.named_parameters():
        if n in encoder and not torch.equal(p,encoder[n]):raise ValueError('checkpoint encoder changed')
    for step in range(completed+1,513):
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
        'model_dual_step_max_normalized':differences,'prediction_gate':bool(qualified and np.mean(errors[1])<=.01),
        'diagnostic_only':True,'q_supplied_once_at_time_s':.1,'future_q_supervision':False,'encoder_updated':False,
        'identity':identity}
    atomic_json(folder/'result.json',result);return result

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--max-seconds',type=float,default=4500.);p.add_argument('--device',default='cuda')
    p.add_argument('--resume',action='store_true')
    p.add_argument('--recovery-cost-seconds',type=float,default=0.,help='additional measured recovery/QA cost, charged once on this invocation')
    a=p.parse_args()
    if not 0<a.max_seconds<=4500:p.error('finite cap<=4500 required')
    if not a.resume and a.recovery_cost_seconds!=0:p.error('recovery cost requires --resume')
    a.output.mkdir(parents=True,exist_ok=a.resume);torch.set_num_threads(1);device=resolve_device(a.device)
    state={'status':'RUNNING','pid':os.getpid(),'rows':[],'max_seconds':a.max_seconds,'diagnostic_only':True,
        'code_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'command':sys.argv}
    state['dependencies']={name:hashlib.sha256((ROOT/'experiments'/name).read_bytes()).hexdigest() for name in
        ('closure_causal_recovery_20261019.py','closure_training_runtime.py','symbolic_closure_rom.py','closure_discovery_benchmarks.py')}
    prior=0.
    if a.resume:
        previous=json.loads((a.output/'state.json').read_text())
        prior=recovery_spend(a.output,previous,state['code_sha256'],state['dependencies'],a.max_seconds,a.recovery_cost_seconds)
        recovery=a.output/('recovery_'+uuid.uuid4().hex);recovery.mkdir()
        atomic_json(recovery/'previous_state.json',previous)
        for name in ('receipt.json','runtime_interruption.json'):
            if (a.output/name).exists():(recovery/name).write_bytes((a.output/name).read_bytes())
        atomic_json(recovery/'adoption.json',{'original_code_sha256':ORIGINAL_CODE_SHA,
            'repair_code_sha256':state['code_sha256'],'prior_elapsed_seconds':prior,
            'additional_recovery_cost_seconds':a.recovery_cost_seconds,'remaining_seconds':a.max_seconds-prior})
    state['prior_elapsed_seconds']=prior
    atomic_json(a.output/'state.json',state);deadline=START+a.max_seconds-prior
    try:
        for case in ('node','edge','saturation'):
            for seed in (5101,5102,5103):
                if time.perf_counter()>=deadline:raise TimeoutError('diagnostic batch cap')
                folder=a.output/f'{case}_{seed}';state['current']=folder.name;atomic_json(a.output/'state.json',state)
                try:row=one(case,seed,device,folder,deadline,resume=a.resume and folder.exists())
                except Exception as error:
                    row={'case':case,'seed':seed,'status':'unresolved','error':repr(error),'traceback':traceback.format_exc()}
                    folder.mkdir(exist_ok=True)
                    failure=folder/'failure.json'
                    if failure.exists():failure=folder/('failure_recovery_'+uuid.uuid4().hex+'.json')
                    atomic_json(failure,row)
                state['rows'].append(row);atomic_json(a.output/'state.json',state)
        state['status']='COMPLETED'
    except BaseException as error:state.update(status='INTERRUPTED',error=repr(error));raise
    finally:
        run_seconds=time.perf_counter()-START
        state['elapsed_seconds']=prior+run_seconds;atomic_json(a.output/'state.json',state)
        atomic_json(a.output/'receipt.json',{'stage':'P1','seconds':run_seconds,'cumulative_seconds':state['elapsed_seconds'],
            'prior_elapsed_seconds':prior,'status':state['status'],'includes_children':False})

if __name__=='__main__':main()
