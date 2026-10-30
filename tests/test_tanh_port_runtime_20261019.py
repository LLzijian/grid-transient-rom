"""Fixed P0 equivalence, causal loss, Adam recovery and reference export checks."""
from __future__ import annotations
import copy
import io
import json
from pathlib import Path
import sys
import time
import traceback
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'experiments'),str(ROOT/'src')]
from benchmark_tanh_port_20261019 import parser, supervised


def run_checks(output, device='cuda', max_seconds=1800):
    import numpy as np
    import torch
    from closure_training_runtime import TrainingClosureROM, resolve_device, synchronize
    from closure_causal_history_residual_20261019 import CausalHistoryResidual, loss, make_optimizer
    from tanh_port_runtime_20261019 import select_runtime, TERM
    from benchmark_tanh_port_20261019 import engineering_inputs,modes,update
    torch.set_num_threads(1)
    device=resolve_device(device)
    deadline=time.perf_counter()+max_seconds
    output=Path(output)
    report={'status':'running','field_trajectory_loss_tolerance':{'atol':1e-11,'rtol':1e-9},
            'gradient_update_state_tolerance':{'atol':1e-10,'rtol':1e-8}, 'checks':[], 'max_abs_by_check':{}}
    def check(a,b,name,gradient=False):
        if time.perf_counter()>=deadline:
            raise TimeoutError('registered engineering deadline')
        if a is None or b is None:
            assert a is None and b is None, name+' None pattern'
        elif torch.is_tensor(a):
            assert a.shape==b.shape and a.dtype==b.dtype, name+' shape/dtype'
            atol,rtol=(1e-10,1e-8) if gradient else (1e-11,1e-9)
            torch.testing.assert_close(a,b,atol=atol,rtol=rtol,msg=lambda msg:name+': '+msg)
            delta=float((a-b).detach().abs().max()) if a.numel() else 0.
            report['max_abs_by_check'][name]=delta
        elif isinstance(a,dict):
            assert a.keys()==b.keys(),name+' keys'
            for k in a:check(a[k],b[k],name+'/'+str(k),gradient)
        elif isinstance(a,(list,tuple)):
            assert len(a)==len(b),name+' length'
            for i,(aa,bb) in enumerate(zip(a,b)):check(aa,bb,name+'/'+str(i),gradient)
        else:assert a==b,name
    def parameter_check(a,b,name):
        check(list(dict(a.named_parameters())),list(dict(b.named_parameters())),name+'/order')
        for (n,p),(nn,pp) in zip(a.named_parameters(),b.named_parameters()):
            assert p.requires_grad==pp.requires_grad,name+'/'+n+' mask'
            check(p,pp,name+'/'+n,True)
    def gradients(a,b,name):
        for (n,p),(nn,pp) in zip(a.named_parameters(),b.named_parameters()):
            check(p.grad,pp.grad,name+'/'+n,True)
    def state_path(model,initial,steps=50):
        rows=[initial]; state=initial
        model.begin_graph_cache()
        try:
            for _ in range(steps):
                if time.perf_counter()>=deadline:raise TimeoutError('trajectory deadline')
                state=model.rk4(state,.01);rows.append(state)
            return torch.stack(rows,1)
        finally:model.end_graph_cache()
    try:
        start,z,observed,scales,meta=engineering_inputs(device)
        (output/'identity.json').write_text(json.dumps(meta,indent=2),encoding='utf-8')
        ref,fast=select_runtime(start),select_runtime(start,fast=True)
        assert fast.supports_fast_field()
        parameter_check(ref,fast,'initial_clone')
        # Every coefficient includes weak/strong and saturation states, internal/external forcing.
        for coefficient in (0.,.4,-.2):
            for external in (False,True):
                label=f'field_{coefficient}_{external}'
                with torch.no_grad():
                    ref.coefficients.fill_(coefficient);fast.coefficients.fill_(coefficient)
                r=z.clone().requires_grad_(); f=z.clone().requires_grad_()
                u=torch.full((12,10),.13,dtype=z.dtype,device=device) if external else None
                for model in (ref,fast):model.zero_grad(set_to_none=True)
                rr,ff=ref.field(r,u),fast.field(f,u)
                check(rr,ff,label)
                rr.square().mean().backward();ff.square().mean().backward()
                check(r.grad,f.grad,label+'/input_gradient',True)
                gradients(ref,fast,label+'/parameter_gradients')
                if external:assert ref.coefficients.grad is None and fast.coefficients.grad is None
                # Original rk4 has no external-input API. Verify its exact inherited internal path.
                with torch.no_grad():check(ref.rk4(z,.01),fast.rk4(z,.01),label+'/rk4')
                report['checks'].append(label)
        # Full 40-state trajectory and its initial-state gradient, including nonlinear q.
        with torch.no_grad():
            ref.coefficients.fill_(.4);fast.coefficients.fill_(.4)
        for model in (ref,fast):model.zero_grad(set_to_none=True)
        r=z.clone().requires_grad_();f=z.clone().requires_grad_()
        rr,ff=state_path(ref,r),state_path(fast,f)
        check(rr,ff,'full_50step_40state_trajectory')
        rr.square().mean().backward();ff.square().mean().backward()
        check(r.grad,f.grad,'full_trajectory_input_gradient',True)
        gradients(ref,fast,'full_trajectory_parameter_gradients')
        report['checks'].append('full_50step_40state_trajectory_and_gradients')
        # Explicit fallback for other canonical AST, multi-term and additional storage.
        for terms in ([{'channel':'port','expression':{'op':'var','name':'q'}}],
                      [{'channel':'port','expression':{'op':'tanh','scale':1.,'arg':{'op':'var','name':'q'}}}],
                      [TERM,{'channel':'node','expression':{'op':'mul','args':[{'op':'var','name':'p'},{'op':'var','name':'q'}]}}]):
            a=TrainingClosureROM.from_config(start.config(),terms=terms).to(device)
            b=select_runtime(a,fast=True)
            assert not b.supports_fast_field()
            with torch.no_grad():check(a.field(z),b.field(z),'fallback/'+str(terms))
        b=select_runtime(start,fast=True)
        b.structured.storage_specs=[object()]
        assert not b.supports_fast_field()
        report['checks'].append('explicit_fallback_selection')
        all_modes=modes(start,observed)
        for kind,i in [('gru',0),('history',2)]:
            a,b=all_modes[i][1],all_modes[i+1][1]
            parameter_check(a,b,kind+'/initial')
            oa,ob=make_optimizer(a),make_optimizer(b)
            # Exact causal normalized 50-step loss, observation-input and all parameter gradients.
            aa=observed.clone().requires_grad_();bb=observed.clone().requires_grad_()
            la,lb=loss(a,aa,scales,deadline),loss(b,bb,scales,deadline)
            check(la,lb,kind+'/normalized_loss')
            la.backward();lb.backward()
            check(aa.grad,bb.grad,kind+'/observation_input_gradients',True)
            gradients(a,b,kind+'/causal_loss_parameter_gradients')
            with torch.no_grad():
                initial_a=torch.cat((observed[:,-51],a.encode(observed[:,:11])),-1)
                initial_b=torch.cat((observed[:,-51],b.encode(observed[:,:11])),-1)
                check(state_path(a,initial_a),state_path(b,initial_b),kind+'/causal_full_trajectory')
            for step in range(2):
                va=update(a,oa,observed,scales,deadline);vb=update(b,ob,observed,scales,deadline)
                check(va,vb,kind+f'/adam_loss_{step}')
                gradients(a,b,kind+f'/adam_gradients_{step}')
                parameter_check(a,b,kind+f'/adam_parameters_{step}')
                check(oa.state_dict(),ob.state_dict(),kind+f'/adam_state_{step}',True)
            # Serialize and restore both model and optimizer, then verify the next update.
            restored=[]
            for tag,model,opt in [('reference',a,oa),('fast',b,ob)]:
                buffer=io.BytesIO()
                torch.save({'model':model.state_dict(),'optimizer':opt.state_dict()},buffer)
                buffer.seek(0);saved=torch.load(buffer,map_location='cpu',weights_only=True)
                recovered=copy.deepcopy(model)
                recovered.load_state_dict(saved['model'])
                recovered_opt=make_optimizer(recovered)
                recovered_opt.load_state_dict(saved['optimizer'])
                update(model,opt,observed,scales,deadline)
                update(recovered,recovered_opt,observed,scales,deadline)
                parameter_check(model,recovered,kind+'/'+tag+'/recovery_next_update')
                check(opt.state_dict(),recovered_opt.state_dict(),kind+'/'+tag+'/recovery_next_adam_state',True)
                restored.append((model,opt))
            parameter_check(a,b,kind+'/recovered_pair')
            check(oa.state_dict(),ob.state_dict(),kind+'/recovered_pair_state',True)
            # Export format consumed by unchanged original reference loaders.
            path=output/(kind+'_disposable_export.json')
            cpu=copy.deepcopy(b).cpu();cpu.export(path)
            loaded=(CausalHistoryResidual if kind=='history' else TrainingClosureROM).from_export(path).to(device)
            parameter_check(b,loaded,kind+'/reference_export_parameters')
            with torch.no_grad():
                check(b.field(z),loaded.field(z),kind+'/reference_export_field')
                check(state_path(b,z),state_path(loaded,z),kind+'/reference_export_trajectory')
            path.unlink() # fit values discarded; retain source and check evidence only
            report['checks'].append(kind+'/causal_loss_all_gradients_adam_recovery_reference_export')
        synchronize(device)
        report['status']='passed'
    except BaseException as error:
        report['status']='failed';report['error']=repr(error);report['traceback']=traceback.format_exc()
        raise
    finally:
        (output/'correctness.json').write_text(json.dumps(report,indent=2),encoding='utf-8')


def test_registered_correctness(tmp_path):
    run_checks(tmp_path,device='cpu',max_seconds=1800)

if __name__=='__main__':
    args=parser().parse_args()
    if args.child:run_checks(args.output,args.device,args.max_seconds)
    else:raise SystemExit(supervised(args,Path(__file__)))

