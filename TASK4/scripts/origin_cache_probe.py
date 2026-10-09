"""Bounded exact-mathematics executor comparison; never selects a model."""
import _bootstrap
import argparse
import gc
import time
import numpy as np
import torch
import psutil
from task4.paths import ROOT,project_path,read_json,write_json
from task4.training import load_checkpoint,check_authorized
from task4.data import load_events,digest
from task4.accelerate import trace_field
from task4.objectives import AnchorTargets,ShootingBank,batch_objective


def main():
    p=argparse.ArgumentParser();p.add_argument('--checkpoint',required=True);p.add_argument('--output',required=True)
    args=p.parse_args();check_authorized(False);torch.set_num_threads(1)
    out=project_path(args.output)
    if out.exists():raise FileExistsError(out)
    source=project_path(args.checkpoint)
    source_hash=digest(source)
    events,physics,_=load_events(read_json(ROOT/'configs/project.json'),'train')
    batch=[next(e for e in events if e.label==label) for label in ['stable','unstable']]
    report={'checkpoint_sha256':source_hash,'event_ids':[e.id for e in batch],
            'kind':'exact_executor_test_not_candidate_selection','test_payload_read':False,
            'cases':[],'complete':False,'equivalence_passed':False,
            'memory_note':'peak RSS is cumulative in this sequential process, not an isolated per-case memory comparison'}
    refs={}
    for reuse in [False,True]:
        model,encoder,head,obj=load_checkpoint(source)
        model.reuse_origin_terms=reuse;model.checkpoint_steps=0
        trace_field(model)
        cfg=dict(obj['config']);params=[p for mod in [model,encoder,head] for p in mod.parameters() if p.requires_grad]
        opt=torch.optim.Adam(params,lr=cfg['lr']);target=AnchorTargets(physics,cfg['anchor_horizons'])
        case={'reuse_origin_terms':reuse,'trace_build_seconds':model.trace_build_seconds,'rows':[]}
        report['cases'].append(case)
        for horizon in [.02,1.,5.]:
            opt.zero_grad(set_to_none=True);start=time.monotonic()
            loss,terms=batch_objective(model,encoder,head,batch,horizon,cfg,target,np.random.default_rng(11),ShootingBank(),opt,None,1.)
            loss.backward();seconds=time.monotonic()-start
            grad=np.concatenate([p.grad.detach().numpy().ravel() if p.grad is not None else np.zeros(p.numel()) for p in params])
            row={'horizon':horizon,'seconds':seconds,'loss':float(loss.detach()),'gradient_l2':float(np.linalg.norm(grad)),
                 'rss_bytes':psutil.Process().memory_info().rss,'peak_rss_bytes':getattr(psutil.Process().memory_info(),'peak_wset',None)}
            if not reuse:refs[horizon]=(row['loss'],grad.copy())
            else:
                ref_loss,ref_grad=refs[horizon]
                row['loss_abs_diff']=abs(row['loss']-ref_loss)
                row['gradient_max_abs_diff']=float(np.max(np.abs(grad-ref_grad)))
                np.testing.assert_allclose(row['loss'],ref_loss,rtol=1e-9,atol=1e-10)
                np.testing.assert_allclose(grad,ref_grad,rtol=1e-8,atol=1e-9)
            case['rows'].append(row);write_json(out,report)
            print({'reuse':reuse,**row},flush=True)
            del loss,terms,grad
        del model,encoder,head,opt,target,params;gc.collect()
    if digest(source)!=source_hash:raise RuntimeError('Frozen source changed')
    report.update(complete=True,equivalence_passed=True)
    write_json(out,report)


if __name__=='__main__':main()
