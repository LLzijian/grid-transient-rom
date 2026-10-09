"""Bounded equal-objective executor benchmark; no fitted candidate selection."""
import _bootstrap
import copy,gc,json,time
import numpy as np
import psutil,torch
from task4.paths import ROOT,read_json,write_json
from task4.training import load_checkpoint,check_authorized
from task4.data import load_events
from task4.objectives import AnchorTargets,ShootingBank,batch_objective
from task4.accelerate import trace_field

check_authorized(False);torch.set_num_threads(1)
base,enc,head,obj=load_checkpoint(ROOT/'artifacts/T4_0001/P_short/last.pt')
events,physics,_=load_events(read_json(ROOT/'configs/project.json'),'train')
batch=[next(e for e in events if e.label==v) for v in ['stable','unstable']]
cfg=dict(obj['config']);report={'checkpoint_step':obj['step'],'source':'P_short complete-update checkpoint','cases':[]}
if (ROOT/'reports/T4_0001_executor_probe.json').exists():report=read_json(ROOT/'reports/T4_0001_executor_probe.json')
report['rejected_combination']={'trace':True,'checkpoint_steps':10,'reason':'TorchScript/checkpoint incompatibility; failed logs retained, no scientific candidate'}
for traced,cp in [(False,10),(True,0)]:
    if any(c['trace']==traced and c['checkpoint_steps']==cp for c in report['cases']):continue
    model=copy.deepcopy(base);encoder=copy.deepcopy(enc);h=copy.deepcopy(head)
    model.checkpoint_steps=cp
    if traced:trace_field(model)
    optim=torch.optim.Adam(list(model.parameters())+list(encoder.parameters())+list(h.parameters()),lr=cfg['lr'])
    target=AnchorTargets(physics,cfg['anchor_horizons']);bank=ShootingBank()
    case={'trace':traced,'checkpoint_steps':cp,'trace_seconds':getattr(model,'trace_build_seconds',0.),'rows':[]}
    for horizon in [.02,.25,1.]:
        optim.zero_grad(set_to_none=True);tick=time.monotonic()
        loss,terms=batch_objective(model,encoder,h,batch,horizon,cfg,target,np.random.default_rng(11),bank,optim,None,1.)
        loss.backward()
        memory=psutil.Process().memory_info()
        case['rows'].append({'horizon':horizon,'seconds':time.monotonic()-tick,'loss':float(loss.detach()),
             'model_gradient_l2':float(sum(p.grad.square().sum() for p in model.parameters() if p.grad is not None).sqrt()),
             'rss_bytes':memory.rss,'peak_rss_bytes':getattr(memory,'peak_wset',None)})
        print(json.dumps({'case':[traced,cp],**case['rows'][-1]}),flush=True)
        del loss,terms
    report['cases'].append(case);write_json(ROOT/'reports/T4_0001_executor_probe.json',report)
    del model,encoder,h,optim,target,bank;gc.collect()
