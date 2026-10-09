"""Bounded actual P updates for resource/loss planning, never promoted."""
import _bootstrap
import json,time,os
import numpy as np
import psutil
import torch
from task4.paths import ROOT,read_json,write_json
from task4.data import load_events,train_scales
from task4.dynamics import PowerGraph
from task4.encoder import HistoryEncoder,PredictiveHead
from task4.objectives import AnchorTargets,ShootingBank,batch_objective

torch.set_num_threads(1);torch.manual_seed(401)
cfg=read_json(ROOT/'configs/experiment.json');events,phys,facts=load_events(read_json(ROOT/'configs/project.json'),'train')
model=PowerGraph(phys,train_scales(events,phys));model.checkpoint_steps=10
model.save_deadline=float(os.environ['TASK4_SAVE_DEADLINE_UNIX'])-5
encoder=HistoryEncoder(model);head=PredictiveHead(model,cfg['anchor_horizons']);bank=ShootingBank()
optim=torch.optim.Adam(list(model.parameters())+list(encoder.parameters())+list(head.parameters()),lr=cfg['lr'])
target=AnchorTargets(phys,cfg['anchor_horizons']);rng=np.random.default_rng(401)
batch=[next(e for e in events if e.label==label) for label in ['stable','unstable']]
report={'scope':'TRAIN engineering pilot; no model selection or promotion','data':facts,'rows':[],'checkpoint_steps':10}
for horizon in [.25,1.,5.]:
    start=time.monotonic();optim.zero_grad(set_to_none=True)
    loss,terms=batch_objective(model,encoder,head,batch,horizon,cfg,target,rng,bank,optim,None,1.)
    loss.backward();norm=torch.nn.utils.clip_grad_norm_(list(model.parameters())+list(encoder.parameters())+list(head.parameters()),100.,error_if_nonfinite=True)
    optim.step()
    mem=psutil.Process().memory_info()
    row={'horizon':horizon,'batch_size':len(batch),'seconds':time.monotonic()-start,'rss_bytes':mem.rss,
         'peak_rss_bytes':getattr(mem,'peak_wset',None),'gradient_norm':float(norm),'loss':float(loss.detach()),
         'terms':{k:float(v.detach()) for k,v in terms.items()}}
    report['rows'].append(row);write_json(ROOT/'reports/T4_0001_joint_cost.json',report);print(json.dumps(row),flush=True)
    del loss,terms
