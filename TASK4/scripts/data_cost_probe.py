"""First real TRAIN audit and finite cost probe, no development/TEST payloads."""
import _bootstrap
import json,time,os
import numpy as np
import torch
from task4.paths import ROOT,read_json,write_json
from task4.data import load_events,train_scales,digest,source_root
from task4.dynamics import PowerGraph
from task4.encoder import HistoryEncoder
from task4.objectives import one_event_loss,m3_numpy

torch.set_num_threads(1);torch.manual_seed(401)
cfg=read_json(ROOT/'configs/project.json')
events,phys,facts=load_events(cfg,'train');scales=train_scales(events,phys)
model=PowerGraph(phys,scales);encoder=HistoryEncoder(model)
out={'data':facts,'scales':scales.tolist(),'physics':phys.as_dict(),
     'model_parameters':sum(p.numel() for p in model.parameters()),
     'encoder_parameters':sum(p.numel() for p in encoder.parameters()),'state_dim':model.dim,
     'source_hashes':{},'measurements':[],'torch':torch.__version__,'device':'cpu'}
kiki=source_root(cfg)
for rel in [cfg['baseline']]+[d+'/'+f for d in cfg['datasets'] for f in ['splits.json','dataset_index.csv']]:
    out['source_hashes'][rel]=digest(kiki/rel)
write_json(ROOT/'reports/T4_0001_data_cost.json',out)
print(json.dumps({k:v for k,v in out.items() if k not in ['physics','source_hashes']},ensure_ascii=False),flush=True)
event=next(e for e in events if e.label=='stable' and len(e.y)>100)
for horizon in [.05,.25,.5]:
    t=time.monotonic();model.zero_grad();encoder.zero_grad()
    loss,_=one_event_loss(model,encoder,event,horizon,.01,True)
    loss.backward()
    row={'id':event.id,'horizon':horizon,'loss':float(loss.detach()),'seconds':time.monotonic()-t,
         'model_gradient_l2':float(torch.sqrt(sum(p.grad.square().sum() for p in model.parameters() if p.grad is not None))),
         'encoder_gradient_l2':float(torch.sqrt(sum(p.grad.square().sum() for p in encoder.parameters() if p.grad is not None)))}
    out['measurements'].append(row);write_json(ROOT/'reports/T4_0001_data_cost.json',out);print(json.dumps(row),flush=True)
    del loss
