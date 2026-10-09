import _bootstrap
import argparse,json
from task4.paths import ROOT,read_json
from task4.training import fit
p=argparse.ArgumentParser();p.add_argument('--config',default='configs/experiment.json');p.add_argument('--output',required=True);p.add_argument('--arm',choices=['P','A','U']);p.add_argument('--resume');p.add_argument('--warm-start');a=p.parse_args()
cfg=read_json(ROOT/a.config)
if a.arm:cfg['arm']=a.arm
print(json.dumps(fit(cfg,read_json(ROOT/'configs/project.json'),a.output,a.resume,a.warm_start),ensure_ascii=False,indent=2))
