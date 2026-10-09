import _bootstrap
import argparse,json,tempfile,shutil
from task4.paths import ROOT,read_json
from task4.training import fit
p=argparse.ArgumentParser();p.add_argument('--output',default='artifacts/synthetic_smoke');a=p.parse_args()
c=read_json(ROOT/'configs/smoke.json')
print(json.dumps(fit(c,read_json(ROOT/'configs/project.json'),a.output),ensure_ascii=False,indent=2))
