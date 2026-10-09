import _bootstrap
import argparse,json,sys
from task4.paths import ROOT,read_json
from task4.data import metadata_preflight
p=argparse.ArgumentParser();p.add_argument('--source',help='Override sibling KIKI root for this process only');a=p.parse_args()
cfg=read_json(ROOT/'configs/project.json')
if a.source:cfg['kiki_root']=a.source
try:
    result=metadata_preflight(cfg)
except (OSError,ValueError,KeyError) as e:
    print(json.dumps({'status':'BLOCKED_DATA','detail':str(e),'action':'Inspect ../kiki metadata/path; do not copy TEST or fabricate data'},ensure_ascii=False,indent=2));sys.exit(2)
print(json.dumps(result,ensure_ascii=False,indent=2))
