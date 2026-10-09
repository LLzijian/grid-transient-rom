import _bootstrap
import argparse,json,sys
from task4.paths import ROOT,read_json
from task4.workflow import Workflow
from task4.runner import run_managed
p=argparse.ArgumentParser(description='Record root-agent decisions and enforce lifecycle/budget gates')
s=p.add_subparsers(dest='cmd',required=True)
s.add_parser('status')
a=s.add_parser('authorize');a.add_argument('--instruction',required=True);a.add_argument('--cap-seconds',type=float,required=True)
a=s.add_parser('open-loop');a.add_argument('--id',required=True);a.add_argument('--spec',required=True);a.add_argument('--reserve-seconds',type=float,required=True)
a=s.add_parser('run');a.add_argument('--cap-seconds',type=float,default=7200);a.add_argument('command',nargs=argparse.REMAINDER)
a=s.add_parser('close-loop');a.add_argument('--decision',choices=['accept','retain','reject','inconclusive'],required=True);a.add_argument('--evidence',required=True);a.add_argument('--candidate');a.add_argument('--code-ref')
a=s.add_parser('propose-direction');a.add_argument('--proposal',required=True)
a=s.add_parser('approve-direction');a.add_argument('--instruction',required=True)
a=s.add_parser('reject-direction');a.add_argument('--instruction',required=True)
a=s.add_parser('pause');a.add_argument('--reason',required=True)
a=s.add_parser('account');a.add_argument('--seconds',type=float,required=True);a.add_argument('--reason',required=True)
a=p.parse_args();w=Workflow()
if a.cmd=='authorize':w.authorize(a.instruction,a.cap_seconds)
elif a.cmd=='open-loop':w.open_loop(a.id,a.spec,a.reserve_seconds)
elif a.cmd=='run':
    cmd=a.command
    if cmd and cmd[0]=='--':cmd=cmd[1:]
    if not cmd:raise ValueError('Pass a real command after --')
    state=w.load();cfg=read_json(ROOT/'configs/project.json')
    result=run_managed(cmd,state['active_loop'],a.cap_seconds,cfg['budget']['memory_gib'])
    print(json.dumps(result,ensure_ascii=False,indent=2));sys.exit(0 if result['exit_code']==0 and result['reason']=='exit' else 1)
elif a.cmd=='close-loop':w.close_loop(a.decision,a.evidence,a.candidate,a.code_ref)
elif a.cmd=='propose-direction':w.propose_direction(a.proposal)
elif a.cmd=='approve-direction':w.approve_direction(a.instruction)
elif a.cmd=='reject-direction':w.reject_direction(a.instruction)
elif a.cmd=='pause':w.pause(a.reason)
elif a.cmd=='account':w.account(a.seconds,a.reason)
print(json.dumps(w.load(),ensure_ascii=False,indent=2))
