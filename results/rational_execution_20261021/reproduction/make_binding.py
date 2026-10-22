"""Generate a fixed-role source binding; no execution or cost admission."""
import argparse,hashlib,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[4];BASE=Path(__file__).resolve().parent
GROUPS=('0_rational','0_tanh','1_rational','1_tanh')
sys.path[:0]=[str(ROOT/'experiments'),str(ROOT/'src')]
from rational_execution_20261021 import (REPAIR_INPUTS,repair_source_names,validate_qa_attempt,validate_qa05_pass,
    RECONSTRUCTION_INPUTS,reconstruction_source_names,RUNTIME_SELECTION,QA05_INPUTS,qa05_source_names,require_unused_qa05_temp)
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 p=argparse.ArgumentParser();p.add_argument('phase',choices=('qa','run','evaluate'));p.add_argument('--group',choices=GROUPS);p.add_argument('--source-commit',required=True);p.add_argument('--qa-attempt',type=int,choices=(1,2,3,4,5),default=1);a=p.parse_args()
 if len(a.source_commit)!=40 or any(c not in '0123456789abcdef' for c in a.source_commit):raise ValueError('full source commit required')
 if (a.phase=='run')!=(a.group is not None) or (a.phase!='qa' and a.qa_attempt!=1):raise ValueError('fixed binding role required')
 role='run_'+a.group if a.phase=='run' else ('qa%02d'%a.qa_attempt if a.phase=='qa' else 'evaluate01')
 target=BASE/'bindings'/(role+'.json');assert not target.exists() and not (BASE/'native_active.lock').exists()
 if a.phase=='qa' and ((BASE/role).exists() or (BASE/('native_'+role)).exists()):raise ValueError('exclusive QA attempt, never overwrite')
 proof=ROOT/'results/rational_control_20261021/main_verification.json';old=ROOT/'private_data/closure_mechanism_20261019/P0/rational_control01/native_qa01/execution.json'
 paths={ROOT/r['path'] for r in read(proof)['exact_ast_closure']}
 paths.update(ROOT/n for n in ('AGENTS.md','docs/engineering_loop.md','reports/rational_execution_20261021_protocol.md','reports/astra6_rational_execution_20261021.md','reports/rational_execution_20261021_plan_review.md','reports/rational_execution_20261021_implementation_review.md','reports/rational_execution_20261021_execution_setup.md','results/rational_execution_20261021/authority.json','experiments/rational_execution_20261021.py','experiments/uncapped_windows_job_20261021.py','tests/test_rational_execution_20261021.py','tests/test_uncapped_windows_job_20261021.py'))
 paths.update((proof,old,Path(__file__).resolve(),BASE/'launch.py'))
 if a.phase!='qa' or a.qa_attempt==3:paths.update(ROOT/n for n in repair_source_names())
 if a.phase!='qa' or a.qa_attempt in (4,5):
  paths.update(ROOT/n for n in reconstruction_source_names());paths.add(ROOT/RUNTIME_SELECTION)
  paths.update((ROOT/n).resolve() for n in read(ROOT/RUNTIME_SELECTION)['files'])
 if a.phase!='qa' or a.qa_attempt==5:paths.update(ROOT/n for n in qa05_source_names())
 if a.phase=='qa' and a.qa_attempt==5:require_unused_qa05_temp()
 if a.qa_attempt==2 or (BASE/'native_qa02/receipt.json').exists():paths.add(ROOT/'reports/rational_execution_20261021_repair.md')
 prior=sorted(BASE.glob('native_*/receipt.json'));paths.update(prior)
 binding={'source_commit':a.source_commit,'phase':a.phase,'group':a.group,'qa_attempt':a.qa_attempt if a.phase=='qa' else None,'prior_receipts':[p.relative_to(ROOT).as_posix() for p in prior]}
 for key,path in (('authority',ROOT/'results/rational_execution_20261021/authority.json'),('mathematical_verification',proof),('original_qa_receipt',old)):
  binding[key]={'path':path.relative_to(ROOT).as_posix(),'sha256':sha(path)}
 if a.phase!='qa' or a.qa_attempt==3:
  for key,(name,digest) in REPAIR_INPUTS.items():binding[key]={'path':name,'sha256':digest}
 if a.phase!='qa' or a.qa_attempt in (4,5):
  for key,(name,digest) in RECONSTRUCTION_INPUTS.items():binding[key]={'path':name,'sha256':digest}
  binding['runtime_selection']={'path':RUNTIME_SELECTION,'sha256':sha(ROOT/RUNTIME_SELECTION)}
 if a.phase!='qa' or a.qa_attempt==5:
  for key,(name,digest) in QA05_INPUTS.items():binding[key]={'path':name,'sha256':digest}
 if a.phase!='qa':paths.add(BASE/'qa05/complete.json')
 if a.phase=='evaluate':
  selection=BASE/'selection.json';paths.add(selection);frozen=read(selection)
  binding['selection_manifest']={'path':selection.relative_to(ROOT).as_posix(),'sha256':sha(selection)}
  for group in GROUPS:
   for n in ('selection.json','result.json','complete.json'):paths.add(BASE/('run_'+group)/n)
  for row in frozen['rows']:
   for n in row['files']:paths.add(Path(row['path'])/n)
   for k in ('train_raw','trace'):paths.add(Path(row[k]['path']))
 binding['files']={(p.resolve().relative_to(ROOT).as_posix() if p.resolve().is_relative_to(ROOT) else str(p.resolve())):sha(p) for p in sorted(paths)}
 if a.phase=='qa':validate_qa_attempt(binding,a.source_commit,a.qa_attempt)
 else:validate_qa05_pass(binding,a.source_commit)
 target.parent.mkdir(parents=True,exist_ok=True)
 with target.open('x',encoding='utf-8') as f:json.dump(binding,f,indent=2);f.write('\n')
 print(json.dumps({'binding':str(target),'files':len(binding['files']),'prior_receipts':len(prior),'source_commit':a.source_commit,'no_execution':True}))
if __name__=='__main__':main()
