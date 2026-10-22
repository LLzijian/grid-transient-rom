"""One fixed baseline or one registered scheme observation bundle. No scientific runs."""
import argparse,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[5];HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT))
from experiments import uncapped_windows_job_20261021 as job
from experiments.rational_python_runtime_20261022 import python_command

def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))
def ref(path):return {'path':str(Path(path).resolve()),'sha256':job.sha(path)}
def main():
    p=argparse.ArgumentParser();p.add_argument('--scheme',choices=('reference','direct_base_venv','venv_pythonw'),required=True)
    p.add_argument('--attempt',type=int,choices=(1,2),required=True);p.add_argument('--source-commit',required=True);a=p.parse_args()
    if len(a.source_commit)!=40 or any(c not in '0123456789abcdef' for c in a.source_commit):raise ValueError('real source commit required')
    out=HERE/(a.scheme+str(a.attempt));out.mkdir(exist_ok=False)
    logical=ROOT/'.venv-gpu/Scripts/python.exe';worker=HERE/'probe_worker.py'
    sources=[Path(__file__).resolve(),worker,ROOT/'experiments/uncapped_windows_job_20261021.py',ROOT/'experiments/rational_python_runtime_20261022.py']
    files={str(path):job.sha(path) for path in sources};summary={'scheme':a.scheme,'attempt':a.attempt,'source_commit':a.source_commit,'observations':{}}
    for role in (('normal',) if a.scheme=='reference' else ('normal','orphan')):
        target=out/role;target.mkdir()
        args=[str(worker),'--role',role,'--scheme',a.scheme,'--out',str(target),'--source-commit',a.source_commit]
        command=({'executable':str(logical),'logical_executable':str(logical),'arguments':args} if a.scheme=='reference' else python_command(logical,args,a.scheme))
        try:
            receipt=job.supervise(command,out/('native_'+role),out/'active.lock',files,ROOT,
                worker_receipt=target/'worker.json',on_worker_owned=lambda identity:(target/'owned').write_text('owned',encoding='utf-8'),
                provenance={'source_commit':a.source_commit,'phase':'runtime_observation','group':role})
            summary['observations'][role]={'receipt':ref(out/('native_'+role)/'receipt.json'),'status':receipt['status']}
        except Exception as error:
            summary['observations'][role]={'error':repr(error)}
        job.atomic(out/'summary.json',summary)
    if a.scheme!='reference':
        baseline=HERE/'reference1/normal/metadata.json';candidate=out/'normal/metadata.json';child=out/'orphan/child.json'
        if all(path.is_file() for path in (baseline,candidate,child)):
            job.atomic(out/'parity.json',{'baseline':ref(baseline),'candidate':ref(candidate),'child_identity':ref(child)})
    print(json.dumps(summary))
if __name__=='__main__':main()
