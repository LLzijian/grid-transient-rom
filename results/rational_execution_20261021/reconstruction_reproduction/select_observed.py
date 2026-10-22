"""Freeze the first actually qualified registered runtime; no process launch."""
import json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[5];HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT))
from experiments import uncapped_windows_job_20261021 as job
from experiments.rational_python_runtime_20261022 import python_command,validate_selection
def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))
def ref(path):return {'path':str(path.resolve()),'sha256':job.sha(path)}
def main():
    scheme=sys.argv[1];attempt=int(sys.argv[2]);observed=HERE/(scheme+str(attempt))
    normal=observed/'native_normal/receipt.json';orphan=observed/'native_orphan/receipt.json'
    baseline=HERE/'reference1/normal/metadata.json';parity=observed/'parity.json'
    command=python_command(ROOT/'.venv-gpu/Scripts/python.exe',[],scheme)
    selection={'status':'SELECTED','scheme':scheme,'observation_attempt':attempt,
        'observation_source_commit':read(normal)['source_commit'],
        **{key:command[key] for key in ('logical_executable','executable','expected_worker_image')},
        'normal_receipt':ref(normal),'orphan_receipt':ref(orphan),'baseline_reference':ref(baseline),'parity_evidence':ref(parity)}
    files={Path(path) for path in read(normal)['source_before']}
    files.update(Path(command[key]) for key in ('logical_executable','executable','expected_worker_image'))
    files.update((normal,orphan,baseline,parity,Path(command['logical_executable']).parent.parent/'pyvenv.cfg',Path(read(baseline)['python_dll']['path'])))
    files.update(Path(item['path']) for item in read(parity).values())
    selection['files']={str(path.resolve()):job.sha(path) for path in sorted(files)}
    temporary=HERE/'proposed_selection.json'
    with temporary.open('x',encoding='utf-8') as stream:json.dump(selection,stream,indent=2)
    validate_selection(temporary)
    target=HERE/'runtime_selection.json'
    with target.open('x',encoding='utf-8') as stream:json.dump(selection,stream,indent=2)
    print(json.dumps({'qualified':True,'path':str(target),'sha256':job.sha(target),'files':len(files)}))
if __name__=='__main__':main()
