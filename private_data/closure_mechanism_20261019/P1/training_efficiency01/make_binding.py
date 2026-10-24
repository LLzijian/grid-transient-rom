"""Freeze four existing source0/2048 states; no optimization is executed."""
import hashlib,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[4];BASE=Path(__file__).resolve().parent
OLD=ROOT/'private_data/closure_mechanism_20261019/P1/horizon_pair01'
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def ref(p):return {'path':str(Path(p).resolve().relative_to(ROOT)),'sha256':sha(p)}
def main():
    import numpy as np
    commit=sys.argv[1]
    if len(commit)!=40 or any(c not in '0123456789abcdef' for c in commit):raise ValueError('full frozen source required')
    native=read(OLD/'native_run/receipt.json');selection=read(OLD/'science/selection.json');binding=read(OLD/'binding.json')
    if native.get('status')!='COMPLETED' or native.get('source_commit')!='ebd45248a25905dbf5af4a3cea8c4073a02994db' or native['source_before']!=native.get('source_after') or selection.get('actual_updates')!=24576 or not selection.get('all_twelve_frozen'):
        raise ValueError('complete old science required')
    files={}
    for name,digest in native['source_before'].items():
        path=(ROOT/name).resolve()
        if sha(path)!=digest:raise ValueError('old closed source changed '+name)
        files[str(path)]=digest
    value={'source_commit':commit,'no_process_quotas':True,'cases':[]}
    for name,path in [('old_native',OLD/'native_run/receipt.json'),('old_complete',OLD/'run/complete.json'),
        ('selection',OLD/'science/selection.json'),('train',ROOT/binding['train_inputs']['6201']['path']),
        ('pool',ROOT/binding['pool_manifest']['path'])]:value[name]=ref(path);files[str(path.resolve())]=sha(path)
    pool=read(ROOT/value['pool']['path'])
    with np.load(ROOT/value['train']['path'],allow_pickle=False) as arrays:coarse_sha=hashlib.sha256(arrays['coarse'].tobytes()).hexdigest()
    for candidate,future in ((2,50),(2,200),(3,50),(3,200)):
        row=next(r for r in selection['rows'] if r['seed']==6301 and r['candidate']==candidate and r['future']==future)
        case={'candidate':candidate,'future':future,'identity':{'source_commit':selection['source_commit'],'seed':6301,
            'train_seed':6201,'train_file_sha256':value['train']['sha256'],'train_sha256':coarse_sha,
            'candidate_key':pool['keys'][candidate],'candidate_index':candidate,'fit_future_steps':future},'snapshots':{}}
        for step in (0,2048):
            case['snapshots'][str(step)]={}
            for kind,filename in [('model','model.json'),('training','training.pt')]:
                path=Path(row['path'])/f'checkpoint_{step:05d}'/filename
                if sha(path)!=row['checkpoints'][str(step)][filename]:raise ValueError('original checkpoint hash mismatch')
                case['snapshots'][str(step)][kind]=ref(path);files[str(path.resolve())]=sha(path)
        value['cases'].append(case)
    extras=['experiments/closure_training_efficiency_20261024.py','tests/test_closure_training_efficiency_20261024.py',
        str((BASE/'launch.py').relative_to(ROOT)),str((BASE/'make_binding.py').relative_to(ROOT)),
        'reports/training_efficiency_20261024_protocol.md','reports/astra6_training_efficiency_20261024_raw.txt',
        'reports/training_efficiency_20261024_review.md']
    for name in extras:files[str((ROOT/name).resolve())]=sha(ROOT/name)
    value['files']=dict(sorted(files.items()))
    with (BASE/'binding.json').open('x',encoding='utf-8') as f:json.dump(value,f,indent=2,sort_keys=True,allow_nan=False)
    print(json.dumps({'files':len(files),'old_sources':len(native['source_before']),'binding_sha256':sha(BASE/'binding.json')}))
if __name__=='__main__':main()
