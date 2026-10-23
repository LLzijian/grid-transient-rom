"""Register existing train inputs and exact closed-source bytes; never run science."""
import ast,hashlib,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[4];BASE=Path(__file__).resolve().parent
OLD=ROOT/'private_data/closure_mechanism_20261019/P1/bounded_edge_grammar01'
def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def ref(path):return {'path':str(Path(path).resolve().relative_to(ROOT)),'sha256':sha(path)}
def main():
    commit=sys.argv[1]
    if len(commit)!=40 or any(c not in '0123456789abcdef' for c in commit):raise ValueError('full frozen source required')
    native=read(OLD/'native_run/receipt.json')
    if native.get('source_commit')!='c9ec666fc7e8dd5ad04f63d9bea3ada33d4ec7a2' or native.get('status')!='COMPLETED' or native['source_before']!=native.get('source_after'):
        raise ValueError('old closed source unavailable')
    files={}
    for name,digest in native['source_before'].items():
        path=(ROOT/name).resolve()
        if sha(path)!=digest:raise ValueError('old closed source changed '+name)
        files[str(path)]=digest
    value={'source_commit':commit,'no_process_quotas':True,'train_inputs':{},'train_qualifications':{}}
    for name,path in [('old_selection',OLD/'science/selection.json'),('old_complete',OLD/'run/complete.json'),
        ('old_native',OLD/'native_run/receipt.json'),('pool_manifest',OLD/'frozen_pool/pool.json'),
        ('teacher_identity',OLD/'frozen_pool/teacher.json')]:value[name]=ref(path);files[str(path.resolve())]=sha(path)
    for seed in (6201,6202,6203):
        for group,filename in [('train_inputs','train_observations.npz'),('train_qualifications','teacher_qualification.json')]:
            path=OLD/'science'/str(seed)/filename;value[group][str(seed)]=ref(path);files[str(path.resolve())]=sha(path)
    extras=['experiments/horizon_pair_20261023.py','tests/test_horizon_pair_20261023.py',
        str((BASE/'launch.py').relative_to(ROOT)),str((BASE/'make_binding.py').relative_to(ROOT)),
        'reports/horizon_pair_20261023_protocol.md','reports/astra6_horizon_pair_20261023.md',
        'reports/astra6_horizon_pair_20261023_raw.txt','reports/horizon_pair_20261023_review.md']
    for name in extras:
        path=ROOT/name
        if path.suffix=='.py':ast.parse(path.read_text(encoding='utf-8'))
        files[str(path.resolve())]=sha(path)
    value['files']=dict(sorted(files.items()))
    with (BASE/'binding.json').open('x',encoding='utf-8') as stream:json.dump(value,stream,indent=2,sort_keys=True,allow_nan=False)
    print(json.dumps({'source_commit':commit,'files':len(files),'old_sources':len(native['source_before']),'binding_sha256':sha(BASE/'binding.json')}))
if __name__=='__main__':main()
