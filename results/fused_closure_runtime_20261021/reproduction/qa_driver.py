"""Native-supervised CPU/CUDA QA, consuming prebound frozen synthetic files."""
from pathlib import Path
import argparse,json,os,sys,time,threading
ROOT=Path(__file__).resolve().parents[4];BASE=Path(__file__).resolve().parent
sys.path[:0]=[str(ROOT/'tests'),str(ROOT/'experiments'),str(ROOT/'src'),str(ROOT)]
p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--binding',type=Path,required=True);p.add_argument('--max-seconds',type=float,required=True);p.add_argument('--source-commit',required=True);a=p.parse_args()
a.output=a.output.resolve()
if a.output not in (BASE/'qa01',BASE/'qa02') or a.max_seconds!=340.:raise ValueError('exact registered QA')
a.output.mkdir(parents=True,exist_ok=False)
from closure_p3_disagreement_20261020 import own_creation_filetime_utc
(a.output/'worker.json').write_text(json.dumps({'pid':os.getpid(),'creation_filetime_utc':own_creation_filetime_utc(),'executable':str(Path(sys.executable).resolve()),'argv':sys.argv}),encoding='utf-8')
started=time.perf_counter()
def expired():os._exit(124)
timer=threading.Timer(a.max_seconds,expired);timer.daemon=True;timer.start()
from fused_closure_audit_20261021 import verify_binding,verify_fixture
os.environ['FUSED_CLOSURE_FIXTURE']=str(BASE/'prepare01');verify_binding(a.binding);verify_fixture()
import torch
torch.set_num_threads(1)
cuda={'available':torch.cuda.is_available(),'torch':torch.__version__,'cuda_build':torch.version.cuda,'device_count':torch.cuda.device_count()}
if not cuda['available'] or cuda['device_count']<1:raise RuntimeError('actual CUDA QA required')
cuda['device_name']=torch.cuda.get_device_name(0)
(a.output/'cuda.json').write_text(json.dumps(cuda,indent=2),encoding='utf-8')
import pytest
code=pytest.main([str(ROOT/'tests/test_fused_closure_runtime_20261021.py'),str(ROOT/'tests/test_fused_closure_audit_20261021.py'),'-q','-rA','--disable-warnings','--basetemp',str(a.output/'pytest_tmp')])
verify_binding(a.binding);verify_fixture()
(a.output/'result.json').write_text(json.dumps({'status':'COMPLETED' if code==0 else 'FAILED','exit_code':int(code),'seconds':time.perf_counter()-started,'cuda':cuda,'source_commit':a.source_commit}),encoding='utf-8')
timer.cancel();raise SystemExit(int(code))
