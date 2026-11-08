"""One synthetic QA execution, no private model/data loading."""
from pathlib import Path
import argparse,json,os,sys,time
ROOT=Path(__file__).resolve().parents[4]
sys.path[:0]=[str(ROOT),str(ROOT/"experiments"),str(ROOT/"src")]
p=argparse.ArgumentParser();p.add_argument("--output",type=Path,required=True);p.add_argument("--max-seconds",type=float,required=True);p.add_argument("--source-commit",required=True)
a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
from closure_p3_disagreement_20261020 import own_creation_filetime_utc
worker={"pid":os.getpid(),"creation_filetime_utc":own_creation_filetime_utc(),"executable":str(Path(sys.executable).resolve()),"argv":sys.argv}
(a.output/"worker.json").write_text(json.dumps(worker),encoding="utf-8")
started=time.perf_counter()
import threading
def expired():
    (a.output/"timeout.json").write_text(json.dumps({"self_timeout":True,"limit_seconds":a.max_seconds}),encoding="utf-8")
    os._exit(124)
timer=threading.Timer(a.max_seconds,expired);timer.daemon=True;timer.start()
import torch
torch.set_num_threads(1)
os.environ['FUSED_CLOSURE_FIXTURE']=str(ROOT/'private_data/closure_mechanism_20261019/P0/fused_runtime01/prepare01')
cuda={'torch':torch.__version__,'cuda_build':torch.version.cuda,'cuda_available':torch.cuda.is_available(),'device_count':torch.cuda.device_count()}
if not cuda['cuda_available'] or cuda['device_count']<1:
    (a.output/'cuda_failure.json').write_text(json.dumps(cuda),encoding='utf-8')
    raise RuntimeError('actual CUDA QA device required')
cuda['device_name']=torch.cuda.get_device_name(torch.cuda.current_device())
(a.output/'cuda.json').write_text(json.dumps(cuda,indent=2),encoding='utf-8')
os.environ['W_ENTRY_FIXTURE_ROOT']=str(a.output/'entry_fixture')
import pytest
import torch
cuda_facts={"available":torch.cuda.is_available(),"torch_version":torch.__version__,"device":torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}
code=pytest.main([str(ROOT/"tests/test_frequency_dissipation_runtime_20261021.py"),str(ROOT/"tests/test_frequency_dissipation_20261021.py"),str(ROOT/"tests/test_frequency_dissipation_native_20261021.py"),"-q","-rA","--disable-warnings","--basetemp",str(a.output/"pytest_tmp")])
result={"status":"COMPLETED" if code==0 else "FAILED","exit_code":int(code),"seconds":time.perf_counter()-started,"no_private_data":True,"cuda_facts":cuda_facts,"source_commit":a.source_commit}
(a.output/"result.json").write_text(json.dumps(result,indent=2),encoding="utf-8")
timer.cancel()
raise SystemExit(int(code))
