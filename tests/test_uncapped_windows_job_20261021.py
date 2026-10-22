"""Tiny real owned Windows subtrees, without scientific data."""
import json,os,subprocess,sys,time
from pathlib import Path
import pytest
from experiments import uncapped_windows_job_20261021 as j
pytestmark=pytest.mark.skipif(os.name!='nt',reason='Windows API')
ROOT=Path(__file__).resolve().parents[1]
def wait_file(path):
 end=time.monotonic()+20
 while not Path(path).is_file():
  if time.monotonic()>end:raise AssertionError('test handshake absent')
  time.sleep(.02)
 return json.loads(Path(path).read_text())
def command(code):return {'executable':sys.executable,'arguments':['-c',code]}
def child_script(path):
 return "import sys,threading;sys.path.insert(0,"+repr(str(ROOT/'experiments'))+");from uncapped_windows_job_20261021 import identity,atomic;atomic("+repr(str(path))+",identity());threading.Event().wait()"
def test_normal_owned_subtree_and_exact_role(tmp_path,monkeypatch):
 requested_flags=[];real_api=j.api
 class RecordingAPI:
  def __init__(self,api):self.api=api
  def __getattr__(self,name):return getattr(self.api,name)
  def CreateProcessW(self,*args):
   requested_flags.append(args[5]);return self.api.CreateProcessW(*args)
 monkeypatch.setattr(j,'api',lambda:RecordingAPI(real_api()))
 r=j.supervise(command("import subprocess,sys;subprocess.run([sys.executable,'-c','pass'],check=True)"),tmp_path/'normal',tmp_path/'lock',{},ROOT,provenance={'source_commit':'a'*40,'group':'0_rational','phase':'qa'})
 assert r['status']=='COMPLETED' and r['job_active_processes']==0 and r['job_total_processes']>=2
 assert r['source_commit']=='a'*40 and r['group']=='0_rational' and r['phase']=='qa'
 assert r['worker']['creation_filetime_utc']>2**53 and r['cleanup_verified'] and not (tmp_path/'lock').exists()
 assert r['runtime_limits'] is r['cpu_limits'] is r['active_process_limit'] is None
 assert r['creation_flags']==0x4|0x8|0x80000
 assert requested_flags==[r['creation_flags']]
 assert r['creation_flags']&0x8 and not r['creation_flags']&(0x08000000|0x10)

def test_exit_zero_parent_leaving_handshaken_live_child_fails_and_cleans(tmp_path):
 path=tmp_path/'child.json'
 code="import subprocess,sys,time,pathlib;subprocess.Popen([sys.executable,'-c',"+repr(child_script(path))+"]);p=pathlib.Path("+repr(str(path))+");exec('while not p.exists(): time.sleep(.02)');sys.exit(0)"
 r=j.supervise(command(code),tmp_path/'zero_parent_orphan',tmp_path/'lock',{},ROOT)
 assert r['launcher_exit_code']==r['worker_exit_code']==0
 assert r['status']=='FAILED' and r['orphans_cleaned'] and r['job_active_processes']==0
 assert r['cleanup_verified'] and j.identity_state(wait_file(path))=='GONE'
 assert not (tmp_path/'lock').exists()
def test_worker_error_cleans_owned_child(tmp_path):
 path=tmp_path/'child.json'
 code="import subprocess,sys,time,pathlib;subprocess.Popen([sys.executable,'-c',"+repr(child_script(path))+"]);p=pathlib.Path("+repr(str(path))+");exec('while not p.exists(): time.sleep(.02)');sys.exit(3)"
 r=j.supervise(command(code),tmp_path/'failure',tmp_path/'lock',{},ROOT)
 assert r['status']=='FAILED' and r['exit_code']==3 and r['orphans_cleaned'] and r['job_active_processes']==0
 assert j.identity_state(wait_file(path))=='GONE'
def test_supervisor_os_exit_cleans_tree_via_noninherited_handle(tmp_path):
 path=tmp_path/'child.json';out=tmp_path/'outer';lock=tmp_path/'lock'
 worker="import subprocess,sys,time,pathlib,threading;subprocess.Popen([sys.executable,'-c',"+repr(child_script(path))+"]);p=pathlib.Path("+repr(str(path))+");exec('while not p.exists(): time.sleep(.02)');threading.Event().wait()"
 script=tmp_path/'supervisor.py'
 script.write_text("import sys,os,time,pathlib\nsys.path.insert(0,"+repr(str(ROOT/'experiments'))+")\nfrom uncapped_windows_job_20261021 import supervise\ndef crash(w):\n while not pathlib.Path("+repr(str(path))+").exists():time.sleep(.02)\n os._exit(121)\nsupervise("+repr(command(worker))+","+repr(str(out))+","+repr(str(lock))+",{},"+repr(str(ROOT))+",on_started=crash)\n",encoding='utf-8')
 p=subprocess.Popen([sys.executable,str(script)]);child=wait_file(path);assert p.wait()==121
 end=time.monotonic()+20
 while j.identity_state(child)=='RUNNING' and time.monotonic()<end:time.sleep(.02)
 assert j.identity_state(child)=='GONE'
 started=json.loads((out/'started.json').read_text());assert j.identity_state(started['worker'])=='GONE'
 assert lock.exists()
def test_exit_before_resume_still_cleans_atomic_job_worker(tmp_path):
 out=tmp_path/'suspended';lock=tmp_path/'lock';script=tmp_path/'creator.py'
 script.write_text("import sys,os\nsys.path.insert(0,"+repr(str(ROOT/'experiments'))+")\nfrom uncapped_windows_job_20261021 import supervise\nsupervise("+repr(command('raise RuntimeError(\"must not run before resume\")'))+","+repr(str(out))+","+repr(str(lock))+",{},"+repr(str(ROOT))+",on_created=lambda w:os._exit(121))\n",encoding='utf-8')
 p=subprocess.Popen([sys.executable,str(script)]);assert p.wait()==121
 worker=json.loads((out/'started.json').read_text())['worker']
 end=time.monotonic()+20
 while j.identity_state(worker)=='RUNNING' and time.monotonic()<end:time.sleep(.02)
 assert j.identity_state(worker)=='GONE' and lock.exists()
def test_actual_worker_receipt_is_retained_inside_redirector_job(tmp_path):
 receipt=tmp_path/'worker.json';release=tmp_path/'release'
 code="import sys,time,pathlib;sys.path.insert(0,"+repr(str(ROOT/'experiments'))+");from uncapped_windows_job_20261021 import identity,atomic;atomic("+repr(str(receipt))+",{**identity(),'phase':'qa','group':None,'source_commit':'"+'b'*40+"','qa_attempt':2});p=pathlib.Path("+repr(str(release))+");exec('while not p.exists():time.sleep(.01)')"
 r=j.supervise(command(code),tmp_path/'actual',tmp_path/'lock',{},ROOT,worker_receipt=receipt,on_worker_owned=lambda w:release.write_text('owned'),provenance={'phase':'qa','group':None,'source_commit':'b'*40,'qa_attempt':2})
 assert r['status']=='COMPLETED' and r['launcher_exit_code']==r['worker_exit_code']==0
 actual=json.loads(receipt.read_text());assert r['worker']['pid']==actual['pid'] and r['worker']['creation_filetime_utc']==actual['creation_filetime_utc']
 assert r['worker']['physical_image'] and r['launcher']['physical_image'] and r['actual_worker_state']=='GONE' and r['job_active_processes']==0
 assert r['creation_flags']==0x4|0x8|0x80000 and not r['orphans_cleaned'] and r['cleanup_verified']
def test_stale_creation_and_exclusive_lock(tmp_path):
 current=j.identity();stale={**current,'creation_filetime_utc':current['creation_filetime_utc']+1}
 assert j.identity_state(stale)=='REUSED_PID'
 lock=tmp_path/'lock';lock.write_text(json.dumps({'supervisor':current,'worker':None}))
 with pytest.raises(FileExistsError):j.supervise(command('pass'),tmp_path/'never',lock,{},ROOT)
 with pytest.raises(ValueError):j.reconcile_lock(lock,{'supervisor':stale,'worker':None},tmp_path/'missing')
 assert lock.exists() and not (tmp_path/'never').exists()

