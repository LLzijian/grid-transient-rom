"""One preregistered normal-tree identity diagnostic; not QA."""
import ctypes,json,sys,threading,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[5]
sys.path.insert(0,str(ROOT/'experiments'))
import uncapped_windows_job_20261021 as j
BASE=Path(__file__).resolve().parent
source=ROOT/'experiments/uncapped_windows_job_20261021.py'
original_sha=j.sha(source)
real_api=j.api
state={'job':None,'retained':{},'first_seen':{},'before_cleanup':None,'errors':[]}
stop=threading.Event()
gate=threading.Lock()
class Proxy:
 def __init__(self,k):self.k=k
 def __getattr__(self,n):return getattr(self.k,n)
 def CreateJobObjectW(self,*args):
  handle=self.k.CreateJobObjectW(*args);state['job']=handle;return handle
def wrapped_api():return Proxy(real_api())
j.api=wrapped_api
k=real_api()
def observe():
 with gate:
  pids=j.owned_pids(k,state['job'])
  for pid in pids:
   if pid not in state['retained']:
    handle=k.OpenProcess(0x1000|0x100000,False,pid)
    if not handle:
     if ctypes.get_last_error()==87:continue
     raise ctypes.WinError(ctypes.get_last_error())
    member=j.W.BOOL();j.checked(k.IsProcessInJob(handle,state['job'],ctypes.byref(member)))
    if not member.value:k.CloseHandle(handle);continue
    state['retained'][pid]=handle
    state['first_seen'][str(pid)]={'pid':pid,**j.times(k,handle),'physical_image':j.physical_image(k,handle),'IsProcessInJob':bool(member.value),'wait_state':int(k.WaitForSingleObject(handle,0))}
  rows=[]
  for pid,handle in state['retained'].items():
   member=j.W.BOOL();j.checked(k.IsProcessInJob(handle,state['job'],ctypes.byref(member)))
   exitcode=j.W.DWORD();j.checked(k.GetExitCodeProcess(handle,ctypes.byref(exitcode)))
   rows.append({'pid':pid,**j.times(k,handle),'physical_image':j.physical_image(k,handle),'IsProcessInJob':bool(member.value),'wait_state':int(k.WaitForSingleObject(handle,0)),'exit_code':int(exitcode.value),'listed_now':pid in pids})
  return rows
def watch():
 while not stop.is_set():
  try:observe()
  except BaseException as exc:state['errors'].append(repr(exc));return
  stop.wait(.02)
real_settle=j.settle_job
def settle(k_arg,job,known):
 stop.set();monitor.join()
 state['before_cleanup']=observe()
 j.atomic(BASE/'identity_before_original_cleanup.json',{'diagnostic_not_QA':True,'original_source_sha256':original_sha,'first_seen':state['first_seen'],'before_cleanup':state['before_cleanup'],'errors':state['errors']})
 return real_settle(k_arg,job,known)
j.settle_job=settle
monitor=threading.Thread(target=watch)
started=time.perf_counter()
try:
 code="import subprocess,sys,time;subprocess.run([sys.executable,'-c','import time;time.sleep(.3)'],check=True);time.sleep(.1)"
 result=j.supervise({'executable':str(ROOT/'.venv-gpu/Scripts/python.exe'),'arguments':['-c',code]},BASE/'original_normal_tree',BASE/'diagnostic.lock',{str(source):original_sha},ROOT,on_started=lambda _:monitor.start(),provenance={'purpose':'single normal venv tree identity diagnostic, not QA'})
 j.atomic(BASE/'diagnostic_summary.json',{'diagnostic_not_QA':True,'receipt':str(BASE/'original_normal_tree/receipt.json'),'receipt_sha256':j.sha(BASE/'original_normal_tree/receipt.json'),'source_before':original_sha,'source_after':j.sha(source),'status':result['status'],'before_cleanup':state['before_cleanup'],'first_seen':state['first_seen'],'errors':state['errors'],'job_cpu_user_seconds':result.get('job_cpu_user_seconds'),'job_cpu_kernel_seconds':result.get('job_cpu_kernel_seconds'),'supervisor_wall_seconds':result['wall_seconds'],'diagnostic_wall_seconds':time.perf_counter()-started})
 print(json.dumps(json.loads((BASE/'diagnostic_summary.json').read_text()),indent=2))
finally:
 stop.set()
 if monitor.ident:monitor.join()
 for handle in state['retained'].values():k.CloseHandle(handle)
