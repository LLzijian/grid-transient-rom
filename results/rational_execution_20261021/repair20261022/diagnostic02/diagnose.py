"""Second and final authorized identity observation; not QA."""
import ctypes,json,sys,threading,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[6]
sys.path.insert(0,str(ROOT/'experiments'))
import uncapped_windows_job_20261021 as j
BASE=Path(__file__).resolve().parent
source=ROOT/'experiments/uncapped_windows_job_20261021.py'
original_sha=j.sha(source)
assert original_sha=='22beb521459461895f25034dc98f9524872669d6ca588344021b0f0cd03c43f7'
real_api=j.api
k=real_api()
kept={}
state={'diagnostic_not_QA':True,'cumulative_observation_attempt':2,'source_before':original_sha,'first_seen':{},'snapshots':{},'field_errors':[],'observer_errors':[],'observation_count':0}
job_holder={}
stop=threading.Event()
gate=threading.Lock()
def stamp():return int(time.time_ns()//100+116444736000000000)
def field(row,name,call):
 try:row[name]=call()
 except BaseException as exc:
  row[name]=None;state['field_errors'].append({'pid':row.get('pid'),'field':name,'observed_filetime_utc':stamp(),'error':repr(exc)})
def persist():
 try:j.atomic(BASE/'identity_observations.json',state)
 except BaseException as exc:state['observer_errors'].append({'operation':'persist','error':repr(exc)})
class Proxy:
 def __init__(self,real):self.real=real
 def __getattr__(self,name):return getattr(self.real,name)
 def CreateJobObjectW(self,*args):
  job=self.real.CreateJobObjectW(*args);job_holder['handle']=job;return job
j.api=lambda:Proxy(real_api())
def membership(handle):
 value=j.W.BOOL();j.checked(k.IsProcessInJob(handle,job_holder['handle'],ctypes.byref(value)));return bool(value.value)
def exitcode(handle):
 value=j.W.DWORD();j.checked(k.GetExitCodeProcess(handle,ctypes.byref(value)));return int(value.value)
def observe(label):
 try:
  with gate:
   pids=j.owned_pids(k,job_holder['handle'])
   state['observation_count']+=1
   for pid in pids:
    if pid in kept:continue
    handle=k.OpenProcess(0x1000|0x100000,False,pid)
    if not handle:
     state['field_errors'].append({'pid':pid,'field':'OpenProcess','observed_filetime_utc':stamp(),'error':int(ctypes.get_last_error())});persist();continue
    kept[pid]=handle
    row={'pid':pid,'identity_query_filetime_utc':stamp()}
    field(row,'times_at_first_query',lambda:j.times(k,handle))
    field(row,'physical_image_at_first_query',lambda:j.physical_image(k,handle))
    field(row,'IsProcessInJob_at_first_query',lambda:membership(handle))
    field(row,'wait_state_at_first_query',lambda:int(k.WaitForSingleObject(handle,0)))
    state['first_seen'][str(pid)]=row
    persist() # Publish each member before touching any subsequent member.
   rows=[]
   for pid,handle in kept.items():
    cached=state['first_seen'][str(pid)]
    row={'pid':pid,'observed_filetime_utc':stamp(),'listed_now':pid in pids,'cached_identity_query_filetime_utc':cached['identity_query_filetime_utc'],'cached_creation_filetime_utc':(cached.get('times_at_first_query') or {}).get('creation_filetime_utc'),'cached_physical_image':cached.get('physical_image_at_first_query')}
    field(row,'current_times',lambda:j.times(k,handle))
    field(row,'IsProcessInJob',lambda:membership(handle))
    field(row,'wait_state',lambda:int(k.WaitForSingleObject(handle,0)))
    field(row,'exit_code',lambda:exitcode(handle))
    rows.append(row)
   state['snapshots'][label]=rows
   persist()
 except BaseException as exc:
  state['observer_errors'].append({'operation':'observe '+label,'error':repr(exc),'observed_filetime_utc':stamp()});persist()
def watch():
 while not stop.is_set():observe('last_running_snapshot');stop.wait(.02)
monitor=threading.Thread(target=watch)
real_settle=j.settle_job
def settle(k_arg,job,known):
 stop.set()
 if monitor.ident:monitor.join()
 observe('after_launcher_gone_before_original_settle')
 try:return real_settle(k_arg,job,known)
 finally:observe('after_original_settle')
j.settle_job=settle
started=time.perf_counter()
try:
 code="import subprocess,sys,time;subprocess.run([sys.executable,'-c','import time;time.sleep(.3)'],check=True);time.sleep(.1)"
 result=j.supervise({'executable':str(ROOT/'.venv-gpu/Scripts/python.exe'),'arguments':['-c',code]},BASE/'original_normal_tree',BASE/'diagnostic.lock',{str(source):original_sha},ROOT,on_started=lambda _:monitor.start(),provenance={'purpose':'authorized second normal venv tree observation, not QA'})
 state['supervisor_result']={name:result.get(name) for name in ('status','launcher','launcher_exit_code','worker_exit_code','orphans_cleaned','job_total_processes','job_active_processes','cleanup_verified','wall_seconds','job_cpu_user_seconds','job_cpu_kernel_seconds')}
except BaseException as exc:
 state['diagnostic_error']=repr(exc)
 raise
finally:
 stop.set()
 if monitor.ident:monitor.join()
 state['source_after']=j.sha(source)
 state['diagnostic_wall_seconds']=time.perf_counter()-started
 receipt=BASE/'original_normal_tree/receipt.json'
 if receipt.is_file():state['receipt_path']=str(receipt);state['receipt_sha256']=j.sha(receipt)
 persist()
 for handle in kept.values():k.CloseHandle(handle)
 print(json.dumps(state,indent=2))
