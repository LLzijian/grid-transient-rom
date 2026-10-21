"""Windows Job ownership without elapsed, CPU or process-count quotas."""
from __future__ import annotations
import ctypes,hashlib,json,os,subprocess,sys,time,traceback
from ctypes import wintypes as W
from pathlib import Path

def atomic(path,value):
    path=Path(path);temp=path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(value,indent=2,allow_nan=False),encoding='utf-8');os.replace(temp,path)
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def verify_files(files):
    result={}
    for name,expected in files.items():
        p=Path(name).resolve()
        if p.suffix.lower() in ('.h5','.hdf5'):raise ValueError('mixed dataset hashing forbidden')
        actual=sha(p)
        if actual!=expected:raise ValueError('bound source changed: '+str(p))
        result[str(p)]=actual
    return result

class STARTUPINFO(ctypes.Structure):
    _fields_=[('cb',W.DWORD),('lpReserved',W.LPWSTR),('lpDesktop',W.LPWSTR),('lpTitle',W.LPWSTR),('dwX',W.DWORD),('dwY',W.DWORD),('dwXSize',W.DWORD),('dwYSize',W.DWORD),('dwXCountChars',W.DWORD),('dwYCountChars',W.DWORD),('dwFillAttribute',W.DWORD),('dwFlags',W.DWORD),('wShowWindow',W.WORD),('cbReserved2',W.WORD),('lpReserved2',ctypes.POINTER(ctypes.c_byte)),('hStdInput',W.HANDLE),('hStdOutput',W.HANDLE),('hStdError',W.HANDLE)]
class STARTUPINFOEX(ctypes.Structure):_fields_=[('StartupInfo',STARTUPINFO),('lpAttributeList',ctypes.c_void_p)]
class PROCESS_INFORMATION(ctypes.Structure):_fields_=[('hProcess',W.HANDLE),('hThread',W.HANDLE),('dwProcessId',W.DWORD),('dwThreadId',W.DWORD)]
class SECURITY_ATTRIBUTES(ctypes.Structure):_fields_=[('nLength',W.DWORD),('lpSecurityDescriptor',ctypes.c_void_p),('bInheritHandle',W.BOOL)]
class BASIC_LIMIT(ctypes.Structure):
    _fields_=[('PerProcessUserTimeLimit',ctypes.c_longlong),('PerJobUserTimeLimit',ctypes.c_longlong),('LimitFlags',W.DWORD),('MinimumWorkingSetSize',ctypes.c_size_t),('MaximumWorkingSetSize',ctypes.c_size_t),('ActiveProcessLimit',W.DWORD),('Affinity',ctypes.c_size_t),('PriorityClass',W.DWORD),('SchedulingClass',W.DWORD)]
class IO_COUNTERS(ctypes.Structure):_fields_=[(n,ctypes.c_ulonglong) for n in ('ReadOperationCount','WriteOperationCount','OtherOperationCount','ReadTransferCount','WriteTransferCount','OtherTransferCount')]
class EXTENDED_LIMIT(ctypes.Structure):_fields_=[('BasicLimitInformation',BASIC_LIMIT),('IoInfo',IO_COUNTERS),('ProcessMemoryLimit',ctypes.c_size_t),('JobMemoryLimit',ctypes.c_size_t),('PeakProcessMemoryUsed',ctypes.c_size_t),('PeakJobMemoryUsed',ctypes.c_size_t)]
class ACCOUNTING(ctypes.Structure):
    _fields_=[('TotalUserTime',ctypes.c_longlong),('TotalKernelTime',ctypes.c_longlong),('ThisPeriodTotalUserTime',ctypes.c_longlong),('ThisPeriodTotalKernelTime',ctypes.c_longlong),('TotalPageFaultCount',W.DWORD),('TotalProcesses',W.DWORD),('ActiveProcesses',W.DWORD),('TotalTerminatedProcesses',W.DWORD)]
def api():
    if os.name!='nt':raise OSError('Windows only')
    k=ctypes.WinDLL('kernel32',use_last_error=True)
    signatures={
      'GetCurrentProcess':([],W.HANDLE),'GetProcessTimes':([W.HANDLE]+[ctypes.POINTER(W.FILETIME)]*4,W.BOOL),
      'OpenProcess':([W.DWORD,W.BOOL,W.DWORD],W.HANDLE),'CloseHandle':([W.HANDLE],W.BOOL),
      'CreateJobObjectW':([ctypes.c_void_p,W.LPCWSTR],W.HANDLE),'SetInformationJobObject':([W.HANDLE,ctypes.c_int,ctypes.c_void_p,W.DWORD],W.BOOL),
      'QueryInformationJobObject':([W.HANDLE,ctypes.c_int,ctypes.c_void_p,W.DWORD,ctypes.c_void_p],W.BOOL),'AssignProcessToJobObject':([W.HANDLE,W.HANDLE],W.BOOL),'TerminateJobObject':([W.HANDLE,W.UINT],W.BOOL),
      'TerminateProcess':([W.HANDLE,W.UINT],W.BOOL),'ResumeThread':([W.HANDLE],W.DWORD),'WaitForSingleObject':([W.HANDLE,W.DWORD],W.DWORD),
      'GetExitCodeProcess':([W.HANDLE,ctypes.POINTER(W.DWORD)],W.BOOL),'SetHandleInformation':([W.HANDLE,W.DWORD,W.DWORD],W.BOOL),
      'CreateFileW':([W.LPCWSTR,W.DWORD,W.DWORD,ctypes.POINTER(SECURITY_ATTRIBUTES),W.DWORD,W.DWORD,W.HANDLE],W.HANDLE),
      'InitializeProcThreadAttributeList':([ctypes.c_void_p,W.DWORD,W.DWORD,ctypes.POINTER(ctypes.c_size_t)],W.BOOL),
      'UpdateProcThreadAttribute':([ctypes.c_void_p,W.DWORD,ctypes.c_size_t,ctypes.c_void_p,ctypes.c_size_t,ctypes.c_void_p,ctypes.c_void_p],W.BOOL),
      'DeleteProcThreadAttributeList':([ctypes.c_void_p],None),
      'CreateProcessW':([W.LPCWSTR,W.LPWSTR,ctypes.c_void_p,ctypes.c_void_p,W.BOOL,W.DWORD,ctypes.c_void_p,W.LPCWSTR,ctypes.c_void_p,ctypes.POINTER(PROCESS_INFORMATION)],W.BOOL)}
    for name,(args,ret) in signatures.items():f=getattr(k,name);f.argtypes=args;f.restype=ret
    return k
def checked(value):
    if not value:raise ctypes.WinError(ctypes.get_last_error())
    return value
def times(k,handle):
    ts=[W.FILETIME() for _ in range(4)];checked(k.GetProcessTimes(handle,*[ctypes.byref(t) for t in ts]))
    values=[(int(t.dwHighDateTime)<<32)|int(t.dwLowDateTime) for t in ts]
    return {'creation_filetime_utc':values[0],'exit_filetime_utc':values[1],'kernel_100ns':values[2],'user_100ns':values[3]}
def identity():
    k=api();return {'pid':os.getpid(),'executable':str(Path(sys.executable).resolve()),**times(k,k.GetCurrentProcess())}
def identity_state(value):
    k=api();handle=k.OpenProcess(0x1000|0x100000,False,int(value['pid']))
    if not handle:
        if ctypes.get_last_error()==87:return 'GONE'
        raise ctypes.WinError(ctypes.get_last_error())
    try:
        if times(k,handle)['creation_filetime_utc']!=value['creation_filetime_utc']:return 'REUSED_PID'
        code=k.WaitForSingleObject(handle,0)
        if code==0:return 'GONE'
        if code==258:return 'RUNNING'
        raise ctypes.WinError(ctypes.get_last_error())
    finally:k.CloseHandle(handle)
def reconcile_lock(lock,expected,closure_receipt):
    """Explicit root-directed reconciliation; never delete a live or changed owner."""
    lock=Path(lock);actual=json.loads(lock.read_text(encoding='utf-8'))
    if actual!=expected:raise ValueError('lock identity differs')
    if identity_state(actual['supervisor'])!='GONE':raise ValueError('supervisor still live or PID reused')
    if actual.get('worker') and identity_state(actual['worker'])!='GONE':raise ValueError('worker still live or PID reused')
    proof=json.loads(Path(closure_receipt).read_text(encoding='utf-8'))
    if proof.get('supervisor')!=actual['supervisor'] or proof.get('worker')!=actual.get('worker') or proof.get('job_active_processes')!=0 or proof.get('cleanup_verified') is not True:raise ValueError('matching actual Job closure receipt required')
    lock.unlink()

def supervise(command,output,lock,files,working_directory,worker_receipt=None,on_started=None,on_created=None,provenance=None):
    """No time limit. Job membership owns cleanup, never a naked PID."""
    k=api();output=Path(output).resolve();lock=Path(lock).resolve()
    supervisor=identity();owner={'supervisor':supervisor,'worker':None}
    lock.parent.mkdir(parents=True,exist_ok=True)
    with lock.open('x',encoding='utf-8') as f:json.dump(owner,f)
    record={**(provenance or {}),'status':'UNRESOLVED','supervisor':supervisor,'worker':None,'command':command,'runtime_limits':None,'cpu_limits':None,'active_process_limit':None}
    job=None;pi=PROCESS_INFORMATION();logs=[];attrs=None;attrs_ready=False;assigned=False;clean=False;started=time.perf_counter();raised=None
    try:
        output.mkdir(parents=True,exist_ok=False);record['source_before']=verify_files(files)
        job=checked(k.CreateJobObjectW(None,None));checked(k.SetHandleInformation(job,1,0))
        limits=EXTENDED_LIMIT();limits.BasicLimitInformation.LimitFlags=0x2000 # KILL_ON_JOB_CLOSE only.
        checked(k.SetInformationJobObject(job,9,ctypes.byref(limits),ctypes.sizeof(limits)))
        sa=SECURITY_ATTRIBUTES(ctypes.sizeof(SECURITY_ATTRIBUTES),None,True)
        for name in ('stdout.txt','stderr.txt'):
            handle=k.CreateFileW(str(output/name),0x40000000,3,ctypes.byref(sa),1,0x80,None)
            if handle in (None,ctypes.c_void_p(-1).value):raise ctypes.WinError(ctypes.get_last_error())
            logs.append(handle)
        size=ctypes.c_size_t();k.InitializeProcThreadAttributeList(None,2,0,ctypes.byref(size));attrs=ctypes.create_string_buffer(size.value)
        checked(k.InitializeProcThreadAttributeList(attrs,2,0,ctypes.byref(size)))
        attrs_ready=True
        inherited=(W.HANDLE*2)(*logs);checked(k.UpdateProcThreadAttribute(attrs,0,0x20002,inherited,ctypes.sizeof(inherited),None,None))
        jobs=(W.HANDLE*1)(job)
        checked(k.UpdateProcThreadAttribute(attrs,0,0x2000D,jobs,ctypes.sizeof(jobs),None,None)) # Atomic creation in Job; unsupported Windows fails closed.
        si=STARTUPINFOEX();si.StartupInfo.cb=ctypes.sizeof(si);si.StartupInfo.dwFlags=0x100;si.StartupInfo.hStdOutput=logs[0];si.StartupInfo.hStdError=logs[1];si.StartupInfo.hStdInput=None;si.lpAttributeList=ctypes.cast(attrs,ctypes.c_void_p)
        exe=str(Path(command['executable']).resolve());args=[exe,*command['arguments']]
        checked(k.CreateProcessW(exe,ctypes.create_unicode_buffer(subprocess.list2cmdline(args)),None,None,True,0x4|0x08000000|0x80000,None,str(Path(working_directory).resolve()),ctypes.byref(si),ctypes.byref(pi)))
        assigned=True
        worker={'pid':int(pi.dwProcessId),'executable':exe,**times(k,pi.hProcess)};owner['worker']=worker;record['worker']=worker
        atomic(lock,owner)
        atomic(output/'started.json',{'supervisor':supervisor,'worker':worker,'source_before':record['source_before']})
        if on_created:on_created(worker)
        if k.ResumeThread(pi.hThread)==0xffffffff:raise ctypes.WinError(ctypes.get_last_error())
        if on_started:on_started(worker)
        # Infinite wait: cancellation raises into the owned-Job cleanup below.
        while True:
            result=k.WaitForSingleObject(pi.hProcess,500)
            if result==0:break
            if result!=258:raise ctypes.WinError(ctypes.get_last_error())
        code=W.DWORD();checked(k.GetExitCodeProcess(pi.hProcess,ctypes.byref(code)));record['exit_code']=int(code.value)
        account=ACCOUNTING();checked(k.QueryInformationJobObject(job,1,ctypes.byref(account),ctypes.sizeof(account),None))
        record['orphans_cleaned']=bool(account.ActiveProcesses)
        if account.ActiveProcesses:checked(k.TerminateJobObject(job,125))
        while True:
            checked(k.QueryInformationJobObject(job,1,ctypes.byref(account),ctypes.sizeof(account),None))
            if account.ActiveProcesses==0:break
            time.sleep(.05) # Cleanup synchronization only; no deadline.
        clean=True;record['job_active_processes']=0
        record['job_cpu_user_seconds']=account.TotalUserTime/1e7;record['job_cpu_kernel_seconds']=account.TotalKernelTime/1e7
        record['job_total_processes']=int(account.TotalProcesses);record['worker_end']=times(k,pi.hProcess)
        if worker_receipt:
            receipt=json.loads(Path(worker_receipt).read_text(encoding='utf-8'))
            if receipt.get('pid')!=worker['pid'] or receipt.get('creation_filetime_utc')!=worker['creation_filetime_utc'] or Path(receipt.get('executable','')).resolve()!=Path(exe):raise ValueError('actual worker receipt identity mismatch')
            record['worker_receipt']=receipt
        record['source_after']=verify_files(files)
        record['status']='COMPLETED' if code.value==0 and not record['orphans_cleaned'] else 'FAILED'
    except BaseException as error:
        raised=error;record.update(error=repr(error),traceback=traceback.format_exc())
    finally:
        try:
            if pi.hProcess and not clean:
                if assigned:checked(k.TerminateJobObject(job,126))
                else:checked(k.TerminateProcess(pi.hProcess,126)) # Retained suspended handle before assignment.
                if assigned:
                    account=ACCOUNTING()
                    while True:
                        checked(k.QueryInformationJobObject(job,1,ctypes.byref(account),ctypes.sizeof(account),None))
                        if account.ActiveProcesses==0:break
                        time.sleep(.05)
                    record.update(job_active_processes=0,job_cpu_user_seconds=account.TotalUserTime/1e7,job_cpu_kernel_seconds=account.TotalKernelTime/1e7,job_total_processes=int(account.TotalProcesses))
                else:k.WaitForSingleObject(pi.hProcess,0xffffffff);record['job_active_processes']=0
                clean=True;record['worker_end']=times(k,pi.hProcess)
        except BaseException as error:
            record['cleanup_error']=repr(error)
            if raised is None:raised=error
        record['cleanup_verified']=clean or not pi.hProcess;record['wall_seconds']=time.perf_counter()-started
        try:record['supervisor_end']=times(k,k.GetCurrentProcess())
        except BaseException as error:record['supervisor_identity_error']=repr(error)
        record['supervisor_end_observed_filetime_utc']=int(time.time_ns()//100+116444736000000000)
        try:
            record.setdefault('source_after',verify_files(files))
        except BaseException as error:
            record['source_after_error']=repr(error);record['status']='UNRESOLVED'
            if raised is None:raised=error
        try:
            if output.is_dir():atomic(output/'receipt.json',record)
        finally:
            if attrs_ready:k.DeleteProcThreadAttributeList(attrs)
            for handle in (pi.hThread,pi.hProcess,*logs,job):
                if handle:k.CloseHandle(handle)
        if record['cleanup_verified'] and json.loads(lock.read_text(encoding='utf-8'))==owner:lock.unlink()
    if raised:raise raised
    return record

