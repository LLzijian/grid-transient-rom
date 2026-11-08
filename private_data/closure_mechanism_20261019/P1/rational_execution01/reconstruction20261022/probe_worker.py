"""Fixed descendant cases and fourteen CUDA scalars; no scientific datasets."""
import argparse,ctypes,json,os,subprocess,sys,threading,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[5]
sys.path.insert(0,str(ROOT))
from experiments import uncapped_windows_job_20261021 as job
from experiments.rational_python_runtime_20261022 import python_command,command_environment

def start_child(command,trace,capture=False):
    flags=subprocess.DETACHED_PROCESS if os.name=='nt' else 0
    options={'env':command_environment(command),'creationflags':flags}
    if capture:options.update(stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True,encoding='utf-8')
    child=subprocess.Popen([command['executable'],*command['arguments']],**options)
    job.atomic(trace,[{'via':'subprocess','creationflags':flags,'child_pid':child.pid,
        **{key:command[key] for key in ('executable','logical_executable','arguments')}}])
    return child

def wait_file(path,child=None,error=None):
    while not path.is_file():
        if error and error.is_file():raise RuntimeError(error.read_text(encoding='utf-8'))
        if child is not None and child.poll() is not None:raise RuntimeError('child exited before handshake: '+str(path))
        time.sleep(.01)
    return json.loads(path.read_text(encoding='utf-8'))

def metadata(out):
    import torch
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic=True;torch.backends.cudnn.benchmark=False
    k=ctypes.WinDLL('kernel32',use_last_error=True)
    k.GetModuleFileNameW.argtypes=[ctypes.c_void_p,ctypes.c_wchar_p,ctypes.c_uint32];k.GetModuleFileNameW.restype=ctypes.c_uint32
    buffer=ctypes.create_unicode_buffer(32768)
    if not k.GetModuleFileNameW(ctypes.pythonapi._handle,buffer,len(buffer)):raise ctypes.WinError(ctypes.get_last_error())
    dll=Path(buffer.value).resolve()
    value=(torch.arange(1,15,device='cuda',dtype=torch.float64)**2).cpu().tolist();torch.cuda.synchronize()
    job.atomic(out/'metadata.json',{'python_version':sys.version.split()[0],
        'python_dll':{'path':str(dll),'sha256':job.sha(dll)},
        'logical_executable':str(Path(sys.executable).resolve()),
        'physical_executable':job.physical_image(job.api(),job.api().GetCurrentProcess()),
        'prefix':sys.prefix,'base_prefix':sys.base_prefix,'package_paths':sys.path,
        'torch_version':torch.__version__,'torch_file':torch.__file__,
        'cuda_version':torch.version.cuda,'cuda_device':torch.cuda.get_device_name(0),
        'deterministic_algorithms':torch.are_deterministic_algorithms_enabled(),
        'cudnn_deterministic':torch.backends.cudnn.deterministic,'cudnn_benchmark':torch.backends.cudnn.benchmark,
        'num_threads':torch.get_num_threads(),'interop_threads':torch.get_num_interop_threads(),'tensor14':value})

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--role',choices=('normal','grandchild','pipes','orphan','supervisor_exit','child','branch','leaf','pipe_child','blocked_child'),required=True)
    p.add_argument('--scheme',choices=('direct_base_venv',),required=True)
    p.add_argument('--out',type=Path,required=True);p.add_argument('--source-commit',required=True)
    a=p.parse_args();a.out.mkdir(parents=True,exist_ok=True)
    def command(role):
        return python_command(sys.executable,[str(Path(__file__).resolve()),'--role',role,'--scheme',a.scheme,
            '--out',str(a.out),'--source-commit',a.source_commit],a.scheme)
    def publish(name,group=None):
        row=job.identity()
        if group:row.update(source_commit=a.source_commit,phase='runtime_observation',group=group)
        job.atomic(a.out/name,row)
    if a.role in ('child','branch','leaf','pipe_child','blocked_child'):
        publish('grandchild.json' if a.role=='leaf' else 'child.json', 'supervisor_exit' if a.role=='blocked_child' else None)
        if a.role=='branch':
            child=start_child(command('leaf'),a.out/'spawns_child.json')
            wait_file(a.out/'grandchild.json',child)
            if child.wait()!=0:raise RuntimeError('grandchild failed')
        elif a.role=='pipe_child':
            print('fixed stdout');print('fixed stderr',file=sys.stderr)
        elif a.role=='blocked_child' or (a.out/'immortal').exists():threading.Event().wait()
        return
    publish('worker.json',a.role)
    while not (a.out/'owned').is_file():time.sleep(.01)
    if a.role=='normal':metadata(a.out)
    if a.role=='supervisor_exit':
        native=job.api();flags=[]
        class RecordingAPI:
            def __getattr__(self,name):return getattr(native,name)
            def CreateProcessW(self,*args):
                flags.append(int(args[5]));return native.CreateProcessW(*args)
        job.api=lambda:RecordingAPI()
        cmd=command('blocked_child')
        def created(identity):
            job.atomic(a.out/'spawns_parent.json',[{'via':'native_job','creationflags':flags[-1],
                'child_pid':identity['pid'],**{key:cmd[key] for key in ('executable','logical_executable','arguments')}}])
        def crash(identity):
            wait_file(a.out/'witness_ready.json',error=a.out/'witness_error.json')
            os._exit(121)
        job.supervise(cmd,a.out/'inner_native',a.out/'inner.lock',{},ROOT,
            worker_receipt=a.out/'child.json',on_created=created,on_worker_owned=crash,
            provenance={'source_commit':a.source_commit,'phase':'runtime_observation','group':a.role})
        raise RuntimeError('fixed supervisor exit did not occur')
    if a.role=='orphan':(a.out/'immortal').write_text('negative application child',encoding='utf-8')
    child_role={'grandchild':'branch','pipes':'pipe_child'}.get(a.role,'child')
    child=start_child(command(child_role),a.out/'spawns_parent.json',capture=a.role=='pipes')
    wait_file(a.out/'child.json',child)
    if a.role=='pipes':
        stdout,stderr=child.communicate()
        job.atomic(a.out/'pipes.json',{'stdout':stdout,'stderr':stderr,'returncode':child.returncode})
        if stdout!='fixed stdout\n' or stderr!='fixed stderr\n' or child.returncode!=0:raise RuntimeError('fixed pipe contents differ')
    elif a.role!='orphan' and child.wait()!=0:raise RuntimeError('normal child failed')

if __name__=='__main__':main()
