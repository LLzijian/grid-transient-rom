"""Registered runtime observation only: identities and fourteen CUDA scalars; no datasets."""
import argparse,ctypes,json,subprocess,sys,threading,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[5]
sys.path.insert(0,str(ROOT))
from experiments import uncapped_windows_job_20261021 as job
from experiments.rational_python_runtime_20261022 import python_command,command_environment

def main():
    p=argparse.ArgumentParser();p.add_argument('--role',choices=('normal','orphan','child'),required=True)
    p.add_argument('--scheme',required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--source-commit',required=True)
    a=p.parse_args();a.out.mkdir(parents=True,exist_ok=True)
    if sys.stdout is None:sys.stdout=(a.out/'pythonw_stdout.txt').open('a',encoding='utf-8',buffering=1)
    if sys.stderr is None:sys.stderr=(a.out/'pythonw_stderr.txt').open('a',encoding='utf-8',buffering=1)
    if a.role=='child':
        job.atomic(a.out/'child.json',job.identity())
        if (a.out/'immortal').exists():threading.Event().wait()
        return
    job.atomic(a.out/'worker.json',{**job.identity(),'source_commit':a.source_commit,'phase':'runtime_observation','group':a.role})
    while not (a.out/'owned').exists():time.sleep(.01)
    if a.role=='normal':
        import torch
        torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.deterministic=True;torch.backends.cudnn.benchmark=False
        k=ctypes.WinDLL('kernel32',use_last_error=True)
        k.GetModuleFileNameW.argtypes=[ctypes.c_void_p,ctypes.c_wchar_p,ctypes.c_uint32];k.GetModuleFileNameW.restype=ctypes.c_uint32
        buffer=ctypes.create_unicode_buffer(32768)
        if not k.GetModuleFileNameW(ctypes.pythonapi._handle,buffer,len(buffer)):raise ctypes.WinError(ctypes.get_last_error())
        dll=Path(buffer.value).resolve()
        value=(torch.arange(1,15,device='cuda',dtype=torch.float64)**2).cpu().tolist();torch.cuda.synchronize()
        job.atomic(a.out/'metadata.json',{'python_version':sys.version.split()[0],
            'python_dll':{'path':str(dll),'sha256':job.sha(dll)},
            'logical_executable':str(Path(sys.executable).resolve()),
            'physical_executable':job.physical_image(job.api(),job.api().GetCurrentProcess()),
            'prefix':sys.prefix,'base_prefix':sys.base_prefix,'package_paths':sys.path,
            'torch_version':torch.__version__,'torch_file':torch.__file__,
            'cuda_version':torch.version.cuda,'cuda_device':torch.cuda.get_device_name(0),
            'deterministic_algorithms':torch.are_deterministic_algorithms_enabled(),
            'cudnn_deterministic':torch.backends.cudnn.deterministic,'cudnn_benchmark':torch.backends.cudnn.benchmark,
            'num_threads':torch.get_num_threads(),'interop_threads':torch.get_num_interop_threads(),'tensor14':value})
    else:(a.out/'immortal').write_text('negative application child',encoding='utf-8')
    args=[str(Path(__file__).resolve()),'--role','child','--scheme',a.scheme,'--out',str(a.out),'--source-commit',a.source_commit]
    command=({'executable':sys.executable,'arguments':args} if a.scheme=='reference' else python_command(sys.executable,args,a.scheme))
    child=subprocess.Popen([command['executable'],*command['arguments']],env=command_environment(command))
    while not (a.out/'child.json').exists():
        if child.poll() is not None:raise RuntimeError('child exited before identity handshake')
        time.sleep(.01)
    if a.role=='normal':
        if child.wait()!=0:raise RuntimeError('normal child failed')

if __name__=='__main__':main()
