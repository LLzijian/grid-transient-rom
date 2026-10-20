"""One fixed post-freeze JSON extraction worker; supervised externally."""
import argparse
import ctypes
from ctypes import wintypes
import hashlib
import json
import os
from pathlib import Path
import runpy
import sys
import threading
import time
import traceback

BASE=Path(__file__).resolve().parents[1]
ROOT=BASE.parents[3]
SCRIPT=BASE/'postprocess/extract_frequency_dissipation_v2.py'
SCRIPT_SHA='ddeeab6c556a1961b55a437b15e5dd24674b575da088ee8c992185cd8f5800ed'
SOURCE_COMMIT='399e50d73531dd79a4613cf7e235fc4db061f6ee'
MANIFEST_SHA='97e8a957dc9dff098368b69234d3976ffad4bc6a408ebe550541c3d46dc1d81c'

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def write(path,value):
    with Path(path).open('x',encoding='utf-8') as stream:
        json.dump(value,stream,indent=2,allow_nan=False);stream.write('\n')

def identity():
    kernel=ctypes.WinDLL('kernel32',use_last_error=True)
    kernel.GetCurrentProcess.argtypes=[]
    kernel.GetCurrentProcess.restype=wintypes.HANDLE
    kernel.GetProcessTimes.argtypes=[wintypes.HANDLE]+[ctypes.POINTER(wintypes.FILETIME)]*4
    kernel.GetProcessTimes.restype=wintypes.BOOL
    times=[wintypes.FILETIME() for _ in range(4)]
    if not kernel.GetProcessTimes(kernel.GetCurrentProcess(),*[ctypes.byref(t) for t in times]):
        raise ctypes.WinError(ctypes.get_last_error())
    return {'pid':os.getpid(),'creation_filetime_utc':(times[0].dwHighDateTime<<32)|times[0].dwLowDateTime,
            'executable':str(Path(sys.executable).resolve()),'argv':sys.argv,'cwd':str(Path.cwd().resolve())}

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--max-seconds',type=float,required=True)
    parser.add_argument('--source-commit',required=True)
    parser.add_argument('--extra',nargs=argparse.REMAINDER,required=True)
    a=parser.parse_args()
    output=a.output.resolve()
    if BASE!=(ROOT/'private_data/closure_mechanism_20261019/P2/frequency_dissipation01').resolve():raise ValueError('fixed W study location required')
    if a.max_seconds!=40. or a.source_commit!=SOURCE_COMMIT:raise ValueError('fixed source and forty-second worker required')
    if not output.is_relative_to(BASE) or output==BASE:raise ValueError('exclusive study-local worker folder required')
    output.mkdir(parents=True,exist_ok=False)
    started=time.perf_counter()
    def expired():
        try:write(output/'timeout.json',{'self_timeout':True,'max_seconds':40.,'elapsed_seconds':time.perf_counter()-started})
        finally:os._exit(124)
    timer=threading.Timer(40.,expired);timer.daemon=True;timer.start()
    original_argv=sys.argv[:]
    try:
        write(output/'worker.json',identity())
        args=a.extra
        def flag(name):
            if args.count(name)!=1 or args.index(name)+1>=len(args):raise ValueError('one explicit '+name+' required')
            return args[args.index(name)+1]
        target=Path(flag('--output')).resolve()
        if not target.is_relative_to(ROOT) or target.is_relative_to(output) or target.suffix!='.json' or target.exists():raise ValueError('new separate workspace JSON output required')
        if Path(flag('--manifest')).resolve()!=BASE/'frozen_evaluation_endpoints.json' or flag('--manifest-sha256')!=MANIFEST_SHA:raise ValueError('fixed complete W freeze required')
        if args.count('--after-complete')!=1:raise ValueError('explicit after-complete required')
        if '--max-seconds' in args and float(flag('--max-seconds'))!=40.:raise ValueError('fixed extractor deadline required')
        if sha(SCRIPT)!=SCRIPT_SHA:raise ValueError('fixed pure JSON extractor changed')
        sys.argv=[str(SCRIPT),*args]
        try:runpy.run_path(str(SCRIPT),run_name='__main__')
        except SystemExit as exc:
            if exc.code not in (None,0):raise
        if sha(SCRIPT)!=SCRIPT_SHA:raise ValueError('extractor changed during execution')
        result=json.loads(target.read_text(encoding='utf-8-sig'))
        if result.get('status')!='COMPLETED':raise ValueError('extractor did not produce complete facts')
        write(output/'result.json',{'status':'COMPLETED','source_commit':SOURCE_COMMIT,'extractor_sha256':SCRIPT_SHA,
              'output_path':str(target),'output_sha256':sha(target),'elapsed_seconds':time.perf_counter()-started})
        return 0
    except BaseException as exc:
        write(output/'failure.json',{'status':'FAILED','error':repr(exc),'traceback':traceback.format_exc(),'elapsed_seconds':time.perf_counter()-started})
        raise
    finally:
        sys.argv=original_argv;timer.cancel()

if __name__=='__main__':raise SystemExit(main())
