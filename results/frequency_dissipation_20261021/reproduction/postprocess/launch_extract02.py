"""Fixed one-shot JSON metadata supervision; never launches science jobs."""
import hashlib,json,os,subprocess,sys,time,traceback
from pathlib import Path
BASE=Path(__file__).resolve().parents[1]
ROOT=BASE.parents[3]
SOURCE='399e50d73531dd79a4613cf7e235fc4db061f6ee'
COMMAND=BASE/'commands/extract02.json'
OUTPUT=BASE/'native_extract02'
WORKER=BASE/'postprocess/extract02/worker.json'
COMMAND_SHA='0f281f87a94cad554f3c82b8cd9f7c5202f426de8ad82531fdfd829aeae3e0f3'
FROZEN_SHA='97e8a957dc9dff098368b69234d3976ffad4bc6a408ebe550541c3d46dc1d81c'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def write(p,v):
    with Path(p).open('x',encoding='utf-8') as stream:json.dump(v,stream,indent=2,allow_nan=False);stream.write('\n')
def require(ok,message):
    if not ok:raise ValueError(message)
def files():
    return [Path(__file__).resolve(),COMMAND,BASE/'postprocess/extract_worker_v2.py',BASE/'postprocess/extract_frequency_dissipation_v2.py',ROOT/'private_data/closure_mechanism_20261019/P2/quartic_memory01/postprocess/extract_quartic_memory.py',ROOT/'experiments/extract_spatial_extrema_20261020.py',ROOT/'experiments/native_process_receipt_20261020.ps1',BASE/'frozen_evaluation_endpoints.json']
def snapshot():return {str(p.relative_to(ROOT)):sha(p) for p in files()}
def main():
    require(BASE==(ROOT/'private_data/closure_mechanism_20261019/P2/frequency_dissipation01').resolve(),'fixed study location')
    require(not OUTPUT.exists() and not WORKER.parent.exists() and not (BASE/'native_active.lock').exists(),'unique metadata output and no live science owner')
    names=('native_qa01','native_cost01','native_train_6101_w','native_train_6102_w','native_train_6103_w','native_eval_6101_w','native_eval_6102_w','native_eval_6103_w')
    prior=[BASE/name/'execution.json' for name in names]
    old=BASE/'native_extract01';old_record=read(old/'execution.json');old_inner=read(old/'inner/receipt.json');old_outer=read(old/'outer/native.json')
    require(old_record.get('source_before')==old_record.get('source_after'),'metadata01 source mismatch')
    require(old_inner.get('native_receipt_complete') is True and old_outer.get('native_receipt_complete') is True and old_inner.get('cleanup_verified') is True and old_inner.get('fixture_identity_after')==old_inner.get('supervised_identity_after')=='GONE' and old_inner.get('exit_code')==old_outer.get('exit_code')==1,'metadata01 ownership unresolved')
    require(old_record.get('charged_seconds')==2.556612499989569,'metadata01 fee mismatch')
    previous_metadata={'execution_sha256':sha(old/'execution.json'),'inner_sha256':sha(old/'inner/receipt.json'),'outer_sha256':sha(old/'outer/native.json'),'charged_seconds':old_record['charged_seconds'],'separate_previous_charge_not_added_again':True}
    require(len(prior)==8,'all eight science jobs required')
    categories={'qa':0,'cost':0,'train':0,'evaluate':0};receipts={}
    for path in prior:
        r=read(path);inner=r.get('inner',{});outer=r.get('outer',{})
        require(r.get('status')=='COMPLETED' and r.get('source_before')==r.get('source_after'),'science receipt incomplete/source mismatch')
        require(inner.get('exit_code')==outer.get('exit_code')==0 and inner.get('native_receipt_complete') is True and outer.get('native_receipt_complete') is True,'science exits incomplete')
        require(inner.get('cleanup_verified') is True and inner.get('fixture_identity_after')==inner.get('supervised_identity_after')=='GONE','science process ownership unresolved')
        require(r['category'] in categories,'unexpected prior job');categories[r['category']]+=1;receipts[str(path)]=sha(path)
    require(categories=={'qa':1,'cost':1,'train':3,'evaluate':3},'science categories incomplete')
    expected={COMMAND:COMMAND_SHA,BASE/'frozen_evaluation_endpoints.json':FROZEN_SHA,BASE/'postprocess/extract_worker_v2.py':'af9decc36e551c4c35ad6d423c775bd662c48a11654fdaec3a703be3af966d7c',BASE/'postprocess/extract_frequency_dissipation_v2.py':'ddeeab6c556a1961b55a437b15e5dd24674b575da088ee8c992185cd8f5800ed',ROOT/'private_data/closure_mechanism_20261019/P2/quartic_memory01/postprocess/extract_quartic_memory.py':'b4fbbd8beba780f7afd9513368bc9cdbd62fea488523494d5a9ac5b5475f954c'}
    for path,digest in expected.items():require(sha(path)==digest,'fixed extraction input changed: '+str(path))
    command=read(COMMAND)
    require(Path(command['executable']).resolve()==ROOT/'.venv-gpu/Scripts/python.exe' and command['arguments_line']==subprocess.list2cmdline(command['arguments']),'command executable/argv differs')
    before=snapshot();OUTPUT.mkdir(exist_ok=False)
    ps=Path(os.environ['SystemRoot'])/'System32/WindowsPowerShell/v1.0/powershell.exe'
    args=[str(ps),'-NoProfile','-NonInteractive','-ExecutionPolicy','Bypass','-File',str(ROOT/'experiments/native_process_receipt_20261020.ps1'),'-CommandJson',str(COMMAND),'-WorkingDirectory',str(ROOT),'-Output',str(OUTPUT/'inner'),'-TimeoutSeconds','70','-MaxSeconds','100','-SourceCommit',SOURCE,'-FixtureIdentityJson',str(WORKER)]
    record={'category':'metadata_extract','status':'RUNNING','source_commit':SOURCE,'source_before':before,'prior_receipt_sha256':receipts,'command_sha256':COMMAND_SHA,'native_cap':100,'ps_timeout':70,'self_seconds':40,'includes_children_once':True,'previous_metadata01':previous_metadata}
    write(OUTPUT/'launch.json',record);started=time.perf_counter()
    try:
        sys.path[:0]=[str(ROOT/'experiments'),str(ROOT/'src')]
        from fixed_edge_pair_20261019 import native_job
        outer=native_job(args,OUTPUT/'outer',100.,worker_json=WORKER);record['outer']=outer;record['charged_seconds']=outer['seconds']
        inner=read(OUTPUT/'inner/receipt.json');record['inner']=inner
        worker=read(WORKER);record['worker']=worker
        result=read(WORKER.parent/'result.json')
        record.update(inner=inner,worker=worker,worker_result=result,source_after=snapshot())
        valid=(outer.get('status')=='COMPLETED' and outer.get('exit_code')==inner.get('exit_code')==0 and outer.get('native_receipt_complete') is True and inner.get('native_receipt_complete') is True and inner.get('cleanup_verified') is True and inner.get('fixture_identity_after')==inner.get('supervised_identity_after')=='GONE' and inner.get('timeout') is False and worker.get('pid')==inner.get('fixture_pid')==outer.get('actual_worker_pid') and isinstance(worker.get('creation_filetime_utc'),int) and worker.get('creation_filetime_utc')==inner.get('fixture_creation_filetime_utc') and Path(worker.get('executable','')).resolve()==ROOT/'.venv-gpu/Scripts/python.exe' and result.get('status')=='COMPLETED' and record['source_before']==record['source_after'])
        record['status']='COMPLETED' if valid else 'UNRESOLVED_EXECUTION'
    except BaseException as exc:
        record.update(status='UNRESOLVED_EXECUTION',error=repr(exc),traceback=traceback.format_exc())
        if (OUTPUT/'outer/native.json').is_file():
            record['outer']=read(OUTPUT/'outer/native.json');record['charged_seconds']=record['outer'].get('seconds')
        if (OUTPUT/'inner/receipt.json').is_file():record['inner']=read(OUTPUT/'inner/receipt.json')
        if WORKER.is_file():record['worker']=read(WORKER)
        record['source_after']=snapshot()
    finally:
        record['launcher_seconds']=time.perf_counter()-started;write(OUTPUT/'execution.json',record)
    print(json.dumps({'status':record['status'],'charged_seconds':record.get('charged_seconds'),'receipt':str(OUTPUT/'execution.json')}))
    return 0 if record['status']=='COMPLETED' else 1
if __name__=='__main__':raise SystemExit(main())
