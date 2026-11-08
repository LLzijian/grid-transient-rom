"""Static edge-polynomial runtime audit, timings and conditional finite study."""
from __future__ import annotations
import argparse,copy,ctypes,hashlib,json,os,subprocess,sys,threading,time,uuid
from pathlib import Path
START=time.perf_counter();ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'experiments'),str(ROOT/'src')]
SEEDS=(6101,6102,6103);CAPS={'qa':620.,'probe':20.,'benchmark':960.,'run':570.,'evaluate':80.}
POOL=ROOT/'private_data/closure_mechanism_20261019/P0/finite_grammar_discovery01/freeze_pool01'
POOL_SHA='00040fbf83705eb660474de660ef96fec0424a299fd3650df4c1f77b605a8925'
TEACHER_SHA='5e31e4a454c45d56aa8ca2586cb39cde1ce96cbcb485fd7316fdbda6af36157c'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def atomic(p,v):
    p=Path(p);tmp=p.with_name(p.name+'.'+uuid.uuid4().hex+'.tmp')
    tmp.write_text(json.dumps(v,indent=2,sort_keys=True,allow_nan=False),encoding='utf-8');os.replace(tmp,p)
def study():
    import finite_grammar_discovery_20261021 as old
    return old
def bound(binding,key):return study().bound(binding,key)
def passed(binding,key):return study().passed(binding,key)
def read_pool(binding):
    if binding.get('pool_manifest')!={'path':POOL.relative_to(ROOT).as_posix()+'/pool.json','sha256':POOL_SHA}:raise ValueError('fixed old pool')
    return study().read_pool(binding)
def teacher(binding):
    if binding.get('teacher_identity')!={'path':POOL.relative_to(ROOT).as_posix()+'/teacher.json','sha256':TEACHER_SHA}:raise ValueError('fixed old teacher')
    return study().teacher(binding)
def fixture(*a,**kw):return study().fixture(*a,**kw)
def qualify_teacher(*a,**kw):return study().qualify_teacher(*a,**kw)
def predict(*a,**kw):return study().predict(*a,**kw)
def score(*a,**kw):return study().score(*a,**kw)
def choose(*a,**kw):return study().choose(*a,**kw)
def counts():return study().counts()
def rank_model(*a,**kw):return study().rank_model(*a,**kw)
def make(*a,**kw):
    from edge_polynomial_runtime_20261021 import make as compiled
    return compiled(*a,**kw)
def frozen_selection(binding,commit):
    frozen=bound(binding,'selection_manifest')
    if not frozen.get('all_three_frozen') or frozen.get('source_commit')!=commit or [r['seed'] for r in frozen.get('rows',[])]!=list(SEEDS):raise ValueError('all three frozen same-source endpoints before val')
    return frozen
def cost_plan():
    return {'updates':2304,'rankings':27,'teacher_pairs':3,'final_evaluations':3,'checkpoint_saves':45,'restores':9,'constructions':27,'final_loads':3,'raw_writes':9,'json_writes':64,'json_payload_bytes':262144,'startup_multiplier':2,'science_margin':1.25,'support':180,'T_cap':576,'native_cap':720,'complete_cap':900}
def engineering_gate(rows):
    import statistics
    ratios=[];ut=[];rt=[]
    for index in range(6):
        by={p:{op:statistics.median([r['seconds'] for r in rows if r['index']==index and r['path']==p and r['operation']==op and not r['warmup']]) for op in ('update','rank')} for p in ('old','new')}
        if any(not (v>0) for p in by.values() for v in p.values()):raise ValueError('complete positive paired timings')
        ratios.append(by['new']['rank']/by['old']['rank']);ut.append([by['old']['update'],by['new']['update']]);rt.append([by['old']['rank'],by['new']['rank']])
    rankratio=sum(x[1] for x in rt)/sum(x[0] for x in rt);updateratio=sum(x[1] for x in ut)/sum(x[0] for x in ut)
    return {'pass':all(x<=1 for x in ratios) and rankratio<=.9 and updateratio<=1.05,'rank_ratios':ratios,'rank_sum_ratio':rankratio,'update_sum_ratio':updateratio}
def benchmark(output,binding,deadline):
    import numpy as np,torch
    from closure_causal_history_residual_20261019 import train_update,checkpoint,CausalHistoryResidual
    from edge_polynomial_runtime_20261021 import make_reference,make,adapt_wrapper
    qa=passed(binding,'qa_result');p=read_pool(binding);t=teacher(binding)
    probes=binding.get('cold_probe_receipts',[])
    if len(probes)!=3:raise ValueError('three actual cold native probes required')
    proof=bound(binding,'cold_probe_wall_proof')
    if proof.get('status')!='COMPLETED' or proof.get('source_commit')!=qa.get('source_commit') or len(proof.get('rows',[]))!=3:raise ValueError('three actual outside-observed cold launches required')
    cold=[];internal_cold=[]
    for item,external in zip(probes,proof['rows']):
        r=bound({'r':item},'r')
        if r.get('status')!='COMPLETED' or not r.get('phase_gate_pass') or r.get('source_before')!=r.get('source_after'):raise ValueError('cold probe failed')
        wall=external.get('full_startup_seconds');generation=external.get('generation_shell_seconds');launch=external.get('launch_shell_seconds')
        if any(not isinstance(x,(int,float)) or isinstance(x,bool) or not np.isfinite(x) or x<=0 for x in (generation,launch,wall)) or wall!=generation+launch:raise ValueError('actual generation plus full launch timings required')
        if external.get('native_receipt_path')!=item['path'] or external.get('native_receipt_sha256')!=item['sha256'] or external.get('attempt')!=r.get('attempt') or r.get('source_commit')!=proof['source_commit'] or wall<max(r['complete_launch_wall_seconds'],r['charged_seconds']):raise ValueError('outside cold observation/native identity mismatch')
        cold.append(wall);internal_cold.append(r['complete_launch_wall_seconds'])
    c,f,sc=fixture(t['term'],8901,0,deadline,True)
    obs=torch.as_tensor(c[:,:61],device='cuda');scale=torch.as_tensor(sc,device='cuda')
    samples=[];construct=[];reference_construct=[];models=[]
    def timed(fn):
        if time.perf_counter()>deadline:raise TimeoutError('benchmark deadline')
        torch.cuda.synchronize();start=time.perf_counter();v=fn();torch.cuda.synchronize()
        return time.perf_counter()-start,v
    def snapshot(m,opt,rng):
        return (copy.deepcopy(m.state_dict()),copy.deepcopy(opt.state_dict()),copy.deepcopy(rng.bit_generator.state),torch.get_rng_state().clone(),[v.clone() for v in torch.cuda.get_rng_state_all()])
    def reset(m,opt,rng,s):
        m.load_state_dict(s[0]);opt.load_state_dict(s[1]);rng.bit_generator.state=copy.deepcopy(s[2]);torch.set_rng_state(s[3]);torch.cuda.set_rng_state_all(s[4]);torch.cuda.synchronize()
    for index,term in enumerate(p['selected']):
        pairs={}
        for path,factory in [('old',make_reference),('new',make)]:
            duration,(m,opt)=timed(lambda factory=factory:factory(term,8901,c));(construct if path=='new' else reference_construct).append(duration)
            with torch.no_grad():m.base.coefficients.fill_(.01)
            rng=np.random.default_rng(np.random.SeedSequence([8901,7]));pairs[path]=(m,opt,rng)
        # Snapshot is reference state and empty Adam; every path receives identical RNG and bytes.
        common=snapshot(*pairs['old']);trained=None;last_loss=None
        for op in ('update','rank'):
            for rep in range(4):
                order=('old','new') if rep%2==0 else ('new','old')
                for path in order:
                    m,opt,rng=pairs[path];reset(m,opt,rng,common)
                    duration,value=timed(lambda:train_update(m,opt,rng,obs,scale,deadline) if op=='update' else rank_model(m,c,f,sc,deadline))
                    if path=='new' and op=='update' and rep==3:trained=snapshot(m,opt,rng);last_loss=float(value[0])
                    samples.append({'index':index,'path':path,'operation':op,'rep':rep,'warmup':rep==0,'seconds':duration,'reset_outside_timer':True})
        # One nonempty Adam state from an actual complete update, retained for I/O measurements.
        reset(*pairs['new'],trained);models.append((*pairs['new'],last_loss))
        atomic(output/'partial_timings.json',{'status':'INCOMPLETE','completed_candidates':index+1,'samples':samples,'construction_seconds':construct,'reference_construction_seconds':reference_construct,'complete_T_available':False})
    saves=[];restores=[];loads=[];rawtimes=[];jsontimes=[];g=[];v=[]
    for n in range(45):
        m,opt,rng,loss=models[n%6];folder=output/f'io_{n}';folder.mkdir()
        ident={'timing_only':True,'padding':b'0'*131072};step=1
        duration,_=timed(lambda:checkpoint(folder,m,opt,rng,step,[loss],ident));saves.append(duration)
        if n<9:
            def restore_io():
                latest=json.loads((folder/'latest.json').read_text(encoding='utf-8'))
                actual=folder/'checkpoint_00001'
                if latest.get('updates')!=1 or latest.get('checkpoint')!='checkpoint_00001' or list(folder.glob('pending_*')) or any(p.name>actual.name for p in folder.glob('checkpoint_*')):raise ValueError('cost-only actual step1/latest reconciliation')
                if latest.get('sha256')!={p.name:sha(p) for p in actual.iterdir()}:raise ValueError('both checkpoint files changed')
                restored=adapt_wrapper(CausalHistoryResidual.from_export(folder/'checkpoint_00001/model.json').cuda())
                state=torch.load(folder/'checkpoint_00001/training.pt',map_location='cpu',weights_only=False)
                if not state['optimizer']['state'] or state['identity']!=ident or state['completed_updates']!=1 or state['trace']!=[loss]:raise ValueError('nonempty Adam and actual step1 identity required')
                if len(state['cuda_rng'])!=torch.cuda.device_count():raise ValueError('CUDA identity changed')
                m.load_state_dict(restored.state_dict());opt.load_state_dict(state['optimizer']);rng.bit_generator.state=state['fit_numpy_rng'];torch.set_rng_state(state['torch_rng']);torch.cuda.set_rng_state_all(state['cuda_rng'])
                if any(not torch.equal(v,restored.state_dict()[k]) for k,v in m.state_dict().items()) or not torch.equal(torch.get_rng_state(),state['torch_rng']) or any(not torch.equal(a,b) for a,b in zip(torch.cuda.get_rng_state_all(),state['cuda_rng'])):raise ValueError('exact restored state/RNG mismatch')
                if getattr(m.base,'_runtime_cache',None) is not None:raise ValueError('restore must clear graph cache')
            duration,_=timed(restore_io);restores.append(duration)
        if n<3:
            duration,_=timed(lambda:adapt_wrapper(CausalHistoryResidual.from_export(folder/'checkpoint_00001/model.json').cuda()));loads.append(duration)
    for n in range(21):
        duration,_=timed(lambda:make(p['selected'][n%6],8901,c));construct.append(duration)
    for n in range(3):
        def pair():
            a,b,s=fixture(t['term'],8910+n,0,deadline,True);qualify_teacher(a,b,s)
            d,e,_=fixture(t['term'],8910+n,1,deadline,True);qualify_teacher(d,e,s)
        duration,_=timed(pair);g.append(duration)
    for model in models:
        m=model[0]
        def final():
            mc=predict(m,c,.01,deadline);mf=predict(m,c,.005,deadline)
            for a,b in ((0,100),(0,500),(200,500)):score(mc,mf,c[:,11:],f[:,11:],sc,a,b)
        duration,_=timed(final);v.append(duration)
    raw_assets=[]
    for n in range(9):
        def write_raw():
            path=output/f'raw_{n}.npz';np.savez(path,coarse=c,fine=f,scales=sc)
            return {'path':str(path),'sha256':sha(path),'bytes':path.stat().st_size}
        duration,asset=timed(write_raw);rawtimes.append(duration);raw_assets.append(asset)
    for n in range(64):
        duration,_=timed(lambda:atomic(output/f'metadata_{n}.json',{'timing_only':True,'padding':'0'*262144}));jsontimes.append(duration)
    u=max(r['seconds'] for r in samples if r['path']=='new' and r['operation']=='update');r=max(x['seconds'] for x in samples if x['path']=='new' and x['operation']=='rank')
    # Unknown survivors are charged each operation's slowest observed full cost.
    actual_fixed_operation_sum=sum(saves)+sum(restores)+sum(construct)+sum(loads)+sum(rawtimes)+sum(jsontimes)
    c_all=45*max(saves)+9*max(restores)+27*max(construct)+3*max(loads)+9*max(rawtimes)+64*max(jsontimes)+2*max(cold)
    T=2304*u+27*r+3*max(g)+3*max(v)+c_all;eng=engineering_gate(samples)
    result={'status':'PASS' if eng['pass'] and T<=576 and 1.25*T+180<=900 else 'REJECT','P1_complete_cap_seconds':900,'P1_native_cap_seconds':720,'engineering':eng,'T_seconds':T,'total_estimate_seconds':1.25*T+180,'samples':samples,'u_max':u,'r_max':r,'g_pair_samples':g,'v_final_samples':v,'save_seconds':saves,'restore_seconds':restores,'construction_seconds':construct,'reference_construction_seconds':reference_construct,'final_load_seconds':loads,'raw_write_seconds':rawtimes,'json_write_seconds':jsontimes,'cold_launch_seconds':cold,'c_all':c_all,'count_plan':cost_plan(),'science_scores':None}
    result.update(actual_fixed_operation_sum_seconds=actual_fixed_operation_sum,raw_assets=raw_assets,internal_cold_launch_seconds=internal_cold,cold_probe_full_rows=proof['rows'],cold_probe_wall_proof=binding['cold_probe_wall_proof'],v_final_candidates_measured=6,formal_v_final_count=3,partial_writes_benchmark_only=6)
    atomic(output/'benchmark.json',result);return result

def search_seed(terms,keys,seed,coarse,fine,sc,output,commit,deadline):
    """Selector inputs intentionally exclude teacher identity and all val arrays."""
    import numpy as np,torch
    from closure_causal_history_residual_20261019 import train_update,checkpoint,restore
    obs=torch.as_tensor(coarse[:,:61],device='cuda');scale=torch.as_tensor(sc,device='cuda')
    identity={'source_commit':commit,'seed':seed,'train_sha256':hashlib.sha256(coarse.tobytes()).hexdigest()}
    active=list(range(6));history=[];unresolved=False
    for target,keep in ((32,2),(128,1),(512,1)):
        rows=[]
        for index in active:
            folder=output/f'candidate_{index}';folder.mkdir(exist_ok=True)
            m,opt=make(terms[index],seed,coarse);rng=np.random.default_rng(np.random.SeedSequence([seed,7]))
            ident={**identity,'candidate_key':keys[index]}
            if target==32:
                trace=[];step=0;checkpoint(folder,m,opt,rng,0,trace,ident)
            else:step,trace=restore(folder,m,opt,rng,ident)
            if step!={32:0,128:32,512:128}[target]:raise ValueError('continuation is not exact previous stage')
            try:
                while step<target:
                    value,_=train_update(m,opt,rng,obs,scale,deadline);trace.append(value);step+=1
            except TimeoutError:raise
            except (RuntimeError,FloatingPointError,ValueError) as e:
                failure={'index':index,'key':keys[index],'score':{'valid':False,'count':16,'mean':None,'B':None,
                         'error':repr(e),'category':'numerical_or_implementation_failure'},'actual_updates':step,'trace':trace}
                atomic(folder/f'failure_before_{target}.json',failure);rows.append(failure);continue
            checkpoint(folder,m,opt,rng,step,trace,ident)
            try:metric=rank_model(m,coarse,fine,sc,deadline)
            except TimeoutError:raise
            except (RuntimeError,FloatingPointError,ValueError) as e:
                metric={'valid':False,'count':16,'mean':None,'B':None,'error':repr(e),'category':'numerical_or_implementation_failure'}
            rows.append({'index':index,'key':keys[index],'score':metric,'path':str(folder/f'checkpoint_{step:05d}'),
                         'files':{n:sha(folder/f'checkpoint_{step:05d}'/n) for n in ('model.json','training.pt')}})
        selected,ambiguous=choose(rows,keep);history.append({'updates':target,'rows':rows,'ambiguous_boundary':ambiguous})
        atomic(output/f'ranking_{target}.json',history[-1])
        if not selected:unresolved=True;break
        active=[r['index'] for r in selected]
    return {'seed':seed,'status':'UNRESOLVED' if unresolved else 'FROZEN','history':history,
            'selected':None if unresolved else selected[0]}


def run_study(output,binding,commit,deadline):
    import numpy as np
    admission=passed(binding,'admission')
    if admission.get('P1_complete_cap_seconds')!=900:raise ValueError('wrong admission')
    p=read_pool(binding);t=teacher(binding);rows=[]
    for seed in SEEDS:
        folder=output/str(seed);folder.mkdir()
        try:
            c,f,sc=fixture(t['term'],seed,0,deadline)
            np.savez(folder/'train_observations.npz',coarse=c,fine=f,scales=sc)
            qualification=qualify_teacher(c,f,sc);atomic(folder/'teacher_qualification.json',qualification)
            if not qualification['valid']:
                rows.append({'seed':seed,'status':'UNRESOLVED','selected':None,'history':[],
                             'teacher_qualification':qualification,'train_observations':{'path':str(folder/'train_observations.npz'),'sha256':sha(folder/'train_observations.npz')}})
                continue
            row=search_seed(p['selected'],p['keys'],seed,c,f,sc,folder,commit,deadline)
            row['train_observations']={'path':str(folder/'train_observations.npz'),'sha256':sha(folder/'train_observations.npz')}
        except TimeoutError:raise
        except (RuntimeError,FloatingPointError,ValueError) as e:
            row={'seed':seed,'status':'UNRESOLVED','selected':None,'history':[],
                 'error':repr(e),'category':'teacher_or_implementation_failure'}
            atomic(folder/'failure.json',row)
        rows.append(row)
    checkpoint_dirs=list(output.glob('*/candidate_*/checkpoint_*'))
    successful_updates=0
    for folder in output.glob('*/candidate_*'):
        latest=json.loads((folder/'latest.json').read_text(encoding='utf-8')) if (folder/'latest.json').is_file() else {'updates':0}
        completed=latest['updates']
        for path in folder.glob('failure_before_*.json'):
            completed=max(completed,json.loads(path.read_text(encoding='utf-8')).get('actual_updates',0))
        successful_updates+=completed
    result={'status':'COMPLETED','all_three_frozen':all(r['status']=='FROZEN' for r in rows),
            'source_commit':commit,'rows':rows,'selection':'train_only','planned_counts':counts(),
            'recorded_completed_counts':{'successful_updates':successful_updates,'checkpoint_directories':len(checkpoint_dirs),
                'fully_published_candidate_rankings':sum(len(stage['rows']) for row in rows for stage in row['history'])}}
    atomic(output/'selection.json',result);return result
def evaluate_study(output,binding,commit,deadline):
    import numpy as np
    from closure_causal_history_residual_20261019 import CausalHistoryResidual
    from closure_expression_identity import mechanism_signature
    frozen=bound(binding,'selection_manifest');p=read_pool(binding)
    if not frozen.get('all_three_frozen') or frozen.get('source_commit')!=commit or [r['seed'] for r in frozen['rows']]!=list(SEEDS):raise ValueError('all three same-source endpoints required before val')
    for row in frozen['rows']:
        for n,h in row['selected']['files'].items():
            if sha(Path(row['selected']['path'])/n)!=h:raise ValueError('endpoint changed')
        if sha(row['train_observations']['path'])!=row['train_observations']['sha256']:raise ValueError('train scales changed')
    t=teacher(binding);rows=[]
    for row in frozen['rows']:
        seed=row['seed'];selected=row['selected'];folder=output/str(seed);folder.mkdir()
        with np.load(row['train_observations']['path']) as train:sc=train['scales'].copy()
        try:
            c,f,_=fixture(t['term'],seed,1,deadline);np.savez(folder/'val_observations.npz',coarse=c,fine=f)
            qualification=qualify_teacher(c,f,sc);atomic(folder/'teacher_qualification.json',qualification)
            if not qualification['valid']:
                rows.append({'seed':seed,'valid':False,'windows':None,'teacher_qualification':qualification,
                             'error':'teacher_numerical_failure','raw':{'path':str(folder/'val_observations.npz'),'sha256':sha(folder/'val_observations.npz')}});continue
            m=CausalHistoryResidual.from_export(Path(selected['path'])/'model.json').cuda()
            from edge_polynomial_runtime_20261021 import adapt_wrapper
            m=adapt_wrapper(m)
            mc=predict(m,c,.01,deadline);mf=predict(m,c,.005,deadline)
            np.savez(folder/'predictions.npz',coarse=mc,fine=mf)
            windows={name:score(mc,mf,c[:,11:],f[:,11:],sc,start,end) for name,start,end in [('one_second',0,100),('five_seconds',0,500),('primary',200,500)]}
            key=t['key'];eliminated=next((stage['updates'] for stage in row['history'] if not any(r['key']==key for r in choose(stage['rows'],2 if stage['updates']==32 else 1)[0])),None)
            rows.append({'seed':seed,'valid':all(w['valid'] for w in windows.values()),'windows':windows,'teacher_qualification':qualification,
                         'signature_match':mechanism_signature(p['selected'][selected['index']])==mechanism_signature(t['term']),
                         'teacher_eliminated_at':eliminated,'files':{n:{'path':str(folder/n),'sha256':sha(folder/n)} for n in ('val_observations.npz','predictions.npz')}})
        except TimeoutError:raise
        except (RuntimeError,FloatingPointError,ValueError) as error:
            rows.append({'seed':seed,'valid':False,'windows':None,'error':repr(error),
                         'category':'teacher_or_model_failure','files':{n:{'path':str(folder/n),'sha256':sha(folder/n)} for n in ('val_observations.npz','predictions.npz') if (folder/n).is_file()}})
    return {'status':'COMPLETED','rows':rows,'finite_pool_consistent':len(rows)==3 and all(r['valid'] and r.get('signature_match',False) for r in rows) and not any(stage['ambiguous_boundary'] for row in frozen['rows'] for stage in row['history'])}



def execute(phase,out,binding,commit,deadline):
    import torch
    torch.set_num_threads(1)
    if not torch.cuda.is_available():raise RuntimeError('actual CUDA required')
    read_pool(binding);teacher(binding)
    if phase=='qa':
        if binding.get('qa_retry_approval'):
            approval=bound(binding,'qa_retry_approval')
            if approval.get('status')!='APPROVED' or approval.get('source_commit')!=commit or not approval.get('declared_file_diff'):raise ValueError('explicit implementation repair authority')
        tests=['tests/test_edge_polynomial_runtime_20261021.py','tests/test_edge_polynomial_study_20261021.py','tests/test_finite_grammar_discovery_20261021.py']
        cmd=[sys.executable,'-m','pytest','-q','-rA','--basetemp',str(out/'qa_artifacts'),*[str(ROOT/x) for x in tests]]
        proc=subprocess.run(cmd,cwd=ROOT,capture_output=True,text=True,timeout=max(.1,deadline-time.perf_counter()))
        (out/'pytest_stdout.txt').write_text(proc.stdout,encoding='utf-8');(out/'pytest_stderr.txt').write_text(proc.stderr,encoding='utf-8')
        return {'status':'PASS' if proc.returncode==0 else 'FAIL','pytest_returncode':proc.returncode,'stdout':proc.stdout,'stderr':proc.stderr,'torch':torch.__version__,'cuda':torch.version.cuda}
    if phase=='probe':
        from edge_polynomial_runtime_20261021 import make
        import numpy as np
        # Cold process, one compiled full-pool model and actual CUDA field, no scores/data fixture.
        obs=np.zeros((16,61,30));terms=read_pool(binding)['selected'];built=[]
        for term in terms:
            m,opt=make(term,8951,obs);built.append(m)
            with torch.no_grad():m.base.field(torch.zeros((16,40),device='cuda',dtype=torch.float64))
        torch.cuda.synchronize()
        return {'status':'PASS','models_constructed':len(built),'fields':len(built),'worker_cold_elapsed_seconds':time.perf_counter()-START,'science_scores':None}
    if phase=='benchmark':return benchmark(out,binding,deadline)
    if phase=='run':return run_study(out,binding,commit,deadline)
    if phase=='evaluate':
        frozen_selection(binding,commit)
        return evaluate_study(out,binding,commit,deadline)
    raise ValueError('unknown phase')
def receipt(out,phase,commit,binding):
    from ctypes import wintypes
    k=ctypes.WinDLL('kernel32',use_last_error=True);k.GetCurrentProcess.restype=wintypes.HANDLE;k.GetCurrentProcess.argtypes=[]
    k.GetProcessTimes.argtypes=[wintypes.HANDLE]+[ctypes.POINTER(wintypes.FILETIME)]*4;k.GetProcessTimes.restype=wintypes.BOOL
    times=[wintypes.FILETIME() for _ in range(4)]
    if not k.GetProcessTimes(k.GetCurrentProcess(),*[ctypes.byref(x) for x in times]):raise ctypes.WinError(ctypes.get_last_error())
    atomic(out/'worker.json',{'pid':os.getpid(),'creation_filetime_utc':(times[0].dwHighDateTime<<32)|times[0].dwLowDateTime,'phase':phase,'source_commit':commit,'binding':str(binding),'executable':sys.executable,'command':sys.argv})
def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('phase',choices=CAPS)
    for n in ('output','binding'):p.add_argument('--'+n,type=Path,required=True)
    p.add_argument('--source-commit',required=True);p.add_argument('--max-seconds',type=float,required=True);a=p.parse_args(argv)
    if a.max_seconds not in (CAPS[a.phase],270. if a.phase=='qa' else CAPS[a.phase]) or len(a.source_commit)!=40 or any(c not in '0123456789abcdef' for c in a.source_commit):p.error('registered caps/full commit')
    out=a.output.resolve();category='P1' if a.phase in ('run','evaluate') else 'P0';base=ROOT/'private_data/closure_mechanism_20261019'/category/'edge_polynomial_runtime01'
    if not out.is_relative_to(base) or out==base:raise ValueError('exclusive registered study output')
    out.mkdir(parents=True,exist_ok=False);deadline=START+a.max_seconds;receipt(out,a.phase,a.source_commit,a.binding.resolve())
    def timeout():atomic(out/'timeout.json',{'status':'TIMEOUT'});os._exit(124)
    timer=threading.Timer(max(.001,deadline-time.perf_counter()),timeout);timer.daemon=True;timer.start()
    try:
        binding=json.loads(a.binding.read_text(encoding='utf-8'))
        if a.phase=='qa' and a.max_seconds==270. and not binding.get('qa_retry_approval'):raise ValueError('QA270 requires bound approved defect repair')
        result=execute(a.phase,out,binding,a.source_commit,deadline)
        result.update(source_commit=a.source_commit,worker_phase_elapsed_seconds=time.perf_counter()-START);atomic(out/'result.json',result);atomic(out/'complete.json',{'status':'COMPLETED','result':result})
        return 0 if result['status'] in ('PASS','COMPLETED','REJECT') else 2
    except BaseException as e:
        import traceback
        atomic(out/'failure.json',{'status':'UNRESOLVED','error':repr(e),'traceback':traceback.format_exc()});raise
    finally:timer.cancel()
if __name__=='__main__':raise SystemExit(main())

