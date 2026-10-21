"""Finite six-candidate continuation study; numerical work requires bound admission."""
from __future__ import annotations
import argparse,copy,ctypes,functools,hashlib,json,os,subprocess,sys,threading,time,uuid
from pathlib import Path
START=time.perf_counter();ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'experiments'),str(ROOT/'src')]
SEEDS=(6101,6102,6103)
CAPS={'freeze-pool':40.,'qa':420.,'cost':100.,'run':570.,'evaluate':80.}


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def atomic(p,value):
    p=Path(p);tmp=p.with_name(p.name+'.'+uuid.uuid4().hex+'.tmp')
    tmp.write_text(json.dumps(value,indent=2,sort_keys=True,allow_nan=False),encoding='utf-8');os.replace(tmp,p)
def bound(binding,key):
    v=binding.get(key,{})
    p=ROOT/v.get('path','')
    if not p.is_file() or sha(p)!=v.get('sha256'):raise ValueError('missing/changed '+key)
    return json.loads(p.read_text(encoding='utf-8'))
def passed(binding,key):
    v=bound(binding,key);v=v.get('result',v)
    if v.get('status')!='PASS':raise ValueError('failed '+key)
    return v


def variables(ast):
    if ast['op']=='var':return {ast['name']}
    return set().union(*(variables(a) for a in ast.get('args',[])),*( [variables(ast['arg'])] if 'arg' in ast else []))
def generate_pool():
    from closure_discovery_benchmarks import pool
    from closure_expression_identity import mechanism_signature
    from closure_discovery_support import term_key
    unique={}
    for t in sorted(pool(),key=lambda t:(hashlib.sha256(term_key(t).encode('utf-8')).hexdigest(),term_key(t))):
        v=variables(t['expression']);sig=mechanism_signature(t)
        if t['channel']=='edge' and v&{'qi','qj'} and v&{'s','c'} and sig is not None:
            unique.setdefault(sig,t)
    eligible=sorted(unique.values(),key=lambda t:(hashlib.sha256(term_key(t).encode('utf-8')).hexdigest(),term_key(t)))
    if len(eligible)<6:raise ValueError('fewer than six mechanisms; no replacement')
    return {'eligible_pool':eligible,'selected':eligible[:6],'keys':[term_key(t) for t in eligible[:6]],
            'generation_rule':'edge; q and s/c; nonzero signature dedup; SHA256(term_key) first6'}
def read_pool(binding):
    p=bound(binding,'pool_manifest')
    actual=generate_pool()
    for k in actual:
        if p.get(k)!=actual[k]:raise ValueError('frozen pool differs')
    return p
def teacher(binding):
    import numpy as np
    p=read_pool(binding);t=bound(binding,'teacher_identity')
    index=int(np.random.default_rng(np.random.SeedSequence([20261021,6100])).integers(6))
    if t.get('index')!=index or t.get('term')!=p['selected'][index] or t.get('key')!=p['keys'][index] or t.get('coefficient')!=.15:
        raise ValueError('teacher identity changed')
    if t.get('pool_sha256')!=binding['pool_manifest']['sha256']:raise ValueError('teacher pool mismatch')
    return t


def fixture(term,seed,stream,deadline,timing=False):
    import numpy as np,torch
    from closure_discovery_benchmarks import toy_source
    from closure_causal_recovery_20261019 import copy_runtime,trajectory
    src=toy_source();m=copy_runtime(src,[term],'cuda')
    rng=np.random.default_rng(np.random.SeedSequence([seed,20261021,991 if timing else 6100]).spawn(2)[stream])
    initial=rng.normal(size=(16,40))*np.where(np.arange(16)[:,None]<8,.08,.3)
    initial[:,30:]=rng.normal(size=(16,10))*np.where(np.arange(16)[:,None]<8,.2,1.2)
    z=torch.as_tensor(initial,device='cuda',dtype=torch.float64)
    with torch.no_grad():
        m.coefficients.fill_(.15)
        c=trajectory(m,z,510,.01,deadline).cpu().numpy()
        f=trajectory(m,z,1020,.005,deadline).cpu().numpy()[:,::2]
    return c,f,np.maximum(c[:,:61].std((0,1)),1e-6)
def qualify_teacher(coarse,fine,scales):
    import numpy as np
    if coarse.shape!=(16,511,30) or fine.shape!=(16,511,30) or np.asarray(scales).shape!=(30,):
        raise ValueError('teacher requires16x511x30 and30 fixed scales')
    scales=np.asarray(scales)
    if not np.isfinite(scales).all() or (scales<1e-6).any():raise ValueError('invalid train scales')
    with np.errstate(over='ignore',invalid='ignore',divide='ignore'):
        difference=np.max(abs(coarse-fine)/scales,(1,2))
    finite=np.isfinite(coarse).all((1,2)) & np.isfinite(fine).all((1,2))
    eligible=finite & np.isfinite(difference) & (difference<=1e-3)
    return {'valid':bool(eligible.all()),'count':16,'eligible':eligible.tolist(),
            'step_max':[float(x) if np.isfinite(x) else None for x in difference],
            'failure_category':None if eligible.all() else 'teacher_numerical_failure'}
def make(term,seed,obs):
    import torch
    from closure_causal_recovery_20261019 import runtime_model
    from closure_discovery_benchmarks import toy_source
    from closure_causal_history_residual_20261019 import CausalHistoryResidual,make_optimizer
    m=CausalHistoryResidual.from_train(runtime_model(toy_source(),[term],seed,'cuda'),torch.as_tensor(obs[:,:11],device='cuda'))
    return m,make_optimizer(m)
def predict(m,obs,dt,deadline):
    import torch
    from closure_causal_recovery_20261019 import trajectory
    with torch.no_grad():
        h=torch.as_tensor(obs[:,:11],device='cuda');z=torch.cat((h[:,-1],m.encode(h)),-1);factor=round(.01/dt)
        return trajectory(m,z,500*factor,dt,deadline)[:,factor::factor].cpu().numpy()
def score(mc,mf,tc,tf,sc,start=200,end=500):
    import numpy as np
    if any(a.shape!=(16,500,30) for a in (mc,mf,tc,tf)):raise ValueError('full16x500x30 required')
    with np.errstate(over='ignore',invalid='ignore',divide='ignore'):
        delta=np.maximum(np.max(abs(mc[:,:end]-mf[:,:end])/sc,(1,2)),np.max(abs(tc[:,:end]-tf[:,:end])/sc,(1,2)))
        finite=np.logical_and.reduce([np.isfinite(a[:,:end]).all((1,2)) for a in (mc,mf,tc,tf)])
        eligible=finite & np.isfinite(delta) & (delta<=1e-3)
        e=(mc[:,start:end]-tc[:,start:end])/sc
        d=(abs(mc[:,start:end]-mf[:,start:end])+abs(tc[:,start:end]-tf[:,start:end]))/sc
        mse=(e*e).mean((1,2));B=(2*abs(e)*d+d*d).mean((1,2))
    clean=lambda v:[float(x) if np.isfinite(x) else None for x in v]
    valid=bool(eligible.all() and np.isfinite(mse).all() and np.isfinite(B).all())
    return {'valid':valid,'count':16,'indices':[start,end],'per_event_mse':clean(mse),'per_event_B':clean(B),
            'eligible':eligible.tolist(),'step_max':clean(delta),'mean':float(mse.mean()) if valid else None,
            'B':float(B.mean()) if valid else None,'weak_mean':float(mse[:8].mean()) if valid else None,
            'strong_mean':float(mse[8:].mean()) if valid else None,'worst':float(mse.max()) if valid else None}
def rank_model(m,coarse,fine,sc,deadline):
    return score(predict(m,coarse,.01,deadline),predict(m,coarse,.005,deadline),coarse[:,11:],fine[:,11:],sc)
def choose(rows,keep):
    valid=sorted((r for r in rows if r['score']['valid']),
                 key=lambda r:(r['score']['mean']-r['score']['B'],r['key']))
    groups=[]
    for r in valid:
        left=r['score']['mean']-r['score']['B'];right=r['score']['mean']+r['score']['B']
        if groups and left<=groups[-1]['right']:
            groups[-1]['rows'].append(r);groups[-1]['right']=max(groups[-1]['right'],right)
        else:groups.append({'right':right,'rows':[r]})
    ordered=[];ambiguous=False
    for g in groups:
        block=sorted(g['rows'],key=lambda r:r['key'])
        for r in block:
            r['interval']=[r['score']['mean']-r['score']['B'],r['score']['mean']+r['score']['B']]
            r['overlap_group']=[x['key'] for x in block]
        if len(ordered)<keep<len(ordered)+len(block):ambiguous=True
        ordered.extend(block)
    if len(ordered)<keep:return [],False
    return ordered[:keep],ambiguous
def counts():return {'updates_per_seed':6*32+2*(128-32)+(512-128),'total_updates':2304,'checkpoints_per_seed':15,'total_checkpoints':45,'total_restores':9,'total_rankings':27,'total_constructions':27,'final_evaluation_loads':3}


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


def run_cost(output,binding,deadline):
    import numpy as np,torch
    from closure_causal_history_residual_20261019 import train_update,checkpoint,restore,CausalHistoryResidual
    passed(binding,'qa_result');p=read_pool(binding);t=teacher(binding);begin=time.perf_counter()
    c,f,sc=fixture(t['term'],7901,0,deadline,True)
    obs=torch.as_tensor(c[:,:61],device='cuda');scale=torch.as_tensor(sc,device='cuda')
    updates=[];ranks=[];save_times=[];restore_times=[];load_times=[]
    for i,term in enumerate(p['selected']):
        start=time.perf_counter();m,opt=make(term,7901,c);torch.cuda.synchronize();load_times.append(time.perf_counter()-start)
        rng=np.random.default_rng(np.random.SeedSequence([7901,7]));folder=output/f'cost_only_{i}';folder.mkdir()
        identity={'cost_only':True,'candidate':i,'padding':b'0'*131072}
        start=time.perf_counter();checkpoint(folder,m,opt,rng,0,[],identity);torch.cuda.synchronize();save_times.append(time.perf_counter()-start)
        start=time.perf_counter();v,_=train_update(m,opt,rng,obs,scale,deadline);torch.cuda.synchronize();updates.append(time.perf_counter()-start)
        # Actual one-step cost checkpoint; restore() only accepts registered
        # multiples32, so restoration timing uses the actual stage0 checkpoint.
        start=time.perf_counter();checkpoint(folder,m,opt,rng,1,[v],identity);torch.cuda.synchronize();save_times.append(time.perf_counter()-start)
        start=time.perf_counter();metric=rank_model(m,c,f,sc,deadline);torch.cuda.synchronize();ranks.append(time.perf_counter()-start)
        start=time.perf_counter()
        # Cost-only actual step1 snapshot has nonempty Adam. The scientific
        # restore() keeps its original multiples32 guard; no counterfeit32.
        actual=folder/'checkpoint_00001'
        restored=CausalHistoryResidual.from_export(actual/'model.json').cuda();m.load_state_dict(restored.state_dict())
        state=torch.load(actual/'training.pt',map_location='cpu',weights_only=False)
        if state['completed_updates']!=1 or len(state['trace'])!=1 or state['identity']!=identity:raise ValueError('cost restoration identity')
        opt.load_state_dict(state['optimizer']);rng.bit_generator.state=state['fit_numpy_rng']
        torch.set_rng_state(state['torch_rng']);torch.cuda.set_rng_state_all(state['cuda_rng'])
        torch.cuda.synchronize();restore_times.append(time.perf_counter()-start)
        start=time.perf_counter();CausalHistoryResidual.from_export(folder/'checkpoint_00000/model.json').cuda();torch.cuda.synchronize();load_times.append(time.perf_counter()-start)
    start=time.perf_counter()
    gc,gf,gsc=fixture(t['term'],7902,0,deadline,True);qualify_teacher(gc,gf,gsc)
    vc,vf,_=fixture(t['term'],7902,1,deadline,True);qualify_teacher(vc,vf,gsc)
    torch.cuda.synchronize();gpair=time.perf_counter()-start
    start=time.perf_counter();pc=predict(m,c,.01,deadline);pf=predict(m,c,.005,deadline)
    for a,b in ((0,100),(0,500),(200,500)):score(pc,pf,c[:,11:],f[:,11:],sc,a,b)
    torch.cuda.synchronize();vfinal=max(time.perf_counter()-start,max(ranks))
    start=time.perf_counter()
    np.savez(output/'cost_only_observations.npz',coarse=c,fine=f,scales=sc)
    raw_time=time.perf_counter()-start;start=time.perf_counter()
    atomic(output/'cost_only_metadata.json',{'cost_only':True,'padding':'0'*262144})
    json_time=time.perf_counter()-start;metadata=raw_time+json_time
    # Predeclared upper strategy: each of45 complete snapshots uses the slowest
    # measured atomic padded save; nine restores,27 constructions,3 final
    # loads,9 real raw-array files and64 conservative JSON writes.
    c_all=45*max(save_times)+9*max(restore_times)+30*max(load_times)+9*raw_time+64*json_time+(time.perf_counter()-START)
    T=2304*max(updates)+27*max(ranks)+3*gpair+3*vfinal+c_all
    return {'status':'PASS' if 1.25*T+180<=900 else 'REJECT','P1_complete_cap_seconds':900,
            'T_seconds':T,'total_estimate_seconds':1.25*T+180,'u_max':max(updates),'r_max':max(ranks),
            'g_pair':gpair,'v_final':vfinal,'c_all':c_all,'updates':updates,'rank_seconds':ranks,
            'save_seconds':save_times,'restore_seconds':restore_times,'load_seconds':load_times,
            'metadata_seconds':metadata,'raw_write_seconds':raw_time,'json_write_seconds':json_time,
            'cost_elapsed':time.perf_counter()-begin,'science_scores':None,
            'c_all_strategy':{'atomic_padded_checkpoints':45,'continuation_restores':9,
                              'constructions':27,'final_evaluation_loads':3,'raw_array_writes':9,'json_writes':64,
                              'startup_and_source_allowance':'entire measured cost elapsed since pre-import START'},**counts()}


def execute(phase,out,binding,commit,deadline):
    import numpy as np,torch
    torch.set_num_threads(1)
    if phase=='freeze-pool':
        p=generate_pool();p['source_commit']=commit;atomic(out/'pool.json',p)
        index=int(np.random.default_rng(np.random.SeedSequence([20261021,6100])).integers(6))
        atomic(out/'teacher.json',{'index':index,'term':p['selected'][index],'key':p['keys'][index],
                                  'coefficient':.15,'pool_sha256':sha(out/'pool.json')})
        return {'status':'COMPLETED','pool_sha256':sha(out/'pool.json'),'teacher_sha256':sha(out/'teacher.json')}
    if phase=='qa':
        read_pool(binding);cmd=[sys.executable,'-m','pytest','-q',str(ROOT/'tests/test_finite_grammar_discovery_20261021.py')]
        if binding.get('qa_retry_approval'):
            raise ValueError('QA02 authority withdrawn before first execution; unique QA only')
        proc=subprocess.run(cmd,cwd=ROOT,capture_output=True,text=True,timeout=max(.1,deadline-time.perf_counter()))
        (out/'pytest_stdout.txt').write_text(proc.stdout,encoding='utf-8');(out/'pytest_stderr.txt').write_text(proc.stderr,encoding='utf-8')
        return {'status':'PASS' if proc.returncode==0 else 'FAIL','pytest_returncode':proc.returncode,'stdout':proc.stdout,'stderr':proc.stderr}
    if phase=='cost':return run_cost(out,binding,deadline)
    if phase=='run':return run_study(out,binding,commit,deadline)
    if phase=='evaluate':return evaluate_study(out,binding,commit,deadline)
    raise ValueError('unknown phase')


def receipt(out,phase,commit,binding):
    class FT(ctypes.Structure):_fields_=[('low',ctypes.c_uint32),('high',ctypes.c_uint32)]
    k=ctypes.WinDLL('kernel32',use_last_error=True);k.GetCurrentProcess.restype=ctypes.c_void_p
    k.GetProcessTimes.argtypes=[ctypes.c_void_p]+[ctypes.POINTER(FT)]*4;times=[FT() for _ in range(4)]
    if not k.GetProcessTimes(k.GetCurrentProcess(),*[ctypes.byref(x) for x in times]):raise ctypes.WinError(ctypes.get_last_error())
    atomic(out/'worker.json',{'pid':os.getpid(),'creation_filetime_utc':(int(times[0].high)<<32)|int(times[0].low),
                            'phase':phase,'source_commit':commit,'binding':str(binding.resolve()),'command':sys.argv,'executable':sys.executable})
def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('phase',choices=CAPS)
    for n in ('output','binding'):p.add_argument('--'+n,type=Path,required=True)
    p.add_argument('--source-commit',required=True);p.add_argument('--max-seconds',type=float,required=True);a=p.parse_args(argv)
    if a.max_seconds!=CAPS[a.phase] or len(a.source_commit)!=40 or any(c not in '0123456789abcdef' for c in a.source_commit):p.error('registered cap/full source commit required')
    out=a.output.resolve();category='P1' if a.phase in ('run','evaluate') else 'P0'
    allowed=(ROOT/'private_data/closure_mechanism_20261019'/category/'finite_grammar_discovery01').resolve()
    if out==allowed or not out.is_relative_to(allowed):raise ValueError('exclusive phase output required')
    out.mkdir(parents=True,exist_ok=False);receipt(out,a.phase,a.source_commit,a.binding);deadline=START+a.max_seconds
    def timeout():atomic(out/'timeout.json',{'status':'TIMEOUT'});os._exit(124)
    timer=threading.Timer(max(.001,deadline-time.perf_counter()),timeout);timer.daemon=True;timer.start()
    try:
        binding=json.loads(a.binding.read_text(encoding='utf-8'));result=execute(a.phase,out,binding,a.source_commit,deadline)
        result.update(source_commit=a.source_commit,worker_phase_elapsed_seconds=time.perf_counter()-START)
        atomic(out/'result.json',result);atomic(out/'complete.json',{'status':'COMPLETED','result':result})
        return 0 if result['status'] in ('PASS','COMPLETED') or (a.phase=='cost' and result['status']=='REJECT') else 2
    except BaseException as error:
        import traceback
        atomic(out/'failure.json',{'status':'UNRESOLVED','error':repr(error),'traceback':traceback.format_exc()});raise
    finally:timer.cancel()
if __name__=='__main__':raise SystemExit(main())
