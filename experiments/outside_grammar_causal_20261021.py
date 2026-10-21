"""Frozen tanh learner against an outside-grammar rational teacher; no project readers."""
from __future__ import annotations
import argparse,json,os,sys,threading,time
from pathlib import Path
START=time.perf_counter();ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'experiments'),str(ROOT/'src')]
from finite_grammar_discovery_20261021 import atomic as _atomic,sha,bound,passed,receipt
SEEDS=(6201,6202,6203)
TERM={'channel':'port','expression':{'op':'tanh','arg':{'op':'var','name':'q'},'scale':2.}}
CAPS={'qa':370.,'cost':370.,'run':570.,'evaluate':80.}
JSON_LIMIT=262144


def atomic(path,value):
    if len(json.dumps(value,indent=2,sort_keys=True,allow_nan=False).encode('utf-8'))>JSON_LIMIT:
        raise ValueError('registered JSON serialization envelope exceeded')
    _atomic(path,value)
def serialization_envelopes():
    # Source maps stay in the external binding; scientific identities carry
    # only the source commit and actual observation identity, never that map.
    maximum=1.7976931348623157e308
    summary={k:{'parameter_'+str(i)+'x'*128:maximum for i in range(96)} for k in ('first','last','max')}
    row={'seed':6201,'status':'FROZEN','path':'x'*1024,'files':{'model.json':'f'*64,'training.pt':'f'*64},
         'train_raw':{'path':'x'*1024,'sha256':'f'*64,'bytes':9999999},'trace':[maximum]*512,
         'raw_gradient_norms':{'total_raw_norm':[maximum]*512,'clip_factor':[maximum]*512},
         'parameter_gradient_summary':{'path':'x'*1024,'sha256':'f'*64}}
    identity={'source_commit':'f'*40,'seed':6201,'train_raw':row['train_raw'],'learner_term':TERM}
    training_identity_bytes=len(json.dumps({'identity':identity,'trace':[maximum]*512}).encode('utf-8'))
    full={'status':'COMPLETED','source_commit':'f'*40,'all_three_frozen':True,'rows':[row]*3,'selection':'fixed_learner_no_search','worker_phase_elapsed_seconds':maximum}
    complete_bytes=len(json.dumps({'status':'COMPLETED','result':full},indent=2,sort_keys=True).encode('utf-8'))
    summary_bytes=len(json.dumps(summary,indent=2,sort_keys=True).encode('utf-8'))
    if training_identity_bytes>131072 or complete_bytes>JSON_LIMIT or summary_bytes>JSON_LIMIT:raise ValueError('full fixed serialization budget does not fit')
    return {'training_identity_and_trace_bytes':training_identity_bytes,'formal_complete_envelope_bytes':complete_bytes}
def update(m,opt,rng,obs,scale,deadline):
    import math
    from closure_causal_history_residual_20261019 import train_update
    value,gradients=train_update(m,opt,rng,obs,scale,deadline)
    total=math.sqrt(sum(v*v for v in gradients.values()))
    return value,gradients,total,min(1.,100./(total+1e-6))
def advance(m,opt,rng,obs,scale,deadline,trace,gradient_trace,gradient_summary):
    value,g,total,clip_factor=update(m,opt,rng,obs,scale,deadline);trace.append(value)
    gradient_trace['total_raw_norm'].append(total);gradient_trace['clip_factor'].append(clip_factor)
    if len(g)>96 or any(len(n)>140 for n in g):raise ValueError('gradient-name serialization envelope exceeded')
    if not gradient_summary['first']:gradient_summary['first']=dict(g)
    gradient_summary['last']=dict(g)
    for name,norm in g.items():gradient_summary['max'][name]=max(norm,gradient_summary['max'].get(name,0.))
    return value


class RationalTeacher:
    """Only the generator uses the hidden teacher state."""
    def __init__(self,base):self.base=base
    @property
    def mass(self):return self.base.mass
    def begin_graph_cache(self):self.base.begin_graph_cache()
    def end_graph_cache(self):self.base.end_graph_cache()
    def field(self,z,u=None):
        import torch
        q=z[...,30:40];port=.4*q/(1+q.abs())
        if u is not None:port=port+torch.as_tensor(u,device=z.device,dtype=z.dtype)
        return self.base.field(z,port)
    def rk4(self,z,dt,u=None):
        a=self.field(z,u);b=self.field(z+dt*a/2,u);c=self.field(z+dt*b/2,u);d=self.field(z+dt*c,u)
        return z+dt*(a+2*b+2*c+d)/6


def fixture(seed,stream,deadline,timing=False):
    import numpy as np,torch
    from closure_discovery_benchmarks import toy_source
    from closure_causal_recovery_20261019 import copy_runtime,trajectory
    teacher=RationalTeacher(copy_runtime(toy_source(),[],'cuda'))
    rng=np.random.default_rng(np.random.SeedSequence([seed,20261021,9201,991 if timing else 0]).spawn(2)[stream])
    initial=rng.normal(size=(16,40))*np.where(np.arange(16)[:,None]<8,.08,.3)
    initial[:,30:]=rng.normal(size=(16,10))*np.where(np.arange(16)[:,None]<8,.2,1.2)
    z=torch.as_tensor(initial,device='cuda',dtype=torch.float64);steps=60 if stream==0 else 110
    with torch.no_grad():
        coarse=trajectory(teacher,z,steps,.01,deadline).cpu().numpy()
        fine=trajectory(teacher,z,2*steps,.005,deadline).cpu().numpy()[:,::2]
    scales=np.maximum(coarse[:,:61].std((0,1)),1e-6)
    return coarse,fine,scales
def make(seed,observed):
    import torch
    from closure_causal_recovery_20261019 import runtime_model
    from closure_discovery_benchmarks import toy_source
    from closure_causal_history_residual_20261019 import CausalHistoryResidual,make_optimizer
    m=CausalHistoryResidual.from_train(runtime_model(toy_source(),[TERM],seed,'cuda'),torch.as_tensor(observed[:,:11],device='cuda'))
    return m,make_optimizer(m)
def qualify(coarse,fine,scales):
    import numpy as np
    if coarse.shape!=fine.shape or coarse.shape[0]!=16 or coarse.shape[-1]!=30 or np.asarray(scales).shape!=(30,):raise ValueError('all16 observed-coordinate events required')
    scales=np.asarray(scales)
    if not np.isfinite(scales).all() or (scales<1e-6).any():raise ValueError('fixed finite train scales required')
    with np.errstate(over='ignore',invalid='ignore',divide='ignore'):delta=np.max(abs(coarse-fine)/scales,(1,2))
    finite=np.isfinite(coarse).all((1,2)) & np.isfinite(fine).all((1,2))
    eligible=finite & np.isfinite(delta) & (delta<=1e-3)
    return {'valid':bool(eligible.all()),'count':16,'eligible':eligible.tolist(),
            'normalized_step_max':[float(v) if np.isfinite(v) else None for v in delta],
            'failure_category':None if eligible.all() else 'numerical_failure'}
def predict(m,observed,dt,deadline):
    import torch
    from closure_causal_history_residual_20261019 import prediction
    with torch.no_grad():return prediction(m,torch.as_tensor(observed,device='cuda'),dt,deadline).cpu().numpy()
def metrics(mc,mf,tc,tf,sc):
    import numpy as np
    tq=qualify(tc,tf,sc);mq=qualify(mc,mf,sc)
    if tc.shape[1]!=mc.shape[1]+11:raise ValueError('history/future alignment differs')
    blocks={}
    for name,a,b in [('all',0,30),('delta',0,10),('omega',10,20),('p',20,30)]:
        with np.errstate(over='ignore',invalid='ignore',divide='ignore'):
            event=np.mean(((mc[...,a:b]-tc[:,11:,a:b])/sc[a:b])**2,(1,2))
        valid=tq['valid'] and mq['valid'] and bool(np.isfinite(event).all())
        blocks[name]={'count':16,'per_event_mse':[float(x) if np.isfinite(x) else None for x in event],
                      'mean':float(event.mean()) if valid else None,'weak_mean':float(event[:8].mean()) if valid else None,
                      'strong_mean':float(event[8:].mean()) if valid else None,'worst':float(event.max()) if valid else None,
                      'median':float(np.median(event)) if valid else None,'above_0_01':int((event>.01).sum()) if valid else None}
    return {'valid':tq['valid'] and mq['valid'] and blocks['all']['mean'] is not None,'blocks':blocks,
            'teacher_qualification':tq,'model_qualification':mq}
def full_evaluation(m,tr,tf,va,vf,sc,deadline):
    train_c=predict(m,tr,.01,deadline);train_f=predict(m,tr,.005,deadline)
    val_c=predict(m,va,.01,deadline);val_f=predict(m,va,.005,deadline)
    return {'train':metrics(train_c,train_f,tr,tf,sc),'val':metrics(val_c,val_f,va,vf,sc)},(train_c,train_f,val_c,val_f)
def raw(path,**arrays):
    import numpy as np
    if any(a.shape[-1]!=30 for k,a in arrays.items() if k!='scales'):raise ValueError('only observed30 raw arrays')
    if Path(path).exists():raise ValueError('refuse raw overwrite')
    np.savez(path,**arrays)
    return {'path':str(Path(path).resolve()),'sha256':sha(path),'bytes':Path(path).stat().st_size}


def verify_checkpoint(directory,expected_step,expected_trace,expected_identity,files,train_raw):
    """Verify actual saved training metadata; a directory name is not evidence."""
    import torch
    directory=Path(directory)
    if set(files)!={'model.json','training.pt'}:raise ValueError('missing endpoint pair')
    for name,digest in files.items():
        if sha(directory/name)!=digest:raise ValueError('frozen endpoint changed')
    if sha(train_raw['path'])!=train_raw['sha256']:raise ValueError('train observations changed')
    state=torch.load(directory/'training.pt',map_location='cpu',weights_only=False)
    if (state.get('completed_updates')!=expected_step or len(state.get('trace',[]))!=expected_step
            or state.get('trace')!=expected_trace or state.get('identity')!=expected_identity):
        raise ValueError('saved endpoint count/trace/source identity differs')
    optimizer=state.get('optimizer',{})
    if not optimizer.get('state') or not optimizer.get('param_groups'):
        raise ValueError('saved endpoint has empty Adam')
    steps=[s.get('step') for s in optimizer['state'].values() if 'step' in s]
    if not steps or any(int(s)!=expected_step for s in steps):raise ValueError('Adam update count differs')
    return state


def validate_selection(value,commit):
    if value.get('all_three_frozen') is not True or value.get('source_commit')!=commit or [r.get('seed') for r in value.get('rows',[])]!=list(SEEDS):raise ValueError('all three same-source endpoints required before val')
    for row in value['rows']:
        directory=Path(row['path'])
        if directory.name!='checkpoint_00512' or directory.parent.name!=str(row['seed']):raise ValueError('wrong frozen endpoint')
        if row.get('status')!='FROZEN' or len(row.get('trace',[]))!=512:raise ValueError('incomplete row')
        identity={'source_commit':commit,'seed':row['seed'],'train_raw':row['train_raw'],'learner_term':TERM}
        verify_checkpoint(directory,512,row['trace'],identity,row['files'],row['train_raw'])


def run(output,binding,commit,deadline):
    import numpy as np,torch
    from closure_causal_history_residual_20261019 import train_update,checkpoint,restore
    admission=passed(binding,'admission_result')
    if admission.get('P1_complete_cap_seconds')!=900:raise ValueError('admission cap differs')
    serialization_envelopes();rows=[]
    for seed in SEEDS:
        folder=output/str(seed);folder.mkdir();trace=[];gradient_trace={'total_raw_norm':[],'clip_factor':[]};gradient_summary={'first':{},'last':{},'max':{}}
        try:
            tr,tf,sc=fixture(seed,0,deadline);identity_raw=raw(folder/'train.npz',coarse=tr,fine=tf,scales=sc)
            qualification=qualify(tr,tf,sc);atomic(folder/'teacher_qualification.json',qualification)
            if not qualification['valid']:
                rows.append({'seed':seed,'status':'UNRESOLVED','train_raw':identity_raw,'qualification':qualification});continue
            m,opt=make(seed,tr);rng=np.random.default_rng(np.random.SeedSequence([seed,7]));obs=torch.as_tensor(tr,device='cuda');scale=torch.as_tensor(sc,device='cuda')
            identity={'source_commit':commit,'seed':seed,'train_raw':identity_raw,'learner_term':TERM}
            checkpoint(folder,m,opt,rng,0,trace,identity)
            for step in range(1,513):
                advance(m,opt,rng,obs,scale,deadline,trace,gradient_trace,gradient_summary)
                if step in (32,128,512):checkpoint(folder,m,opt,rng,step,trace,identity)
                if step==32:
                    m,opt=make(seed,tr);recovered,replay=restore(folder,m,opt,rng,identity)
                    if recovered!=32 or replay!=trace:raise ValueError('32 continuation failed')
                    trace=replay
            path=folder/'checkpoint_00512'
            atomic(folder/'parameter_gradient_summary.json',gradient_summary)
            rows.append({'seed':seed,'status':'FROZEN','path':str(path),'files':{n:sha(path/n) for n in ('model.json','training.pt')},
                         'train_raw':identity_raw,'trace':trace,'raw_gradient_norms':gradient_trace,
                         'parameter_gradient_summary':{'path':str(folder/'parameter_gradient_summary.json'),'sha256':sha(folder/'parameter_gradient_summary.json')}})
        except TimeoutError:raise
        except (RuntimeError,FloatingPointError,ValueError) as error:
            atomic(folder/'parameter_gradient_summary.json',gradient_summary)
            rows.append({'seed':seed,'status':'UNRESOLVED','error':repr(error),'trace':trace,'raw_gradient_norms':gradient_trace,
                         'parameter_gradient_summary':{'path':str(folder/'parameter_gradient_summary.json'),'sha256':sha(folder/'parameter_gradient_summary.json')},
                         'raw_files':[{'path':str(p),'sha256':sha(p)} for p in folder.glob('*.npz')]})
            atomic(folder/'failure.json',rows[-1])
    result={'status':'COMPLETED','source_commit':commit,'all_three_frozen':all(r['status']=='FROZEN' for r in rows),'rows':rows,'selection':'fixed_learner_no_search'}
    atomic(output/'selection.json',result);return result
def evaluate(output,binding,commit,deadline):
    import numpy as np
    from closure_causal_history_residual_20261019 import CausalHistoryResidual
    frozen=bound(binding,'selection_manifest');validate_selection(frozen,commit);rows=[]
    for row in frozen['rows']:
        seed=row['seed'];folder=output/str(seed);folder.mkdir()
        try:
            with np.load(row['train_raw']['path']) as data:tr=data['coarse'].copy();tf=data['fine'].copy();sc=data['scales'].copy()
            va,vf,_=fixture(seed,1,deadline);valraw=raw(folder/'val.npz',coarse=va,fine=vf)
            m=CausalHistoryResidual.from_export(Path(row['path'])/'model.json').cuda()
            result,pred=full_evaluation(m,tr,tf,va,vf,sc,deadline)
            predraw=raw(folder/'predictions.npz',train_coarse=pred[0],train_fine=pred[1],val_coarse=pred[2],val_fine=pred[3])
            rows.append({'seed':seed,'status':'COMPLETED','evaluation':result,'val_raw':valraw,'prediction_raw':predraw})
        except TimeoutError:raise
        except (RuntimeError,FloatingPointError,ValueError) as error:
            rows.append({'seed':seed,'status':'UNRESOLVED','error':repr(error),'count':16,
                         'raw_files':[{'path':str(p),'sha256':sha(p)} for p in folder.glob('*.npz')]})
    validate_selection(frozen,commit)
    return {'status':'COMPLETED','rows':rows,'scope':'single fixed learner capability; no structural recovery or promotion'}


def cost(output,binding,deadline):
    import numpy as np,torch
    from closure_causal_history_residual_20261019 import train_update,checkpoint,restore,CausalHistoryResidual
    # Heavy imports and actual QA/source proof precede the measured body.
    from closure_causal_recovery_20261019 import runtime_model,copy_runtime,trajectory
    from closure_discovery_benchmarks import toy_source
    passed(binding,'qa_result')
    serialization_envelopes()
    begin=time.perf_counter();panels={k:[] for k in ('update','g_train','g_val','v_complete','save','restore','construct','load','raw_write','json_write')}
    progressive=output/'partial_timings.json'
    def measured(key,fn):
        torch.cuda.synchronize();start=time.perf_counter();value=fn();torch.cuda.synchronize()
        panels[key].append(time.perf_counter()-start);return value
    def generated(seed,stream,scales=None):
        c,f,own_scales=fixture(seed,stream,deadline,True)
        try:qualification=qualify(c,f,own_scales if scales is None else scales)
        except ValueError as error:qualification={'valid':False,'count':16,'eligible':[False]*16,'error':repr(error),'failure_category':'numerical_failure'}
        return c,f,own_scales,qualification
    for seed in SEEDS:
        folder=output/f'timing_{seed}';folder.mkdir()
        tr,tf,sc,tq=measured('g_train',lambda:generated(seed,0))
        va,vf,_,vq=measured('g_val',lambda:generated(seed,1,sc))
        train_raw=measured('raw_write',lambda:raw(folder/'train.npz',coarse=tr,fine=tf,scales=sc))
        measured('raw_write',lambda:raw(folder/'val.npz',coarse=va,fine=vf))
        if not tq['valid'] or not vq['valid']:
            atomic(folder/'teacher_qualification_failure.json',{'train':tq,'val':vq})
            raise FloatingPointError('timing teacher qualification failed; all16 retained')
        m,opt=measured('construct',lambda:make(seed,tr));rng=np.random.default_rng(np.random.SeedSequence([seed,7]))
        obs=torch.as_tensor(tr,device='cuda');scale=torch.as_tensor(sc,device='cuda');trace=[]
        gradient_trace={'total_raw_norm':[],'clip_factor':[]};gradient_summary={'first':{},'last':{},'max':{}}
        for step in range(32):
            measured('update',lambda:advance(m,opt,rng,obs,scale,deadline,trace,gradient_trace,gradient_summary))
            atomic(progressive,{'status':'INCOMPLETE','completed_timing_seed':seed,'updates_in_seed':step+1,'panels':panels})
        identity={'cost_only':True,'seed':seed,'actual_updates':32,'identity_padding':b'0'*131072}
        saved=[]
        for index in range(4):
            savefolder=folder/f'save{index}';savefolder.mkdir();saved.append(savefolder)
            measured('save',lambda:checkpoint(savefolder,m,opt,rng,32,trace,identity))
        m,opt=measured('construct',lambda:make(seed,tr))
        recovered,replay=measured('restore',lambda:restore(saved[0],m,opt,rng,identity))
        if recovered!=32 or replay!=trace:raise ValueError('actual cost restore incomplete')
        endpoint=saved[0]/'checkpoint_00032'
        frozen_files={n:sha(endpoint/n) for n in ('model.json','training.pt')}
        def loaded_and_verified():
            verify_checkpoint(endpoint,32,trace,identity,frozen_files,train_raw)
            model=CausalHistoryResidual.from_export(endpoint/'model.json').cuda()
            with np.load(train_raw['path']) as data:
                loaded_train=data['coarse'].copy();loaded_fine=data['fine'].copy();loaded_scale=data['scales'].copy()
            verify_checkpoint(endpoint,32,trace,identity,frozen_files,train_raw)
            return model,loaded_train,loaded_fine,loaded_scale
        m,tr,tf,sc=measured('load',loaded_and_verified)
        evaluation,pred=measured('v_complete',lambda:full_evaluation(m,tr,tf,va,vf,sc,deadline))
        measured('raw_write',lambda:raw(folder/'predictions.npz',train_coarse=pred[0],train_fine=pred[1],val_coarse=pred[2],val_fine=pred[3]))
        atomic(progressive,{'status':'INCOMPLETE','completed_seed':seed,'panels':panels})
    overhead=len(json.dumps({'cost_only':True,'padding':''},indent=2,sort_keys=True).encode('utf-8'))
    for index in range(64):measured('json_write',lambda:atomic(output/f'cost_only_metadata_{index:02d}.json',{'cost_only':True,'padding':'0'*(JSON_LIMIT-overhead)}))
    measured_total=sum(sum(values) for values in panels.values());body=time.perf_counter()-begin
    return {'status':'COMPLETED','cost_panels_complete':True,'panels':panels,
            'body_wall_seconds':body,'body_measured_seconds':measured_total,'body_unassigned_seconds':max(0.,body-measured_total),
            'pre_body_seconds':begin-START,'counts':{k:len(v) for k,v in panels.items()},
            'science_scores':None,'actual_timing_updates':96,'formal_updates':1536,'formal_seeds':list(SEEDS)}


def execute(phase,out,binding,commit,deadline):
    import torch
    torch.set_num_threads(1)
    if phase=='qa':
        import subprocess
        command=[sys.executable,'-m','pytest','-q',str(ROOT/'tests/test_outside_grammar_causal_20261021.py'),'--basetemp',str(out/'pytest_tmp')]
        proc=subprocess.run(command,cwd=ROOT,capture_output=True,text=True,timeout=max(.1,deadline-time.perf_counter()))
        (out/'pytest_stdout.txt').write_text(proc.stdout,encoding='utf-8');(out/'pytest_stderr.txt').write_text(proc.stderr,encoding='utf-8')
        return {'status':'PASS' if proc.returncode==0 else 'FAIL','pytest_returncode':proc.returncode,'stdout':proc.stdout,'stderr':proc.stderr}
    if phase=='cost':return cost(out,binding,deadline)
    if phase=='run':return run(out,binding,commit,deadline)
    return evaluate(out,binding,commit,deadline)
def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('phase',choices=CAPS)
    for key in ('output','binding'):p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--source-commit',required=True);p.add_argument('--max-seconds',type=float,required=True);a=p.parse_args(argv)
    if a.max_seconds!=CAPS[a.phase] or len(a.source_commit)!=40 or any(c not in '0123456789abcdef' for c in a.source_commit):p.error('registered cap and full commit required')
    out=a.output.resolve();category='P1' if a.phase in ('run','evaluate') else 'P0'
    allowed=(ROOT/'private_data/closure_mechanism_20261019'/category/'outside_grammar_causal01').resolve()
    if out==allowed or not out.is_relative_to(allowed):raise ValueError('exclusive phase directory required')
    out.mkdir(parents=True,exist_ok=False);receipt(out,a.phase,a.source_commit,a.binding);deadline=START+a.max_seconds
    def timeout():atomic(out/'timeout.json',{'status':'TIMEOUT'});os._exit(124)
    timer=threading.Timer(max(.001,deadline-time.perf_counter()),timeout);timer.daemon=True;timer.start()
    try:
        binding=json.loads(a.binding.read_text(encoding='utf-8'));result=execute(a.phase,out,binding,a.source_commit,deadline)
        result.update(source_commit=a.source_commit,worker_phase_elapsed_seconds=time.perf_counter()-START)
        atomic(out/'result.json',result);atomic(out/'complete.json',{'status':'COMPLETED','result':result})
        return 0 if result['status'] in ('PASS','COMPLETED') else 2
    except BaseException as error:
        import traceback
        atomic(out/'failure.json',{'status':'UNRESOLVED','error':repr(error),'traceback':traceback.format_exc()});raise
    finally:timer.cancel()
if __name__=='__main__':raise SystemExit(main())
