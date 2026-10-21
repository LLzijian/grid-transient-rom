"""Paired fixed rational/tanh causal study; numerical imports are worker-lazy."""
from __future__ import annotations
import argparse,copy,hashlib,json,os,sys,threading,time
from pathlib import Path
START=time.perf_counter();ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'experiments'),str(ROOT/'src')]
from finite_grammar_discovery_20261021 import sha,bound,passed,receipt
from outside_grammar_causal_20261021 import atomic,raw,qualify,metrics,advance,RationalTeacher
SEEDS=(6301,6302,6303);GROUPS=('0_rational','0_tanh','1_rational','1_tanh')
CAPS={'qa':220.,'cost':420.,'run':2700.,'evaluate':600.};FORMAT='rational_control_causal_v1'
_MODEL=None


def model_class():
    global _MODEL
    if _MODEL is not None:return _MODEL
    import torch
    from closure_causal_history_residual_20261019 import CausalHistoryResidual
    from closure_training_runtime import TrainingClosureROM
    class StudyModel(CausalHistoryResidual):
        def __init__(self,base,scales,source_mass=None,form='rational'):
            super().__init__(base,scales,source_mass)
            if form not in ('rational','tanh') or base.terms or base.coefficients.numel():raise ValueError('fixed form and empty ordinary base required')
            self.form=form;self.port_coefficient=torch.nn.Parameter(base.mass.new_zeros(1))
        @classmethod
        def from_train(cls,base,train_histories,form):
            from closure_causal_history_residual_20261019 import history_features
            if train_histories.shape!=(16,11,30):raise ValueError('exact16 train histories required')
            scale=history_features(train_histories,base.mass.detach()).square().mean((0,1)).sqrt().clamp_min(1e-6).detach()
            return cls(base,scale,form=form)
        def field(self,z,u=None):
            z=torch.as_tensor(z,device=self.mass.device,dtype=self.mass.dtype);q=z[...,30:40]
            feature=q/(1+q.abs()) if self.form=='rational' else torch.tanh(2*q)
            port=self.port_coefficient[0]*feature
            if u is not None:port=port+torch.as_tensor(u,device=z.device,dtype=z.dtype)
            return self.base.field(z,port)
        def rk4(self,z,dt,u=None):
            a=self.field(z,u);b=self.field(z+dt*a/2,u);c=self.field(z+dt*b/2,u);d=self.field(z+dt*c,u)
            return z+dt*(a+2*b+2*c+d)/6
        def export(self,path):
            if getattr(self.base,'_runtime_cache',None) is not None:raise RuntimeError('close graph before export')
            value={'format':FORMAT,'form':self.form,'base':copy.deepcopy(self.base).cpu()._export_payload(),
                   'history_scales':self.history_scales.detach().cpu().tolist(),'source_mass':self.source_mass.detach().cpu().tolist(),
                   'history_coefficients':self.history_coefficients.detach().cpu().tolist(),
                   'port_coefficient':self.port_coefficient.detach().cpu().tolist(),
                   'trainable_parameters':[n for n,p in self.named_parameters() if p.requires_grad]}
            atomic(path,value);return value
        @classmethod
        def from_export(cls,path):
            value=json.loads(Path(path).read_text(encoding='utf-8'))
            expected={'format','form','base','history_scales','source_mass','history_coefficients','port_coefficient','trainable_parameters'}
            if set(value)!=expected or value['format']!=FORMAT or value['form'] not in ('rational','tanh'):raise ValueError('wrong study format/form')
            b=value['base']
            if b.get('format')!='symbolic_closure_v1' or b.get('terms')!=[] or b.get('coefficients')!=[]:raise ValueError('ordinary empty base required')
            base=TrainingClosureROM.from_config(b['config'],terms=[])
            state={k:torch.as_tensor(v,dtype=base.state_dict()[k].dtype) for k,v in b['state_dict'].items()};base.load_state_dict(state,strict=True)
            base.provenance=copy.deepcopy(b.get('provenance',{}))
            m=cls(base,value['history_scales'],value['source_mass'],value['form'])
            a=torch.as_tensor(value['port_coefficient'],dtype=m.mass.dtype);h=torch.as_tensor(value['history_coefficients'],dtype=m.mass.dtype)
            if a.shape!=(1,) or h.shape!=(6,) or not bool(torch.isfinite(a).all() and torch.isfinite(h).all()):raise ValueError('invalid closure/history coefficients')
            with torch.no_grad():m.port_coefficient.copy_(a);m.history_coefficients.copy_(h)
            names=set(dict(m.named_parameters()));flags=set(value['trainable_parameters'])
            if not flags<=names:raise ValueError('unknown parameter flags')
            for n,p in m.named_parameters():p.requires_grad_(n in flags)
            return m
    _MODEL=StudyModel;return StudyModel
def __getattr__(name):
    if name=='StudyModel':return model_class()
    raise AttributeError(name)
def optimizer(m):
    import torch
    groups={'encoder':[],'closure':[],'skeleton':[]}
    for name,p in m.base.named_parameters():
        if p.requires_grad:groups['encoder' if ('gru' in name or 'head' in name) else 'closure' if name in ('coefficients','_template_scalars.3','_template_scalars.4') else 'skeleton'].append(p)
    if m.port_coefficient.requires_grad:groups['closure'].append(m.port_coefficient)
    rates={'encoder':3e-4,'closure':1e-3,'skeleton':1e-2}
    result=[{'params':p,'lr':rates[n]} for n,p in groups.items() if p]
    if m.history_coefficients.requires_grad:result.append({'params':[m.history_coefficients],'lr':.01})
    return torch.optim.Adam(result)
def make(seed,observed,form):
    import torch
    from closure_causal_recovery_20261019 import runtime_model
    from closure_discovery_benchmarks import toy_source
    from closure_causal_history_residual_20261019 import history_features
    base=runtime_model(toy_source(),[],seed,'cuda');h=torch.as_tensor(observed[:,:11],device='cuda')
    if h.shape!=(16,11,30):raise ValueError('exact16 training histories required')
    m=model_class().from_train(base,h,form=form);return m,optimizer(m)
def array_sha(a):return hashlib.sha256(a.tobytes()).hexdigest()
def data(seed,split,noise,deadline,timing=False,train_scales=None):
    import numpy as np,torch
    from closure_causal_recovery_20261019 import copy_runtime,trajectory
    from closure_discovery_benchmarks import toy_source
    if split not in (0,1) or noise not in (0,1) or (split==1 and train_scales is None):raise ValueError('fixed split/noise and train-only scales required')
    tail=991 if timing else 0
    rng=np.random.default_rng(np.random.SeedSequence([seed,20261021,9301,tail]).spawn(2)[split])
    initial=rng.normal(size=(16,40))*np.where(np.arange(16)[:,None]<8,.08,.3)
    initial[:,30:]=rng.normal(size=(16,10))*np.where(np.arange(16)[:,None]<8,.2,1.2)
    teacher=RationalTeacher(copy_runtime(toy_source(),[],'cuda'));z=torch.as_tensor(initial,device='cuda',dtype=torch.float64);steps=60 if split==0 else 110
    with torch.no_grad():c=trajectory(teacher,z,steps,.01,deadline).cpu().numpy();f=trajectory(teacher,z,2*steps,.005,deadline).cpu().numpy()[:,::2]
    scales=np.maximum(c.std((0,1)),1e-6) if split==0 else np.array(train_scales,copy=True)
    observed=c.copy();nrng=np.random.default_rng(np.random.SeedSequence([seed,20261021,9302,tail]).spawn(2)[split])
    if noise:
        if split==0:observed+=.01*scales*nrng.normal(size=c.shape)
        else:observed[:,:11]+=.01*scales*nrng.normal(size=c[:,:11].shape)
    identity={'clean_sha256':array_sha(c),'fine_sha256':array_sha(f),'observed_sha256':array_sha(observed),'scales_sha256':array_sha(scales),'seed':seed,'noise':noise,'split':split}
    return {'coarse':c,'fine':f,'observed':observed,'scales':scales,'identity':identity,'qualification':qualify(c,f,scales)}
def save_data(path,d):return raw(path,coarse=d['coarse'],fine=d['fine'],observed=d['observed'],scales=d['scales'])
def load_data(path):
    import numpy as np
    with np.load(path) as f:return {k:f[k].copy() for k in ('coarse','fine','observed','scales')}
def predict(m,obs,dt,deadline):
    import torch
    from closure_causal_history_residual_20261019 import prediction
    with torch.no_grad():return prediction(m,torch.as_tensor(obs,device='cuda'),dt,deadline).cpu().numpy()
def complete(m,tr,va,deadline):
    import numpy as np
    tc=predict(m,tr['observed'],.01,deadline);tf=predict(m,tr['observed'],.005,deadline)
    vc=predict(m,va['observed'],.01,deadline);vf=predict(m,va['observed'],.005,deadline);s=tr['scales']
    result={'train':metrics(tc,tf,tr['coarse'],tr['fine'],s),'val':metrics(vc,vf,va['coarse'],va['fine'],s)}
    objective=np.mean(((tc-tr['observed'][:,11:])/s)**2)
    result['noisy_train_objective']=float(objective) if np.isfinite(objective) else None
    return result,{'train_coarse':tc,'train_fine':tf,'val_coarse':vc,'val_fine':vf}
def fileid(path):return {'path':str(Path(path).resolve()),'sha256':sha(path)}
def serialization_envelopes():
    huge=1.7976931348623157e308
    trace={'loss':[huge]*512,'gradient':{'total_raw_norm':[huge]*512,'clip_factor':[huge]*512},'parameter_summary':{k:{'x'*120+str(i):huge for i in range(96)} for k in ('first','last','max')}}
    row={'key':'1_rational_6303','path':'x'*1024,'train_raw':{'path':'x'*1024,'sha256':'f'*64},'trace':{'path':'x'*1024,'sha256':'f'*64},'identity':{'source_commit':'f'*40,'key':'1_rational_6303','data_identity':{k:'f'*64 for k in ('clean_sha256','fine_sha256','observed_sha256','scales_sha256')},'train_raw':{'path':'x'*1024,'sha256':'f'*64}}}
    trace_bytes=len(json.dumps(trace,indent=2,sort_keys=True).encode());selection_bytes=len(json.dumps({'rows':[row]*12},indent=2,sort_keys=True).encode())+8192
    identity_bytes=len(json.dumps({'identity':row['identity'],'trace':[huge]*512}).encode())
    if max(trace_bytes,selection_bytes)>262144 or identity_bytes>131072:raise ValueError('fixed serialization envelope exceeded')
    return {'trace_bytes':trace_bytes,'selection_bytes':selection_bytes,'checkpoint_identity_bytes':identity_bytes}
def validate_optimizer(opt,payload,step,gradient_names=()):
    import torch
    if not opt.get('state') or not opt.get('param_groups'):raise ValueError('nonempty Adam state/groups required')
    flags=payload['trainable_parameters'];groups={'encoder':[],'closure':[],'skeleton':[]}
    for name in flags:
        if not name.startswith('base.'):continue
        n=name[5:];groups['encoder' if ('gru' in n or 'head' in n) else 'closure' if n in ('coefficients','_template_scalars.3','_template_scalars.4') else 'skeleton'].append(name)
    if 'port_coefficient' in flags:groups['closure'].append('port_coefficient')
    expected=[(names,rate) for names,rate in ((groups['encoder'],3e-4),(groups['closure'],1e-3),(groups['skeleton'],1e-2)) if names]
    if 'history_coefficients' in flags:expected.append((['history_coefficients'],.01))
    if len(expected)!=len(opt['param_groups']):raise ValueError('Adam group mapping differs')
    parameters={};name_ids={}
    for (names,rate),group in zip(expected,opt['param_groups']):
        ids=group.get('params',[])
        if len(ids)!=len(names) or group.get('lr')!=rate:raise ValueError('Adam parameter group differs')
        for pid,name in zip(ids,names):
            if pid in parameters:raise ValueError('duplicate optimizer parameter id')
            value=payload['base']['state_dict'][name[5:]] if name.startswith('base.') else payload[name]
            parameters[pid]=torch.as_tensor(value,dtype=torch.float64);name_ids[name]=pid
    for pid,state in opt['state'].items():
        if pid not in parameters or not {'step','exp_avg','exp_avg_sq'}<=set(state):raise ValueError('incomplete or unassociated Adam state')
        counter=torch.as_tensor(state['step'])
        if counter.numel()!=1 or not bool(torch.isfinite(counter).all()) or float(counter)!=step:raise ValueError('Adam step differs')
        for key in ('exp_avg','exp_avg_sq'):
            value=state[key]
            if not torch.is_tensor(value) or value.shape!=parameters[pid].shape or not bool(torch.isfinite(value).all()):raise ValueError('invalid Adam moment shape/value')
    for name in gradient_names:
        if name not in name_ids:raise ValueError('gradient parameter absent from optimizer')
        pid=name_ids[name]
        if parameters[pid].numel() and pid not in opt['state']:raise ValueError('observed gradient missing Adam state')


def verify(row,step=512,cost=False):
    import torch
    directory=Path(row['path'])
    if not cost and (directory.name!='checkpoint_00512' or row.get('status')!='FROZEN'):raise ValueError('incomplete512 endpoint')
    for n,h in row['files'].items():
        if sha(directory/n)!=h:raise ValueError('endpoint changed')
    if set(row['files'])!={'model.json','training.pt'} or sha(row['train_raw']['path'])!=row['train_raw']['sha256']:raise ValueError('missing endpoint/data pair')
    state=torch.load(directory/'training.pt',map_location='cpu',weights_only=False)
    trace_payload=json.loads(Path(row['trace']['path']).read_text())
    trace=trace_payload['loss']
    if sha(row['trace']['path'])!=row['trace']['sha256']:raise ValueError('trace changed')
    if cost and (trace!=row['cost_trace'] or trace_payload.get('cost_only',{}).get('actual_updates')!=step):raise ValueError('actual cost trace differs')
    if state.get('completed_updates')!=step or len(state.get('trace',[]))!=step or state['trace']!=trace or state['identity']!=row['identity']:raise ValueError('actual step/trace/identity differs')
    payload=json.loads((directory/'model.json').read_text())
    if payload.get('format')!=FORMAT or payload.get('form')!=row['form']:raise ValueError('model form differs')
    gradient_names=trace_payload.get('parameter_summary',{}).get('max',{})
    validate_optimizer(state.get('optimizer',{}),payload,step,gradient_names)
def validate_selection(frozen,commit):
    expected={g+'_'+str(s) for g in GROUPS for s in SEEDS}
    if frozen.get('all_twelve_frozen') is not True or frozen.get('source_commit')!=commit or {r.get('key') for r in frozen.get('rows',[])}!=expected or len(frozen['rows'])!=12:raise ValueError('all12 real endpoints required before val')
    pair={}
    for row in frozen['rows']:
        identity=row['identity'];data_identity=identity['data_identity']
        if (identity.get('source_commit')!=commit or identity.get('key')!=row['key'] or identity.get('form')!=row['form']
                or row['key']!=f"{row['noise']}_{row['form']}_{row['seed']}"
                or data_identity.get('seed')!=row['seed'] or data_identity.get('noise')!=row['noise'] or data_identity.get('split')!=0):raise ValueError('source/form/noise/seed differs')
        verify(row);key=(row['seed'],row['noise']);identity=row['identity']['data_identity']
        if key in pair and pair[key]!=identity:raise ValueError('paired observations differ')
        pair[key]=identity
def run(out,binding,commit,deadline,group):
    import numpy as np,torch
    from closure_causal_history_residual_20261019 import checkpoint,restore
    if passed(binding,'admission_result').get('P1_complete_cap_seconds')!=12000:raise ValueError('wrong admission')
    serialization_envelopes();noise=int(group[0]);form=group[2:];rows=[]
    for seed in SEEDS:
        key=group+'_'+str(seed);folder=out/key;folder.mkdir();trace=[];grad={'total_raw_norm':[],'clip_factor':[]};summary={'first':{},'last':{},'max':{}}
        try:
            d=data(seed,0,noise,deadline);train_raw=save_data(folder/'train.npz',d)
            if not d['qualification']['valid']:rows.append({'key':key,'status':'UNRESOLVED','qualification':d['qualification'],'train_raw':train_raw});continue
            m,opt=make(seed,d['observed'],form);rng=np.random.default_rng(np.random.SeedSequence([seed,7]));obs=torch.as_tensor(d['observed'],device='cuda');sc=torch.as_tensor(d['scales'],device='cuda')
            identity={'source_commit':commit,'key':key,'data_identity':d['identity'],'train_raw':train_raw,'form':form}
            checkpoint(folder,m,opt,rng,0,trace,identity)
            for step in range(1,513):
                advance(m,opt,rng,obs,sc,deadline,trace,grad,summary)
                if step in (32,128,512):checkpoint(folder,m,opt,rng,step,trace,identity)
                if step==32:
                    m,opt=make(seed,d['observed'],form);actual,trace=restore(folder,m,opt,rng,identity)
                    if actual!=32:raise ValueError('wrong continuation count')
            atomic(folder/'trace.json',{'loss':trace,'gradient':grad,'parameter_summary':summary})
            path=folder/'checkpoint_00512';rows.append({'key':key,'seed':seed,'noise':noise,'form':form,'status':'FROZEN','path':str(path),
                 'files':{n:sha(path/n) for n in ('model.json','training.pt')},'train_raw':train_raw,'trace':fileid(folder/'trace.json'),'identity':identity,
                 'loss_first32':sum(trace[:32])/32,'loss_last32':sum(trace[-32:])/32})
        except TimeoutError:raise
        except (RuntimeError,FloatingPointError,ValueError) as e:
            atomic(folder/'failed_trace.json',{'loss':trace,'gradient':grad,'parameter_summary':summary})
            rows.append({'key':key,'status':'UNRESOLVED','error':repr(e),'trace':fileid(folder/'failed_trace.json'),
                         'raw_files':[fileid(p) for p in folder.glob('*.npz')]})
    result={'status':'COMPLETED','source_commit':commit,'group':group,'rows':rows,'all_group_frozen':all(r['status']=='FROZEN' for r in rows)}
    atomic(out/'selection.json',result);return result
def evaluate(out,binding,commit,deadline):
    frozen=bound(binding,'selection_manifest');validate_selection(frozen,commit);rows=[]
    for row in frozen['rows']:
        folder=out/row['key'];folder.mkdir()
        try:
            tr=load_data(row['train_raw']['path']);va=data(row['seed'],1,row['noise'],deadline,train_scales=tr['scales']);vr=save_data(folder/'val.npz',va)
            m=model_class().from_export(Path(row['path'])/'model.json').cuda();result,pred=complete(m,tr,va,deadline)
            pr=raw(folder/'predictions.npz',**pred);rows.append({'key':row['key'],'seed':row['seed'],'noise':row['noise'],'form':row['form'],'status':'COMPLETED','evaluation':result,'val_raw':vr,'prediction_raw':pr})
        except TimeoutError:raise
        except (RuntimeError,FloatingPointError,ValueError) as e:rows.append({'key':row['key'],'status':'UNRESOLVED','error':repr(e),'count':16,'raw_files':[fileid(p) for p in folder.glob('*.npz')]})
    validate_selection(frozen,commit);differences=[]
    for noise in (0,1):
        for seed in SEEDS:
            a=next(r for r in rows if r['key']==f'{noise}_rational_{seed}');b=next(r for r in rows if r['key']==f'{noise}_tanh_{seed}')
            valid=all(r.get('evaluation',{}).get('val',{}).get('valid',False) for r in (a,b))
            differences.append({'noise':noise,'seed':seed,'val_tanh_minus_rational':b['evaluation']['val']['blocks']['all']['mean']-a['evaluation']['val']['blocks']['all']['mean'] if valid else None})
    # Preserve all event/block evidence in each model's own bounded file;
    # the12-model receipt contains compact summaries and their exact identities.
    for row in rows:
        if 'evaluation' in row:
            path=out/row['key']/'evaluation.json';atomic(path,row['evaluation']);row['evaluation_file']=fileid(path)
            row['evaluation']={split:{'valid':value['valid'],'blocks':{name:{k:v for k,v in block.items() if k!='per_event_mse'} for name,block in value['blocks'].items()}}
                               for split,value in row['evaluation'].items() if split in ('train','val')}
    noise_summary=[]
    for noise in (0,1):
        values=[r['val_tanh_minus_rational'] for r in differences if r['noise']==noise]
        noise_summary.append({'noise':noise,'count':3,'mean_difference':sum(values)/3 if all(v is not None for v in values) else None,
                              'directions':[None if v is None else 'positive' if v>0 else 'negative' if v<0 else 'zero' for v in values]})
    return {'status':'COMPLETED','rows':rows,'paired_differences':differences,'noise_summary':noise_summary}
def cost(out,binding,deadline):
    import numpy as np,torch
    from closure_causal_history_residual_20261019 import checkpoint,restore
    from closure_causal_recovery_20261019 import runtime_model,copy_runtime,trajectory
    from closure_discovery_benchmarks import toy_source
    model_class();passed(binding,'qa_result');serialization_envelopes();begin=time.perf_counter()
    panels={k:[] for k in ('update','g_train','g_val','v_complete','save','restore','construct','load','raw_write','json_write')}
    def measured(k,fn):
        torch.cuda.synchronize();start=time.perf_counter();value=fn();torch.cuda.synchronize();panels[k].append(time.perf_counter()-start);return value
    for group in GROUPS:
        noise=int(group[0]);form=group[2:]
        for seed in SEEDS:
            folder=out/(group+'_'+str(seed));folder.mkdir()
            tr=measured('g_train',lambda:data(seed,0,noise,deadline,True));va=measured('g_val',lambda:data(seed,1,noise,deadline,True,tr['scales']))
            train_raw=measured('raw_write',lambda:save_data(folder/'train.npz',tr));measured('raw_write',lambda:save_data(folder/'val.npz',va))
            if not tr['qualification']['valid'] or not va['qualification']['valid']:raise FloatingPointError('timing teacher qualification failed; raw retained')
            m,opt=measured('construct',lambda:make(seed,tr['observed'],form));rng=np.random.default_rng(np.random.SeedSequence([seed,7]))
            obs=torch.as_tensor(tr['observed'],device='cuda');scale=torch.as_tensor(tr['scales'],device='cuda');trace=[];grad={'total_raw_norm':[],'clip_factor':[]};summary={'first':{},'last':{},'max':{}}
            for step in range(32):
                measured('update',lambda:advance(m,opt,rng,obs,scale,deadline,trace,grad,summary))
                atomic(out/'partial_timings.json',{'status':'INCOMPLETE','group':group,'seed':seed,'step':step+1,'panels':panels})
            identity={'cost_only':True,'form':form,'seed':seed,'data_identity':tr['identity'],'train_raw':train_raw,'padding':b'0'*131072}
            saved=[]
            for i in range(4):
                folder_save=folder/f'save{i}';folder_save.mkdir();saved.append(folder_save)
                measured('save',lambda:checkpoint(folder_save,m,opt,rng,32,trace,identity))
            m,opt=measured('construct',lambda:make(seed,tr['observed'],form));actual,replay=measured('restore',lambda:restore(saved[0],m,opt,rng,identity))
            if actual!=32 or trace!=replay:raise ValueError('timing restore failed')
            trace_payload={'loss':trace,'gradient':grad,'parameter_summary':summary,'cost_only':{'actual_updates':32,'padding':''}}
            trace_bytes=len(json.dumps(trace_payload,indent=2,sort_keys=True,allow_nan=False).encode('utf-8'))
            if trace_bytes>262144:raise ValueError('actual cost trace exceeds envelope')
            trace_payload['cost_only']['padding']='0'*(262144-trace_bytes)
            atomic(folder/'cost_only_trace.json',trace_payload)
            endpoint=saved[0]/'checkpoint_00032';row={'path':str(endpoint),'files':{n:sha(endpoint/n) for n in ('model.json','training.pt')},'train_raw':train_raw,'form':form,'identity':identity,'cost_trace':trace,'trace':fileid(folder/'cost_only_trace.json')}
            def loaded():
                verify(row,32,True);model=model_class().from_export(endpoint/'model.json').cuda();d=load_data(train_raw['path']);verify(row,32,True);return model,d
            m,train_copy=measured('load',loaded);result,pred=measured('v_complete',lambda:complete(m,train_copy,va,deadline));measured('raw_write',lambda:raw(folder/'predictions.npz',**pred))
    overhead=len(json.dumps({'cost_only':True,'padding':''},indent=2,sort_keys=True).encode())
    for i in range(256):measured('json_write',lambda:atomic(out/f'cost_only_metadata_{i:03d}.json',{'cost_only':True,'padding':'0'*(262144-overhead)}))
    body=time.perf_counter()-begin;measured_sum=sum(map(sum,panels.values()))
    return {'status':'COMPLETED','cost_panels_complete':True,'panels':panels,'counts':{k:len(v) for k,v in panels.items()},'body_wall_seconds':body,
            'body_measured_seconds':measured_sum,'body_unassigned_seconds':max(0.,body-measured_sum),'pre_body_seconds':begin-START,'science_scores':None}
def execute(phase,out,binding,commit,deadline,group):
    import torch
    torch.set_num_threads(1)
    if phase=='qa':
        import subprocess
        command=[sys.executable,'-m','pytest','-q',str(ROOT/'tests/test_rational_control_20261021.py'),'--basetemp',str(out/'pytest_tmp')]
        p=subprocess.run(command,cwd=ROOT,capture_output=True,text=True,timeout=max(.1,deadline-time.perf_counter()))
        (out/'pytest_stdout.txt').write_text(p.stdout,encoding='utf-8');(out/'pytest_stderr.txt').write_text(p.stderr,encoding='utf-8')
        return {'status':'PASS' if p.returncode==0 else 'FAIL','pytest_returncode':p.returncode,'stdout':p.stdout,'stderr':p.stderr}
    if phase=='cost':return cost(out,binding,deadline)
    if phase=='run':return run(out,binding,commit,deadline,group)
    return evaluate(out,binding,commit,deadline)
def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('phase',choices=CAPS)
    for n in ('output','binding'):p.add_argument('--'+n,type=Path,required=True)
    p.add_argument('--source-commit',required=True);p.add_argument('--max-seconds',type=float,required=True);p.add_argument('--group',choices=GROUPS);a=p.parse_args(argv)
    if a.max_seconds!=CAPS[a.phase] or len(a.source_commit)!=40 or any(c not in '0123456789abcdef' for c in a.source_commit) or (a.phase=='run')!=(a.group is not None):p.error('registered cap/commit and run-only group required')
    out=a.output.resolve();category='P1' if a.phase in ('run','evaluate') else 'P0';allowed=(ROOT/'private_data/closure_mechanism_20261019'/category/'rational_control01').resolve()
    if out==allowed or not out.is_relative_to(allowed):raise ValueError('exclusive phase directory required')
    out.mkdir(parents=True,exist_ok=False);receipt(out,a.phase,a.source_commit,a.binding);deadline=START+a.max_seconds
    def timeout():atomic(out/'timeout.json',{'status':'TIMEOUT'});os._exit(124)
    timer=threading.Timer(max(.001,deadline-time.perf_counter()),timeout);timer.daemon=True;timer.start()
    try:
        binding=json.loads(a.binding.read_text(encoding='utf-8'));result=execute(a.phase,out,binding,a.source_commit,deadline,a.group)
        result.update(source_commit=a.source_commit,worker_phase_elapsed_seconds=time.perf_counter()-START);atomic(out/'result.json',result);atomic(out/'complete.json',{'status':'COMPLETED','result':result})
        return 0 if result['status'] in ('PASS','COMPLETED') else 2
    except BaseException as e:
        import traceback
        atomic(out/'failure.json',{'status':'UNRESOLVED','error':repr(e),'traceback':traceback.format_exc()});raise
    finally:timer.cancel()
if __name__=='__main__':raise SystemExit(main())
