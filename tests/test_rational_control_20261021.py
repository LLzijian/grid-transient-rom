"""Engineering fixtures only; no formal seeds or research scores."""
import copy,json,sys,types
from pathlib import Path
import numpy as np
import pytest
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'experiments'),str(ROOT/'src')]
import rational_control_20261021 as study
from closure_causal_history_residual_20261019 import loss,prediction,train_update,checkpoint,restore


def obs():
    rng=np.random.default_rng(9401);return rng.normal(0,.02,(16,13,30)),np.full(30,.02)
def make(form):
    torch.set_num_threads(1);assert torch.cuda.is_available(),'CUDA QA cannot skip'
    data,sc=obs();m,opt=study.make(9401,data,form)
    with torch.no_grad():m.port_coefficient.fill_(.021)
    return m,opt,torch.as_tensor(data,device='cuda'),torch.as_tensor(sc,device='cuda')
def close(a,b):torch.testing.assert_close(a,b,atol=1e-10,rtol=1e-9)
def exact(a,b):
    if torch.is_tensor(a):assert a.dtype==b.dtype and torch.equal(a.cpu(),b.cpu())
    elif isinstance(a,np.ndarray):np.testing.assert_array_equal(a,b)
    elif isinstance(a,dict):
        assert a.keys()==b.keys()
        for k in a:exact(a[k],b[k])
    elif isinstance(a,(tuple,list)):
        assert len(a)==len(b)
        for x,y in zip(a,b):exact(x,y)
    else:assert a==b
def capture(m,opt,rng):return copy.deepcopy({'model':m.state_dict(),'optimizer':opt.state_dict(),'numpy':rng.bit_generator.state,'cpu':torch.get_rng_state(),'cuda':torch.cuda.get_rng_state_all()})


@pytest.mark.parametrize('form',['rational','tanh'])
def test_ports_zero_negative_and_rk4_recomputes(form):
    m,opt,observed,sc=make(form);z=torch.zeros((3,40),device='cuda',dtype=torch.float64);z[:,30:]=torch.tensor([-2.,0.,.3],device='cuda')[:,None]
    q=z[:,30:];port=m.port_coefficient*(q/(1+q.abs()) if form=='rational' else torch.tanh(2*q))
    difference=m.field(z)-m.base.field(z)
    close(difference[:,10:20],port/m.mass);close(difference[:,:10],torch.zeros_like(difference[:,:10]));close(difference[:,20:],torch.zeros_like(difference[:,20:]))
    calls=[]
    def basefield(base,state,u=None):
        calls.append(u.clone());result=torch.zeros_like(state);result[...,10:20]=u;result[...,30:]=1.;return result
    m.base.field=types.MethodType(basefield,m.base);m.rk4(torch.zeros((1,40),device='cuda',dtype=torch.float64),.1)
    assert len(calls)==4 and calls[0].abs().sum()==0 and calls[-1].abs().sum()>0
    assert sum(id(p)==id(m.port_coefficient) for g in opt.param_groups for p in g['params'])==1
    assert next(g['lr'] for g in opt.param_groups if any(p is m.port_coefficient for p in g['params']))==.001


def test_common_initial_state_and_nonzero_tanh_full_field_raw_gradient_clip_adam():
    from closure_causal_recovery_20261019 import runtime_model
    from closure_discovery_benchmarks import toy_source
    from closure_causal_history_residual_20261019 import CausalHistoryResidual,make_optimizer
    data,_=obs();r,ro=study.make(9401,data,'rational');m,opt=study.make(9401,data,'tanh')
    exact(r.state_dict(),m.state_dict());exact(ro.state_dict(),opt.state_dict())
    old=CausalHistoryResidual.from_train(runtime_model(toy_source(),[{'channel':'port','expression':{'op':'tanh','arg':{'op':'var','name':'q'},'scale':2.}}],9401,'cuda'),torch.as_tensor(data[:,:11],device='cuda'));other=make_optimizer(old)
    with torch.no_grad():m.port_coefficient.fill_(.03);old.base.coefficients.fill_(.03)
    z=torch.zeros((3,40),device='cuda',dtype=torch.float64);z[:,30:]=torch.tensor([-1.,0.,.2],device='cuda')[:,None]
    close(m.field(z),old.field(z))
    observed=torch.as_tensor(data,device='cuda');scale=torch.full((30,),.02,device='cuda',dtype=torch.float64)
    a=loss(m,observed,scale)*1e6;b=loss(old,observed,scale)*1e6;close(a,b);a.backward();b.backward()
    left=dict(m.named_parameters());right=dict(old.named_parameters())
    for name,p in left.items():
        if name=='base.coefficients':assert p.numel()==0;continue
        q=right['base.coefficients' if name=='port_coefficient' else name]
        assert (p.grad is None)==(q.grad is None),name
        if p.grad is not None:close(p.grad,q.grad)
    na=torch.nn.utils.clip_grad_norm_(m.parameters(),100.);nb=torch.nn.utils.clip_grad_norm_(old.parameters(),100.);assert na>100 and nb>100;close(na,nb)
    opt.step();other.step()
    for name,p in left.items():
        if name!='base.coefficients':close(p,right['base.coefficients' if name=='port_coefficient' else name])


@pytest.mark.parametrize('form',['rational','tanh'])
def test_cpu_cuda_gradient_export_independent_numpy_and_exact_resume(form,tmp_path):
    from symbolic_closure_rom import numpy_field
    m,opt,observed,sc=make(form);cpu=copy.deepcopy(m).cpu();other=study.optimizer(cpu)
    a=loss(m,observed,sc);b=loss(cpu,observed.cpu(),sc.cpu());close(a.cpu(),b);a.backward();b.backward()
    for (n,p),(q,v) in zip(m.named_parameters(),cpu.named_parameters()):
        assert n==q and (p.grad is None)==(v.grad is None)
        if p.grad is not None:close(p.grad.cpu(),v.grad)
    opt.zero_grad(set_to_none=True);rng=np.random.default_rng(13);trace=[];identity={'qa_only':True,'form':form}
    for _ in range(32):v,_=train_update(m,opt,rng,observed,sc);trace.append(v)
    checkpoint(tmp_path,m,opt,rng,32,trace,identity)
    for _ in range(2):v,_=train_update(m,opt,rng,observed,sc);trace.append(v)
    expected=capture(m,opt,rng)
    with torch.no_grad():output=prediction(m,observed).clone()
    torch.rand(7);torch.rand(7,device='cuda');rng.random(7)
    step,replay=restore(tmp_path,m,opt,rng,identity);assert step==32
    for _ in range(2):v,_=train_update(m,opt,rng,observed,sc);replay.append(v)
    exact(expected,capture(m,opt,rng));assert trace==replay
    with torch.no_grad():torch.testing.assert_close(output,prediction(m,observed),atol=0,rtol=0)
    m.export(tmp_path/'model.json');loaded=study.model_class().from_export(tmp_path/'model.json').cuda();assert loaded.form==form
    exact(m.state_dict(),loaded.state_dict());z=torch.cat((observed[:2,10],m.encode(observed[:2,:11])),-1)
    payload=json.loads((tmp_path/'model.json').read_text());zn=z.detach().cpu().numpy();q=zn[:,30:]
    field=numpy_field(payload['base'],zn);feature=q/(1+np.abs(q)) if form=='rational' else np.tanh(2*q)
    field[:,10:20]+=payload['port_coefficient'][0]*feature/np.asarray(payload['base']['physical']['mass'])
    close(m.field(z).cpu(),torch.as_tensor(field))
    def independent_field(state):
        result=numpy_field(payload['base'],state);q=state[:,30:]
        feature=q/(1+np.abs(q)) if form=='rational' else np.tanh(2*q)
        result[:,10:20]+=payload['port_coefficient'][0]*feature/np.asarray(payload['base']['physical']['mass'])
        return result
    independent=zn.copy();actual=z.detach().clone();dt=.01
    with torch.no_grad():
        m.begin_graph_cache()
        try:
            for _ in range(50):
                a=independent_field(independent);b=independent_field(independent+dt*a/2)
                c=independent_field(independent+dt*b/2);d=independent_field(independent+dt*c)
                independent=independent+dt*(a+2*b+2*c+d)/6
                actual=m.rk4(actual,dt)
                close(actual.cpu(),torch.as_tensor(independent))
                close(actual[:,:30].cpu(),torch.as_tensor(independent[:,:30]))
        finally:m.end_graph_cache()


def test_optimizer_rejects_missing_step_and_empty_groups(tmp_path):
    m,opt,observed,sc=make('rational');rng=np.random.default_rng(71)
    _,gradients=train_update(m,opt,rng,observed,sc)
    m.export(tmp_path/'model.json');payload=json.loads((tmp_path/'model.json').read_text())
    state=opt.state_dict();study.validate_optimizer(state,payload,1,gradients)
    incomplete=copy.deepcopy(state);del next(iter(incomplete['state'].values()))['step']
    with pytest.raises(ValueError,match='incomplete'):study.validate_optimizer(incomplete,payload,1,gradients)
    empty=copy.deepcopy(state);empty['param_groups']=[]
    with pytest.raises(ValueError,match='groups'):study.validate_optimizer(empty,payload,1,gradients)


def test_noise_paired_scales_history_causality_and_clean_qualification():
    torch.set_num_threads(1);assert torch.cuda.is_available()
    c=study.data(9402,0,0,None,True);n=study.data(9402,0,1,None,True);again=study.data(9402,0,1,None,True)
    np.testing.assert_array_equal(c['coarse'],n['coarse']);np.testing.assert_array_equal(c['scales'],n['scales']);assert n['identity']==again['identity']
    assert not np.array_equal(c['observed'][:,:11],n['observed'][:,:11]);assert not np.array_equal(c['observed'][:,11:],n['observed'][:,11:])
    v=study.data(9402,1,1,None,True,c['scales']);np.testing.assert_array_equal(v['observed'][:,11:],v['coarse'][:,11:]);np.testing.assert_array_equal(c['scales'],v['scales'])
    m,_=study.make(9402,n['observed'],'rational');calls=[];original=m.encode
    def counted(h):calls.append(tuple(h.shape));return original(h)
    m.encode=counted;study.predict(m,n['observed'],.01,None);assert calls==[(16,11,30)]
    with pytest.raises(ValueError):m.encode(torch.as_tensor(n['observed'][:,:12],device='cuda'))
    clean=n['coarse'].copy();fine=n['fine'].copy();fine[0,0,0]=np.nan
    assert not study.qualify(clean,fine,n['scales'])['valid']


def test_twelve_freeze_barrier_negative512_and_serialization(tmp_path,monkeypatch):
    def forbidden(*args,**kwargs):raise AssertionError('val before12 endpoints')
    monkeypatch.setattr(study,'data',forbidden)
    with pytest.raises(ValueError):study.evaluate(tmp_path,{},'f'*40,None)
    with pytest.raises(ValueError):study.validate_selection({'all_twelve_frozen':True,'source_commit':'f'*40,'rows':[]},'f'*40)
    folder=tmp_path/'checkpoint_00512';folder.mkdir();(folder/'model.json').write_text('{}');train=tmp_path/'train.npz';train.write_bytes(b'negative engineering fixture')
    identity={'qa_only':True};torch.save({'completed_updates':32,'trace':[0.]*32,'identity':identity,'optimizer':{'state':{0:{'step':torch.tensor(32.)}}}},folder/'training.pt')
    trace=tmp_path/'trace.json';trace.write_text(json.dumps({'loss':[0.]*512}))
    row={'path':str(folder),'status':'FROZEN','files':{n:study.sha(folder/n) for n in ('model.json','training.pt')},'train_raw':study.fileid(train),'identity':identity,'trace':study.fileid(trace),'form':'rational'}
    with pytest.raises(ValueError,match='step/trace/identity'):study.verify(row)
    envelope=study.serialization_envelopes();assert envelope['trace_bytes']<=262144 and envelope['checkpoint_identity_bytes']<=131072
