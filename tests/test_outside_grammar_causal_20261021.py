"""Independent engineering inputs only; no formal teacher samples or project data."""
import copy,json,sys
from pathlib import Path
import numpy as np
import pytest
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'experiments'),str(ROOT/'src')]
import outside_grammar_causal_20261021 as study


def observations():
    rng=np.random.default_rng(93013);t=np.arange(61)[None,:,None]*.01
    obs=rng.normal(0,.02,(16,1,30))+rng.normal(0,.002,(16,1,30))*t
    return obs,np.maximum(obs.std((0,1)),1e-6)
def model():
    torch.set_num_threads(1);assert torch.cuda.is_available(),'registered CUDA QA cannot skip'
    obs,sc=observations();m,opt=study.make(93013,obs)
    with torch.no_grad():m.base.coefficients.fill_(.021)
    return m,opt,torch.as_tensor(obs,device='cuda'),torch.as_tensor(sc,device='cuda')
def exact(a,b):
    if torch.is_tensor(a):assert a.dtype==b.dtype and torch.equal(a.cpu(),b.cpu())
    elif isinstance(a,np.ndarray):np.testing.assert_array_equal(a,b)
    elif isinstance(a,dict):
        assert a.keys()==b.keys()
        for k in a:exact(a[k],b[k])
    elif isinstance(a,(list,tuple)):
        assert len(a)==len(b)
        for x,y in zip(a,b):exact(x,y)
    else:assert a==b
def capture(m,opt,rng):
    return copy.deepcopy({'model':m.state_dict(),'optimizer':opt.state_dict(),'numpy':rng.bit_generator.state,
                          'cpu':torch.get_rng_state(),'cuda':torch.cuda.get_rng_state_all()})


def test_teacher_rational_port_zero_negative_and_physical_direction():
    from closure_causal_recovery_20261019 import copy_runtime
    from closure_discovery_benchmarks import toy_source
    base=copy_runtime(toy_source(),[],'cpu')
    with torch.no_grad():base.mass.copy_(torch.linspace(.7,1.6,10,dtype=torch.float64));base.structured.mass.copy_(base.mass)
    teacher=study.RationalTeacher(base);z=torch.zeros((3,40),dtype=torch.float64)
    z[0,30:]=-2.;z[1,30:]=0.;z[2,30:]=.3
    expected=.4*z[:,30:]/(1+z[:,30:].abs())
    difference=teacher.field(z)-base.field(z)
    torch.testing.assert_close(difference[:,10:20],expected/base.mass,atol=1e-10,rtol=1e-9)
    torch.testing.assert_close(difference[:,:10],torch.zeros_like(difference[:,:10]),atol=1e-10,rtol=1e-9)
    torch.testing.assert_close(difference[:,20:],torch.zeros_like(difference[:,20:]),atol=1e-10,rtol=1e-9)
    torch.testing.assert_close(teacher.field(z,torch.ones((3,10)))-teacher.field(z),
        torch.cat((torch.zeros((3,10)),torch.ones((3,10))/base.mass,torch.zeros((3,20))),-1),atol=1e-10,rtol=1e-9)


def test_teacher_rk4_recomputes_port_at_every_substep():
    class Base:
        mass=torch.ones(10,dtype=torch.float64)
        def __init__(self):self.ports=[]
        def field(self,z,u):
            self.ports.append(u.clone());out=torch.zeros_like(z);out[...,10:20]=u;out[...,30:]=1.;return out
    base=Base();teacher=study.RationalTeacher(base);z=torch.zeros((1,40),dtype=torch.float64)
    result=teacher.rk4(z,.1)
    values=[.4*q/(1+abs(q)) for q in (0.,.05,.05,.1)]
    assert len(base.ports)==4
    for actual,value in zip(base.ports,values):torch.testing.assert_close(actual,torch.full_like(actual,value))
    torch.testing.assert_close(result[:,10:20],torch.full((1,10),.1*(values[0]+2*values[1]+2*values[2]+values[3])/6,dtype=torch.float64))


def test_learner_full_half_second_cpu_reference_grad_clip_adam_numpy_export(tmp_path):
    from closure_causal_history_residual_20261019 import loss,make_optimizer
    from symbolic_closure_rom import numpy_field
    m,opt,obs,sc=model();ref=copy.deepcopy(m).cpu();other=make_optimizer(ref)
    a=loss(m,obs,sc)*1e6;b=loss(ref,obs.cpu(),sc.cpu())*1e6
    torch.testing.assert_close(a.cpu(),b,atol=1e-10,rtol=1e-9);a.backward();b.backward()
    left=dict(m.named_parameters());right=dict(ref.named_parameters())
    for name,p in left.items():
        assert (p.grad is None)==(right[name].grad is None)
        if p.grad is not None:torch.testing.assert_close(p.grad.cpu(),right[name].grad,atol=1e-10,rtol=1e-9)
    n1=torch.nn.utils.clip_grad_norm_(m.parameters(),100.);n2=torch.nn.utils.clip_grad_norm_(ref.parameters(),100.)
    assert n1>100 and n2>100
    opt.step();other.step()
    for name,p in left.items():torch.testing.assert_close(p.cpu(),right[name],atol=1e-10,rtol=1e-9)
    m.export(tmp_path/'model.json');payload=json.loads((tmp_path/'model.json').read_text())['base']
    z=torch.cat((obs[:2,10],m.encode(obs[:2,:11])),-1)
    torch.testing.assert_close(m.field(z).cpu(),torch.as_tensor(numpy_field(payload,z.detach().cpu().numpy())),atol=1e-10,rtol=1e-9)


def test_once_history_and_no_future_observations():
    m,opt,obs,sc=model();calls=[];original=m.encode
    def counted(h):calls.append(tuple(h.shape));return original(h)
    m.encode=counted;study.predict(m,obs.cpu().numpy(),.01,None)
    assert calls==[(16,11,30)]
    with pytest.raises(ValueError):m.encode(obs[:,:12])
    changed=obs.clone();changed[:,11:]+=999
    torch.testing.assert_close(original(obs[:,:11]),original(changed[:,:11]),atol=0,rtol=0)


def test_complete16_teacher_history_qualification_and_no_badrow_drop():
    tc=np.zeros((16,61,30));tf=tc.copy();mc=tc[:,11:].copy();mf=mc.copy();sc=np.ones(30)
    assert study.metrics(mc,mf,tc,tf,sc)['valid']
    tf[7,0,1]=.002
    result=study.metrics(mc,mf,tc,tf,sc)
    assert not result['valid'] and result['blocks']['all']['mean'] is None
    assert result['blocks']['all']['count']==16 and len(result['blocks']['all']['per_event_mse'])==16
    with pytest.raises(ValueError):study.qualify(tc[:15],tf[:15],sc)


def test_exact_32_checkpoint_resume_34_all_states_and_output(tmp_path):
    from closure_causal_history_residual_20261019 import train_update,checkpoint,restore,prediction
    m,opt,obs,sc=model();rng=np.random.default_rng(93);trace=[];identity={'qa_only':True}
    for _ in range(32):v,_=train_update(m,opt,rng,obs,sc);trace.append(v)
    checkpoint(tmp_path,m,opt,rng,32,trace,identity)
    for _ in range(2):v,_=train_update(m,opt,rng,obs,sc);trace.append(v)
    final=capture(m,opt,rng)
    with torch.no_grad():expected=prediction(m,obs).clone()
    torch.rand(7);torch.rand(7,device='cuda');rng.random(7)
    step,replay=restore(tmp_path,m,opt,rng,identity);assert step==32 and len(replay)==32
    for _ in range(2):v,_=train_update(m,opt,rng,obs,sc);replay.append(v)
    exact(final,capture(m,opt,rng));assert trace==replay
    with torch.no_grad():torch.testing.assert_close(expected,prediction(m,obs),atol=0,rtol=0)


def test_val_generation_requires_all_three_endpoints(monkeypatch,tmp_path):
    def forbidden(*args,**kwargs):raise AssertionError('formal val requested before freeze')
    monkeypatch.setattr(study,'fixture',forbidden)
    with pytest.raises(ValueError):study.evaluate(tmp_path,{},'f'*40,None)
    with pytest.raises(ValueError):study.validate_selection({'all_three_frozen':False,'rows':[]},'f'*40)
    with pytest.raises(ValueError):study.validate_selection({'all_three_frozen':True,'source_commit':'f'*40,'rows':[{'seed':6201}]},'f'*40)


def test_512_path_does_not_establish_actual_512_updates(monkeypatch,tmp_path):
    def forbidden(*args,**kwargs):raise AssertionError('val accessed before actual checkpoint verification')
    monkeypatch.setattr(study,'fixture',forbidden)
    train=tmp_path/'train.npz';train.write_bytes(b'explicit negative fixture; not scientific data')
    train_raw={'path':str(train),'sha256':study.sha(train)};rows=[]
    for seed in study.SEEDS:
        directory=tmp_path/str(seed)/'checkpoint_00512';directory.mkdir(parents=True)
        (directory/'model.json').write_text('{}')
        identity={'source_commit':'f'*40,'seed':seed,'train_raw':train_raw,'learner_term':study.TERM}
        # Intentional wrong count in a path named512, never positive evidence.
        torch.save({'completed_updates':32,'trace':[0.]*32,'identity':identity,
                    'optimizer':{'state':{0:{'step':torch.tensor(32.)}},'param_groups':[{'params':[0]}]}},directory/'training.pt')
        rows.append({'seed':seed,'status':'FROZEN','path':str(directory),'files':{n:study.sha(directory/n) for n in ('model.json','training.pt')},
                     'train_raw':train_raw,'trace':[0.]*512})
    value={'all_three_frozen':True,'source_commit':'f'*40,'rows':rows}
    with pytest.raises(ValueError,match='count/trace/source'):
        study.validate_selection(value,'f'*40)


def test_fixed_serialization_envelopes_fit_timed_padding():
    limits=study.serialization_envelopes()
    assert limits['training_identity_and_trace_bytes']<=131072
    assert limits['formal_complete_envelope_bytes']<=262144
