"""Independent recursive/NumPy and exact new-backend recovery QA."""
import copy,json,sys
from pathlib import Path
import numpy as np
import pytest
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'experiments'),str(ROOT/'src')]
import edge_polynomial_runtime_20261021 as new
from closure_causal_history_residual_20261019 import make_optimizer,train_update,loss,prediction,checkpoint,restore
from symbolic_closure_rom import evaluate_expression_torch,numpy_field,validate_expression


def terms():
    from finite_grammar_discovery_20261021 import generate_pool
    return generate_pool()['selected']
def data():
    rng=np.random.default_rng(8212);t=np.arange(61)[None,:,None]*.01
    obs=rng.normal(0,.01,(16,1,30))+rng.normal(0,.001,(16,1,30))*t
    return obs,np.maximum(obs.std((0,1)),1e-6)
def models(term):
    torch.set_num_threads(1);assert torch.cuda.is_available(),'registered CUDA QA cannot skip'
    obs,sc=data();ref,_=new.make_reference(term,8212,obs)
    with torch.no_grad():
        ref.base.coefficients.fill_(.017)
        ref.history_coefficients.copy_(torch.linspace(-.001,.001,6,device='cuda'))
    current=new.adapt_wrapper(ref)
    return ref,current,torch.as_tensor(obs,device='cuda'),torch.as_tensor(sc,device='cuda')
def close(a,b,trajectory=False):
    torch.testing.assert_close(a,b,atol=1e-9 if trajectory else 1e-10,rtol=1e-8 if trajectory else 1e-9)
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


@pytest.mark.parametrize('index',range(6))
def test_six_feature_values_gradients_field_and_complete_dual_5s(index):
    from closure_causal_recovery_20261019 import trajectory
    term=terms()[index];ref,m,obs,sc=models(term);assert m.base.polynomial_enabled
    features=torch.randn((3,10,6),device='cuda',dtype=torch.float64)*.04
    features[0]=0.;features[1]=-features[1].abs();features.requires_grad_(True)
    value,gradient=m.base.feature_value_gradient(features)
    v,g=evaluate_expression_torch(term['expression'],features,'edge')
    close(value[...,0],v);close(gradient[...,0,:],g)
    close(torch.autograd.grad(value.sum(),features)[0],g)
    z=torch.cat((obs[:,10],m.encode(obs[:,:11])),-1)
    close(m.field(z),ref.field(z));close(m.field(torch.zeros_like(z)),ref.field(torch.zeros_like(z)))
    u=torch.full((16,10),.012,device='cuda',dtype=torch.float64);close(m.field(z,u),ref.field(z,u))
    za=z.detach().clone().requires_grad_(True);zb=z.detach().clone().requires_grad_(True)
    close(torch.autograd.grad(m.field(za).square().sum(),za)[0],
          torch.autograd.grad(ref.field(zb).square().sum(),zb)[0])
    left=[(n,p) for n,p in m.named_parameters() if p.requires_grad]
    right=[(n,p) for n,p in ref.named_parameters() if p.requires_grad]
    assert [n for n,_ in left]==[n for n,_ in right]
    ga=torch.autograd.grad(m.field(z.detach()).square().sum(),[p for _,p in left],allow_unused=True)
    gb=torch.autograd.grad(ref.field(z.detach()).square().sum(),[p for _,p in right],allow_unused=True)
    for (name,_),a,b in zip(left,ga,gb):
        assert (a is None)==(b is None),name
        if a is not None:close(a,b)
    for dt in (.01,.005):
        with torch.no_grad():
            a=prediction(m,torch.cat((obs[:,:11],obs.new_zeros((16,500,30))),1),dt)
            b=prediction(ref,torch.cat((obs[:,:11],obs.new_zeros((16,500,30))),1),dt)
        assert a.shape==b.shape==(16,500,30);close(a,b,True)


def test_half_second_raw_gradients_triggered_clip_and_adam():
    ref,m,obs,sc=models(terms()[0]);opt=make_optimizer(m);other=make_optimizer(ref)
    # A fixed engineering stress multiplier exercises the actual clip branch;
    # both full autonomous objectives and all parameter gradients are compared.
    a=loss(m,obs,sc)*1e6;b=loss(ref,obs,sc)*1e6;close(a,b)
    a.backward();b.backward();left=dict(m.named_parameters());right=dict(ref.named_parameters())
    for name,p in left.items():
        assert (p.grad is None)==(right[name].grad is None)
        if p.grad is not None:close(p.grad,right[name].grad)
    n1=torch.nn.utils.clip_grad_norm_(m.parameters(),100.);n2=torch.nn.utils.clip_grad_norm_(ref.parameters(),100.)
    assert n1>100 and n2>100;close(n1,n2)
    opt.step();other.step()
    for name,p in left.items():close(p,right[name])
    for x,y in zip(opt.state.values(),other.state.values()):
        for key in x:close(x[key],y[key])


def test_nonunit_mass_and_power_scale_preserve_physical_field():
    term={'channel':'edge','expression':{'op':'mul','args':[{'op':'var','name':'pi'},
          {'op':'var','name':'qj'},{'op':'var','name':'s'}]}}
    ref,m,obs,sc=models(term)
    with torch.no_grad():
        for model in (ref,m):
            model.mass.copy_(torch.linspace(.7,2.3,10,device='cuda',dtype=torch.float64))
            model.base.p_scale.copy_(torch.linspace(.3,1.7,10,device='cuda',dtype=torch.float64))
            model.base._clear_runtime()
    z=torch.linspace(-.07,.11,40,device='cuda',dtype=torch.float64).repeat(3,1)
    close(m.field(z),ref.field(z));close(m.rk4(z,.01),ref.rk4(z,.01))


def edge(expr):return {'channel':'edge','expression':validate_expression(expr,'edge')}
def v(name):return {'op':'var','name':name}


@pytest.mark.parametrize('expr',[
    {'op':'add','args':[{'op':'const','value':1},v('c')]},
    {'op':'pow4','arg':v('qi')},
    {'op':'mul','args':[v('pi'),v('qj'),v('s')]},
])
def test_centering_physical_c_degree_four_and_edge_reversal(expr):
    ref,m,obs,sc=models(edge(expr));assert m.base.polynomial_enabled
    z=torch.cat((obs[:3,10],m.encode(obs[:3,:11])),-1)
    close(m.field(z),ref.field(z))
    m0=new.adapt_wrapper(ref)
    with torch.no_grad():
        seen=set()
        for name,b in m0.named_buffers():
            if name.endswith('phi') or name.endswith('base_D') or name.endswith('.D'):
                if b.data_ptr() not in seen:seen.add(b.data_ptr());b.neg_()
    close(m.field(z),m0.field(z))
    with torch.no_grad():
        m.base.phi.zero_();ref.base.phi.zero_()
    close(m.field(z),ref.field(z))


def test_fallback_mixed_channels_and_unsupported():
    # A port tanh is legal in the original grammar and forces whole fallback.
    term={'channel':'port','expression':{'op':'tanh','arg':v('q'),'scale':1.}}
    ref,m,obs,sc=models(term);assert not m.base.polynomial_enabled
    z=torch.cat((obs[:3,10],m.encode(obs[:3,:11])),-1);close(m.field(z),ref.field(z))
    changed=m.base.with_terms([terms()[0],term]);assert not changed.polynomial_enabled


def test_pure_constant_edge_is_centered_zero_without_changing_model_legality():
    # Production model validation rejects zero mechanisms. Exercise the exact
    # compiler boundary directly rather than bypass that validation in a model.
    term=edge({'op':'const','value':1})
    value,derivatives,slope=new.compile_polynomial(term)
    assert value=={} and derivatives==[{}]*6 and slope==0
    features=torch.zeros((2,10,6),dtype=torch.float64)
    v,g=evaluate_expression_torch(term['expression'],features,'edge')
    assert not bool(v.any()) and not bool(g.any())


def test_conversion_rng_flags_export_numpy_cpu_cuda_and_legacy_checkpoint(tmp_path):
    ref,m,obs,sc=models(terms()[0]);cpu=torch.get_rng_state();cuda=torch.cuda.get_rng_state_all()
    clone=new.adapt_wrapper(ref);exact(cpu,torch.get_rng_state());exact(cuda,torch.cuda.get_rng_state_all())
    assert [(n,p.requires_grad) for n,p in clone.named_parameters()]==[(n,p.requires_grad) for n,p in ref.named_parameters()]
    assert not {id(p) for p in clone.parameters()} & {id(p) for p in ref.parameters()}
    z=torch.cat((obs[:2,10],m.encode(obs[:2,:11])),-1)
    before=m.field(z);m.cpu().cuda();close(m.field(z),before)
    m.export(tmp_path/'model.json');payload=json.loads((tmp_path/'model.json').read_text())['base']
    expected=torch.as_tensor(numpy_field(payload,z.detach().cpu().numpy()),device='cuda');close(m.field(z),expected)
    from closure_causal_history_residual_20261019 import CausalHistoryResidual
    loaded=new.adapt_wrapper(CausalHistoryResidual.from_export(tmp_path/'model.json').cuda());close(loaded.field(z),m.field(z))
    opt=make_optimizer(ref);rng=np.random.default_rng(13);identity={'legacy_qa':True}
    legacy=tmp_path/'legacy';legacy.mkdir();checkpoint(legacy,ref,opt,rng,0,[],identity)
    newopt=make_optimizer(m);step,trace=restore(tmp_path/'legacy',m,newopt,rng,identity)
    assert step==0 and trace==[] and isinstance(m.base,new.EdgePolynomialROM)
    close(m.field(z),ref.field(z))


def test_new_backend_exact_32_resume_34(tmp_path):
    ref,m,obs,sc=models(terms()[0]);opt=make_optimizer(m);rng=np.random.default_rng(8);trace=[];identity={'new_backend_qa':True}
    for _ in range(32):value,_=train_update(m,opt,rng,obs,sc);trace.append(value)
    checkpoint(tmp_path,m,opt,rng,32,trace,identity)
    for _ in range(2):value,_=train_update(m,opt,rng,obs,sc);trace.append(value)
    expected=capture(m,opt,rng)
    with torch.no_grad():output=prediction(m,obs).clone()
    torch.rand(7);torch.rand(7,device='cuda');rng.random(7)
    step,replay=restore(tmp_path,m,opt,rng,identity);assert step==32
    for _ in range(2):value,_=train_update(m,opt,rng,obs,sc);replay.append(value)
    exact(expected,capture(m,opt,rng));assert trace==replay
    with torch.no_grad():actual=prediction(m,obs)
    torch.testing.assert_close(actual,output,rtol=0,atol=0)


def test_shared_basis_limit_and_cache_lifetime():
    ref,m,obs,sc=models(terms()[0]);z=torch.cat((obs[:2,10],m.encode(obs[:2,:11])),-1)
    assert len(m.base._poly_basis)<=256
    original=m.field(z);m.begin_graph_cache()
    try:
        close(m.field(z),original)
        with pytest.raises(RuntimeError):m.begin_graph_cache()
    finally:m.end_graph_cache()
    assert m.base._poly_graph_weights is None
    changed=m.base.with_terms([terms()[1]]);assert changed.polynomial_enabled
    m.base.load_state_dict(m.base.state_dict());assert m.base._poly_graph_weights is None
    # A new optimizer graph must use current parameter values after an update.
    opt=make_optimizer(m);opt.zero_grad(set_to_none=True);m.begin_graph_cache()
    try:m.field(z.detach()).square().sum().backward()
    finally:m.end_graph_cache()
    opt.step();assert m.base._poly_graph_directions is None
    fresh=new.adapt_wrapper(m)
    m.begin_graph_cache()
    try:close(m.field(z.detach()),fresh.field(z.detach()))
    finally:m.end_graph_cache()
