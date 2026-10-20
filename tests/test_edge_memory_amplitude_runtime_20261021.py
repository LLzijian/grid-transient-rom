"""E8 synthetic QA; only root may execute the sole registered native QA."""
import copy
import hashlib
import json
import os
from pathlib import Path
import sys
import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'experiments'), str(ROOT/'src'), str(ROOT/'tests')]
import edge_memory_amplitude_runtime_20261021 as prt
fused=prt
drt=prt
from closure_training_runtime import TrainingClosureROM, TensorEventStore, objective
from train_local_memory_0098 import graph
from train_closure import make_optimizer
from symbolic_closure_rom import numpy_field, term_key

DEVICES = ['cpu'] + (['cuda'] if torch.cuda.is_available() else [])
FIELD = dict(rtol=2e-11, atol=1e-11)
LOSS = dict(rtol=2e-10, atol=2e-10)
GRAD = dict(rtol=2e-9, atol=2e-9)


def prepared(device):
    """Root QA must set the explicit frozen fixture directory; no silent fallback."""
    folder = os.environ.get('FUSED_CLOSURE_FIXTURE')
    if not folder:
        raise RuntimeError('FUSED_CLOSURE_FIXTURE must identify frozen prepare01 assets')
    from fused_closure_audit_20261021 import load_fixture
    models, events, rng, manifest = load_fixture(device)
    assert len(events)==12 and all(e['x'].shape==(531,30) for e in events)
    restore_rng(rng, device)
    return {k:m.to(device) for k,m in models.items()}, events, rng


def restore_rng(state, device):
    torch.set_rng_state(state['cpu'])
    np.random.set_state(state['numpy'])
    if device == 'cuda': torch.cuda.set_rng_state(state['cuda'])


def pair(device, kind='E'):
    models, events, rng = prepared(device)
    fast = prt.build_E(models['N'])
    with torch.no_grad():fast.coefficients[prt.a_E_index(fast)]=-.03
    payload=copy.deepcopy(fast).cpu()._export_payload()
    ref=TrainingClosureROM.from_config(payload['config'],terms=payload['terms'])
    ref.load_state_dict(fast.state_dict())
    flags={n:p.requires_grad for n,p in fast.named_parameters()}
    for n,p in ref.named_parameters():p.requires_grad_(flags[n])
    ref=ref.to(device)
    ids=tuple(id(p) for p in fast.parameters())
    names=[(n,p.shape,p.requires_grad) for n,p in fast.named_parameters()]
    prt.enable_runtime(fast)
    assert tuple(id(p) for p in fast.parameters())==ids
    assert names==[(n,p.shape,p.requires_grad) for n,p in fast.named_parameters()]
    assert len(fast.terms)==8 and len(prt.coefficient_indices(fast))==1
    assert not any(t['channel']=='node' and t['expression'].get('op')=='pow4' for t in fast.terms)
    return ref,fast,events,rng


def gradients_equal(a,b):
    assert list(dict(a.named_parameters())) == list(dict(b.named_parameters()))
    for p,q in zip(a.parameters(),b.parameters()):
        assert (p.grad is None) == (q.grad is None)
        if p.grad is not None: torch.testing.assert_close(p.grad,q.grad,**GRAD)


def optimizer_equal(a,b):
    for pa,pb in zip(a.param_groups,b.param_groups):
        assert {k:v for k,v in pa.items() if k!='params'} == {k:v for k,v in pb.items() if k!='params'}
    assert len(a.state) == len(b.state)
    for x,y in zip(a.state.values(),b.state.values()):
        assert x.keys() == y.keys()
        for k in x: torch.testing.assert_close(x[k],y[k],**GRAD)


@pytest.mark.parametrize('device', DEVICES)
@pytest.mark.parametrize('kind', ['E'])
def test_fields_state_parameter_gradients_external_units(device,kind):
    ref, fast, events, _ = pair(device,kind)
    with torch.no_grad():
        for model in (ref,fast):
            mass=torch.linspace(.7,1.6,10,device=device,dtype=torch.float64)
            model.mass.copy_(mass);model.structured.mass.copy_(mass)
    z = torch.tensor(np.concatenate((events[0]['x'][11:15], np.full((4,10),.13)),axis=1),device=device)
    assert not torch.all(ref.mass == 1)
    for u in (None, torch.linspace(-.1,.2,10,device=device,dtype=torch.float64)):
        ref.zero_grad(set_to_none=True); fast.zero_grad(set_to_none=True)
        za=z.detach().clone().requires_grad_(); zb=z.detach().clone().requires_grad_()
        a,b=ref.field(za,u),fast.field(zb,u)
        torch.testing.assert_close(a,b,**FIELD)
        probe=torch.arange(40,dtype=z.dtype,device=device)/40
        (a*probe).sum().backward(); (b*probe).sum().backward()
        torch.testing.assert_close(za.grad,zb.grad,**GRAD); gradients_equal(ref,fast)
        cpu=copy.deepcopy(ref).cpu()
        np.testing.assert_allclose(b.detach().cpu(),numpy_field(cpu,z.cpu(),None if u is None else u.cpu()),**FIELD)
    shifted=z.clone();shifted[:,:10]+=.37
    torch.testing.assert_close(fast.field(z),fast.field(shifted),**FIELD)
    u0=torch.zeros(10,dtype=z.dtype,device=device);u1=u0+.04
    delta=fast.field(z,u1)-fast.field(z,u0)
    expected=torch.zeros_like(z);expected[:,10:20]=.04/fast.mass
    torch.testing.assert_close(delta,expected,**FIELD)


def ragged(events):
    # Genuine available counts are 1,7,23,500; no padding or invented samples.
    return [dict(e,x=e['x'][:n].copy(),torque=e['torque'][:n].copy())
            for e,n in zip(events,(12,18,34,531))]


@pytest.mark.parametrize('device', DEVICES)
@pytest.mark.parametrize('kind', ['E'])
@pytest.mark.parametrize('horizon', [.03,5.])
def test_full_original_objective_components_gradients_and_adam(device,kind,horizon):
    ref,fast,events,rng=pair(device,kind)
    events=ragged(events);store=TensorEventStore(events,device);samples=[(e,10) for e in events]
    assert store.prepare(samples,5).counts.tolist()==[1.,7.,23.,500.]
    opts=[make_optimizer(m) for m in (ref,fast)]
    outputs=[]
    for model in (ref,fast):
        restore_rng(rng,device)
        with graph(model):
            value,terms=objective(model,store,samples,horizon,.01)
            value.backward()
        outputs.append((value,terms))
    torch.testing.assert_close(outputs[0][0],outputs[1][0],**LOSS)
    assert outputs[0][1].keys()==outputs[1][1].keys()
    for k in outputs[0][1]: torch.testing.assert_close(outputs[0][1][k],outputs[1][1][k],**LOSS)
    gradients_equal(ref,fast)
    norms=[]
    for model,opt in zip((ref,fast),opts):
        norms.append(torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],100.))
        opt.step()
    torch.testing.assert_close(*norms,**GRAD)
    for p,q in zip(ref.parameters(),fast.parameters()): torch.testing.assert_close(p,q,**GRAD)
    optimizer_equal(*opts)
    assert fast._e_cache is None and fast._runtime_cache is None
    # Consecutive complete updates exercise rebuilt mappings after Adam changes.
    for model,opt in zip((ref,fast),opts):
        if model is fast: fused.one_update(model,opt,store,samples,.03,.01)
        else:
            opt.zero_grad(set_to_none=True)
            with graph(model):
                v,_=objective(model,store,samples,.03,.01);v.backward()
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],100.)
            opt.step()
    for p,q in zip(ref.parameters(),fast.parameters()): torch.testing.assert_close(p,q,**GRAD)
    optimizer_equal(*opts)


@pytest.mark.parametrize('device', DEVICES)
@pytest.mark.parametrize('kind', ['E'])
def test_nonempty_adam_rng_restore_next_update_and_export(tmp_path,device,kind):
    ref,fast,events,rng=pair(device,kind)
    events=ragged(events);store=TensorEventStore(events,device);samples=[(e,10) for e in events]
    opt=make_optimizer(fast)
    fused.one_update(fast,opt,store,samples,.03,.01)
    assert opt.state
    fast.export(tmp_path/'model.json')
    saved=json.loads((tmp_path/'model.json').read_text());assert saved['coefficients']==saved['state_dict']['coefficients']
    # A real temporary checkpoint holds nonempty Adam and all RNG streams.
    restore_rng(rng,device)
    snapshot={'cpu':torch.get_rng_state().clone(),'numpy':copy.deepcopy(np.random.get_state()),
              'cuda':torch.cuda.get_rng_state().clone() if device=='cuda' else rng['cuda']}
    torch.save({'optimizer':opt.state_dict(),'rng':snapshot},tmp_path/'checkpoint.pt')
    checkpoint=torch.load(tmp_path/'checkpoint.pt',map_location='cpu',weights_only=False)
    restored=prt.ERuntime.from_export(tmp_path/'model.json').to(device)
    other=make_optimizer(restored);other.load_state_dict(checkpoint['optimizer'])
    for p,q in zip(fast.parameters(),restored.parameters()): assert p.requires_grad==q.requires_grad
    sequences=[]
    for model,optimizer in ((fast,opt),(restored,other)):
        restore_rng(checkpoint['rng'],device)
        probes=(torch.rand(4,device=device),np.random.random(4))
        logs=fused.one_update(model,optimizer,store,samples,.03,.01)
        sequences.append((probes,logs))
    assert torch.equal(sequences[0][0][0],sequences[1][0][0])
    assert np.array_equal(sequences[0][0][1],sequences[1][0][1])
    for p,q in zip(fast.parameters(),restored.parameters()): torch.testing.assert_close(p,q,**GRAD)
    optimizer_equal(opt,other)
    # The old class still restores the fused export without any new schema.
    old=type(ref).from_export(tmp_path/'model.json').to(device)
    z=torch.zeros(2,40,dtype=torch.float64,device=device)
    torch.testing.assert_close(old.field(z),prt.ERuntime.from_export(tmp_path/'model.json').to(device).field(z),**FIELD)
    saved['coefficients'][0]+=.01;(tmp_path/'bad.json').write_text(json.dumps(saved))
    with pytest.raises(ValueError): prt.ERuntime.from_export(tmp_path/'bad.json')


@pytest.mark.parametrize('kind',['E'])
def test_cache_freshness_graph_lifecycle_failure_flags_and_order(kind):
    ref,fast,events,_=pair('cpu',kind)
    z=torch.tensor(np.concatenate((events[0]['x'][11:13],np.full((2,10),.12)),axis=1))
    before=fast.field(z).detach()
    with torch.no_grad():fast.coefficients[0].add_(.1);ref.coefficients[0].add_(.1)
    assert not torch.equal(before,fast.field(z))
    torch.testing.assert_close(fast.field(z),ref.field(z),**FIELD)
    fast.begin_graph_cache();cache=fast._e_cache
    assert cache['linear'].grad_fn is not None
    fast.end_graph_cache();assert fast._e_cache is None
    fast.begin_graph_cache()
    with pytest.raises(ValueError):fast.field(torch.zeros(3,39))
    assert fast._e_cache is None and fast._runtime_cache is None
    fast.begin_graph_cache();fast.load_state_dict(fast.state_dict());assert fast._e_cache is None
    fast.begin_graph_cache();fast.to('cpu');assert fast._e_cache is None
    reordered=fast.with_terms(list(reversed(fast.terms)))
    torch.testing.assert_close(reordered.field(z),fast.field(z),**FIELD)
    invalid=copy.deepcopy(fast);invalid.terms[0]=invalid.terms[1]
    with pytest.raises(ValueError):prt.enable_runtime(invalid)


@pytest.mark.parametrize('device',DEVICES)
def test_causal_rollout_one_initialization_and_original_auxiliary_positions(device,monkeypatch):
    _,model,events,_=pair(device,'E');events=ragged(events)
    store=TensorEventStore(events,device);samples=[(e,10) for e in events]
    calls=[];encode=model.encode
    def traced(h): calls.append(h.detach().clone());return encode(h)
    monkeypatch.setattr(model,'encode',traced)
    with graph(model): model.rollout(store.histories(samples),3,.01)
    assert len(calls)==1
    calls.clear()
    with graph(model): objective(model,store,samples,.03,.01)
    assert len(calls)==3
    assert torch.equal(calls[0],store.histories(samples))
    assert torch.equal(calls[1],store.histories(samples))
    assert torch.equal(calls[2],store.histories(samples,adjacent=True))


@pytest.mark.parametrize('kind',['E'])
@pytest.mark.parametrize('fault',['parameter','gradient','moment'])
def test_nonfinite_update_rejected_and_cache_cleared(kind,fault):
    _,model,events,_=pair('cpu',kind);opt=make_optimizer(model)
    events=ragged(events);store=TensorEventStore(events,'cpu');samples=[(e,10) for e in events]
    hook=None
    if fault=='parameter':
        with torch.no_grad():model.coefficients[0]=float('nan')
    elif fault=='gradient':hook=model.coefficients.register_hook(lambda g:g*float('nan'))
    else:
        model.coefficients.sum().backward();opt.step();opt.zero_grad(set_to_none=True)
        opt.state[model.coefficients]['exp_avg'][0]=float('nan')
    try:
        with pytest.raises(FloatingPointError):fused.one_update(model,opt,store,samples,.03,.01)
    finally:
        if hook is not None:hook.remove()
    assert model._runtime_cache is None and model._e_cache is None


def test_signed_edge_no_projection_and_exact_ast_export_guards(tmp_path):
    _,model,_,_=pair('cpu')
    with torch.no_grad():model.coefficients[prt.a_E_index(model)]=-.4
    prt.require_finite(model)
    model.export(tmp_path/'signed.json')
    loaded=prt.ERuntime.from_export(tmp_path/'signed.json')
    assert float(loaded.coefficients[prt.a_E_index(loaded)])==-.4
    assert list(prt.coefficient_indices(loaded))==['a_E']
    with pytest.raises(ValueError):model.with_terms(prt.TERMS[:-1])
    with pytest.raises(ValueError):prt.build_E(model)
    model.begin_graph_cache()
    with pytest.raises(RuntimeError):model.export(tmp_path/'active.json')
    assert model._e_cache is not None
    model.end_graph_cache()


@pytest.mark.parametrize('device',DEVICES)
def test_zero_edge_embeds_n_public_raw_gradients_and_new_coefficient_connected(device):
    models,events,rng=prepared(device)
    n=models['N'];e=prt.build_E(n)
    assert float(e.coefficients[prt.a_E_index(e)])==0.
    events=ragged(events);store=TensorEventStore(events,device);samples=[(v,10) for v in events]
    h=store.histories(samples)
    torch.testing.assert_close(n.encode(h),e.encode(h),rtol=0,atol=0)
    z=torch.randn(3,40,dtype=torch.float64,device=device)*.22
    torch.testing.assert_close(n.field(z),e.field(z),**FIELD)
    for model in (n,e):
        restore_rng(rng,device)
        with graph(model):
            loss,_=objective(model,store,samples,.03,.01);loss.backward()
    for name,left in n.named_parameters():
        right=dict(e.named_parameters())[name]
        if name=='coefficients':
            for i,t in enumerate(n.terms):
                j=[term_key(t) for t in e.terms].index(term_key(t))
                torch.testing.assert_close(left.grad[i],right.grad[j],**GRAD)
        else:
            assert (left.grad is None)==(right.grad is None)
            if left.grad is not None:torch.testing.assert_close(left.grad,right.grad,**GRAD)
    e.zero_grad(set_to_none=True)
    grad=torch.autograd.grad(e.field(z).square().sum(),e.coefficients)[0]
    assert abs(float(grad[prt.a_E_index(e)]))>1e-14


@pytest.mark.parametrize('device',DEVICES)
def test_edge_analytic_energy_gradients_fields_origin_and_endpoint_symmetry(device):
    models,events,_=prepared(device);n=models['N'];e=prt.build_E(n)
    with torch.no_grad():e.coefficients[prt.a_E_index(e)]=-.13
    z=(torch.randn(3,40,dtype=torch.float64,device=device)*.24).requires_grad_()
    x=e.interface_to_state(z)
    hn,gn=n.energy_and_gradient(x);he,ge=e.energy_and_gradient(x)
    parts=prt.extra_energy_components(e,z)
    torch.testing.assert_close(he-hn,parts['energy'],**FIELD)
    delta=torch.cat((parts['g_delta'],torch.zeros_like(z[:,10:30]),parts['g_q']),-1)
    torch.testing.assert_close(ge-gn,delta,**FIELD)
    torch.testing.assert_close(torch.autograd.grad(parts['energy'].sum(),z,retain_graph=True)[0],delta,**FIELD)
    J,R=e.matrices();B=e.port_matrix();u=e.port_values(x)
    actual=(ge*e.interface_to_state(e.field(z))).sum(-1)
    wanted=-(ge*(ge@R.T)).sum(-1)+((ge@B)*u).sum(-1)
    torch.testing.assert_close(actual,wanted,**LOSS)
    torch.testing.assert_close(e.field(z)-n.field(z),parts['field_delta']+parts['field_q'],**FIELD)
    external=torch.linspace(-.2,.3,10,dtype=z.dtype,device=device)
    torch.testing.assert_close(e.field(z,external)-n.field(z,external),parts['field_delta']+parts['field_q'],**FIELD)
    # Independent NumPy analytic gradients, using no Torch gradient implementation.
    zz=z.detach().cpu().numpy();D=e.base_D.detach().cpu().numpy();S=np.abs(D)
    phi=e.phi.detach().cpu().numpy();k=e.base_coupling.detach().cpu().numpy();k=k/max(k.mean(),1e-15)
    r=zz[:,30:]@S.T;angle=phi+zz[:,:10]@D.T;c=np.cos(angle)-np.cos(phi)
    np.testing.assert_allclose(parts['g_delta'].detach().cpu(),.13*(k*r*r*np.sin(angle))@D,**FIELD)
    np.testing.assert_allclose(parts['g_q'].detach().cpu(),-.26*(k*r*c)@S,**FIELD)
    # Central differences check all state directions on the added energy alone.
    eps=1e-6
    for i in range(40):
        plus=z.detach().clone();minus=z.detach().clone();plus[:,i]+=eps;minus[:,i]-=eps
        fd=(prt.extra_energy_components(e,plus)['energy']-prt.extra_energy_components(e,minus)['energy'])/(2*eps)
        torch.testing.assert_close(fd,delta[:,i],rtol=2e-8,atol=2e-9)
    zero=torch.zeros(40,dtype=z.dtype,device=device,requires_grad=True)
    def energy(v):return prt.extra_energy_components(e,v)['energy']
    torch.testing.assert_close(torch.autograd.functional.jacobian(energy,zero),torch.zeros_like(zero),**FIELD)
    torch.testing.assert_close(torch.autograd.functional.hessian(energy,zero),zero.new_zeros(40,40),**FIELD)
    from symbolic_closure_rom import evaluate_expression_torch
    features=torch.randn(4,6,dtype=z.dtype,device=device)
    rev=features[:,[1,0,3,2,4,5]].clone();rev[:,4]*=-1
    v,g=evaluate_expression_torch(prt.EDGE_TERM['expression'],features,'edge')
    vr,gr=evaluate_expression_torch(prt.EDGE_TERM['expression'],rev,'edge')
    torch.testing.assert_close(v,vr,**FIELD)
    restored=gr[:,[1,0,3,2,4,5]].clone();restored[:,4]*=-1
    torch.testing.assert_close(g,restored,**FIELD)


@pytest.mark.parametrize('device',DEVICES)
def test_edge_orientation_full_model_pairing(device):
    ref,fast,events,_=pair(device)
    z=torch.randn(4,40,dtype=torch.float64,device=device)*.2
    for original in (ref,fast):
        reverse=copy.deepcopy(original)
        snapshot={name:value.detach().clone() for name,value in original.state_dict().items()}
        changed={name:value.clone() for name,value in snapshot.items()}
        flipped=('base_D','structured.D','phi','structured.phi')
        for name in flipped:changed[name]=-snapshot[name].clone()
        reverse.load_state_dict(changed)
        for name in flipped:
            assert torch.equal(reverse.state_dict()[name],-snapshot[name])
        assert not torch.equal(reverse.base_D,original.base_D)
        reverse.end_graph_cache()
        for u in (None,torch.linspace(-.2,.1,10,device=device,dtype=z.dtype)):
            torch.testing.assert_close(reverse.field(z,u),original.field(z,u),**FIELD)




def relabel_field_model(model,perm):
    """Exact fixed-toy field-state relabel, not a replacement matrix implementation."""
    state={k:v.detach().clone() for k,v in model.state_dict().items()}
    p2=torch.cat((perm,perm+10))
    nodes={'mass','p_scale','input_scale_delta','input_scale_omega','delta_star','rate','gain',
           'structured.mass','structured.delta_star','structured.gain'}
    nodes.update('structured.'+name+suffix for name in ('damping','jpi_p','rpp') for suffix in ('','_mask'))
    blocks={'input_scale','structured.scales','structured.storage_base','structured.storage_base_mask'}
    columns={'base_D','structured.D','port_c_weight','port_s_weight'}
    squares={'W','neighbor_A'}
    touched=nodes|blocks|columns|squares
    assert touched<=state.keys(), touched-state.keys()
    # Every fixed field buffer is explicitly classified; edge order is retained.
    unchanged={'active_edge','base_coupling','phi','template_mask','structured.phi',
               'structured.coupling_mask','structured.dissipation_mask'}
    buffers=set(dict(model.named_buffers()))
    assert buffers<=touched|unchanged, buffers-(touched|unchanged)
    structured_params={n for n,_ in model.named_parameters() if n.startswith('structured.')}
    assert structured_params<=touched|{'structured.coupling'}, structured_params-touched
    assert not model.structured.exchange_specs and not model.structured.dissipation_specs
    for name in nodes:state[name]=state[name].index_select(0,perm)
    for name in blocks:state[name]=state[name].index_select(0,p2)
    for name in columns:state[name]=state[name].index_select(1,perm)
    for name in squares:state[name]=state[name].index_select(0,perm).index_select(1,perm)
    result=copy.deepcopy(model);result.load_state_dict(state);result.end_graph_cache()
    for name in unchanged:
        if name in state:assert torch.equal(result.state_dict()[name],model.state_dict()[name])
    return result


@pytest.mark.parametrize('device',DEVICES)
@pytest.mark.parametrize('kind',['E'])
def test_full_graph_and_bone_relabel_conjugates_jr_and_field(device,kind):
    ref,fast,events,_=pair(device,kind)
    # Make node quantities asymmetric so omitted permutations cannot pass by
    # uniform-coefficient coincidence. These are QA-only derived frozen models.
    with torch.no_grad():
        for model in (ref,fast):
            node=torch.linspace(.7,1.6,10,dtype=torch.float64,device=device)
            for target in (model.mass,model.structured.mass,model.p_scale,
                           model.input_scale_delta,model.input_scale_omega,
                           model.gain,model.rate,model.structured.gain):target.copy_(node)
            model.input_scale.copy_(torch.cat((node,node)))
            for j,name in enumerate(('damping','jpi_p','rpp')):
                getattr(model.structured,name).copy_(node+.1*j)
                mask=getattr(model.structured,name+'_mask')
                mask.copy_((torch.arange(10,device=device)%3!=j).to(mask.dtype))
            storage=torch.linspace(.3,1.2,20,dtype=torch.float64,device=device)
            model.structured.storage_base.copy_(storage)
            model.structured.scales.copy_(storage)
            model.structured.storage_base_mask.copy_((torch.arange(20,device=device)%4!=1).to(storage.dtype))
    perm=torch.tensor([3,0,5,7,1,9,4,2,8,6],device=device)
    p40=torch.cat([perm+10*j for j in range(4)])
    z=torch.tensor(np.concatenate((events[0]['x'][11:15],np.full((4,10),.17)),axis=1),device=device)
    u=torch.linspace(-.2,.3,10,dtype=torch.float64,device=device)
    transformed=[]
    for model in (ref,fast):
        other=relabel_field_model(model,perm);transformed.append(other)
        for matrix,changed in zip(model.matrices(),other.matrices()):
            torch.testing.assert_close(changed,matrix[p40][:,p40],**FIELD)
        for external in (None,u):
            actual=other.field(z[:,p40],None if external is None else external[perm])
            torch.testing.assert_close(actual,model.field(z,external)[:,p40],**FIELD)
    for external in (None,u):
        port=None if external is None else external[perm]
        torch.testing.assert_close(transformed[0].field(z[:,p40],port),
                                   transformed[1].field(z[:,p40],port),**FIELD)
