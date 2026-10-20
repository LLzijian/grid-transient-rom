"""P8 synthetic QA; only root may execute the sole registered native QA."""
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
import cubic_port_only_runtime_20261021 as prt
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


def pair(device, kind='P'):
    models, events, rng = prepared(device)
    fast = prt.build_P(models['N'])
    with torch.no_grad():fast.coefficients[prt.b_index(fast)]=-.03
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
@pytest.mark.parametrize('kind', ['P'])
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
@pytest.mark.parametrize('kind', ['P'])
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
    assert fast._p_cache is None and fast._runtime_cache is None
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
def test_zero_coefficients_connected_and_extra_contribution_permutation(device):
    ref,fast,events,_=pair(device,'P')
    indices=drt.coefficient_indices(fast)
    for model in (ref,fast):
        with torch.no_grad():
            for i in drt.coefficient_indices(model).values(): model.coefficients[i].zero_()
    z=torch.tensor(np.concatenate((events[0]['x'][11:15],np.full((4,10),.27)),axis=1),device=device)
    for model in (ref,fast): model.field(z).square().sum().backward()
    gradients_equal(ref,fast)
    assert all(abs(float(fast.coefficients.grad[i]))>1e-14 for i in indices.values())
    # Independently verify the added local port under simultaneous machine/mass permutation.
    perm=torch.tensor([3,0,5,7,1,9,4,2,8,6],device=device)
    b=-.03;q=z[:,30:];mass=fast.mass
    assert torch.equal((b*q.pow(3)/mass)[:,perm],b*q[:,perm].pow(3)/mass[perm])
    torch.testing.assert_close((z[:,10:20]*b*q.pow(3)).sum(-1),
        (z[:,10:20][:,perm]*b*q[:,perm].pow(3)).sum(-1),**FIELD)


@pytest.mark.parametrize('device', DEVICES)
@pytest.mark.parametrize('kind', ['P'])
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
    restored=prt.PRuntime.from_export(tmp_path/'model.json').to(device)
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
    torch.testing.assert_close(old.field(z),prt.PRuntime.from_export(tmp_path/'model.json').to(device).field(z),**FIELD)
    saved['coefficients'][0]+=.01;(tmp_path/'bad.json').write_text(json.dumps(saved))
    with pytest.raises(ValueError): prt.PRuntime.from_export(tmp_path/'bad.json')


@pytest.mark.parametrize('kind',['P'])
def test_cache_freshness_graph_lifecycle_failure_flags_and_order(kind):
    ref,fast,events,_=pair('cpu',kind)
    z=torch.tensor(np.concatenate((events[0]['x'][11:13],np.full((2,10),.12)),axis=1))
    before=fast.field(z).detach()
    with torch.no_grad():fast.coefficients[0].add_(.1);ref.coefficients[0].add_(.1)
    assert not torch.equal(before,fast.field(z))
    torch.testing.assert_close(fast.field(z),ref.field(z),**FIELD)
    fast.begin_graph_cache();cache=fast._p_cache
    assert cache['linear'].grad_fn is not None
    fast.end_graph_cache();assert fast._p_cache is None
    fast.begin_graph_cache()
    with pytest.raises(ValueError):fast.field(torch.zeros(3,39))
    assert fast._p_cache is None and fast._runtime_cache is None
    fast.begin_graph_cache();fast.load_state_dict(fast.state_dict());assert fast._p_cache is None
    fast.begin_graph_cache();fast.to('cpu');assert fast._p_cache is None
    reordered=fast.with_terms(list(reversed(fast.terms)))
    torch.testing.assert_close(reordered.field(z),fast.field(z),**FIELD)
    invalid=copy.deepcopy(fast);invalid.terms[0]=invalid.terms[1]
    with pytest.raises(ValueError):prt.enable_runtime(invalid)


@pytest.mark.parametrize('device',DEVICES)
def test_causal_rollout_one_initialization_and_original_auxiliary_positions(device,monkeypatch):
    _,model,events,_=pair(device,'P');events=ragged(events)
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


@pytest.mark.parametrize('kind',['P'])
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
    assert model._runtime_cache is None and model._p_cache is None


def test_signed_b_no_projection_and_exact_ast_export_guards(tmp_path):
    _,model,_,_=pair('cpu')
    with torch.no_grad():model.coefficients[prt.b_index(model)]=-.4
    prt.require_finite(model)
    model.export(tmp_path/'signed.json')
    loaded=prt.PRuntime.from_export(tmp_path/'signed.json')
    assert float(loaded.coefficients[prt.b_index(loaded)])==-.4
    assert list(prt.coefficient_indices(loaded))==['b']
    with pytest.raises(ValueError):model.with_terms(prt.TERMS[:-1])
    with pytest.raises(ValueError):prt.build_P(model)
    model.begin_graph_cache()
    with pytest.raises(RuntimeError):model.export(tmp_path/'active.json')
    assert model._p_cache is not None
    model.end_graph_cache()


@pytest.mark.parametrize('device',DEVICES)
def test_zero_b_embeds_n_initializer_public_raw_gradients_and_energy(device):
    models,events,rng=prepared(device)
    n=models['N'];p=prt.build_P(n)
    assert float(p.coefficients[prt.b_index(p)])==0.
    store=TensorEventStore(ragged(events),device)
    samples=[(e,10) for e in ragged(events)]
    history=store.histories(samples)
    torch.testing.assert_close(n.encode(history),p.encode(history),rtol=0,atol=0)
    z=torch.tensor(np.concatenate((events[0]['x'][11:15],np.full((4,10),.27)),axis=1),device=device)
    torch.testing.assert_close(n.field(z),p.field(z),**FIELD)
    for model in (n,p):
        restore_rng(rng,device)
        with graph(model):
            value,_=objective(model,store,samples,.03,.01)
            value.backward()
    for name,left in n.named_parameters():
        right=dict(p.named_parameters())[name]
        if name=='coefficients':
            for i,term in enumerate(n.terms):
                j=[term_key(t) for t in p.terms].index(term_key(term))
                torch.testing.assert_close(left.grad[i],right.grad[j],**GRAD)
        else:
            assert (left.grad is None)==(right.grad is None)
            if left.grad is not None:torch.testing.assert_close(left.grad,right.grad,**GRAD)
    for b in (-.23,.19):
        with torch.no_grad():p.coefficients[prt.b_index(p)]=b
        difference=p.field(z)-n.field(z)
        expected=torch.zeros_like(z);expected[:,10:20]=b*z[:,30:].pow(3)/p.mass
        torch.testing.assert_close(difference,expected,**FIELD)
        x=p.interface_to_state(z)
        hn,gn=n.energy_and_gradient(x);hp,gp=p.energy_and_gradient(x)
        torch.testing.assert_close(hn,hp,**FIELD);torch.testing.assert_close(gn,gp,**FIELD)
        J,R=p.matrices();B=p.port_matrix();u=p.port_values(x)
        actual=(gp*p.interface_to_state(p.field(z))).sum(-1)
        wanted=-(gp*(gp@R.T)).sum(-1)+((gp@B)*u).sum(-1)
        torch.testing.assert_close(actual,wanted,**LOSS)
        torch.testing.assert_close(prt.extra_port_power(p,z),
                                   (z[:,10:20]*b*z[:,30:].pow(3)).sum(-1),**FIELD)
        # A supplied port replaces every autonomous term, including b*q^3.
        external=torch.full((10,),.17,device=device,dtype=z.dtype)
        torch.testing.assert_close(p.field(z,external),n.field(z,external),**FIELD)


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
@pytest.mark.parametrize('kind',['P'])
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
