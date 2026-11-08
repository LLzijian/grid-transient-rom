"""Finite data-free qualification DEFINITIONS; not executed during preparation.

No private source, plane selection on A96, event loader, NNLS or optimizer step.
The fixed known-ROM controls are scientific qualification, not execution release.
"""
import copy
import hashlib
import json
import math
from pathlib import Path
import sys
import tempfile
from unittest.mock import patch
import numpy as np
import pytest
import torch
from scipy.integrate import solve_ivp
from scipy.linalg import expm

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'experiments'))
from memory_transport_model_20261029 import MemoryTransportROM,VARIANTS,helmert_basis,prepare_plane
from state_gated_memory_model_20261028 import StateGatedMemoryROM
from force_graph_model_20261028 import PAIRS,RIDGE

DTYPE=torch.float64


def config():
    i=np.arange(10,dtype=float);B=np.zeros((10,45))
    for e,(a,b) in enumerate(PAIRS):B[a,e],B[b,e]=1.,-1.
    return dict(mass=(1+i/10).tolist(),D=B.T.tolist(),coupling=[.1]*45,
        delta_star=[0.]*10,damping=(.1+i/100).tolist(),jpi_p=(1+i/30).tolist(),
        storage_base=(1+i/20).tolist(),rpp=(.2+i/100).tolist(),
        gain=[1.]*10,rate=[1.]*10,input_scale_delta=[1.]*10,input_scale_omega=[1.]*10,p_scale=[1.]*10)


def model(variant,coefficient=.5):
    basis=helmert_basis();m=MemoryTransportROM(config(),variant,u=basis[:,0],v=basis[:,1],
        plane_identity=dict(kind='fixed artificial Helmert plane; no source decomposition'))
    with torch.no_grad():
        m.kappa_raw.copy_(torch.log(1/m.kappa_reference));m.lambda_raw.copy_(torch.tensor(math.log(.5),dtype=DTYPE))
        m.a_bar.copy_(.01*(1+torch.arange(45,dtype=DTYPE)/45))
        if variant=='S':m.s_raw.fill_(coefficient)
        if variant=='P':m.d_raw.fill_(coefficient)
    return m


def history():
    t=torch.arange(11,dtype=DTYPE)[:,None]*.01
    h=torch.zeros((11,30),dtype=DTYPE)
    h[:,:10]=.02*torch.sin(torch.arange(10,dtype=DTYPE)[None,:]+t)
    h[:,10:20]=.01*torch.arange(10,dtype=DTYPE)[None,:]+.03*t
    h[:,20:]=.02;h[:-1,20:]=float('nan')
    return h


def operator_numpy(m):
    B=m.B.detach().numpy();u=m.u.numpy();v=m.v.numpy()
    Q=float(m.lambda_.detach())*np.eye(10)+(B*m.a_bar.detach().numpy())@B.T
    if m.variant=='S':Q+=float(m.s.detach())*(np.outer(u,v)-np.outer(v,u))
    if m.variant=='P':Q+=float(m.d.detach())*(np.outer(u,u)+np.outer(v,v))
    return Q


def force_numpy(m,x):
    B=m.B.numpy();phi=m.phi.numpy()
    return -((np.sin(x[...,:10]@B+phi)-np.sin(phi))*m.reference_coupling45.numpy())@B.T-m.damping.numpy()*x[...,10:20]+(m.jpi_p*m.storage_base).numpy()*x[...,20:30]


def exact_history_reference(m,h):
    """Independent full nonsymmetric 30x30 negative-time block exponential.

    No exp(-Q)*exp(-sK) splitting. This includes all ten affine basis columns.
    """
    h=h.detach().numpy();operator=operator_numpy(m);k=float(m.kappa.detach())
    block=np.zeros((30,30));block[:10,:10]=-operator;block[:10,10:20]=k*np.eye(10);block[10:20,20:]=np.eye(10)
    flow=expm(-.01*block)
    pblock=np.zeros((30,30));pblock[:10,:10]=-np.diag((m.rpp*m.storage_base).numpy())
    pblock[:10,10:20]=-np.diag(m.jpi_p.numpy());pblock[10:20,20:]=np.eye(10)
    pflow=expm(-.01*pblock)
    affine=np.zeros((10,11));affine[:,1:]=np.eye(10);rs=[affine.copy()];ps=[h[-1,20:].copy()]
    for i in range(9,-1,-1):
        slope=(h[i+1,10:20]-h[i,10:20])/.01
        augmented=np.zeros((30,11));augmented[:10]=affine
        augmented[10:20,0]=h[i+1,10:20];augmented[20:,0]=slope
        affine=(flow@augmented)[:10];rs.append(affine.copy())
        ps.append((pflow@np.r_[ps[-1],h[i+1,10:20],slope])[:10])
    states=np.stack(list(reversed(rs)));p=np.stack(list(reversed(ps)));rbar,Z=states[:,:,0],states[:,:,1:]
    physical=np.c_[h[:,:20],p];f=force_numpy(m,physical)-rbar
    C=np.stack([.02/6*(Z[i]+4*Z[i+1]+Z[i+2]) for i in range(0,10,2)])
    b=np.stack([.02/6*(f[i]+4*f[i+1]+f[i+2])-m.mass.numpy()*(h[i+2,10:20]-h[i,10:20]) for i in range(0,10,2)])
    rstar=m.rstar.numpy();design=(C*rstar[None,None,:]/(.02*rstar[None,:,None])).reshape(50,10)
    F=Z[0]*rstar[None,:]/rstar[:,None];q=rbar[0]/rstar
    augmented=np.r_[design,math.sqrt(RIDGE)*F];target=np.r_[(b/(.02*rstar)).reshape(-1),-math.sqrt(RIDGE)*q]
    coordinates=np.linalg.lstsq(augmented,target,rcond=None)[0]
    return dict(Zr=Z,rbar=rbar,p_history=p,r0=rstar*coordinates,F=F,q=q)


@pytest.mark.parametrize('variant',VARIANTS)
def test_nonzero_operator_internal_J_R_power_and_linear_generator(variant):
    m=model(variant);z=torch.linspace(-.1,.1,80,dtype=DTYPE).reshape(2,40)
    x=m.interface_to_state(z).requires_grad_();H,grad=m.energy_and_gradient(x);J,R=m.matrices(z)
    torch.testing.assert_close(torch.autograd.grad(H.sum(),x,create_graph=True)[0],grad,atol=1e-12,rtol=0)
    torch.testing.assert_close(J+J.transpose(-1,-2),torch.zeros_like(J),atol=1e-12,rtol=0)
    assert float(torch.linalg.eigvalsh(R).min())>=-1e-12
    field=m.internal_field(x)
    torch.testing.assert_close(field,torch.einsum('...ij,...j->...i',J-R,grad),atol=1e-12,rtol=0)
    torch.testing.assert_close((grad*field).sum(-1),-torch.einsum('...i,...ij,...j->...',grad,R,grad),atol=1e-12,rtol=0)
    c=m._new_context();linear=m._linear_generator(c)
    torch.testing.assert_close(linear[30:,30:],-torch.from_numpy(operator_numpy(m)),atol=1e-12,rtol=0)
    assert m.mode=='M' and m.gate_mode=='A' and m.variant==variant
    assert m.complexity_counts()['raw_trainable_scalar_count']==(47 if variant=='A' else 48)
    assert m.complexity_counts()['fixed_plane_stored_scalars']==20
    torch.testing.assert_close(c['K']@torch.ones(10,dtype=DTYPE),torch.zeros(10,dtype=DTYPE),atol=1e-12,rtol=0)
    torch.testing.assert_close(c['P2']@torch.ones(10,dtype=DTYPE),torch.zeros(10,dtype=DTYPE),atol=1e-12,rtol=0)


@pytest.mark.parametrize('variant',VARIANTS)
@pytest.mark.parametrize('history_dt',(.0025,.00125))
def test_full_negative_time_block_reference_and_causal_prior(variant,history_dt):
    m=model(variant);h=history()
    with torch.no_grad():
        actual=m.history_system(h,history_dt=history_dt);expected=exact_history_reference(m,h)
        for key in ('Zr','rbar','p_history','r0','F','q'):
            reference=torch.from_numpy(expected[key]);error=actual[key]-reference
            if key in ('rbar','r0'):error=error/m.rstar
            if key=='p_history':error=error/m.p_scale
            assert float(error.abs().max())<=1e-7
            torch.testing.assert_close(actual[key],reference,atol=1e-7,rtol=1e-7)
        residual=actual['A']@actual['s']-actual['bscaled'];prior=actual['F']@actual['s']+actual['q']
        assert float((actual['A'].T@residual+RIDGE*actual['F'].T@prior).abs().max())<1e-10
        initialized=m.initialize(h,history_dt=history_dt)
        assert torch.equal(initialized[:30],h[-1])
        contaminated=h.clone();contaminated[:-1,20:]=1e9
        torch.testing.assert_close(m.encode(contaminated,history_dt=history_dt),actual['r0'],atol=0,rtol=0)
        batched=m.history_system(torch.stack((h,h)),history_dt=history_dt)
        torch.testing.assert_close(batched['r0'],actual['r0'].expand(2,10),atol=1e-12,rtol=0)


@pytest.mark.parametrize('variant',VARIANTS)
def test_translation_and_simultaneous_machine_plane_permutation(variant):
    m=model(variant);h=history();z=torch.linspace(-.1,.1,40,dtype=DTYPE)
    shifted=h.clone();shifted[:,:10]+=.7
    torch.testing.assert_close(m.encode(shifted),m.encode(h),atol=1e-12,rtol=0)
    permutation=torch.tensor([7,0,9,4,2,6,1,8,3,5]);cfg=copy.deepcopy(m._source_config)
    for name,value in cfg.items():
        if name=='D':cfg[name]=torch.tensor(value,dtype=DTYPE)[:,permutation].tolist()
        elif len(value)==10:cfg[name]=torch.tensor(value,dtype=DTYPE)[permutation].tolist()
    other=MemoryTransportROM(cfg,variant,u=m.u[permutation],v=m.v[permutation])
    with torch.no_grad():
        other.kappa_raw.copy_(m.kappa_raw);other.lambda_raw.copy_(m.lambda_raw)
        for e,(i,j) in enumerate(PAIRS):other.a_bar[e]=m.a_bar[PAIRS.index(tuple(sorted((int(permutation[i]),int(permutation[j])))))]
        if variant=='S':other.s_raw.copy_(m.s_raw)
        if variant=='P':other.d_raw.copy_(m.d_raw)
    ix=torch.cat([permutation+offset for offset in (0,10,20,30)])
    torch.testing.assert_close(other.field(z[ix]),m.field(z)[ix],atol=1e-12,rtol=0)
    torch.testing.assert_close(other.encode(h[:,ix[:30]]),m.encode(h)[permutation],atol=1e-12,rtol=0)


@pytest.mark.parametrize('variant',VARIANTS)
def test_all_raw_coordinates_and_causal_history_finite_difference(variant):
    m=model(variant);h=history();h[:-1,20:]=0.;h.requires_grad_()
    target=torch.linspace(-.3,.8,40,dtype=DTYPE)
    def loss():return (m.rollout(h,1,.01)[-1]*target).square().sum()
    L=loss();L.backward();epsilon=1e-5
    for name,p in m.named_parameters():
        assert p.grad is not None and torch.isfinite(p.grad).all()
        for coordinate in range(p.numel()):
            initial=p.detach().clone();direction=torch.zeros_like(p).reshape(-1);direction[coordinate]=1.;direction=direction.reshape(p.shape)
            derivative=(p.grad*direction).sum().detach()
            with torch.no_grad():
                p.copy_(initial+epsilon*direction);plus=loss();p.copy_(initial-epsilon*direction);minus=loss();p.copy_(initial)
            torch.testing.assert_close(derivative,(plus-minus)/(2*epsilon),atol=2e-7,rtol=2e-5)
    assert h.grad is not None and torch.isfinite(h.grad).all()
    assert torch.equal(h.grad[:-1,20:],torch.zeros_like(h.grad[:-1,20:]))
    for row in range(11):
        for column in range(30 if row==10 else 20):
            initial=h.detach().clone();derivative=h.grad[row,column].detach()
            with torch.no_grad():
                h[row,column]+=epsilon;plus=loss();h.copy_(initial);h[row,column]-=epsilon;minus=loss();h.copy_(initial)
            torch.testing.assert_close(derivative,(plus-minus)/(2*epsilon),atol=2e-7,rtol=2e-5)


def test_zero_d_has_legal_one_sided_derivative_and_projection_does_not_lock_it():
    m=model('P',0.);z=torch.linspace(-.1,.2,40,dtype=DTYPE)
    value=(m.field(z)[30:]*torch.arange(10,dtype=DTYPE)).sum();derivative=torch.autograd.grad(value,m.d_raw)[0]
    with torch.no_grad():
        m.d_raw.fill_(1e-5);plus=(m.field(z)[30:]*torch.arange(10,dtype=DTYPE)).sum();m.d_raw.zero_()
    torch.testing.assert_close(derivative,(plus-value.detach())/1e-5,atol=1e-10,rtol=1e-8)
    assert derivative.abs()>0
    with torch.no_grad():m.d_raw.fill_(-.1)
    m.project_runtime_();assert m.d_raw.item()==0.
    assert torch.autograd.grad(m.field(z)[30:].square().sum(),m.d_raw)[0].abs()>0


@pytest.mark.parametrize('variant',VARIANTS)
def test_exact_zero_embedding_source_export_and_restoration(variant):
    original=StateGatedMemoryROM(config(),'A')
    with torch.no_grad():
        original.kappa_raw.fill_(.2);original.lambda_raw.fill_(-.3);original.a_bar.fill_(.02)
    h=history();basis=helmert_basis()
    with tempfile.TemporaryDirectory() as folder:
        source=Path(folder)/'A.json';original.export(source)
        m=MemoryTransportROM.from_a_export(source,variant,u=basis[:,0],v=basis[:,1],
            expected_sha256=hashlib.sha256(source.read_bytes()).hexdigest())
        for name,value in original.state_dict().items():assert torch.equal(m.state_dict()[name],value)
        torch.testing.assert_close(m.initialize(h),original.initialize(h),atol=1e-12,rtol=0)
        torch.testing.assert_close(m.rollout(h,5,.01),original.rollout(h,5,.01),atol=1e-12,rtol=0)
        if variant=='S':
            with torch.no_grad():m.s_raw.fill_(-.5)
        if variant=='P':
            with torch.no_grad():m.d_raw.fill_(.5)
        m.source_identity['fixture_ancestry']='known source A';path=Path(folder)/'transport.json';m.export(path)
        restored=MemoryTransportROM.from_export(path)
        assert restored.variant==variant and restored.history_dt==m.history_dt
        assert restored.source_identity==m.source_identity and restored.plane_identity==m.plane_identity
        assert dict((n,p.requires_grad) for n,p in restored.named_parameters())==dict((n,p.requires_grad) for n,p in m.named_parameters())
        for name,value in m.state_dict().items():assert torch.equal(restored.state_dict()[name],value)
        torch.testing.assert_close(restored.rollout(h,5,.01),m.rollout(h,5,.01),atol=0,rtol=0)
        with pytest.raises(ValueError):StateGatedMemoryROM.from_export(path)
        # Full parameter/gradient layout survives; an actual Adam/RNG restoration
        # qualification belongs to the separately registered 36-call study QA.
        assert sum(p.numel() for p in restored.parameters())==(47 if variant=='A' else 48)


@pytest.mark.parametrize('variant,coefficient',[('A',0.),('S',.5),('S',-.5),('P',.5)])
def test_cached_uncached_operator_gradients_and_lifecycle(variant,coefficient):
    m=model(variant,coefficient);h=history();z=torch.linspace(-.1,.1,40,dtype=DTYPE)
    m.begin_graph_cache();context=m._context()
    torch.testing.assert_close(m.field(z),m.field_reference(z),atol=1e-12,rtol=0)
    cached=m.history_system(h);reference=m.history_system(h,reference=True)
    torch.testing.assert_close(cached['r0'],reference['r0'],atol=0,rtol=0)
    cached_objective=m.field(z).square().sum()+cached['r0'].square().sum()
    reference_objective=m.field_reference(z).square().sum()+reference['r0'].square().sum()
    cached_gradients=torch.autograd.grad(cached_objective,tuple(m.parameters()),retain_graph=True)
    reference_gradients=torch.autograd.grad(reference_objective,tuple(m.parameters()))
    for a,b in zip(cached_gradients,reference_gradients):
        torch.testing.assert_close(a,b,atol=1e-12,rtol=1e-12)
    loss=m.rollout(h,2,.01).square().sum();loss.backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in m.parameters())
    with tempfile.TemporaryDirectory() as folder:
        with pytest.raises(RuntimeError):m.export(Path(folder)/'active.json')
    m.end_graph_cache();assert m._cache is None
    m.zero_grad(set_to_none=True);m.rollout(h,2,.01).square().sum().backward();assert m._cache is None
    m.begin_graph_cache()
    with torch.no_grad():m.lambda_raw.add_(.01)
    assert m._context() is not context and m._cache is None
    m.begin_graph_cache();m.train(False);assert m._cache is None
    m.begin_graph_cache();m.to(dtype=DTYPE);assert m._cache is None
    m.begin_graph_cache();m.load_state_dict(m.state_dict());assert m._cache is None
    with pytest.raises(ValueError):m.fast_step(z,.01)
    with pytest.raises(ValueError):m._history_context(m._new_context())
    with pytest.raises(ValueError):m.rollout(h,1,.01,method='fast')
    with pytest.raises(ValueError):m.mode='S'
    with pytest.raises(ValueError):m.gate_mode='C'
    with pytest.raises(ValueError):m.variant='P' if variant!='P' else 'S'
    m.begin_graph_cache()
    with torch.no_grad():
        m.field(z)
        assert m._cache is None


@pytest.mark.parametrize('variant',VARIANTS)
def test_fixed_known_ROM_causal_future_true_RK4_against_DOP853(variant):
    m=model(variant);A=operator_numpy(m);mass=m.mass.numpy();G=m.jpi_p.numpy();Sp=m.storage_base.numpy();Rp=m.rpp.numpy();k=float(m.kappa.detach())
    def rhs(t,x):
        _,w,p,r=np.split(x,4)
        return np.r_[w,(force_numpy(m,x[:30])-r)/mass,-G*w-Rp*Sp*p,k*w-A@r]
    times=np.arange(411)*.01
    for j in range(10):
        initial=np.zeros(40);initial[:10]=-.15/10;initial[j]+=.15
        initial[10+(j+1)%10]=.1;initial[20+(j+2)%10]=.05;initial[30+(j+3)%10]=.1
        truth=solve_ivp(rhs,(0.,4.1),initial,method='DOP853',rtol=1e-12,atol=1e-13,t_eval=times)
        assert truth.success and truth.y.shape==(40,411) and np.isfinite(truth.y).all()
        history_only=torch.from_numpy(truth.y[:30,200:211].T.copy());history_only[:-1,20:]=float('nan')
        with torch.no_grad():
            coarse=m.rollout(history_only,200,.01,history_dt=.0025)
            fine=m.rollout(history_only,400,.005,history_dt=.00125)
        reference=torch.from_numpy(truth.y[:,210:].T.copy())
        for path in (coarse,fine[::2]):
            assert float(((path[0,30:]-reference[0,30:])/m.rstar).abs().max())<=1e-2
            assert torch.all((path[:,:30]-reference[:,:30]).abs().amax(0)<=2e-2)
        torch.testing.assert_close(coarse,fine[::2],atol=1e-7,rtol=1e-7)
        # This is the actual classical RK4 step, not an exact-linear surrogate.
        state=coarse[0];a=m.field(state);b=m.field(state+.005*a);c=m.field(state+.005*b);d=m.field(state+.01*c)
        torch.testing.assert_close(coarse[1],state+.01/6*(a+2*b+2*c+d),atol=0,rtol=0)


def test_plane_cutoff_and_internal_double_root_with_mocked_single_eigh():
    """No real source decomposition: mock outputs on an artificial export only."""
    original=StateGatedMemoryROM(config(),'A')
    with tempfile.TemporaryDirectory() as folder:
        path=Path(folder)/'artificial_A.json';original.export(path)
        digest=hashlib.sha256(path.read_bytes()).hexdigest()
        for separated in (True,False):
            # Deliberately non-exact mocked eigenvalues also verify that the
            # source residual is measured, rather than hardcoded to zero.
            eigenvalues=torch.tensor([.5,.5,2.,3.,4.,5.,6.,7.,8.] if separated else [1.]*9,dtype=DTYPE)
            with patch('torch.linalg.eigh',return_value=(eigenvalues,torch.eye(9,dtype=DTYPE))) as decomposition:
                output=Path(folder)/('separated' if separated else 'unresolved')
                result=prepare_plane(path,expected_sha256=digest,output_directory=output)
            decomposition.assert_called_once()
            assert decomposition.call_args.args[0].shape==(9,9)
            assert result['cutoff_pass']==separated and result['eigh_count']==1
            if separated:
                u,v=np.asarray(result['u']),np.asarray(result['v'])
                np.testing.assert_allclose(np.c_[u,v].T@np.c_[u,v],np.eye(2),atol=1e-12,rtol=0)
                assert abs(u.sum())<1e-12 and abs(v.sum())<1e-12
                assert u[np.argmax(abs(u))]>0 and v[np.argmax(abs(v))]>0
                frame=np.c_[u,v];Q=original.memory_matrix(torch.zeros(10,dtype=DTYPE)).detach().numpy()
                checks=result['plane_checks'];residual=np.linalg.norm(Q@frame-frame*np.asarray([.5,.5])[None,:],axis=0)
                np.testing.assert_allclose(checks['eigenpair_residual_l2_per_mode'],residual,atol=1e-15,rtol=0)
                assert min(checks['eigenpair_residual_l2_per_mode'])>.1
                assert checks['eigenpair_normalization_scale_per_second']==8.
                np.testing.assert_allclose(checks['normalized_eigenpair_residual_l2_per_mode'],residual/8.,atol=1e-15,rtol=0)
                np.testing.assert_allclose(checks['zero_sum_components'],frame.sum(0),atol=1e-15,rtol=0)
                assert checks['orthogonality_frobenius']<=1e-12
            else:
                assert result['status']=='PLANE_UNRESOLVED' and result['u'] is None and result['v'] is None
                assert result['plane_checks'] is None
            assert json.loads((output/'result.json').read_text(encoding='utf-8'))==result
            with patch('torch.linalg.eigh') as forbidden_repeat:
                with pytest.raises(FileExistsError):prepare_plane(path,expected_sha256=digest,output_directory=output)
            forbidden_repeat.assert_not_called()
        failure=Path(folder)/'hash_failure'
        with patch('torch.linalg.eigh') as forbidden_wrong_source:
            with pytest.raises(ValueError):prepare_plane(path,expected_sha256='0'*64,output_directory=failure)
        forbidden_wrong_source.assert_not_called()
        assert json.loads((failure/'result.json').read_text(encoding='utf-8'))['status']=='PLANE_PREPARATION_FAILED'


def test_export_rejects_lambda_boundary_and_corrupt_plane_or_variant_layout():
    with tempfile.TemporaryDirectory() as folder:
        m=model('S');path=Path(folder)/'valid.json';payload=m.export(path)
        for edit in ('plane','variant','lambda_boundary'):
            corrupted=copy.deepcopy(payload)
            if edit=='plane':corrupted['plane']['u'][0]+=.1
            elif edit=='variant':corrupted['variant']='P'
            else:corrupted['state_dict']['lambda_raw']=-1000.
            bad=Path(folder)/(edit+'.json');bad.write_text(json.dumps(corrupted),encoding='utf-8')
            with pytest.raises(ValueError):MemoryTransportROM.from_export(bad)
        with torch.no_grad():m.lambda_raw.fill_(-1000.)
        with pytest.raises(ValueError):m.export(Path(folder)/'no_boundary_rescue.json')


@pytest.mark.parametrize('variant',VARIANTS)
def test_reference_history_can_compute_but_cannot_export_or_restore_checkpoint(variant):
    main=model(variant);reference=MemoryTransportROM(main._source_config,variant,u=main.u,v=main.v,history_dt=.00125)
    reference.load_state_dict(main.state_dict());h=history()
    with torch.no_grad():
        torch.testing.assert_close(reference.initialize(h),main.initialize(h,history_dt=.00125),atol=0,rtol=0)
    with tempfile.TemporaryDirectory() as folder:
        folder=Path(folder);ref_path=folder/'forbidden_reference.json'
        with pytest.raises(ValueError):reference.export(ref_path)
        assert not ref_path.exists()
        path=folder/'primary.json';payload=main.export(path)
        assert MemoryTransportROM.from_export(path).history_dt==.0025
        payload['history_dt']=.00125
        tampered=folder/'tampered_reference.json';tampered.write_text(json.dumps(payload),encoding='utf-8')
        with pytest.raises(ValueError):MemoryTransportROM.from_export(tampered)


class UncachedTransportReference(MemoryTransportROM):
    """Test-only uncached operator route through the identical full objective."""
    def _context(self):return self._new_context()


@pytest.mark.parametrize('variant',VARIANTS)
def test_complete_objective_cache_equivalence_all_gradients(variant):
    import state_gated_memory_study_20261028 as sgm
    m=model(variant);reference=UncachedTransportReference(m._source_config,variant,u=m.u,v=m.v)
    reference.load_state_dict(m.state_dict());events=sgm.old.synthetic_fixture()
    values=[];gradients=[]
    for candidate in (m,reference):
        with sgm.graph(candidate):
            value,_=sgm.objective(candidate,sgm.EventStore(events,'cpu'),[(e,10) for e in events],.02,method='rk4')
            value.backward()
        values.append(value.detach())
        gradients.append({n:p.grad.detach().clone() for n,p in candidate.named_parameters() if p.requires_grad})
        assert all(torch.isfinite(v).all() for v in gradients[-1].values())
        assert candidate._cache is None
    torch.testing.assert_close(values[0],values[1],atol=1e-10,rtol=1e-8)
    assert set(gradients[0])==set(gradients[1])==set(dict(m.named_parameters()))
    for name in gradients[0]:torch.testing.assert_close(gradients[0][name],gradients[1][name],atol=1e-10,rtol=1e-8)


@pytest.mark.parametrize('variant',VARIANTS)
def test_actual_CUDA_field_initialize_short_complete_J_all_gradients(variant):
    """Synthetic mathematical check; real-source fit297/12x8 remains separate."""
    if not torch.cuda.is_available():pytest.skip('actual CUDA required; no CPU substitution')
    import state_gated_memory_study_20261028 as sgm
    events=sgm.old.synthetic_fixture();m=model(variant)
    models=[m,copy.deepcopy(m).to('cuda').double()]
    h=np.stack([e['x'][:11] for e in events]);states=np.c_[h[:,-1],np.full((len(h),10),1e-5)]
    values=[];gradients=[]
    for candidate in models:
        tensors=dict(field=candidate.field(candidate._tensor(states)),initialize=candidate.initialize(h),
                     short_RK4=candidate.rollout(h,4,.01))
        with sgm.graph(candidate):
            J,_=sgm.objective(candidate,sgm.EventStore(events,candidate.mass.device),[(e,10) for e in events],.02,method='rk4')
            J.backward()
        tensors['J']=J
        values.append({k:v.detach().cpu() for k,v in tensors.items()})
        gradients.append({n:p.grad.detach().cpu() for n,p in candidate.named_parameters() if p.requires_grad})
        assert set(gradients[-1])==set(dict(candidate.named_parameters()))
        assert all(torch.isfinite(v).all() for v in gradients[-1].values())
    for key in values[0]:torch.testing.assert_close(values[0][key],values[1][key],atol=1e-10,rtol=1e-8)
    for name in gradients[0]:torch.testing.assert_close(gradients[0][name],gradients[1][name],atol=1e-10,rtol=1e-8)
