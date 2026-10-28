"""CPU float64 mathematical qualification; no optimizer or real event payload."""
import copy
import json
from pathlib import Path
import sys
import tempfile
import time
import numpy as np
import pytest
from scipy.integrate import solve_ivp
from scipy.linalg import expm
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"experiments"))
from state_gated_memory_model_20261028 import StateGatedMemoryROM,ControlDegenerate,GATE_MODES
from force_graph_model_20261028 import ForceGraphROM,PAIRS,RIDGE
from test_force_graph_model_20261028 import fixture,history,state,spatial,explicit_field,DTYPE

torch.set_num_threads(1)
MEASUREMENTS={}


def model(mode):
    m=StateGatedMemoryROM(fixture(),mode);spatial(m)
    with torch.no_grad(): m.kappa_raw.fill_(.21);m.lambda_raw.fill_(-.17)
    return m


def independent_Q(m,d):
    a=m.a_bar.detach().numpy();g=np.array([(1+np.cos(d[i]-d[j]))/2 for i,j in PAIRS])
    if m.gate_mode=="A": weights=a
    elif m.gate_mode=="B": weights=a*np.dot(a,g)/a.sum()
    else: weights=a*g
    Q=float(m.lambda_.detach())*np.eye(10)
    for w,(i,j) in zip(weights,PAIRS):
        b=np.zeros(10);b[i],b[j]=1.,-1.;Q+=w*np.outer(b,b)
    return Q


def independent_flow(m,h,constant=False):
    h=h.detach().numpy();k=float(m.kappa.detach());G=m.jpi_p.numpy();Ap=(m.rpp*m.storage_base).numpy()
    v=np.zeros((20,11));v[:10,0]=h[-1,20:];v[10:,1:]=np.eye(10);out=[v.copy()]
    if constant:
        T=np.zeros((40,40));T[:10,:10]=-np.diag(Ap);T[10:20,10:20]=-independent_Q(m,h[0,:10])
        T[:10,20:30]=-np.diag(G);T[10:20,20:30]=k*np.eye(10);T[20:30,30:]=np.eye(10)
        augmented=np.zeros((40,11));augmented[:20]=v;augmented[20:30,0]=h[-1,10:20]
        augmented[30:,0]=(h[-1,10:20]-h[0,10:20])/.1
        return np.stack([(expm(T*t)@augmented)[:20] for t in np.linspace(-.1,0,11)])
    for i in range(9,-1,-1):
        def rhs(t,y):
            x=h[i+1,:20]+(h[i,:20]-h[i+1,:20])*(-t/.01)
            value=y.reshape(20,11)
            derivative=np.r_[-Ap[:,None]*value[:10],-independent_Q(m,x[:10])@value[10:]]
            derivative[:10,0]-=G*x[10:];derivative[10:,0]+=k*x[10:]
            return derivative.ravel()
        result=solve_ivp(rhs,(0,-.01),v.ravel(),method="DOP853",rtol=1e-12,atol=1e-13)
        assert result.success
        v=result.y[:,-1].reshape(20,11);out.append(v.copy())
    return np.stack(list(reversed(out)))


def reference_initial(m,h,v):
    vb=torch.from_numpy(v[:,:,0]);Z=torch.from_numpy(v[:,:,1:]);Zr=Z[:,10:]
    C=torch.stack([.02/6*(Zr[i]+4*Zr[i+1]+Zr[i+2]) for i in range(0,10,2)])
    physical=torch.cat((h[:,:20],vb[:,:10],torch.zeros(11,10,dtype=DTYPE)),-1)
    f=explicit_field(m,physical)[:,10:20]*m.mass-vb[:,10:]
    b=torch.stack([.02/6*(f[i]+4*f[i+1]+f[i+2])-m.mass*(h[i+2,10:20]-h[i,10:20]) for i in range(0,10,2)])
    A=(C*m.rstar[None,None,:]/(.02*m.rstar[None,:,None])).reshape(50,10)
    F=Zr[0]*m.rstar[None,:]/m.rstar[:,None];q=vb[0,10:]/m.rstar
    design=torch.cat((A,np.sqrt(RIDGE)*F));target=torch.cat(((b/(.02*m.rstar)).flatten(),-np.sqrt(RIDGE)*q))
    s=torch.from_numpy(np.linalg.lstsq(design.detach().numpy(),target.detach().numpy(),rcond=None)[0])
    return m.rstar*s,F,q


@pytest.mark.parametrize("mode",GATE_MODES)
def test_independent_constant_or_noncommuting_LTV_reference_and_affine_prior(mode):
    m=model(mode);h=history(1)[0]
    h[:,:10]=torch.sin(torch.arange(11,dtype=DTYPE)[:,None]*.7+torch.arange(10,dtype=DTYPE)[None,:])*.8
    if mode=="A": h[:,10:20]=torch.linspace(-.03,.06,10,dtype=DTYPE)+torch.arange(11,dtype=DTYPE)[:,None]*.002
    with torch.no_grad():
        s=m.history_system(h);v=independent_flow(m,h,constant=mode=="A");expected,F,q=reference_initial(m,h,v)
        ref=torch.from_numpy(v)
        errZ=(s["Zr"]-ref[:,10:,1:]).abs().max().item()
        errv=((s["rbar"]-ref[:,10:,0])/m.rstar).abs().max().item()
        errp=((s["p_history"]-ref[:,:10,0])/m.p_scale).abs().max().item()
        errinit=((s["r0"]-expected)/m.rstar).abs().max().item()
        MEASUREMENTS[mode]=dict(Z_absolute_error=errZ,normalized_rbar_error=errv,normalized_p_error=errp,normalized_initial_error=errinit,
            raw_initial_absolute_error=float((s["r0"]-expected).abs().max()),history_dt=m.history_dt,
            independent_reference="SciPy expm" if mode=="A" else "independent physical DOP853 rtol1e-12 atol1e-13")
        if mode=="C":
            q0=independent_Q(m,h[0,:10].numpy());q1=independent_Q(m,h[5,:10].numpy())
            commutator_norm=float(np.linalg.norm(q0@q1-q1@q0))
            MEASUREMENTS[mode]["Q_commutator_norm"]=commutator_norm
            assert commutator_norm>1e-5
        assert max(errZ,errv,errp,errinit)<1e-7
        torch.testing.assert_close(s["Zr"],ref[:,10:,1:],atol=1e-7,rtol=1e-7)
        torch.testing.assert_close(s["rbar"],ref[:,10:,0],atol=1e-7,rtol=1e-7)
        torch.testing.assert_close(s["r0"],expected,atol=1e-7,rtol=1e-7)
        torch.testing.assert_close(s["F"],F,atol=1e-7,rtol=1e-7);torch.testing.assert_close(s["q"],q,atol=1e-7,rtol=1e-7)
        residual=s["A"]@s["s"]-s["bscaled"];prior=s["F"]@s["s"]+s["q"]
        assert (s["A"].T@residual+RIDGE*s["F"].T@prior).abs().max()<1e-10
        assert torch.equal(m.initialize(h)[:30],h[-1])
        fine=m.history_system(h,history_dt=.00125)
        torch.testing.assert_close(s["r0"],fine["r0"],atol=1e-7,rtol=1e-7)
        if mode=="C":
            q0=independent_Q(m,h[0,:10].numpy());q1=independent_Q(m,h[5,:10].numpy())
            assert np.linalg.norm(q0@q1-q1@q0)>1e-5


@pytest.mark.parametrize("mode",GATE_MODES)
def test_power_nonunit_chainrule_permutation_angle_translation_and_exports(mode):
    m=model(mode);z=state();x=m.interface_to_state(z).requires_grad_()
    H,grad=m.energy_and_gradient(x);auto=torch.autograd.grad(H.sum(),x,create_graph=True)[0]
    torch.testing.assert_close(auto,grad,atol=1e-12,rtol=0)
    J,R=m.matrices(z);f=m.internal_field(x)
    torch.testing.assert_close(f,torch.einsum("...ij,...j->...i",J-R,grad),atol=1e-12,rtol=0)
    assert torch.linalg.eigvalsh(R).min()>=-1e-12
    assert ((grad*f).sum(-1)+torch.einsum("...i,...ij,...j->...",grad,R,grad)).abs().max()<1e-12
    h=history();polluted=h.clone();polluted[:,:10,20:]=float("nan")
    assert torch.equal(m.initialize(h),m.initialize(polluted))
    shifted=h.clone();shifted[:,:,:10]+=.5
    torch.testing.assert_close(m.encode(h),m.encode(shifted),atol=1e-12,rtol=0)
    cfg=copy.deepcopy(m._source_config);perm=torch.tensor([7,0,9,4,2,6,1,8,3,5])
    for k,v in cfg.items():
        if k=="D": cfg[k]=torch.tensor(v)[:,perm].tolist()
        elif len(v)==10: cfg[k]=torch.tensor(v,dtype=DTYPE)[perm].tolist()
    other=StateGatedMemoryROM(cfg,mode)
    with torch.no_grad():
        other.kappa_raw.copy_(m.kappa_raw);other.lambda_raw.copy_(m.lambda_raw)
        for e,(i,j) in enumerate(PAIRS): other.a_bar[e]=m.a_bar[PAIRS.index(tuple(sorted((int(perm[i]),int(perm[j])))))]
    ix=torch.cat([perm+offset for offset in (0,10,20,30)])
    torch.testing.assert_close(other.field(z[:,ix]),m.field(z)[:,ix],atol=1e-12,rtol=0)
    torch.testing.assert_close(other.encode(h[:,:,ix[:30]]),m.encode(h)[:,perm],atol=1e-12,rtol=0)
    with tempfile.TemporaryDirectory(prefix="gated-export-") as folder:
        p=Path(folder)/"model.json";m.export(p);loaded=StateGatedMemoryROM.from_export(p)
        assert torch.equal(m.initialize(h),loaded.initialize(h)) and loaded.gate_mode==mode
        assert all(torch.equal(v,loaded.state_dict()[k]) for k,v in m.state_dict().items())
    with pytest.raises(TypeError): m.matrices()
    with pytest.raises(ValueError): m.rollout(h,1,.01,method="fast")


@pytest.mark.parametrize("mode",GATE_MODES)
def test_all_raw_and_history_gradients_finite_difference(mode):
    m=model(mode);h=history(1).requires_grad_();target=torch.linspace(-.3,.8,40,dtype=DTYPE)
    def loss(): return (m.rk4(m.initialize(h),.01)*target).square().sum()
    L=loss();L.backward();step=1e-5
    for name,p in m.named_parameters():
        assert p.grad is not None and torch.isfinite(p.grad).all()
        direction=torch.linspace(.2,1.,p.numel(),dtype=DTYPE).reshape(p.shape)
        actual=(p.grad*direction).sum().detach();initial=p.detach().clone()
        with torch.no_grad():
            p.copy_(initial+step*direction);plus=loss();p.copy_(initial-step*direction);minus=loss();p.copy_(initial)
        torch.testing.assert_close(actual,(plus-minus)/(2*step),atol=2e-7,rtol=2e-5)
    assert h.grad is not None and torch.isfinite(h.grad).all()
    for block in (slice(0,10),slice(10,20),slice(20,30)):
        direction=torch.zeros_like(h);direction[:,:,block]=.3
        if block.start==20: direction[:,:10,20:]=0
        initial=h.detach().clone();actual=(h.grad*direction).sum().detach()
        with torch.no_grad():
            h.copy_(initial+step*direction);plus=loss();h.copy_(initial-step*direction);minus=loss();h.copy_(initial)
        torch.testing.assert_close(actual,(plus-minus)/(2*step),atol=2e-7,rtol=2e-5)
    assert torch.equal(h.grad[:,:10,20:],torch.zeros_like(h.grad[:,:10,20:]))


def test_trace_single_edge_zero_graph_near_zero_control_derivatives_and_cache():
    B,C=model("B"),model("C");z=state()*3;d=z[:,:10]
    torch.testing.assert_close(B.memory_matrix(d).diagonal(dim1=-2,dim2=-1).sum(-1),C.memory_matrix(d).diagonal(dim1=-2,dim2=-1).sum(-1),atol=1e-12,rtol=0)
    for m in (B,C):
        with torch.no_grad(): m.a_bar.zero_();m.a_bar[9]=.02
    torch.testing.assert_close(B.field(z),C.field(z),atol=1e-12,rtol=0)
    # Fixed near-zero relative directions preserve the full ratio derivative.
    for scale in (1.,1e-12):
        with torch.no_grad(): B.a_bar.copy_(torch.linspace(.01,.03,45,dtype=DTYPE)*scale)
        delta=d.detach().clone().requires_grad_();value=B.memory_matrix(delta).square().sum()
        ga,gd=torch.autograd.grad(value,(B.a_bar,delta));assert torch.isfinite(ga).all() and torch.isfinite(gd).all()
        direction=torch.linspace(.2,1.,45,dtype=DTYPE)*B.a_bar.detach().sum()
        original=B.a_bar.detach().clone();step=1e-5
        with torch.no_grad():
            B.a_bar.copy_(original+step*direction);plus=B.memory_matrix(delta).square().sum()
            B.a_bar.copy_(original-step*direction);minus=B.memory_matrix(delta).square().sum();B.a_bar.copy_(original)
        torch.testing.assert_close((ga*direction).sum(),(plus-minus)/(2*step),atol=2e-7,rtol=2e-5)
        # Remove the constant lambda term algebraically before normalization,
        # so near-zero derivative tests cannot pass by absolute tiny effects.
        delta=d.detach().clone().requires_grad_();original=B.a_bar.detach().clone();s0=original.sum()
        def normalized_objective():
            w=B.effective_weights(delta)/s0
            return ((B.B*w[...,None,:])@B.B.T).square().sum()
        value=normalized_objective();ga,gd=torch.autograd.grad(value,(B.a_bar,delta))
        with torch.no_grad():
            B.a_bar.copy_(original+step*direction);plus=normalized_objective()
            B.a_bar.copy_(original-step*direction);minus=normalized_objective();B.a_bar.copy_(original)
        torch.testing.assert_close((ga*direction).sum(),(plus-minus)/(2*step),atol=2e-7,rtol=2e-5)
        direction_delta=torch.sin(torch.arange(delta.numel(),dtype=DTYPE)).reshape(delta.shape)
        initial_delta=delta.detach().clone()
        with torch.no_grad():
            delta.copy_(initial_delta+step*direction_delta);plus=normalized_objective()
            delta.copy_(initial_delta-step*direction_delta);minus=normalized_objective();delta.copy_(initial_delta)
        torch.testing.assert_close((gd*direction_delta).sum(),(plus-minus)/(2*step),atol=2e-7,rtol=2e-5)
    for mode in GATE_MODES:
        m=model(mode)
        with torch.no_grad(): m.a_bar.zero_();q=m.memory_matrix(d)
        torch.testing.assert_close(q,(m.lambda_*m.W).expand(3,10,10),atol=1e-12,rtol=0)
        if mode=="B":
            with pytest.raises(ControlDegenerate): m.field(z)
            m.begin_graph_cache()
            with pytest.raises(ControlDegenerate): m.project_runtime_()
            assert m._cache is None
            with tempfile.TemporaryDirectory(prefix="gated-zero-") as folder: m.export(Path(folder)/"zero.json")
        else: m.field(z).square().sum().backward();assert torch.isfinite(m.a_bar.grad).all()
    m=model("C");m.begin_graph_cache();before=m.memory_matrix(d).detach()
    with torch.no_grad(): m.a_bar.add_(.01)
    assert not torch.equal(before,m.memory_matrix(d));assert m._cache is None
    assert sum(p.numel() for p in m.parameters())==47 and not hasattr(m,"raw_gamma")


def test_precision_pair_separate_initialization_configuration_and_legacy_export():
    m=model("C");h=history()
    assert m.history_dt==.0025
    coarse=m.rollout(h,1,.01)
    fine=m.rollout(h,2,.005,history_dt=.00125)
    assert torch.equal(coarse[:,0],m.initialize(h,history_dt=.0025))
    assert torch.equal(fine[:,0],m.initialize(h,history_dt=.00125))
    assert m.history_dt==.0025 and m._cache is None
    assert not torch.equal(coarse[:,0,30:],fine[:,0,30:])
    with tempfile.TemporaryDirectory(prefix="gated-precision-export-") as folder:
        path=Path(folder)/"model.json";m.export(path)
        assert StateGatedMemoryROM.from_export(path).history_dt==.0025
        legacy=StateGatedMemoryROM(fixture(),"C",history_dt=.005);spatial(legacy);legacy.export(path)
        loaded=StateGatedMemoryROM.from_export(path)
        assert loaded.history_dt==.005 and torch.equal(loaded.initialize(h),legacy.initialize(h))
        ref=StateGatedMemoryROM(fixture(),"C",history_dt=.00125)
        with pytest.raises(ValueError,match="qualification only"): ref.export(path)


class Reports:
    def __init__(self): self.rows=[]
    def pytest_runtest_logreport(self,report):
        if report.when=="call" or report.failed:
            self.rows.append(dict(test=report.nodeid,outcome=report.outcome,error=str(report.longrepr) if report.failed else None))


if __name__=="__main__":
    import hashlib
    tick=time.time();reports=Reports();code=pytest.main([__file__,"-q","-p","no:cacheprovider"],plugins=[reports])
    for name,module in list(sys.modules.items()):
        if name.endswith("test_state_gated_memory_model_20261028"): MEASUREMENTS.update(getattr(module,"MEASUREMENTS",{}))
    path=ROOT/"results/state_gated_memory_20261028/model_pre_freeze_qa.json";path.parent.mkdir(exist_ok=True,parents=True)
    prior=json.loads(path.read_text()) if path.exists() else dict(runs=[])
    prior["runs"].append(dict(status="PASS_CPU_MATHEMATICAL_QA" if code==0 else "UNRESOLVED_CPU_MATHEMATICAL_QA",
      tests=reports.rows,measurements=MEASUREMENTS,wall_seconds=time.time()-tick,optimizer_calls=0,device="cpu",dtype="float64",
      real_event_arrays_read=False,cuda_executed=False,training_admission=False,old_legacy_drop_in_qualification="FAILED_NOT_ADOPTED",
      precision_revision=dict(history_primary=.0025,history_reference=.00125,future_primary=.01,future_reference=.005,
          reason="root adopted prospective Astra precision revision; not a pure bug repair; original .005 failure retained"),
      files={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in (Path(__file__),ROOT/"experiments/state_gated_memory_model_20261028.py")},
      unverified=["unique source real-data numerical qualification","CUDA parity","full8s cost","real Adam/RNG recovery","scientific mechanism evidence"]))
    qualified={mode:all(row["outcome"]=="passed" for row in reports.rows if row["test"].endswith("["+mode+"]"))
               and len([row for row in reports.rows if row["test"].endswith("["+mode+"]")])==3 for mode in GATE_MODES}
    shared=[row for row in reports.rows if "independent_constant_or_noncommuting_LTV_reference" not in row["test"]]
    prior.update(status=prior["runs"][-1]["status"],training_admission=False,last_run_index=len(prior["runs"])-1,
                 per_variant_qualified=qualified,shared_checks_qualified=len(shared)==8 and all(row["outcome"]=="passed" for row in shared),
                 legacy_history_005_qualification="FAILED_NUMERICAL_QUALIFICATION")
    path.write_text(json.dumps(prior,indent=2,allow_nan=False)+"\n",encoding="utf-8");raise SystemExit(code)
