"""Finite CPU float64 mathematical fixtures, zero optimizer and no event files."""
import copy
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import time
from types import SimpleNamespace

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))
from force_graph_model_20261028 import ForceGraphROM, MODES, PAIRS, RIDGE
from force_exchange_model_20261028 import ForceExchangeROM
from test_force_graph_model_20261028 import fixture, history, state, spatial, DTYPE
from force_graph_study_20261028 import objective

torch.set_num_threads(1)
MEASUREMENTS = {}


class Reports:
    def __init__(self):
        self.rows = []

    def pytest_runtest_logreport(self, report):
        if report.when == "call" or report.failed:
            self.rows.append(dict(test=report.nodeid, phase=report.when, outcome=report.outcome,
                                  error=str(report.longrepr) if report.failed else None))


def pair(mode):
    old = ForceGraphROM(fixture(), mode)
    spatial(old)
    with torch.no_grad():
        old.kappa_raw.fill_(.21)
        old.lambda_raw.fill_(-.17)
    new = ForceExchangeROM(fixture(), mode)
    new.load_state_dict({**new.state_dict(), **old.state_dict()})
    return old, new


def synthetic_store():
    # Random arithmetic fixtures, deliberately unequal available durations.
    g = torch.Generator().manual_seed(9028)
    rows = {str(i): .06 * torch.randn(n, 30, dtype=DTYPE, generator=g)
            for i, n in enumerate((15, 17, 19))}
    return SimpleNamespace(events=rows), [({"id": key}, 10) for key in rows]


@pytest.mark.parametrize("mode", MODES)
def test_zero_gamma_original_field_init_rollout_full_J_and_all_raw_gradients(mode):
    old, new = pair(mode)
    h, z = history(), state()
    assert torch.allclose(old.field(z), new.field(z), atol=1e-12, rtol=0)
    init_error = float((old.initialize(h) - new.initialize(h)).detach().abs().max())
    rollout_error = float((old.rollout(h, 4, .01) - new.rollout(h, 4, .01)).detach().abs().max())
    a, b = old.history_system(h), new.history_system(h)
    u_error = float((a["u"] - b["u"]).detach().abs().max())
    # Equal ridge penalizes the same historical force, not the terminal force.
    penalty = b["s"] @ b["F"].T + b["q"]
    penalty_error = float((penalty - a["s"]).detach().abs().max())
    residual_a = a["s"] @ a["A"].T - a["bscaled"]
    residual_b = b["s"] @ b["A"].T - b["bscaled"]
    residual_error = float((residual_a - residual_b).detach().abs().max())
    store, samples = synthetic_store()
    losses = []
    for m in (old, new):
        m.begin_graph_cache()
        J, _ = objective(m, store, samples, .08, method="rk4")
        J.backward()
        m.end_graph_cache()
        losses.append(J.detach())
    assert torch.allclose(*losses, atol=1e-12, rtol=0)
    grad_errors = {}
    for name, p in old.named_parameters():
        other = dict(new.named_parameters())[name]
        assert p.grad is not None and other.grad is not None
        assert torch.allclose(p.grad, other.grad, atol=1e-10, rtol=1e-8)
        grad_errors[name] = float((p.grad - other.grad).abs().max())
    assert torch.isfinite(new.raw_gamma.grad) and abs(float(new.raw_gamma.grad)) > 1e-10
    eps = 1e-5
    values = []
    with torch.no_grad():
        for gamma in (-eps, eps):
            new.raw_gamma.fill_(gamma)
            values.append(objective(new, store, samples, .08)[0])
        new.raw_gamma.zero_()
    fd = (values[1] - values[0]) / (2 * eps)
    assert torch.allclose(fd, new.raw_gamma.grad, atol=1e-9, rtol=1e-6)
    MEASUREMENTS[mode] = {"zero_gamma_J": float(losses[0]), "old_raw_gradient_max_error": grad_errors,
                          "gamma_gradient": float(new.raw_gamma.grad), "gamma_centered_fd": float(fd),
                          "zero_gamma_init_max_error": init_error,"zero_gamma_short_rollout_max_error":rollout_error,
                          "historical_force_max_error":u_error,"ridge_argument_max_error":penalty_error,
                          "momentum_residual_max_error":residual_error,"J_absolute_error":float(abs(losses[1]-losses[0]))}
    # Keep every original strict gate and report all decisive measurements even
    # if one floating-point comparison is unresolved.
    assert max(init_error,rollout_error,u_error,penalty_error,residual_error) <= 1e-12, MEASUREMENTS[mode]


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("gamma", (-1.4, 0., 1.7))
def test_exchange_nonunit_matrix_power_and_true_RK4(mode, gamma):
    _, m = pair(mode)
    with torch.no_grad():
        m.raw_gamma.fill_(gamma)
    z = state()
    x = m.interface_to_state(z).detach().requires_grad_()
    H, grad = m.energy_and_gradient(x)
    assert torch.allclose(torch.autograd.grad(H.sum(), x, create_graph=True)[0], grad, atol=1e-12, rtol=0)
    J, R = m.matrices()
    assert torch.equal(J + J.T, torch.zeros_like(J))
    f = m.internal_field(x)
    assert torch.allclose(f, grad @ (J - R).T, atol=1e-12, rtol=0)
    assert torch.allclose((grad * f).sum(-1), -(grad * (grad @ R.T)).sum(-1), atol=1e-12, rtol=0)
    powers = m.exchange_power(z)
    assert powers["net_power"].abs().max() < 1e-12
    if gamma:
        assert powers["p_power"].abs().max() > 1e-5
    a = m.field(z); b = m.field(z + .005 * a)
    c = m.field(z + .005 * b); d = m.field(z + .01 * c)
    assert torch.allclose(m.rk4(z, .01), z + .01 / 6 * (a + 2*b + 2*c + d), atol=1e-12, rtol=0)
    assert torch.allclose(m.internal_field(x) - torch.cat((torch.zeros_like(x[..., :10]),
        -m._potential_gradient(x[..., :10], m._context()), torch.zeros_like(x[..., 20:])), -1),
        x @ m._linear_generator(m._context()).T, atol=1e-12, rtol=0)


@pytest.mark.parametrize("gamma", (-1.3, 0., 1.6))
def test_reverse_20D_independent_fine_RK4_Simpson_signs_and_affine_ridge(gamma):
    _, m = pair("M")
    with torch.no_grad():
        m.raw_gamma.fill_(gamma)
        h = history(1)[0]
        system = m.history_system(h)
        Q = m.lambda_ * torch.eye(10, dtype=DTYPE)
        for e, (i, j) in enumerate(PAIRS):
            b = torch.zeros(10, dtype=DTYPE); b[i], b[j] = 1., -1.
            Q += m.a_bar[e] * torch.outer(b, b)
        def rhs(v, omega):
            p, r = v.split(10)
            return torch.cat((-m.jpi_p * omega - m.rpp*m.storage_base*p
                              + gamma*r/(m.kappa*m.storage_base).sqrt(),
                              m.kappa*omega - Q@r - gamma*(m.kappa*m.storage_base).sqrt()*p))
        def reverse(terminal, forcing=True):
            v = terminal.clone(); out = [v]
            for i in range(9, -1, -1):
                for j in range(50):
                    frac = j/50
                    wr, wl = h[i+1, 10:20], h[i, 10:20]
                    w = wr*(1-frac)+wl*frac
                    wm = wr*(1-frac-.01)+wl*(frac+.01)
                    we = wr*(1-frac-.02)+wl*(frac+.02)
                    if not forcing:
                        w = wm = we = torch.zeros_like(w)
                    dt = -.0002
                    a=rhs(v,w); b=rhs(v+dt*a/2,wm); c=rhs(v+dt*b/2,wm); d=rhs(v+dt*c,we)
                    v = v + dt/6*(a+2*b+2*c+d)
                out.append(v)
            return torch.stack(list(reversed(out)))
        vb = reverse(torch.cat((h[-1,20:], torch.zeros(10,dtype=DTYPE))))
        Z = torch.stack([reverse(torch.eye(20,dtype=DTYPE)[10+j], False) for j in range(10)], -1)
        assert torch.allclose(system["vbar"], vb, atol=1e-12, rtol=0), float((system["vbar"]-vb).abs().max())
        assert torch.allclose(system["Z"], Z, atol=1e-12, rtol=0)
        cnode = Z[:,10:] - (m.jpi_p*m.storage_base)[None,:,None]*Z[:,:10]
        C = torch.stack([.02/6*(cnode[i]+4*cnode[i+1]+cnode[i+2]) for i in range(0,10,2)])
        base = m.base_momentum_field(torch.cat((h[:,:20],vb[:,:10]),-1)) - vb[:,10:]
        b = torch.stack([.02/6*(base[i]+4*base[i+1]+base[i+2])-m.mass*(h[i+2,10:20]-h[i,10:20]) for i in range(0,10,2)])
        assert torch.allclose(system["C"], C, atol=1e-12, rtol=0)
        assert torch.allclose(system["b"], b, atol=1e-12, rtol=0)
        residual=system["s"]@system["A"].T-system["bscaled"]
        penalty=system["s"]@system["F"].T+system["q"]
        stationarity=residual@system["A"]+RIDGE*penalty@system["F"]
        assert stationarity.abs().max() < 1e-12
        assert system["q"].abs().max() > 1e-5
        assert torch.equal(m.initialize(h)[:30], h[-1])


@pytest.mark.parametrize("mode", MODES)
def test_causality_machine_permutation_full_gradients_cache_and_exports(mode):
    _, m = pair(mode)
    with torch.no_grad(): m.raw_gamma.fill_(-.7)
    h=history(); z=state()
    polluted=h.clone(); polluted[:,:10,20:]=float("nan")
    assert torch.equal(m.encode(h),m.encode(polluted))
    perm=torch.tensor([7,0,9,4,2,6,1,8,3,5])
    cfg=copy.deepcopy(m._source_config)
    for key,value in list(cfg.items()):
        if key=="D": cfg[key]=torch.tensor(value)[:,perm].tolist()
        elif len(value)==10: cfg[key]=torch.tensor(value,dtype=DTYPE)[perm].tolist()
    p=ForceExchangeROM(cfg,mode)
    with torch.no_grad():
        p.kappa_raw.copy_(m.kappa_raw); p.lambda_raw.copy_(m.lambda_raw); p.raw_gamma.copy_(m.raw_gamma)
        if mode!="L":
            source=m.k_bar if mode=="S" else m.a_bar
            target=p.k_bar if mode=="S" else p.a_bar
            for e,(i,j) in enumerate(PAIRS): target[e]=source[PAIRS.index(tuple(sorted((int(perm[i]),int(perm[j])))))]
    ix=torch.cat(tuple(perm+offset for offset in (0,10,20,30)))
    assert torch.allclose(p.field(z[:,ix]),m.field(z)[:,ix],atol=1e-12,rtol=0)
    assert torch.allclose(p.encode(h[:,:,ix[:30]]),m.encode(h)[:,perm],atol=1e-12,rtol=0)
    # F and q must participate in backward, not merely in the forward solve.
    s=m.history_system(h)
    grads=torch.autograd.grad(s["r0"].square().sum(),(s["F"],s["q"]),retain_graph=True)
    assert all(torch.isfinite(g).all() and g.abs().max()>1e-12 for g in grads)
    m.rollout(h,3,.01).square().sum().backward()
    assert all(v.grad is not None and torch.isfinite(v.grad).all() for v in m.parameters())
    with torch.inference_mode():
        m.begin_graph_cache(); m.encode(h)
    m.encode(h).square().sum().backward(); assert m._cache is None
    for name, param in m.named_parameters():
        m.begin_graph_cache(); m.encode(h)
        with torch.no_grad(): param.add_(.001)
        m.encode(h); assert m._cache is None
    m.begin_graph_cache(); m.train(); assert m._cache is None
    m.begin_graph_cache(); m.cpu(); assert m._cache is None
    m.begin_graph_cache(); m.load_state_dict(m.state_dict()); assert m._cache is None
    with tempfile.TemporaryDirectory(prefix="exchange-fixture-") as folder:
        path=Path(folder)/"model.json"
        for train_gamma in (False, True):
            m.raw_gamma.requires_grad_(train_gamma)
            m.export(path); loaded=ForceExchangeROM.from_export(path)
            assert loaded.raw_gamma.requires_grad==train_gamma
            assert loaded.source_identity==m.source_identity
            assert all(torch.equal(v,loaded.state_dict()[k]) for k,v in m.state_dict().items())
            assert torch.equal(m.encode(h),loaded.encode(h))
            assert torch.equal(m.field(z),loaded.field(z))
        old,_=pair(mode); old.export(path)
        imported=ForceExchangeROM.from_force_graph_export(path,train_gamma=False)
        assert imported.source_identity["force_graph_export_sha256"]==hashlib.sha256(path.read_bytes()).hexdigest()
        assert torch.allclose(old.encode(h),imported.encode(h),atol=1e-12,rtol=0)
        assert set(n for n,p in imported.named_parameters() if p.requires_grad)==set(dict(old.named_parameters()))


if __name__ == "__main__":
    start=time.time()
    reports=Reports()
    code=pytest.main([__file__, "-q", "-p", "no:cacheprovider"],plugins=[reports])
    for module_name, module in list(sys.modules.items()):
        if module_name.endswith("test_force_exchange_model_20261028"):
            MEASUREMENTS.update(getattr(module, "MEASUREMENTS", {}))
    output=ROOT/"results/force_exchange_20261028/model_pre_freeze_qa.json"
    output.parent.mkdir(parents=True,exist_ok=True)
    prior=json.loads(output.read_text()) if output.exists() else {"runs":[]}
    prior["runs"].append(dict(exit_code=code,wall_seconds=time.time()-start, measurements=MEASUREMENTS,tests=reports.rows,
        formulation="unit reverse interval; delta_omega forcing; augmented QR solve",
        cpu=True,dtype="float64",optimizer_calls=0,real_event_arrays_read=False,val_read=False,test_read=False,
        source_selection=None,files={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in
        (Path(__file__),ROOT/"experiments/force_exchange_model_20261028.py",ROOT/"experiments/force_graph_model_20261028.py")},
        limitations=["CUDA parity, real-source qualification and optimizer/RNG recovery are not tested here",
                     "No full-horizon real-data precision or cost claim; original objective used on short synthetic data"]))
    prior["status"]="PASSED_CPU_FIXTURE_QA" if code==0 else "NUMERICAL_UNRESOLVED"
    prior["training_admission"]=False
    output.write_text(json.dumps(prior,indent=2,allow_nan=False)+"\n")
    raise SystemExit(code)
