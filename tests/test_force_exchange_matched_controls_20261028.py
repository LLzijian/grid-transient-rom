"""Prospective matched-control CPU qualification; never a legacy drop-in gate.

No optimizer, real event/cache reader, source choice, CUDA, or training admission.
Legacy module, tests and all five QA runs are immutable input evidence.
"""
import hashlib
import json
import math
from pathlib import Path
import sys
import time

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))
from force_exchange_model_20261028 import ForceExchangeROM
from force_graph_model_20261028 import MODES, PAIRS, RIDGE
from force_graph_study_20261028 import objective
from test_force_graph_model_20261028 import fixture, history, state, spatial, DTYPE
from test_force_exchange_model_20261028 import synthetic_store

torch.set_num_threads(1)
MEASUREMENTS = {}
IMMUTABLE = {
    "experiments/force_exchange_model_20261028.py": "010ff2210fb84767fa6a5dee4cfe38826ea8526787a277e09db08e5c7b02d1b4",
    "experiments/force_graph_model_20261028.py": "44e5fe2e1403bc7e072d2fb87100eaa694b15da5b4694169de141cc6059089af",
    "tests/test_force_exchange_model_20261028.py": "671c94b5c54478e863b9597f85fc758c3ffa49b529adb7f10fe609c08a39f6ae",
    "results/force_exchange_20261028/model_pre_freeze_qa.json": "f947ac00d1dfc7cfdc46f8fde15069a52da84abcd7470c8fa25b909c5c0137b4",
    "reports/force_exchange_20261028_numerical_decision.md": "67ba846f9e9e95d5753c46278b69ad7ccb6dd623fc80346f5edac8c834194ee2",
}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def matched(mode):
    # Independent construction, equal arithmetic fixture and theta; no chosen F.
    models = [ForceExchangeROM(fixture(), mode, train_gamma=trainable,
              source_identity={"scope": "mathematical fixture, not scientific source selection"})
              for trainable in (False, True)]
    for model in models:
        spatial(model)
        with torch.no_grad():
            model.kappa_raw.fill_(.21)
            model.lambda_raw.fill_(-.17)
    return models


def error(a, b):
    return float((a-b).detach().abs().max())


@pytest.mark.parametrize("mode", MODES)
def test_independent_C0_C1_zero_gamma_match_complete_J_and_old_raw_gradients(mode):
    c0, c1 = matched(mode)
    assert not c0.raw_gamma.requires_grad and c1.raw_gamma.requires_grad
    assert c0.gamma.item() == c1.gamma.item() == 0.
    assert all(torch.equal(v, c1.state_dict()[name]) for name, v in c0.state_dict().items())
    old_names = set(dict(c0.named_parameters())) - {"raw_gamma"}
    assert {name for name,p in c0.named_parameters() if p.requires_grad} == old_names
    assert {name for name,p in c1.named_parameters() if p.requires_grad} == old_names | {"raw_gamma"}
    h,z = history(),state()
    values = {}
    for name, a, b in (
        ("field", c0.field(z), c1.field(z)),
        ("initialize", c0.initialize(h), c1.initialize(h)),
        ("short_RK4_trajectory", c0.rollout(h,4,.01), c1.rollout(h,4,.01))):
        assert torch.allclose(a,b,atol=1e-12,rtol=1e-12)
        values[name+"_max_absolute_difference"] = error(a,b)
    store,samples=synthetic_store()
    losses=[]
    for model in (c0,c1):
        model.begin_graph_cache()
        J,_=objective(model,store,samples,.08,method="rk4")
        J.backward()
        model.end_graph_cache()
        losses.append(J.detach())
    assert torch.allclose(*losses,atol=1e-12,rtol=1e-12)
    values.update(J=float(losses[0]),J_absolute_difference=error(*losses))
    grads={}
    for name in sorted(old_names):
        a=dict(c0.named_parameters())[name].grad
        b=dict(c1.named_parameters())[name].grad
        assert a is not None and b is not None
        assert torch.isfinite(a).all() and torch.isfinite(b).all()
        assert torch.allclose(a,b,atol=1e-12,rtol=1e-12)
        grads[name]=error(a,b)
    assert c0.raw_gamma.grad is None
    assert c1.raw_gamma.grad is not None and torch.isfinite(c1.raw_gamma.grad)
    prior=json.loads((ROOT/"results/force_exchange_20261028/model_pre_freeze_qa.json").read_text())
    recorded=prior["runs"][-1]["measurements"][mode]
    assert math.isclose(float(c1.raw_gamma.grad),recorded["gamma_gradient"],abs_tol=1e-12,rel_tol=1e-12)
    assert math.isclose(float(c1.raw_gamma.grad),recorded["gamma_centered_fd"],abs_tol=1e-9,rel_tol=1e-6)
    values.update(old_raw_gradient_max_absolute_differences=grads,
                  gamma_gradient=float(c1.raw_gamma.grad),reused_centered_difference=recorded["gamma_centered_fd"])
    MEASUREMENTS["matched_"+mode]=values


@pytest.mark.parametrize("mode", MODES)
def test_zero_gamma_independent_reverse_flow_and_physical_initial_state(mode):
    model,_=matched(mode)
    h=history(1)[0]
    with torch.no_grad():
        actual=model.history_system(h)
        # Independent physical RHS, built from pair incidence rather than the
        # implementation's T/B/flow/Z or augmented matrix exponential.
        Q=model.lambda_*torch.eye(10,dtype=DTYPE)
        if mode=="M":
            for e,(i,j) in enumerate(PAIRS):
                b=torch.zeros(10,dtype=DTYPE);b[i],b[j]=1.,-1.
                Q=Q+model.a_bar[e]*torch.outer(b,b)
        def rhs(v,omega):
            p,r=v[:,:10],v[:,10:]
            return torch.cat((-model.jpi_p*omega-model.rpp*model.storage_base*p,
                              model.kappa*omega-r@Q.T),-1)
        # Case 0 carries p0/forcing; ten other cases carry unit terminal r.
        v=torch.zeros(11,20,dtype=DTYPE)
        v[0,:10]=h[-1,20:30]
        v[1:,10:]=torch.eye(10,dtype=DTYPE)
        outputs=[v]
        dt=-.0002
        for i in range(9,-1,-1):
            wr,wl=h[i+1,10:20],h[i,10:20]
            for j in range(50):
                w=[]
                for frac in (j/50,(j+.5)/50,(j+1)/50):
                    forcing=torch.zeros(11,10,dtype=DTYPE)
                    forcing[0]=wr*(1-frac)+wl*frac
                    w.append(forcing)
                a=rhs(v,w[0]);b=rhs(v+dt*a/2,w[1]);c=rhs(v+dt*b/2,w[1]);d=rhs(v+dt*c,w[2])
                v=v+dt/6*(a+2*b+2*c+d)
            outputs.append(v)
        reference=torch.stack(list(reversed(outputs)))
        vb=reference[:,0]
        Z=reference[:,1:].transpose(-1,-2)
        assert torch.allclose(actual["vbar"],vb,atol=1e-12,rtol=0)
        assert torch.allclose(actual["Z"],Z,atol=1e-12,rtol=0)
        nodes=Z[:,10:]-(model.jpi_p*model.storage_base)[None,:,None]*Z[:,:10]
        C=torch.stack([.02/6*(nodes[i]+4*nodes[i+1]+nodes[i+2]) for i in range(0,10,2)])
        # Independent nonlinear physical momentum force at fine-reference nodes.
        force=torch.zeros(11,10,dtype=DTYPE)
        for e,(i,j) in enumerate(PAIRS):
            coupling=model.kappa_reference*model.k_bar[e] if mode=="S" else model.reference_coupling45[e]
            phi=model.delta_star[i]-model.delta_star[j]
            edge=coupling*(torch.sin(phi+h[:,i]-h[:,j])-torch.sin(phi))
            force[:,i]+=edge;force[:,j]-=edge
        momentum=-force-model.damping*h[:,10:20]+model.jpi_p*model.storage_base*vb[:,:10]-vb[:,10:]
        b=torch.stack([.02/6*(momentum[i]+4*momentum[i+1]+momentum[i+2])
                       -model.mass*(h[i+2,10:20]-h[i,10:20]) for i in range(0,10,2)])
        A=(C*model.rstar[None,None,:]/(.02*model.rstar[None,:,None])).reshape(50,10)
        beta=(b/(.02*model.rstar)).flatten()
        F=Z[0,10:]*model.rstar[None,:]/model.rstar[:,None]
        q=vb[0,10:]/model.rstar
        design=torch.cat((A,math.sqrt(RIDGE)*F))
        target=torch.cat((beta,-math.sqrt(RIDGE)*q))
        # Independent NumPy SVD least squares, not implementation QR/solve.
        s=torch.from_numpy(np.linalg.lstsq(design.numpy(),target.numpy(),rcond=None)[0])
        r0=model.rstar*s
        estimated=vb+torch.einsum("tij,j->ti",Z,r0)
        expected_init=torch.cat((h[-1],r0))
        assert torch.allclose(model.initialize(h),expected_init,atol=1e-12,rtol=0)
        assert torch.equal(model.initialize(h)[:30],h[-1])
        for name,observed,expected in (
            ("C",actual["C"],C),("b",actual["b"],b),("F",actual["F"],F),("q",actual["q"],q),
            ("p_history",actual["p_history"],estimated[:,:10]),
            ("r_history",actual["r_history"],estimated[:,10:])):
            assert torch.allclose(observed,expected,atol=1e-12,rtol=0),name
        residual=actual["s"]@actual["A"].T-actual["bscaled"]
        penalty=actual["s"]@actual["F"].T+actual["q"]
        stationarity=residual@actual["A"]+RIDGE*penalty@actual["F"]
        assert stationarity.abs().max()<1e-12
        MEASUREMENTS["independent_zero_"+mode]={
            "vbar_max_absolute_error":error(actual["vbar"],vb),"Z_max_absolute_error":error(actual["Z"],Z),
            "physical_initializer_max_absolute_error":error(model.initialize(h),expected_init),
            "momentum_integral_b_max_absolute_error":error(actual["b"],b),
            "ridge_F_max_absolute_error":error(actual["F"],F),"ridge_q_max_absolute_error":error(actual["q"],q),
            "stationarity_max_absolute_residual":float(stationarity.abs().max()),
            "reference":"independent physical backward RK4 dt=-0.0002, 50 steps per interval; NumPy SVD least squares"}


def test_frozen_legacy_evidence_unchanged():
    for path,digest in IMMUTABLE.items():
        assert sha(ROOT/path)==digest,path
    prior=json.loads((ROOT/"results/force_exchange_20261028/model_pre_freeze_qa.json").read_text())
    assert prior["status"]=="NUMERICAL_UNRESOLVED" and prior["training_admission"] is False
    final=prior["runs"][-1]
    assert final["observed_passed"]==12 and final["observed_failed"]==6
    assert len([t for t in final["tests"] if t["outcome"]=="passed"])==12


class Reports:
    def __init__(self): self.rows=[]
    def pytest_runtest_logreport(self,report):
        if report.when=="call" or report.failed:
            self.rows.append(dict(test=report.nodeid,outcome=report.outcome,
                                  error=str(report.longrepr) if report.failed else None))


if __name__=="__main__":
    start=time.time()
    before={path:sha(ROOT/path) for path in IMMUTABLE}
    reports=Reports()
    code=pytest.main([__file__,"-q","-p","no:cacheprovider"],plugins=[reports])
    for name,module in list(sys.modules.items()):
        if name.endswith("test_force_exchange_matched_controls_20261028"):
            MEASUREMENTS.update(getattr(module,"MEASUREMENTS",{}))
    after={path:sha(ROOT/path) for path in IMMUTABLE}
    immutable_ok=before==after==IMMUTABLE
    passed=code==0 and immutable_ok
    old_qa=ROOT/"results/force_exchange_20261028/model_pre_freeze_qa.json"
    old_record=json.loads(old_qa.read_text())
    output=ROOT/"results/force_exchange_20261028/matched_control_cpu_qualification.json"
    result=dict(status="PASS_FOR_MATCHED_CONTROLS" if passed else "FAILED_MATCHED_CONTROL_CPU_QUALIFICATION",
        exit_code=code,wall_seconds=time.time()-start,tests=reports.rows,measurements=MEASUREMENTS,
        old_legacy_drop_in_qualification="FAILED_NOT_ADOPTED",
        old_qa=dict(path=str(old_qa),sha256=sha(old_qa),status_unchanged=old_record["status"],
                    final_original_test_count=18,final_original_passed=12,final_original_failed=6),
        immutable_before=before,immutable_after=after,immutable_verified=immutable_ok,
        test_sha256=sha(__file__),dtype="float64",device="cpu",optimizer_calls=0,
        real_event_arrays_read=False,val_read=False,test_read=False,scientific_source_selection=None,
        tolerances=dict(matched_atol=1e-12,matched_rtol=1e-12,independent_atol=1e-12,independent_rtol=0),
        training_admission=False,cuda_qualification_completed=False,real_source_qualification_completed=False,
        adam_rng_recovery_completed=False,scientific_training_admission_completed=False,
        preserved_original_passed_checks=[row for row in old_record["runs"][-1]["tests"] if row["outcome"]=="passed"],
        reused_gamma_centered_difference=dict(qa_path=str(old_qa),qa_sha256=sha(old_qa),
            scope="same unchanged source module, mathematical fixtures and full original J; no finite-difference rerun"),
        comparison_scope="F keeps frozen parent initializer; C0/C1 share joint initializer, primary scientific comparison is common96; C0(0)-F separate",
        unverified=["CUDA","unique scientific source","F versus C0(0) real fit numerical qualification",
                    "Adam and full RNG recovery","real scientific training, precision and cost"])
    if output.exists():
        previous=json.loads(output.read_text())
        result["previous_attempts"]=previous.get("previous_attempts",[])+[{k:v for k,v in previous.items() if k!="previous_attempts"}]
    output.write_text(json.dumps(result,indent=2,allow_nan=False)+"\n",encoding="utf-8")
    raise SystemExit(0 if passed else 1)
