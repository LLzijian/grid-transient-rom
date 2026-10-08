"""Loop0082: one frozen 24-amplitude conditional ridge fit and transfer check."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import shutil
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "experiments"), str(ROOT / "src")]
from audit_force_initialization_0080 import METRIC, numeric, point, risk  # noqa: E402
from audit_conditional_latent_0081 import EXPECTED_ENDPOINTS, EXPECTED_WINDOWS, windows  # noqa: E402
from train_joint_latent import BASE, DT, NETWORK, ROOTS, read_development  # noqa: E402
from train_structured_rom import _h_field, load_model  # noqa: E402
from structured_rom import StructuredROM, numpy_field  # noqa: E402
from kiki.models.private_pm import MechanicalROM  # noqa: E402

CHECKPOINT = ROOT / "private_data/task2_structure_0078_extend_G-fixed/step3000"
FORCE = ROOT / "private_data/task2_structure_force_0079"
BASELINE = ROOT / "private_data/task2_force_initialization_0080"
FINE, COARSE, RIDGE = .005, .01, 1e-4
CONDITIONAL_SPEED_SCALE = 1.003367
AUTONOMOUS_METRIC = (.47675314588683515, 1.0033667249602924)
if METRIC != AUTONOMOUS_METRIC: raise AssertionError("inherited autonomous metric changed")


def sha(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def hashes(checkpoint: Path, force: Path, baseline: Path) -> dict[str, str]:
    paths = {"checkpoint_model": checkpoint / "model.pt", "checkpoint_meta": checkpoint / "checkpoint.json",
             "normalization": checkpoint / "normalization.npz", "export": checkpoint / "structured_numeric_export.json",
             "force_arrays": force / "projection_arrays.npz", "force_evidence": force / "evidence.json",
             "baseline_evidence": baseline / "evidence.json", "baseline_trajectories": baseline / "trajectories.npz",
             "base": Path(BASE), "network": Path(NETWORK), "code": Path(__file__),
             "trainer": ROOT / "experiments/train_structured_rom.py", "core": ROOT / "experiments/structured_rom.py",
             "grammar": ROOT / "experiments/structure_grammar.py", "reader": ROOT / "experiments/train_joint_latent.py",
             "audit0080_helper": ROOT / "experiments/audit_force_initialization_0080.py",
             "audit0081_helper": ROOT / "experiments/audit_conditional_latent_0081.py"}
    for root in ROOTS:
        paths[f"{root.name}_csv"] = root / "dataset_index.csv"; paths[f"{root.name}_splits"] = root / "splits.json"
    source = Path(json.loads((checkpoint / "checkpoint.json").read_text()).get("source", ""))
    if source.exists():
        paths["source_model"] = source / "model.pt"; paths["source_normalization"] = source / "normalization.npz"
    return {name: sha(path) for name, path in paths.items()}


def _network() -> dict[str, np.ndarray]:
    net = dict(np.load(NETWORK, allow_pickle=False)); buses = np.unique(np.r_[net["load_buses"], net["generator_buses"]])
    net["observed_buses"] = buses; net["load_positions"] = np.searchsorted(buses, net["load_buses"]); return net


def deterministic_test_z() -> torch.Tensor:
    """Model-only fixed states for coefficient and structural QA, never TRAIN rows."""
    first=np.r_[np.linspace(-.04,.05,10),np.linspace(.03,-.02,10),np.linspace(-.07,.08,10),(.11,-.17,.23)]
    second=np.r_[np.linspace(.02,-.03,10),np.linspace(-.04,.05,10),np.linspace(.06,-.05,10),(-.19,.13,-.07)]
    return torch.as_tensor(np.stack((first,second)),dtype=torch.float64)


def _coi(value: np.ndarray, mass: np.ndarray) -> np.ndarray:
    return value - (value * mass).sum(-1, keepdims=True) / mass.sum()


def weighted_residual(residual: np.ndarray, L: np.ndarray, mass: np.ndarray, p_scale: np.ndarray,
                      force_scale: np.ndarray) -> np.ndarray:
    """The frozen 31-row W_b metric; residual is an interface-field average."""
    if residual.shape[-1] != 33 or L.shape[-2:] != (10, 3): raise ValueError("0082 residual/map shape")
    omega, p, q = residual[..., 10:20], residual[..., 20:30], residual[..., 30:]
    coi = np.sqrt(.5 / 10.) * _coi(omega, mass) / CONDITIONAL_SPEED_SCALE
    common = ((omega * mass).sum(-1, keepdims=True) / mass.sum()) / CONDITIONAL_SPEED_SCALE
    pm = p / (np.sqrt(10.) * p_scale)
    latent = np.sqrt(.1 / 10.) * np.einsum("...ij,...j->...i", L, q) / force_scale
    answer = np.concatenate((coi, common, pm, latent), -1)
    if not np.isfinite(answer).all(): raise FloatingPointError("nonfinite weighted residual")
    return answer


def weighted_columns(columns: np.ndarray, L: np.ndarray, mass: np.ndarray, p_scale: np.ndarray,
                     force_scale: np.ndarray) -> np.ndarray:
    """Apply the same frozen W_b row scaling to 24 field columns."""
    if columns.shape[-2:] != (33, 24) or L.shape[-2:] != (10, 3): raise ValueError("0082 column/map shape")
    omega, p, q = columns[:, 10:20], columns[:, 20:30], columns[:, 30:]
    coi = np.sqrt(.5 / 10.) * _coi(np.moveaxis(omega, 1, -1), mass).transpose(0, 2, 1) / CONDITIONAL_SPEED_SCALE
    common = ((omega * mass[None, :, None]).sum(1, keepdims=True) / mass.sum()) / CONDITIONAL_SPEED_SCALE
    pm = p / (np.sqrt(10.) * p_scale[None, :, None])
    latent = np.sqrt(.1 / 10.) * np.einsum("nij,njk->nik", L, q) / force_scale
    answer = np.concatenate((coi, common, pm, latent), 1)
    if answer.shape[1:] != (31, 24) or not np.isfinite(answer).all(): raise FloatingPointError("nonfinite weighted columns")
    return answer


def solve_ridge(c: np.ndarray, A: np.ndarray, weights: np.ndarray) -> dict[str, np.ndarray | float]:
    """The single preregistered beta-space ridge solve, including inactive columns."""
    c, A, weights = np.asarray(c, float), np.asarray(A, float), np.asarray(weights, float)
    if c.ndim != 2 or A.shape != (len(c), c.shape[1], 24) or weights.shape != (len(c),): raise ValueError("ridge shape")
    if not (np.isfinite(c).all() and np.isfinite(A).all() and np.isfinite(weights).all() and np.all(weights > 0)):
        raise FloatingPointError("nonfinite ridge input")
    if abs(weights.sum() - 1.) > 1e-12: raise AssertionError("window weights must sum one")
    scales = np.sqrt(np.einsum("w,wkj,wkj->j", weights, A, A)); threshold = 1e-12 * float(scales.max(initial=0.))
    active = scales > threshold; beta = np.zeros(24); delta = np.zeros(24)
    B=np.zeros_like(A)
    if active.any():
        B[:, :, active] = A[:, :, active] / scales[active][None, None, :]
        K = np.einsum("w,wki,wkj->ij", weights, B, B) + RIDGE * np.eye(24)
        g = -np.einsum("w,wki,wk->i", weights, B, c)
        beta = np.linalg.solve(K, g); delta[active] = beta[active] / scales[active]
        stationarity = np.linalg.norm(K @ beta - g) / max(1., np.linalg.norm(g))
        singular = np.linalg.svd(B[:, :, active].reshape(-1, active.sum()) * np.repeat(np.sqrt(weights), c.shape[1])[:, None], compute_uv=False)
        condition = float(np.linalg.cond(K))
    else:
        K, g, stationarity, singular, condition = RIDGE*np.eye(24), np.zeros(24), 0., np.zeros(0), 1.
    if not np.isfinite(stationarity) or stationarity > 1e-9: raise AssertionError("ridge stationarity")
    return {"theta_delta": delta, "beta": beta, "scales": scales, "active": active,
            "stationarity_relative": float(stationarity), "design_singular_values": singular,
            "ridge_condition": condition, "K":K, "g":g}


def require_time(deadline: float, stage: str) -> None:
    if time.perf_counter() > deadline: raise TimeoutError(f"0082 {stage} wall cap")


def simpson_rule(fine: bool) -> tuple[tuple[float, ...], tuple[float, ...], float]:
    # Two .005-wide Simpson panels have substep h=.0025, hence h/3=.005/6.
    return ((0., .0025, .005, .0075, .01), (1., 4., 2., 4., 1.), .005 / 6.) if fine else ((0., .005, .01), (1., 4., 1.), .01 / 6.)


def event_weights(event_index: np.ndarray) -> np.ndarray:
    counts=np.bincount(np.asarray(event_index, dtype=int), minlength=346)
    if np.any(counts == 0): raise AssertionError("each TRAIN event requires a window")
    result=1. / (346. * counts[event_index])
    if not np.isclose(result.sum(),1.): raise AssertionError("event weights")
    return result


def _linear(segment: torch.Tensor, duration: torch.Tensor, t: float) -> torch.Tensor:
    step = torch.floor(torch.full_like(duration, t / DT + 1e-12, dtype=torch.float64)).long()
    step = torch.minimum(step, duration - 1); fraction = ((t - step.double() * DT) / DT).clamp(0., 1.)
    row = torch.arange(len(step), device=segment.device); return segment[row, step] + fraction[:, None] * (segment[row, step + 1] - segment[row, step])


def _term_value(model, x: torch.Tensor, port: int, term: int) -> torch.Tensor:
    rom, spec = model.structured, model.structured.port_specs[port]["terms"][term]
    eta = x[:, :rom.n] @ rom.D.T; v = torch.cat((x[:, rom.n:2 * rom.n], x[:, 2 * rom.n:]), -1)
    kind = spec.get("kind", "linear").lower(); index = int(spec.get("index", spec.get("indices", (0,))[0])); scale = float(spec.get("scale", 1.))
    if kind == "linear": return v[:, index] / scale
    if kind in ("sin", "sin_difference"): return torch.sin(rom.phi[index] + eta[:, index]) - torch.sin(rom.phi[index])
    if kind in ("cos", "cos_difference"): return torch.cos(rom.phi[index] + eta[:, index]) - torch.cos(rom.phi[index])
    raise ValueError("unknown frozen port basis")


def coefficient_fields(model, z: torch.Tensor, J: torch.Tensor, R: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """24 exact interface fields per unit signed amplitude: 8 H then 16 port terms."""
    rom, x = model.structured, model.structured.interface_to_state(z); fields = []
    if len(rom.storage_gamma) != 8 or len(rom.port_coefficient) != 2 or any(len(x) != 8 for x in rom.port_coefficient):
        raise AssertionError("frozen 8H/16port support required")
    for spec in rom.storage_specs:
        fields.append(rom.state_to_interface(_h_field(model, x, spec, J, R)))
    for port in range(2):
        for term in range(8):
            fields.append(rom.state_to_interface(_term_value(model, x, port, term)[:, None] * B[:, port]))
    result = torch.stack(fields, -1)
    if result.shape[-2:] != (33, 24) or not torch.isfinite(result).all(): raise FloatingPointError("nonfinite coefficient field")
    return result


def _integrals(model, xseg: torch.Tensor, qseg: torch.Tensor, duration: torch.Tensor, step: float) -> tuple[np.ndarray, np.ndarray]:
    """Fine composite Simpson (.005 panels) or coarse one-panel Simpson (.01)."""
    fine = step == FINE; base = torch.zeros((len(duration), 33), dtype=torch.float64); cols = torch.zeros((len(duration), 33, 24), dtype=torch.float64)
    cached = getattr(model, "_graph_cache", None)
    J, R, B = cached if cached is not None else (*model.structured.matrices(), model.structured.port_matrix())
    for interval in range(int(duration.max())):
        rows = torch.nonzero(duration > interval, as_tuple=False).ravel()
        if not len(rows): break
        xx, qq, dd = xseg[rows], qseg[rows], duration[rows]
        times, weights, factor = simpson_rule(fine)
        for local, weight in zip(times, weights, strict=True):
            z = torch.cat((_linear(xx, dd, interval * DT + local), _linear(qq, dd, interval * DT + local)), -1)
            value = model.field(z); direction = coefficient_fields(model, z, J, R, B)
            if not (torch.isfinite(z).all() and torch.isfinite(value).all() and torch.isfinite(direction).all()): raise FloatingPointError("nonfinite Simpson stage")
            base[rows] += factor * weight * value; cols[rows] += factor * weight * direction
    return base.detach().cpu().numpy(), cols.detach().cpu().numpy()


def _table(train: list[dict[str, Any]], arrays, evidence: dict[str, Any]) -> dict[str, Any]:
    if evidence["train_event_ids"] != [e["id"] for e in train]: raise AssertionError("0079 order")
    idx, ep, qe, L = arrays["G-fixed_event_index"], arrays["G-fixed_endpoint"], arrays["G-fixed_qenc"], arrays["G-fixed_L"]
    lookup = {(int(i), int(t)): row for row, (i, t) in enumerate(zip(idx, ep, strict=True))}
    if len(lookup) != EXPECTED_ENDPOINTS: raise AssertionError("0079 endpoint uniqueness")
    rows=[]
    for i, event in enumerate(train):
        expected = np.arange(10, len(event["x"])); got = np.sort(ep[idx == i])
        if not np.array_equal(got, expected): raise AssertionError("0079 full endpoint map")
        for a, b in windows(len(event["x"])):
            q = np.stack([qe[lookup[i, t]] for t in range(a, b + 1)])
            rows.append((i, a, b, event["x"][a:b + 1], q, L[lookup[i, b]], event["id"], event["label"], str(event["group"])))
    if len(rows) != EXPECTED_WINDOWS: raise AssertionError("window count")
    return {"event_index": np.array([r[0] for r in rows]), "start": np.array([r[1] for r in rows]), "end": np.array([r[2] for r in rows]),
            "x": [r[3] for r in rows], "q": [r[4] for r in rows], "L": np.stack([r[5] for r in rows]),
            "event_id": [r[6] for r in rows], "label": [r[7] for r in rows], "group": [r[8] for r in rows]}


def _design(model, table: dict[str, Any], deadline: float = float("inf"), progress: dict[str, Any] | None = None) -> dict[str, np.ndarray]:
    n = len(table["start"]); c=np.empty((n,31)); A=np.empty((n,31,24)); cf=np.empty_like(c); Af=np.empty_like(A)
    mass=model.mass.detach().cpu().numpy(); pscale=model.p_scale.detach().cpu().numpy(); force=model.force_scale.detach().cpu().numpy()
    weights=event_weights(table["event_index"])
    with torch.no_grad(): model.begin_graph_cache()
    try:
        with torch.no_grad():
            for low in range(0,n,256):
                require_time(deadline,"fit/design")
                high=min(low+256,n); duration=torch.as_tensor(table["end"][low:high]-table["start"][low:high], dtype=torch.long); width=int(duration.max())+1
                if progress is not None: progress.update({"stage":"design","batch":[low,high]})
                x=np.zeros((high-low,width,30)); q=np.zeros((high-low,width,3)); delta=np.zeros((high-low,33))
                for j,(xx,qq) in enumerate(zip(table["x"][low:high],table["q"][low:high],strict=True)):
                    x[j,:len(xx)]=xx; q[j,:len(qq)]=qq; delta[j]=np.r_[xx[-1]-xx[0],qq[-1]-qq[0]]
                fb, fA = _integrals(model,torch.as_tensor(x,dtype=torch.float64),torch.as_tensor(q,dtype=torch.float64),duration,FINE)
                cb, cA = _integrals(model,torch.as_tensor(x,dtype=torch.float64),torch.as_tensor(q,dtype=torch.float64),duration,COARSE)
                tau=duration.numpy()[:,None]*DT; L=table["L"][low:high]
                c[low:high]=weighted_residual((fb-delta)/tau,L,mass,pscale,force); A[low:high]=weighted_columns(fA/tau[:,:,None],L,mass,pscale,force)
                cf[low:high]=weighted_residual((cb-delta)/tau,L,mass,pscale,force); Af[low:high]=weighted_columns(cA/tau[:,:,None],L,mass,pscale,force)
    finally: model.end_graph_cache()
    return {"c":c,"A":A,"weights":weights,"coarse_c":cf,"coarse_A":Af}


def _theta(model) -> np.ndarray:
    return np.r_[[float(x.detach()) for x in model.structured.storage_gamma], *[x.detach().cpu().numpy() for x in model.structured.port_coefficient]]


def set_theta(model, theta: np.ndarray) -> None:
    if theta.shape != (24,) or not np.isfinite(theta).all(): raise ValueError("theta")
    with torch.no_grad():
        for i in range(8): model.structured.storage_gamma[i].copy_(torch.tensor(theta[i]))
        model.structured.port_coefficient[0].copy_(torch.tensor(theta[8:16])); model.structured.port_coefficient[1].copy_(torch.tensor(theta[16:24]))


def _checked_rollout(model, states: torch.Tensor, lengths: np.ndarray, step: float, *, deadline: float = float("inf"), progress: dict[str, Any] | None = None, partial: Path | None = None, event_ids: np.ndarray | None = None) -> np.ndarray:
    out=np.full((len(states),int(lengths.max())+1,33),np.nan); out[:,0]=states.numpy(); current=states.clone()
    with torch.no_grad():
        model.begin_graph_cache()
        try:
            for k in range(1,out.shape[1]):
                require_time(deadline,"forecast")
                ix=np.flatnonzero(lengths>=k)
                if not len(ix): break
                if progress is not None: progress.update({"stage":"forecast","rk_step":k,"active_events":ix.tolist()})
                where=torch.as_tensor(ix); z=current.index_select(0,where)
                if not torch.isfinite(z).all(): raise FloatingPointError(f"nonfinite RK4 input {k}")
                a=model.field(z); zb=z+step*a/2
                if not (torch.isfinite(a).all() and torch.isfinite(zb).all()): raise FloatingPointError(f"nonfinite RK4 a/b input {k}")
                b=model.field(zb); zc=z+step*b/2
                if not (torch.isfinite(b).all() and torch.isfinite(zc).all()): raise FloatingPointError(f"nonfinite RK4 b/c input {k}")
                c=model.field(zc); zd=z+step*c
                if not (torch.isfinite(c).all() and torch.isfinite(zd).all()): raise FloatingPointError(f"nonfinite RK4 c/d input {k}")
                d=model.field(zd)
                nxt=z+step*(a+2*b+2*c+d)/6
                if not all(torch.isfinite(v).all() for v in (z,a,b,c,d,nxt)): raise FloatingPointError(f"nonfinite RK4 stage {k}")
                current=current.index_copy(0,where,nxt); out[ix,k]=nxt.numpy()
        except Exception:
            if partial is not None: np.savez_compressed(partial,partial_forecast=out,lengths=lengths,current_state=current.detach().cpu().numpy(),event_ids=np.asarray(event_ids if event_ids is not None else np.arange(len(lengths))))
            raise
        finally: model.end_graph_cache()
    return out


def _changed(before: dict[str, torch.Tensor], model) -> list[str]:
    return [key for key,value in model.state_dict().items() if not torch.equal(value,before[key])]


def affinity_qa(model, z: torch.Tensor, delta: np.ndarray) -> dict[str, float]:
    """Direct perturbations check every one of the 33 analytic coefficient rows."""
    original=_theta(model); epsilon=1.
    with torch.no_grad():
        model.begin_graph_cache()
        try:
            base=model.field(z); J,R,B=model.structured.matrices()[0],model.structured.matrices()[1],model.structured.port_matrix()
            analytic=coefficient_fields(model,z,J,R,B); errors=[]
            for j in range(24):
                plus=original.copy(); minus=original.copy(); plus[j]+=epsilon; minus[j]-=epsilon
                set_theta(model,plus); fp=model.field(z); set_theta(model,minus); fm=model.field(z)
                errors.append(float(((fp-fm)/(2*epsilon)-analytic[:,:,j]).abs().max()/max(1.,float(analytic[:,:,j].abs().max()))))
            set_theta(model,original); direct=model.field(z); set_theta(model,original+delta); candidate=model.field(z)
            affine=float((candidate - (direct + torch.einsum("bij,j->bi", analytic, torch.as_tensor(delta)))).abs().max() / max(1., float(candidate.abs().max())))
        finally:
            set_theta(model,original); model.end_graph_cache()
    maximum=max(errors,default=0.)
    if not np.isfinite(maximum) or not np.isfinite(affine) or max(maximum,affine)>1e-10: raise AssertionError("coefficient affinity QA")
    return {"direct_column_relative_max":maximum,"candidate_affinity_relative":affine}


def conditional_summary(table: dict[str, Any], residual: np.ndarray) -> dict[str, Any]:
    """Equal windows within event, then equal events; retain physical and latent parts."""
    event=table["event_index"]; labels=np.asarray(table["label"]); groups=np.asarray(table["group"])
    parts={"total":np.sum(residual*residual,1),"physical":np.sum(residual[:,:21]**2,1),"latent":np.sum(residual[:,21:]**2,1)}
    def summary(indices):
        return {name:float(np.sqrt(np.mean([values[event==i].mean() for i in indices]))) for name,values in parts.items()}
    per_event={name:np.asarray([values[event==i].mean() for i in range(346)]) for name,values in parts.items()}
    all_ids=np.arange(346); result={"overall":summary(all_ids),"subsets":{},"groups":{},"initial":{},"per_event":{name:value.tolist() for name,value in per_event.items()},"window_counts":np.bincount(event,minlength=346).tolist()}
    for label,count in (("stable",311),("unstable",35)):
        ids=np.unique(event[labels==label]);
        if len(ids)!=count: raise AssertionError("conditional subset")
        result["subsets"][label]=summary(ids)
    for group in sorted(set(groups)):
        result["groups"][group]=summary(np.unique(event[groups==group]))
    initial=table["start"]==10
    if initial.sum()!=346: raise AssertionError("initial window coverage")
    result["initial"]={name:float(np.sqrt(np.mean(values[initial]))) for name,values in parts.items()}
    return result


def structural_qa(model, candidate: Path, z: torch.Tensor) -> dict[str, float]:
    """Numeric export/reload plus the fixed origin, gauge and energy identities."""
    reloaded=load_model(candidate); reloaded.eval(); payload=json.loads((candidate/"structured_numeric_export.json").read_text()); numeric=StructuredROM.from_export(candidate/"structured_numeric_export.json").double()
    with torch.no_grad():
        field=model.field(z); reloaded_field=reloaded.field(z); internal=model.structured.interface_to_state(z); internal_field=model.structured.field(internal); numeric_field=numeric.field(internal); _,energy=model.structured.energy_and_gradient(internal); J,R=model.structured.matrices(); B=model.structured.port_matrix(); ports=model.structured.port_values(internal)
        zero=torch.zeros((2,33),dtype=torch.float64)
        origin=float(model.field(zero).abs().max()); kinematic=float((field[:,:10]-z[:,10:20]).abs().max())
        shifted=z.clone(); shifted[:,:10]+=0.37; shifted_field=model.field(shifted); gauge=float((shifted_field-field).abs().max()/max(1.,float(field.abs().max())))
        reload=float((field-reloaded_field).abs().max()/max(1.,float(field.abs().max())))
        exported_numpy=torch.as_tensor(numpy_field(payload,internal.detach().cpu().numpy()),dtype=torch.float64)
        export=max(float((internal_field-numeric_field).abs().max()/max(1.,float(internal_field.abs().max()))),float((internal_field-exported_numpy).abs().max()/max(1.,float(internal_field.abs().max()))))
        power=float((torch.sum(energy*internal_field,-1)+torch.sum((energy@R)*energy,-1)-torch.sum((energy@B)*ports,-1)).abs().max()/max(1.,float(internal_field.abs().max())))
        antisym=float((J+J.T).abs().max()); symmetric=float((R-R.T).abs().max()); psd=float(torch.linalg.eigvalsh(R).min())
    if max(origin,kinematic,gauge,reload,export,power,antisym,symmetric,max(0.,-psd))>1e-10: raise AssertionError("origin/gauge/energy/export QA")
    return {"origin_max":origin,"kinematic_max":kinematic,"gauge_field_relative":gauge,"full_reload_relative":reload,"numeric_export_relative":export,"power_identity_relative":power,"J_antisymmetry":antisym,"R_symmetry":symmetric,"R_min_eigenvalue":psd}


def prepare(checkpoint: Path, force: Path, baseline: Path, output: Path) -> dict[str, Any]:
    """Model/source QA only: deliberately does not materialize development trajectories."""
    before=hashes(checkpoint,force,baseline); base=MechanicalROM.load(BASE); model=load_model(checkpoint,base=base); model.eval()
    z=deterministic_test_z()
    with torch.no_grad():
        model.begin_graph_cache(); J,R,B=model.structured.matrices()[0],model.structured.matrices()[1],model.structured.port_matrix(); columns=coefficient_fields(model,z,J,R,B); model.end_graph_cache()
    if not torch.isfinite(columns).all(): raise FloatingPointError("nonfinite analytic QA")
    output.mkdir(parents=False); candidate=output/"candidate"; candidate.mkdir(); torch.save(model.state_dict(),candidate/"model.pt"); shutil.copyfile(checkpoint/"normalization.npz",candidate/"normalization.npz"); model.structured.export(candidate/"structured_numeric_export.json")
    meta=json.loads((checkpoint/"checkpoint.json").read_text()); (candidate/"checkpoint.json").write_text(json.dumps({"arm":"G-fixed","step":3000,"source":meta.get("source") or str(ROOT / "private_data/task2_long_0060_5s")},indent=2))
    affine=affinity_qa(model,z,np.linspace(-.03,.03,24)); structural=structural_qa(model,candidate,z)
    if before != hashes(checkpoint,force,baseline): raise RuntimeError("source changed during prepare")
    result={"loop":"0082","prepare_only":True,"test_read":False,"train_materialized":False,"conditional_speed_scale":CONDITIONAL_SPEED_SCALE,"autonomous_metric":AUTONOMOUS_METRIC,"hashes_before":before,"hashes_after":hashes(checkpoint,force,baseline),"column_shape":list(columns.shape),"affinity":affine,"structural":structural}
    (output/"prepare.json").write_text(json.dumps(result,indent=2)); return result


def run(checkpoint: Path, force: Path, baseline: Path, output: Path) -> dict[str, Any]:
    if output.exists(): raise FileExistsError(output)
    torch.set_num_threads(1); started=time.perf_counter(); before=hashes(checkpoint,force,baseline); current={"stage":"provenance"}; timers={}; output.mkdir()
    try:
        meta=json.loads((checkpoint/"checkpoint.json").read_text())
        if meta.get("arm")!="G-fixed" or meta.get("step")!=3000: raise AssertionError("frozen G-fixed3000 checkpoint required")
        base=MechanicalROM.load(BASE); events,excluded,identities=read_development(base,_network(),full_train=True); train=events["train"]
        if len(train)!=346 or sum(e["label"]=="stable" for e in train)!=311 or sum(e["label"]=="unstable" for e in train)!=35 or len({e["group"] for e in train})!=24 or sum(v=="train" for v in excluded.values())!=23: raise AssertionError("TRAIN identity")
        fe=json.loads((force/"evidence.json").read_text()); arrays=np.load(force/"projection_arrays.npz",allow_pickle=False)
        h79=fe["hashes_before"]["G"]
        if fe["identities"]!=identities or h79["model"]!=sha(checkpoint/"model.pt") or h79["normalization"]!=sha(checkpoint/"normalization.npz") or h79["base"]!=sha(BASE) or h79["network"]!=sha(NETWORK): raise AssertionError("0079 provenance")
        current={"stage":"table"}; table=_table(train,arrays,fe); model=load_model(checkpoint,base=base); model.eval(); frozen={k:v.detach().clone() for k,v in model.state_dict().items()}; theta0=_theta(model); fit_started=time.perf_counter(); global_deadline=started+3600.; fit_deadline=fit_started+1800.
        zqa=deterministic_test_z()
        prefit_affinity=affinity_qa(model,zqa,np.linspace(-.03,.03,24))
        current.update({"stage":"design_and_fit","windows":len(table["start"])}); designed=_design(model,table,min(fit_deadline,global_deadline),current); solved=solve_ridge(designed["c"],designed["A"],designed["weights"]); theta=theta0+solved["theta_delta"]
        postfit_affinity=affinity_qa(model,zqa,solved["theta_delta"]); timers["fit_design_seconds"]=time.perf_counter()-fit_started
        if timers["fit_design_seconds"]>1800: raise TimeoutError("0082 fit/design 30min cap")
        # All candidate fields are affine in the fitted block; save both steps before numerical gates.
        rb=designed["c"]; rc=rb+np.einsum("wkj,j->wk",designed["A"],solved["theta_delta"]); rbcoarse=designed["coarse_c"]; rccoarse=rbcoarse+np.einsum("wkj,j->wk",designed["coarse_A"],solved["theta_delta"])
        np.savez_compressed(output/"fit_arrays.npz",c=designed["c"],A=designed["A"],weights=designed["weights"],event_index=table["event_index"],start=table["start"],end=table["end"],event_id=np.asarray(table["event_id"]),label=np.asarray(table["label"]),group=np.asarray(table["group"]),window_length=table["end"]-table["start"],L_end=table["L"],conditional_speed_scale=CONDITIONAL_SPEED_SCALE,autonomous_metric=np.asarray(AUTONOMOUS_METRIC),theta0=theta0,theta=theta,scales=solved["scales"],beta=solved["beta"],active=solved["active"],K=solved["K"],g=solved["g"],residual_baseline_fine=rb,residual_baseline_coarse=rbcoarse,residual_candidate_fine=rc,residual_candidate_coarse=rccoarse)
        conditional={"baseline":conditional_summary(table,rb),"candidate":conditional_summary(table,rc)}
        (output/"fit_evidence.json").write_text(json.dumps({"fit_seconds":timers["fit_design_seconds"],"conditional_speed_scale":CONDITIONAL_SPEED_SCALE,"autonomous_metric":AUTONOMOUS_METRIC,"prefit_affinity":prefit_affinity,"postfit_affinity":postfit_affinity,"conditional":conditional,"stationarity_relative":solved["stationarity_relative"]},indent=2,allow_nan=False))
        numeric_windows=np.linalg.norm(rb-rbcoarse,axis=1); candidate_numeric=np.linalg.norm(rc-rccoarse,axis=1)
        if max(numeric_windows.max(),candidate_numeric.max())>.001: raise FloatingPointError("conditional Simpson numerical gate")
        set_theta(model,theta); changed=_changed(frozen,model)
        allowed=[f"structured.storage_gamma.{i}" for i in range(8)]+["structured.port_coefficient.0","structured.port_coefficient.1"]
        if not set(changed).issubset(allowed) or any(not torch.equal(frozen[k],model.state_dict()[k]) for k in frozen if k not in allowed): raise AssertionError(f"illegal state change {changed}")
        candidate_dir=output/"candidate"; candidate_dir.mkdir(); torch.save(model.state_dict(),candidate_dir/"model.pt"); shutil.copyfile(checkpoint/"normalization.npz",candidate_dir/"normalization.npz"); model.structured.export(candidate_dir/"structured_numeric_export.json")
        (candidate_dir/"checkpoint.json").write_text(json.dumps({"arm":"G-fixed","step":3000,"source":meta.get("source") or str(ROOT / "private_data/task2_long_0060_5s"),"parent_checkpoint":str(checkpoint),"loop":"0082"},indent=2))
        structural=structural_qa(model,candidate_dir,zqa)
        candidate_hashes={name:sha(candidate_dir/name) for name in ("model.pt","normalization.npz","structured_numeric_export.json","checkpoint.json")}
        # The paired autonomous transfer reuses the verified 0080 baseline only after its own provenance check.
        be=json.loads((baseline/"evidence.json").read_text()); cached=np.load(baseline/"trajectories.npz",allow_pickle=False)
        if be["identities"]!=identities or be["hashes_before"]["model"]!=sha(checkpoint/"model.pt") or be["hashes_before"]["norm"]!=sha(checkpoint/"normalization.npz") or be["hashes_before"]["base"]!=sha(BASE) or be["hashes_before"]["network"]!=sha(NETWORK): raise AssertionError("0080 baseline provenance")
        baseline_fine,baseline_coarse,cached_truth,cached_lengths,cached_ids,cached_q=np.asarray(cached["fine_baseline"]),np.asarray(cached["coarse_baseline"]),np.asarray(cached["truth"]),np.asarray(cached["lengths"]),np.asarray(cached["event_ids"]),np.asarray(cached["qenc"])
        cached_labels,cached_mass,cached_pscale=np.asarray(cached["labels"]),np.asarray(cached["mass"]),np.asarray(cached["p_scale"])
        hist=np.stack([e["x"][:11] for e in train])
        with torch.no_grad(): q=model.encode(torch.as_tensor(hist,dtype=torch.float64)).cpu().numpy()
        lengths=np.array([min(500,len(e["x"])-11) for e in train]); states=torch.as_tensor(np.c_[hist[:,-1],q],dtype=torch.float64)
        expected_truth=np.full_like(cached_truth,np.nan)
        for i,event in enumerate(train): expected_truth[i,:lengths[i]+1]=event["x"][10:10+lengths[i]+1]
        if not(np.array_equal(cached_ids,np.asarray([e["id"] for e in train])) and np.array_equal(cached_labels,np.asarray([e["label"] for e in train])) and np.array_equal(cached_lengths,lengths) and np.array_equal(cached_truth,expected_truth,equal_nan=True) and np.array_equal(cached_mass,model.mass.detach().cpu().numpy()) and np.array_equal(cached_pscale,model.p_scale.detach().cpu().numpy()) and np.allclose(cached_q,q,atol=1e-10,rtol=0)): raise AssertionError("0080 cached baseline alignment")
        baseline_numeric=numeric(baseline_fine,baseline_coarse,lengths,cached_mass,cached_pscale)
        if any(abs(baseline_numeric[i][key]-be["numerics"]["baseline"][i][key])>1e-12 for i in range(346) for key in ("coi","common","p")): raise AssertionError("0080 baseline numeric reproduction")
        baseline_reproduced={}
        for h in (1,2,5):
            values=[]
            for i,n in enumerate(lengths):
                k=min(round(h/DT),n); values.append(float(point(baseline_fine[i,2:2*k+1:2],expected_truth[i,1:k+1],cached_mass,cached_pscale).mean()))
            baseline_reproduced[str(h)]=risk(values)
            if any(abs(baseline_reproduced[str(h)][key]-be["horizons"][str(h)]["baseline"][key])>1e-12 for key in ("mean","tail35","risk")): raise AssertionError("0080 baseline risk reproduction")
        forecast_started=time.perf_counter(); forecast_deadline=forecast_started+1800.
        current={"stage":"candidate_forecast"}; ids=np.asarray([e["id"] for e in train]); fine=_checked_rollout(model,states,lengths*2,FINE,deadline=min(forecast_deadline,global_deadline),progress=current,partial=output/"partial_fine.npz",event_ids=ids); coarse=_checked_rollout(model,states,lengths,COARSE,deadline=min(forecast_deadline,global_deadline),progress=current,partial=output/"partial_coarse.npz",event_ids=ids)
        truth=np.full((346,int(lengths.max())+1,30),np.nan)
        for i,e in enumerate(train): truth[i,:lengths[i]+1]=e["x"][10:10+lengths[i]+1]
        pscale=model.p_scale.detach().cpu().numpy(); mass=model.mass.detach().cpu().numpy(); np.savez_compressed(output/"trajectories.npz",event_ids=np.asarray([e["id"] for e in train]),labels=np.asarray([e["label"] for e in train]),mass=mass,p_scale=pscale,conditional_speed_scale=CONDITIONAL_SPEED_SCALE,autonomous_metric=np.asarray(AUTONOMOUS_METRIC),lengths=lengths,truth=truth,qenc=q,fine_baseline=baseline_fine,coarse_baseline=baseline_coarse,fine_candidate=fine,coarse_candidate=coarse); numerics=numeric(fine,coarse,lengths,mass,pscale)
        if any(v>.001 for row in numerics for v in row.values()): raise FloatingPointError("autonomous numerical gate")
        per=[]; horizons={}
        for h in (1,2,5):
            candidate=[]; baseline_values=[]
            for i,n in enumerate(lengths):
                k=min(round(h/DT),n); candidate.append(float(point(fine[i,2:2*k+1:2],truth[i,1:k+1],mass,pscale).mean())); baseline_values.append(float(point(baseline_fine[i,2:2*k+1:2],truth[i,1:k+1],mass,pscale).mean())); per.append({"event_id":train[i]["id"],"label":train[i]["label"],"horizon":h,"candidate":candidate[-1],"baseline":baseline_values[-1],"available_steps":int(k)})
            horizons[str(h)]={"candidate":risk(candidate),"baseline":risk(baseline_values)}
            if horizons[str(h)]["baseline"] != baseline_reproduced[str(h)]: raise AssertionError("baseline risk changed after candidate rollout")
        timers["forecast_seconds"]=time.perf_counter()-forecast_started
        if timers["forecast_seconds"]>1800: raise TimeoutError("0082 forecast 30min cap")
        if not np.array_equal(_theta(model),theta) or any(not torch.equal(frozen[k],model.state_dict()[k]) for k in frozen if k not in allowed): raise AssertionError("candidate state mutated during forecast")
        after=hashes(checkpoint,force,baseline)
        if before!=after: raise RuntimeError("source changed")
        gates={"total_ratio":conditional["candidate"]["overall"]["total"]/conditional["baseline"]["overall"]["total"],"physical_ratio":conditional["candidate"]["overall"]["physical"]/conditional["baseline"]["overall"]["physical"],"unstable_total_ratio":conditional["candidate"]["subsets"]["unstable"]["total"]/conditional["baseline"]["subsets"]["unstable"]["total"]}
        average={name:float(np.mean([horizons[str(h)][name]["risk"] for h in (1,2,5)])) for name in ("baseline","candidate")}
        unstable=[x for x in per if x["horizon"]==5 and x["label"]=="unstable"]
        severe={name:{"mean":float(np.mean([x[name] for x in unstable])),"worst4":float(np.mean(sorted(x[name] for x in unstable)[-4:]))} for name in ("baseline","candidate")}
        result={"loop":"0082","test_read":False,"validation_materialized_by_reader":True,"conditional_speed_scale":CONDITIONAL_SPEED_SCALE,"autonomous_metric":AUTONOMOUS_METRIC,"identities":identities,"excluded":excluded,"window_count":len(table["start"]),"hashes_before":before,"hashes_after":after,"candidate_hashes":candidate_hashes,"fit":{"stationarity_relative":solved["stationarity_relative"],"ridge_condition":solved["ridge_condition"],"design_singular_values":solved["design_singular_values"].tolist(),"inactive_columns":np.flatnonzero(~solved["active"]).tolist(),"unregularized_baseline":float(np.einsum("w,wk,wk",designed["weights"],rb,rb)),"unregularized_candidate":float(np.einsum("w,wk,wk",designed["weights"],rc,rc)),"penalized_objective":float(np.einsum("w,wk,wk",designed["weights"],rc,rc)+RIDGE*np.dot(solved["beta"],solved["beta"]))},"affinity_qa":{"prefit":prefit_affinity,"postfit":postfit_affinity},"structural_qa":structural,"conditional":conditional,"conditional_gates":gates,"conditional_numeric_max":float(max(numeric_windows.max(),candidate_numeric.max())),"autonomous_numerics":numerics,"autonomous_horizons":horizons,"autonomous_average_risk":average,"unstable_5s":severe,"per_event":per,"timers":timers,"seconds":time.perf_counter()-started,"allowed_state_keys_changed":changed}
        (output/"evidence.json").write_text(json.dumps(result,indent=2,allow_nan=False)); return result
    except Exception as error:
        timers["total_seconds"]=time.perf_counter()-started
        (output/"failure_evidence.json").write_text(json.dumps({"type":type(error).__name__,"message":str(error),"current":current,"timers":timers,"hashes_before":before,"test_read":False},indent=2)); raise


def main() -> None:
    parser=argparse.ArgumentParser(); parser.add_argument("--checkpoint",type=Path,default=CHECKPOINT); parser.add_argument("--force",type=Path,default=FORCE); parser.add_argument("--baseline",type=Path,default=BASELINE); parser.add_argument("--output",type=Path,required=True); parser.add_argument("--prepare-only",action="store_true"); args=parser.parse_args()
    if args.prepare_only: prepare(args.checkpoint,args.force,args.baseline,args.output)
    else: run(args.checkpoint,args.force,args.baseline,args.output)


if __name__ == "__main__": main()
