"""Independent audit of one completed 0078 TRAIN-only extension.

This module reads checkpoints/evidence only.  It never opens trajectory data
or H5 files and does not rerun training or validation.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import torch
try:
    from check_structure_fit_0078 import _monitor_ok
except ModuleNotFoundError:
    from experiments.check_structure_fit_0078 import _monitor_ok


def _tree_equal(a, b) -> bool:
    if torch.is_tensor(a) or torch.is_tensor(b):
        return torch.is_tensor(a) and torch.is_tensor(b) and torch.equal(a, b)
    if isinstance(a, dict) or isinstance(b, dict):
        return isinstance(a, dict) and isinstance(b, dict) and set(a) == set(b) and all(_tree_equal(a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple)) or isinstance(b, (list, tuple)):
        return type(a) is type(b) and len(a) == len(b) and all(_tree_equal(x, y) for x, y in zip(a, b))
    return a == b


def _finite_tree(x) -> bool:
    if torch.is_tensor(x): return bool(torch.isfinite(x).all())
    if isinstance(x, dict): return all(_finite_tree(v) for v in x.values())
    if isinstance(x, (list, tuple)): return all(_finite_tree(v) for v in x)
    if isinstance(x, (float, int)): return math.isfinite(float(x))
    return True


def _finite_number(x) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(float(x))


def extension_horizon(step: int) -> float:
    if not 1201 <= step <= 3000: raise ValueError(step)
    return (1., 2., 5.)[(step - 301) % 3]


def _load_json(path): return json.loads(Path(path).read_text(encoding="utf-8"))
def _state(path): return torch.load(Path(path), map_location="cpu", weights_only=True)


def _checkpoint(path):
    path = Path(path)
    return _load_json(path / "checkpoint.json"), _state(path / "model.pt")


def _optimizer_names(meta): return meta.get("optimizer_parameter_names")


def _optimizer_state_by_name(opt, meta):
    if _optimizer_layout(opt, meta) is None: return None
    names = _optimizer_names(meta)
    out = {}
    for group, group_names in zip(opt["param_groups"], names, strict=True):
        for pid, name in zip(group["params"], group_names, strict=True):
            if name in out: return None
            out[name] = opt.get("state", {}).get(pid, {})
    return out


def _optimizer_layout(opt, meta):
    names = _optimizer_names(meta); groups = opt.get("param_groups", [])
    if not isinstance(names, list) or len(names) != len(groups): return None
    if any(not isinstance(row, list) or len(row) != len(group.get("params", [])) for row, group in zip(names, groups)): return None
    return [(list(row), {k: v for k, v in group.items() if k != "params"}) for row, group in zip(names, groups)]


def _frozen_exact(before, after, requires_grad):
    trainable = {k for k, v in requires_grad.items() if v}
    return set(before) == set(after) and all(k in trainable or torch.equal(before[k], after[k]) for k in before)


def _trace_checks(ev):
    traces = [r for r in ev.get("traces", []) if "horizon_s" in r]
    checks = {
        "trace_count": len(traces) == 1800,
        "trace_steps": [r.get("step") for r in traces] == list(range(1201, 3001)),
        "trace_horizons": all(r.get("horizon_s") == extension_horizon(r["step"]) for r in traces if isinstance(r.get("step"), int) and 1201 <= r["step"] <= 3000),
        "trace_finite": all(isinstance(r.get("terms"),dict) and isinstance(r.get("checks"),dict) and all(_finite_number(r.get(k)) for k in ("loss", "gradient_norm_before_cap", "gradient_norm_clip_return")) and _finite_tree(r["terms"]) and _finite_tree(r["checks"]) for r in traces),
        "trace_invariants": all(isinstance(r.get("checks"), dict) and all(k in r["checks"] for k in ("f0", "probe_finite", "probe_nonzero", "kinematic_max_abs")) and r["checks"]["f0"] <= 1e-10 and r["checks"]["probe_finite"] is True and math.isfinite(float(r["checks"]["probe_nonzero"])) and r["checks"]["probe_nonzero"] > 0 and r["checks"]["kinematic_max_abs"] <= 1e-12 for r in traces),
        "complexity_zero": ev.get("complexity_weight") == 0. and all("complexity" not in r.get("terms", {}) or float(r["terms"]["complexity"]) == 0. for r in traces),
        "no_growth_or_prune": not any("growth" in r or "pruning" in r for r in ev.get("traces", [])),
    }
    return checks


def audit(initial, resume, final, evidence, *, model_loader=None):
    initial, resume, final, evidence = map(Path, (initial, resume, final, evidence))
    mi, si = _checkpoint(initial); mr, sr = _checkpoint(resume); mf, sf = _checkpoint(final)
    oi, or_, of = (_state(p / "optimizer.pt") for p in (initial, resume, final))
    ri, rr, rf = (_state(p / "rng_state.pt") for p in (initial, resume, final))
    ev = _load_json(evidence); checks = {}
    checks["original_resume_model_exact"] = _tree_equal(si, sr)
    checks["original_resume_optimizer_exact"] = _tree_equal(oi, or_)
    checks["original_resume_rng_exact"] = _tree_equal(ri, rr)
    checks["selection_rng_unchanged"] = "selection_numpy" in rr and "selection_numpy" in rf and _tree_equal(rr["selection_numpy"], rf["selection_numpy"])
    checks["final_model_optimizer_finite"] = _finite_tree(sf) and _finite_tree(of)
    checks["resume_metadata_exact"] = all(mr.get(k) == mi.get(k) for k in ("arm", "requires_grad", "optimizer_parameter_names"))
    checks["final_arm_same"] = mf.get("arm") == mi.get("arm") == ev.get("arm")
    checks["checkpoint_steps_paths"] = (mi.get("step") == 1200 and mr.get("step") == 1200 and mf.get("step") == 3000 and ev.get("steps_from") == 1200 and ev.get("steps_to") == 3000 and Path(ev.get("checkpoint", "")).resolve() == initial.resolve() and Path(ev.get("final_checkpoint", "")).resolve() == final.resolve())
    req_i, req_f = mi.get("requires_grad", {}), mf.get("requires_grad", {})
    checks["requires_grad_unchanged"] = req_i == req_f
    checks["optimizer_names_unchanged"] = mi.get("optimizer_parameter_names") == mf.get("optimizer_parameter_names")
    checks["frozen_state_exact"] = _frozen_exact(si, sf, req_i)
    checks["support_masks_exact"] = all(torch.equal(si[k], sf[k]) for k in si if ("support" in k or "mask" in k) and k in sf)
    checks["groups_exact"] = mi.get("extra_groups", []) == mf.get("extra_groups", [])
    checks.update(_trace_checks(ev))
    hb, ha = ev.get("hashes_before"), ev.get("hashes_after")
    checks["hashes_nonempty_unchanged"] = isinstance(hb, dict) and bool(hb) and isinstance(ha, dict) and hb == ha
    checks["monitor346"] = _monitor_ok(ev.get("monitor3000"))
    checks["no_xi"] = mi.get("arm") == "C-free" or not any("xi" in k for k in sf)
    # Adam step/frozen moments compare is done by parameter name, excluding path metadata.
    li, lr, lf = _optimizer_layout(oi, mi), _optimizer_layout(or_, mr), _optimizer_layout(of, mf)
    checks["optimizer_layout_unchanged"] = li is not None and li == lr == lf
    ni, nr, nf = _optimizer_state_by_name(oi, mi), _optimizer_state_by_name(or_, mr), _optimizer_state_by_name(of, mf)
    step_ok, frozen_mom_ok = ni is not None and nr is not None and nf is not None and lf is not None, True
    if step_ok:
        optimizer_names = {name for names, _ in lf for name in names}
        for name, req in req_f.items():
            if name not in optimizer_names:
                if req: step_ok = False
                continue
            if name not in nf or name not in nr: step_ok = False; continue
            if req:
                a, b = nr[name].get("step"), nf[name].get("step")
                step_ok &= a is not None and b is not None and float(b) == float(a) + 1800
            else:
                frozen_mom_ok &= _tree_equal(nr[name], nf[name])
    checks["adam_trainable_advanced_1800"] = bool(step_ok)
    checks["adam_frozen_moments_unchanged"] = bool(frozen_mom_ok)
    if model_loader is not None:
        model = model_loader(final)
        probe = torch.linspace(-.03, .03, 66, dtype=torch.float64).reshape(2, 33)
        with torch.no_grad():
            f = model.field(probe); z = torch.zeros_like(probe); shifted = probe.clone(); shifted[:, :10] += .37
            checks["qa_finite"] = bool(torch.isfinite(f).all())
            checks["qa_f0"] = bool(model.field(z).abs().max() <= 1e-10)
            checks["qa_kinematic"] = bool((f[:, :10] - probe[:, 10:20]).abs().max() <= 1e-12)
            checks["qa_gauge"] = bool(torch.allclose(f, model.field(shifted), atol=1e-10, rtol=1e-10))
        if mf.get("arm") != "C-free":
            export = _load_json(final / "structured_numeric_export.json")
            checks["qa_ports"] = len(export.get("ports", [])) <= 2 and all(len(p.get("terms", [])) <= 8 for p in export.get("ports", []))
    return {"checks": checks, "passed": bool(all(checks.values())), "limitation": "per-step event IDs are not recorded; monitor and trace audit cannot prove historical event sampling"}


def main():
    p = argparse.ArgumentParser(); p.add_argument("--initial", type=Path, required=True); p.add_argument("--resume", type=Path, required=True); p.add_argument("--final", type=Path, required=True); p.add_argument("--evidence", type=Path, required=True); p.add_argument("--output", type=Path, required=True); a = p.parse_args()
    from train_structured_rom import load_model
    out = audit(a.initial, a.resume, a.final, a.evidence, model_loader=load_model)
    a.output.write_text(json.dumps(out, indent=2, allow_nan=False), encoding="utf-8"); print(json.dumps(out)); raise SystemExit(0 if out["passed"] else 1)


if __name__ == "__main__": main()
