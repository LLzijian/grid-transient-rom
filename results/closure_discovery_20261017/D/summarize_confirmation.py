"""JSON-only supplemental summary for the frozen D confirmation cohort."""
from __future__ import annotations
import argparse, hashlib, json, math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
SPEC = ROOT / "results/closure_discovery_20261017/D/reporting_spec.json"

def _finite(x):
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)

def _aggregate(records, horizon):
    label = "unstable" if horizon == "unstable_actual" else "stable"
    hkey = "prefix" if horizon == "unstable_actual" else horizon[:-1]
    cohort = [r for r in records if r.get("label") == label]
    applicable, unavailable, missing = [], [], []
    for r in cohort:
        m = next((v for v in r.get("metrics", []) if v.get("horizon") == hkey), None)
        c = r.get("components", {}).get(hkey)
        if m is None or c is None:
            missing.append(r.get("id")); continue
        if m.get("available") is True and c.get("available") is True:
            applicable.append(r)
        else:
            unavailable.append(r.get("id"))
    out = {"horizon": horizon, "available_event_count": len(applicable), "unavailable_event_count": len(unavailable),
           "unavailable_ids": unavailable, "event_ids": [r.get("id") for r in applicable]}
    errors, reasons = [], []
    frequency_sse = power_sse = frequency_n = power_n = 0.0
    for row in applicable:
        ident = row.get("id")
        if row.get("numerical_pass") is not True:
            reasons.append({"id": ident, "reason": "numerical_pass_false"})
        m = next((v for v in row.get("metrics", []) if v.get("horizon") == hkey), None)
        if m is None or m.get("finite") is not True or not _finite(m.get("normalized_sse")) or not _finite(m.get("n")) or m["n"] <= 0 or m["normalized_sse"] < 0:
            reasons.append({"id": ident, "reason": "missing_or_nonfinite_normalized_metric"}); continue
        component_h = hkey
        comp = row.get("components", {}).get(component_h, {})
        f, p = comp.get("frequency", {}), comp.get("mechanical_power", {})
        if f.get("finite") is not True or p.get("finite") is not True:
            reasons.append({"id": ident, "reason": "component_finite_false"}); continue
        vals = [f.get("normalized_mse"), p.get("normalized_mse"), f.get("physical_sse"), f.get("n_scalar_points"), p.get("physical_sse"), p.get("n_scalar_points")]
        if not all(_finite(v) for v in vals) or vals[0] < 0 or vals[1] < 0 or vals[2] < 0 or vals[3] <= 0 or vals[4] < 0 or vals[5] <= 0:
            reasons.append({"id": ident, "reason": "missing_or_nonfinite_component"}); continue
        e = m["normalized_sse"] / m["n"] + f["normalized_mse"] + p["normalized_mse"]
        if not _finite(e): reasons.append({"id": ident, "reason": "normalized_error_overflow"}); continue
        errors.append((ident, e)); frequency_sse += f["physical_sse"]; frequency_n += f["n_scalar_points"]
        power_sse += p["physical_sse"]; power_n += p["n_scalar_points"]
        if not all(_finite(v) for v in (frequency_sse, frequency_n, power_sse, power_n)):
            reasons.append({"id": ident, "reason": "pooled_component_overflow"})
    if missing: reasons.extend({"id": i, "reason": "missing_horizon_fields"} for i in missing)
    if len(errors) != len(applicable) or reasons:
        out.update({"status": "unresolved", "affected": reasons, "normalized_risk": None,
                    "frequency_rmse": None, "mechanical_power_rmse": None})
        return out
    if not errors:
        out.update({"status": "unresolved", "affected": [], "normalized_risk": None,
                    "frequency_rmse": None, "mechanical_power_rmse": None}); return out
    values = sorted((v for _, v in errors), reverse=True); k = max(1, math.ceil(.1 * len(values)))
    mean, tail = sum(values) / len(values), sum(values[:k]) / k
    risk = .8 * mean + .2 * tail
    if not all(_finite(v) for v in (mean, tail, risk, frequency_sse, frequency_n, power_sse, power_n)):
        out.update({"status": "unresolved", "affected": [{"reason": "aggregate_overflow"}], "normalized_risk": None,
                    "frequency_rmse": None, "mechanical_power_rmse": None}); return out
    out.update({"status": "complete", "affected": [], "normalized_risk": risk,
                "frequency_rmse": math.sqrt(frequency_sse / frequency_n), "frequency_unit": "rad/s",
                "frequency_event_count": len(errors), "frequency_scalar_points": int(frequency_n), "time_points": int(frequency_n),
                "mechanical_power_rmse": math.sqrt(power_sse / power_n),
                "mechanical_power_unit": "p.u. (100 MVA system base)",
                "mechanical_power_event_count": len(errors), "mechanical_power_scalar_points": int(power_n)})
    return out

def summarize_model(evaluation):
    records = evaluation["records"]
    summaries = {h: _aggregate(records, h) for h in ("1s", "2s", "5s", "8s", "unstable_actual")}
    five = summaries["5s"]
    mismatches = []
    for row in records:
        if row.get("label") != "stable": continue
        m = next((v for v in row.get("metrics", []) if v.get("horizon") == "5"), None)
        c = row.get("components", {}).get("5", {})
        if m and c.get("available") is True and m.get("available") is True:
            vals = [m.get("normalized_sse"), m.get("n"), c.get("frequency",{}).get("normalized_mse"), c.get("mechanical_power",{}).get("normalized_mse"), row.get("mse")]
            if not all(_finite(v) for v in vals) or vals[0] < 0 or vals[1] <= 0:
                mismatches.append(row.get("id")); continue
            reconstructed = vals[0] / vals[1] + vals[2] + vals[3]
            if not _finite(reconstructed) or abs(reconstructed - vals[4]) > 1e-10: mismatches.append(row.get("id"))
    five["saved_primary_risk"] = evaluation.get("primary_risk")
    five["per_event_mse_mismatches"] = mismatches
    five["per_event_mse_match"] = not mismatches
    if mismatches:
        five.update({"status":"unresolved", "normalized_risk":None, "frequency_rmse":None, "mechanical_power_rmse":None})
    return {"classification_counts": evaluation.get("classification_counts"), "primary_risk": evaluation.get("primary_risk"),
            "horizons": summaries}

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--output", required=True); a = ap.parse_args()
    root = ROOT / "private_data/closure_discovery_20261017/D"; freeze_p = root / "confirmation_freeze.json"
    primary_p = root / "primary_comparison.json"; val_p = root / "data_val_result.json"
    for p in (freeze_p, primary_p, val_p):
        if not p.is_file(): raise RuntimeError(f"required registered input missing: {p.name}")
    freeze, primary, val = (json.loads(p.read_text(encoding="utf-8")) for p in (freeze_p, primary_p, val_p))
    models = freeze.get("models", [])
    if len(models) != 14 or val.get("attempted_conditions") != 24:
        raise RuntimeError("frozen model cohort or VAL attempt record is incomplete")
    expected_keys = {f"{name}_{seed}" for name in ("fixed","library","main","alternative") for seed in (2701,2702,2703)} | {"delete3_2701","delete2_2701"}
    model_keys = {f"{m.get('name')}_{m.get('seed')}" for m in models}
    if model_keys != expected_keys: raise RuntimeError("frozen model identities do not match the registered 14-key cohort")
    val_count = val.get("val_event_count")
    rows = []
    cache_identity = None; cache_ids = None
    for model in models:
        path = Path(model["path"]); path = path / "model.json" if path.is_dir() else path
        if hashlib.sha256(path.read_bytes()).hexdigest() != model["sha256"]: raise RuntimeError("frozen model SHA mismatch")
        key = f"{model['name']}_{model['seed']}"; ep = root / "evaluation" / f"{key}.json"
        ev = json.loads(ep.read_text(encoding="utf-8"))
        if ev.get("_identity", {}).get("model_sha256") != model["sha256"]: raise RuntimeError(f"evaluation model identity mismatch: {key}")
        identity = ev.get("_identity", {}).get("cohort")
        if identity is None or identity.get("event_count") != val_count: raise RuntimeError(f"evaluation cohort size mismatch: {key}")
        ids = sorted(str(x.get("id")) for x in ev.get("records", []))
        if len(ids) != len(set(ids)) or len(ids) != val_count: raise RuntimeError(f"evaluation record IDs incomplete/duplicated: {key}")
        if cache_identity is None: cache_identity, cache_ids = identity, ids
        elif identity != cache_identity or ids != cache_ids: raise RuntimeError("evaluation caches do not share one cohort identity and record ID set")
        rows.append({"name": model["name"], "seed": model["seed"], **summarize_model(ev)})
    spec_sha = hashlib.sha256(SPEC.read_bytes()).hexdigest()
    result = {"script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "reporting_spec_sha256": spec_sha, "freeze_sha256": hashlib.sha256(freeze_p.read_bytes()).hexdigest(),
              "cache_sha256": {f"{m['name']}_{m['seed']}": hashlib.sha256((root/'evaluation'/f"{m['name']}_{m['seed']}.json").read_bytes()).hexdigest() for m in models},
              "primary_comparison": primary, "val_generation": val, "models": rows, "test_read": False}
    Path(a.output).write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return 0

if __name__ == "__main__": raise SystemExit(main())
