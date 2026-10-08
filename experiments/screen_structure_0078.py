"""Pure fixed-endpoint screen for the preregistered Loop 0078 campaign.

The module does not read H5 files or inspect TEST.  Its CLI accepts only
explicit JSON evidence paths supplied after the three arms finish.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Mapping

S_EVIDENCE = (
    Path("private_data/task2_long_0060_5s_eval/evidence.json"),
    Path("private_data/task2_l1_S_0067_original/evidence.json"),
)
HORIZONS = ("1.0", "2.0", "5.0")


def _finite(x: Any) -> bool:
    return isinstance(x, (int, float)) and math.isfinite(float(x))


def _number(x: Any):
    return float(x) if _finite(x) else None


def _value(obj: Mapping[str, Any], *keys: str):
    cur: Any = obj
    for key in keys:
        if not isinstance(cur, Mapping) or key not in cur:
            return None
        cur = cur[key]
    return cur


def _summary(metrics: Mapping[str, Any], horizon: str):
    aliases = {"1.0": ("1.0", "1"), "2.0": ("2.0", "2"), "5.0": ("5.0", "5"), "8.0": ("8.0", "8")}[horizon]
    for root in (metrics, metrics.get("evaluation", {}) if isinstance(metrics, Mapping) else {}):
        for key in ("summary", "complete", "summaries"):
            for alias in aliases:
                val = _value(root, key, alias)
                if isinstance(val, Mapping): return val
    return None


def _metric(metrics: Mapping[str, Any], section: str, horizon: str, name: str):
    row = _summary(metrics, horizon) if section == "summary" else None
    if row is not None:
        return row.get(name)
    return _value(metrics, section, horizon, name)


def _common5(metrics: Mapping[str, Any]):
    rows = metrics.get("common_frequency") or _value(metrics, "evaluation", "common_frequency")
    if not isinstance(rows, list):
        return None
    selected = [r for r in rows if isinstance(r, Mapping) and str(r.get("horizon")) in ("5", "5.0")
                and r.get("complete") is not False]
    if not selected or not all(_finite(r.get("sse")) and float(r['sse']) >= 0
                              and _finite(r.get("n")) and float(r["n"]) > 0 for r in selected):
        return None
    sse, n = sum(float(r["sse"]) for r in selected), sum(float(r["n"]) for r in selected)
    return math.sqrt(sse / n)


def _l1(metrics: Mapping[str, Any]):
    for root in (metrics, metrics.get("evaluation", {}) if isinstance(metrics, Mapping) else {}):
        for key in ("l1_numerically_gated_counts", "l1", "l1_counts"):
            val = root.get(key) if isinstance(root, Mapping) else None
            if isinstance(val, Mapping):
                return val
    return None


def _finite_summary(metrics: Mapping[str, Any], horizon: str) -> bool:
    row = _summary(metrics, horizon)
    return isinstance(row, Mapping) and all(_finite(row.get(k)) for k in ("mean", "pooled", "cvar10"))


def screens(metrics: Mapping[str, Any], S: Mapping[str, Any], C: Mapping[str, Any]) -> dict[str, Any]:
    """Return fixed 1200-step gate booleans and ratios; missing is failure."""
    ratios: dict[str, Any] = {}
    gates: dict[str, bool] = {}
    for h in HORIZONS:
        vals = {k: _metric(metrics, "summary", h, k) for k in ("mean", "pooled", "cvar10")}
        ref = {k: _metric(S, "summary", h, k) for k in ("mean", "pooled", "cvar10")}
        ok = _finite_summary(metrics, h) and _finite_summary(S, h)
        for k in vals:
            ratio = float(vals[k]) / float(ref[k]) if ok and float(ref[k]) > 0 else None
            ratios[f"S_{h}_{k}"] = ratio
            gates[f"S_{h}_{k}"] = bool(ok and _finite(vals[k]) and _finite(ref[k]) and float(ref[k]) >= 0 and float(vals[k]) <= 1.05 * float(ref[k]))
    candidate5, control5 = _metric(metrics, "summary", "5.0", "pooled"), _metric(C, "summary", "5.0", "pooled")
    ratios["C_free_5.0_pooled"] = float(candidate5) / float(control5) if _finite(candidate5) and _finite(control5) and float(control5) > 0 else None
    gates["G_auto_vs_C_free_5.0_pooled"] = bool(_finite(candidate5) and _finite(control5) and float(candidate5) <= .95 * float(control5))
    gates["matched_control_numerically_valid"] = C.get("model_numerical_gate_pass") is True
    up = metrics.get("unstable_available_prefix") if isinstance(metrics, Mapping) else None
    sup = S.get("unstable_available_prefix") if isinstance(S, Mapping) else None
    ok_up = isinstance(up, Mapping) and isinstance(sup, Mapping) and all(_finite(up.get(k)) and _finite(sup.get(k)) for k in ("pooled", "cvar10"))
    for k in ("pooled", "cvar10"):
        ratio = float(up[k]) / float(sup[k]) if ok_up and float(sup[k]) > 0 else None
        ratios[f"unstable_prefix_S_{k}"] = ratio
        gates[f"unstable_prefix_S_{k}"] = bool(ok_up and float(sup[k]) >= 0 and float(up[k]) <= 1.05 * float(sup[k]))
    cm, cs = _common5(metrics), _common5(S)
    ratios["common5_S"] = float(cm) / float(cs) if _finite(cm) and _finite(cs) and float(cs) > 0 else None
    gates["common5_S"] = bool(_finite(cm) and _finite(cs) and float(cm) <= 1.05 * float(cs))
    l1, numerical = _l1(metrics), metrics.get("model_numerical_gate_pass")
    gates["L1_detect"] = bool(isinstance(l1, Mapping) and _number(l1.get("detected_unstable")) is not None and float(l1["detected_unstable"]) >= 10)
    gates["L1_false_los"] = bool(isinstance(l1, Mapping) and _number(l1.get("false_los")) is not None and float(l1["false_los"]) == 0)
    gates["L1_false_window_settled"] = bool(isinstance(l1, Mapping) and _number(l1.get("false_window_settled")) is not None and float(l1["false_window_settled"]) <= 2)
    gates["numerical_gate"] = numerical is True
    retention = {k:v for k,v in gates.items() if k not in
                 ("G_auto_vs_C_free_5.0_pooled", "matched_control_numerically_valid")}
    return {"gates": gates, "ratios": ratios, "retention_pass": bool(all(retention.values())),
            "screen_pass": bool(all(gates.values()))}


def select_extensions(arms: Mapping[str, Mapping[Any, Mapping[str, Any]]]) -> list[str]:
    """Select at most two eligible arms by deterministic TRAIN-only risk."""
    eligible = []
    for arm, rows in arms.items():
        try:
            r0 = rows.get(0, rows.get("0")); r8 = rows.get(800, rows.get("800_before_growth", rows.get("800"))); r12 = rows.get(1200, rows.get("1200"))
            if not all(isinstance(r, Mapping) for r in (r0, r8, r12)): continue
            vals = [r.get("risk", r.get("average_risk")) for r in (r0, r8, r12)]
            if not all(_finite(v) for v in vals):
                continue
            if float(vals[2]) <= .9 * float(vals[0]) and float(vals[2]) <= .98 * float(vals[1]):
                eligible.append((float(vals[2]), str(arm)))
        except (KeyError, TypeError, AttributeError):
            continue
    return [arm for _, arm in sorted(eligible)[:2]]


def _read(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def main():
    p = argparse.ArgumentParser(); p.add_argument("--g-auto", type=Path, required=True); p.add_argument("--g-fixed", type=Path, required=True); p.add_argument("--c-free", type=Path, required=True); p.add_argument("--s-eval", type=Path, default=S_EVIDENCE[0]); p.add_argument("--s-l1", type=Path, default=S_EVIDENCE[1]); p.add_argument("--output", type=Path, required=True)
    a = p.parse_args(); arms = {"G-auto": _read(a.g_auto), "G-fixed": _read(a.g_fixed), "C-free": _read(a.c_free)}; S = _read(a.s_eval); S_l1 = _read(a.s_l1); S = dict(S); S.setdefault("common_frequency", S_l1.get("common_frequency")); C = arms["C-free"]; result = {arm: screens(value, S, C) for arm, value in arms.items()}; result["provenance"] = {"s_eval": str(a.s_eval), "s_l1": str(a.s_l1)}; a.output.write_text(json.dumps(result, indent=2, allow_nan=False), encoding="utf-8")


if __name__ == "__main__":
    main()
