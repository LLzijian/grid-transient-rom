"""P4 (H3 cheap check): frozen A step64 versus step128 forward evaluation.

Read-only except the declared P4 outputs.

* No training, no optimizer, no new data and no checkpoint writes.
* Both frozen checkpoints are rolled forward on the 267 stable ``fit297`` events with the
  registered fine path (0.005 s true RK4, history dt 0.00125, graph cache, CPU float64,
  one thread), exactly the pipeline that produced the frozen 0.005 s prediction files.
* Only the first 8 s are integrated because the requested metrics are stable 5 s / 8 s.
* ``step128`` is a comparability gate: its 5 s / 8 s risks must reproduce the published
  ``0.8675373355146678`` / ``0.6255711284192054`` at atol 1e-9.  If that gate fails the
  step64 numbers are not released.
* Only per-event scalar risks are retained; no trajectory arrays are stored.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "experiments"), str(ROOT / "src")]

import closure_probe_20261030 as probe  # noqa: E402

P4_SCHEMA = "closure_probe_20261030_p4_checkpoint_v1"
P4_SUMMARY_SCHEMA = "closure_probe_20261030_p4_summary_v1"
P4_PUBLIC_SCHEMA = "closure_probe_20261030_p4_public_v1"

P4_PRIVATE = probe.PRIVATE_ROOT / "p4_step64"
P4_PUBLIC = probe.RESULTS_ROOT / "p4_step64_public.json"
CHECKPOINTS = {
    "step64": probe.TRANSPORT_STUDY / "A_30311/checkpoints/step000064/model.json",
    "step128": probe.TRANSPORT_STUDY / "A_30311/checkpoints/step000128/model.json",
}
PANEL48_PUBLIC = ROOT / "results/memory_transport_20261029/A_30311_completed_public.json"

FINE_DT = 0.005
FINE_STEPS = 1600  # 8 s at 0.005 s; first 1601 rows of the 15 s production rollout
HISTORY_DT_REFERENCE = 0.00125
HORIZONS = {"5": 500, "8": 800}
BLOCK = 16
GATE_ATOL = 1e-9
PANEL48_5S = {"step64": 0.47995563880415043, "step128": 0.5102582083889101}

CURVE_SCHEMA = "closure_probe_20261030_p4b_curve_v1"
CURVE_PUBLIC_SCHEMA = "closure_probe_20261030_p4b_curve_public_v1"
CURVE_STEPS = (0, 16, 32, 48, 64, 80, 96, 112, 128)
CURVE_CHECKPOINT_DIR = probe.TRANSPORT_STUDY / "A_30311/checkpoints"
CURVE_PRIVATE = probe.PRIVATE_ROOT / "p4b_curve"
CURVE_PUBLIC = probe.RESULTS_ROOT / "p4b_curve_public.json"
CURVE_GATE_ATOL = 1e-16
EXISTING_P4 = {
    64: probe.PRIVATE_ROOT / "p4_step64/p4_step64.json",
    128: probe.PRIVATE_ROOT / "p4_step64/p4_step128.json",
}
# Used only if the preceding P4 private artifacts are unavailable.
FALLBACK_P4 = {
    64: {
        "risk_5s": 0.8172485484178245,
        "risk_8s": 0.5989693104826443,
        "common_omega_rmse_8s": 0.3160953510604371,
        "normalized_p_rmse_8s": 0.4416262549657568,
    },
    128: {
        "risk_5s": 0.8675373355146679,
        "risk_8s": 0.6255711284192053,
        "common_omega_rmse_8s": 0.32140212114056016,
        "normalized_p_rmse_8s": 0.4568781908441448,
    },
}


def _sha(path: Path) -> str:
    return probe.sha256_file(path)


def _load_panel48() -> dict:
    """Read the published fixed-panel48 milestones; reference only, never mixed with fit297."""
    payload = json.loads(PANEL48_PUBLIC.read_text(encoding="utf-8"))
    milestones = payload["fixed_panel"]["milestones"]
    value64 = float(milestones["64"]["risks"]["stable_complete_horizons_s"]["5"]["value"])
    value128 = float(milestones["128"]["risks"]["stable_complete_horizons_s"]["5"]["value"])
    reference = {
        "source": PANEL48_PUBLIC.relative_to(ROOT).as_posix(),
        "sha256": _sha(PANEL48_PUBLIC),
        "cohort": "fixed panel48 (36 stable / 12 unstable), not the fit297 stable cohort",
        "step64_5s": value64,
        "step128_5s": value128,
        "delta_128_minus_64_5s": value128 - value64,
        "direction": "rise" if value128 > value64 else ("fall" if value128 < value64 else "flat"),
        "lead_reference_match": bool(
            abs(value64 - PANEL48_5S["step64"]) < 1e-8 and abs(value128 - PANEL48_5S["step128"]) < 1e-8
        ),
    }
    return reference


def _rollout(model, histories: np.ndarray) -> np.ndarray:
    """Registered fine forward rollout for the first 8 s; no gradient, no cache leak."""
    from state_gated_memory_study_20261028 import HISTORY_DT_REFERENCE as registered_history_dt

    if float(registered_history_dt) != HISTORY_DT_REFERENCE:
        raise ValueError("registered reference history dt changed")
    tensor = torch.as_tensor(np.asarray(histories), dtype=torch.float64)
    with torch.no_grad():
        model.begin_graph_cache()
        try:
            path = model.rollout(
                tensor, FINE_STEPS, FINE_DT, history_dt=registered_history_dt
            )
        finally:
            model.end_graph_cache()
    path = path.detach().cpu().numpy()
    if path.shape != (len(histories), FINE_STEPS + 1, 40):
        raise ValueError(f"unexpected rollout shape {path.shape}")
    return path


def _event_row(model, event: dict, path: np.ndarray) -> dict:
    """Per-event 5 s / 8 s risks and the two pooled-8s RMSE components."""
    from train_structured_rom import _point_error

    mass = model.mass.detach().cpu().numpy()
    p_scale = model.p_scale.detach().cpu().numpy()
    row = {"event_id": event["id"]}
    point_maps = {}
    for horizon, steps in HORIZONS.items():
        pred_np = probe.fine_slice(path, steps)
        truth_np = probe.truth_slice(event, steps)
        if not (np.isfinite(pred_np).all() and np.isfinite(truth_np).all()):
            return {"event_id": event["id"], "finite": False}
        pred = torch.as_tensor(pred_np, dtype=torch.float64)
        truth = torch.as_tensor(truth_np, dtype=torch.float64)
        with torch.no_grad():
            point = _point_error(pred, truth, model)
        row[f"point_error_{horizon}"] = float(point.mean())
        point_maps[horizon] = (pred_np, truth_np)
    pred8, truth8 = point_maps["8"]
    omega_error = pred8[:, 10:20] - truth8[:, 10:20]
    common = omega_error @ mass / mass.sum()
    row["common_omega_sse"] = float(common @ common)
    row["common_omega_n"] = int(common.size)
    p_error = (pred8[:, 20:30] - truth8[:, 20:30]) / p_scale[None, :]
    row["normalized_p_sse"] = float(np.sum(p_error**2))
    row["normalized_p_n"] = int(p_error.size)
    row["finite"] = True
    return row


def evaluate_checkpoint(label: str, checkpoint: Path, events: list[dict], identity: dict) -> dict:
    """Forward-only evaluation of one frozen checkpoint on the stable fit297 cohort."""
    started = time.perf_counter()
    torch.set_num_threads(1)
    model = probe.load_model({"export": checkpoint, "loader": "transport"})
    probe._assert_model_scales(model, identity)
    if float(model.history_dt) != 0.0025:
        raise ValueError(f"{label}: history_dt is not 0.0025")
    stable = sorted([event for event in events if event["label"] == "stable"], key=lambda event: event["id"])
    if len(stable) != probe.EXPECTED_STABLE:
        raise ValueError(f"stable denominator changed: {len(stable)}")
    for event in stable:
        steps = probe.scope_steps(event, "8")
        if steps is None:
            raise ValueError(f"{event['id']}: stable complete-8s unavailable")
    rows, unresolved = [], []
    block_times = []
    for first in range(0, len(stable), BLOCK):
        block = stable[first : first + BLOCK]
        histories = np.stack([event["x"][: probe.HISTORY_ROWS] for event in block])
        tick = time.perf_counter()
        paths = _rollout(model, histories)
        block_times.append(time.perf_counter() - tick)
        for i, event in enumerate(block):
            row = _event_row(model, event, paths[i])
            if not row.get("finite"):
                unresolved.append(event["id"])
            else:
                rows.append(row)
    if unresolved:
        raise FloatingPointError(f"{label}: nonfinite rollout for {len(unresolved)} stable events")
    if len(rows) != probe.EXPECTED_STABLE:
        raise ValueError(f"{label}: eligible {len(rows)} != {probe.EXPECTED_STABLE}")
    metrics = {
        "risk_5s": probe.risk_aggregate([row["point_error_5"] for row in rows]),
        "risk_8s": probe.risk_aggregate([row["point_error_8"] for row in rows]),
    }
    pooled = probe.pooled_8s_rmse(rows)
    metrics.update(
        {
            "common_omega_rmse_8s": pooled["common_omega_rmse_8s"],
            "normalized_p_rmse_8s": pooled["normalized_p_rmse_8s"],
            "common_omega_n": pooled["common_omega_n"],
            "normalized_p_n": pooled["normalized_p_n"],
        }
    )
    payload = {
        "schema": P4_SCHEMA,
        "status": "COMPLETED",
        "label": label,
        "role": "forward-only evaluation; no training and no H3 conclusion",
        "checkpoint": {
            "path": checkpoint.relative_to(ROOT).as_posix(),
            "sha256": _sha(checkpoint),
            "history_dt": float(model.history_dt),
            "mass_p_scale_sha256": identity["scales"]["sha256"],
        },
        "cohort": {
            "fit297_events": len(events),
            "stable": probe.EXPECTED_STABLE,
            "eligible": len(rows),
            "unstable_scored": 0,
            "note": "requested metrics are stable-only; the 30 unstable events are not scored",
        },
        "method": {
            "fine_dt": FINE_DT,
            "fine_steps": FINE_STEPS,
            "history_dt_reference": HISTORY_DT_REFERENCE,
            "path": "fine[::2][1:steps+1, :30] versus x[11:11+steps, :30]",
            "horizon_steps": dict(HORIZONS),
            "blocks": len(block_times),
            "block_size": BLOCK,
            "rollout": "true RK4, graph cache, CPU float64, torch.set_num_threads(1)",
            "predictions_stored": False,
            "avoided_fine_array_bytes_per_checkpoint": int(
                probe.EXPECTED_STABLE * (FINE_STEPS + 1) * 40 * 8
            ),
        },
        "metrics": metrics,
        "per_event": rows,
        "runtime": probe._runtime(
            started,
            {
                "rollout_seconds": float(sum(block_times)),
                "rollout_blocks": len(block_times),
            },
        ),
        "limits": "Descriptive forward metrics only; no H3, mechanism or model-selection conclusion.",
    }
    return payload


def _p1_cross_check(rows: list[dict]) -> dict | None:
    """Optional diagnostic: compare fresh step128 per-event risks with the frozen-prediction probe."""
    p1_path = probe.PRIVATE_ROOT / "p1.json"
    if not p1_path.is_file():
        return None
    payload = json.loads(p1_path.read_text(encoding="utf-8"))
    reference = payload.get("models", {}).get("A", {}).get("per_event", {})
    result = {"source": p1_path.relative_to(ROOT).as_posix(), "sha256": _sha(p1_path)}
    for scope, key in (("5", "point_error_5"), ("8", "point_error_8")):
        cells = reference.get(scope, {})
        diffs = [
            abs(row[key] - float(cells[row["event_id"]]["total"]))
            for row in rows
            if row["event_id"] in cells
        ]
        result[scope] = {
            "compared_events": len(diffs),
            "max_abs_diff": max(diffs) if diffs else None,
            "atol_1e-9_passed": bool(diffs) and max(diffs) < GATE_ATOL,
        }
    return result


def _public_copy(summary: dict) -> dict:
    """Aggregate-only public payload; no per-event rows and no trajectory arrays."""
    public = {
        "schema": P4_PUBLIC_SCHEMA,
        "status": summary["status"],
        "role": summary["role"],
        "generated_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "cohort": summary["cohort"],
        "method": summary["method"],
        "checkpoints": summary["checkpoints"],
        "comparability_gate": summary["comparability_gate"],
        "step64": summary["step64"],
        "step128": summary["step128"],
        "delta_128_minus_64": summary["delta_128_minus_64"],
        "panel48_reference": summary["panel48_reference"],
        "direction_5s": summary["direction_5s"],
        "denominators": summary["denominators"],
        "evidence": summary["evidence"],
        "runtime": summary["runtime"],
        "boundaries": [
            "Both checkpoints are frozen; this job only runs forward RK4, no training or optimizer.",
            "Only the 267 stable fit297 events are scored; the 30 unstable events are not scored.",
            "Panel48 values are displayed as a separate fixed cohort reference and are not mixed with fit297.",
            "No H3, mechanism, optimization or model-selection conclusion is asserted.",
        ],
    }
    probe._assert_public(public)
    return public


def run(private_dir: Path, public_path: Path) -> dict:
    """Evaluate step128 (gate) then step64; write private and public summaries."""
    started = time.perf_counter()
    torch.set_num_threads(1)
    record, events = probe.load_frozen_events()
    identity = probe.verify_frozen_scales()
    private_dir = probe._resolve(private_dir)
    private_dir.relative_to(probe._resolve(probe.PRIVATE_ROOT))
    public_path = probe.guard_output(public_path, probe.RESULTS_ROOT)

    step128 = evaluate_checkpoint("step128", CHECKPOINTS["step128"], events, identity)
    probe.write_json(private_dir / "p4_step128.json", step128)

    expected = {
        "risk_5s": probe.PUBLIC_VALUES[("A", "5")],
        "risk_8s": probe.PUBLIC_VALUES[("A", "8")],
        "common_omega_rmse_8s": probe.RMSE_GATES[("A", "common_omega_rmse_8s")],
        "normalized_p_rmse_8s": probe.RMSE_GATES[("A", "normalized_p_rmse_8s")],
    }
    actual = {key: step128["metrics"][key] for key in expected}
    errors = {key: abs(actual[key] - expected[key]) for key in expected}
    gate = {
        "passed": bool(
            errors["risk_5s"] < GATE_ATOL and errors["risk_8s"] < GATE_ATOL
        ),
        "atol": GATE_ATOL,
        "required": ["risk_5s", "risk_8s"],
        "expected": expected,
        "actual": actual,
        "abs_error": errors,
        "rmse_abs_error": {
            key: errors[key] for key in ("common_omega_rmse_8s", "normalized_p_rmse_8s")
        },
        "note": (
            "risk 5/8 s are the required comparability gate; RMSE and per-event cross-check are "
            "additional diagnostics"
        ),
    }
    cross_check = _p1_cross_check(step128["per_event"]) if gate["passed"] else None
    if cross_check is not None:
        gate["frozen_prediction_cross_check"] = cross_check

    panel48 = _load_panel48()
    step64 = None
    if gate["passed"]:
        step64 = evaluate_checkpoint("step64", CHECKPOINTS["step64"], events, identity)
        probe.write_json(private_dir / "p4_step64.json", step64)
        delta = {
            key: step128["metrics"][key] - step64["metrics"][key]
            for key in ("risk_5s", "risk_8s", "common_omega_rmse_8s", "normalized_p_rmse_8s")
        }
        direction = {
            "full_cohort_5s_delta": delta["risk_5s"],
            "full_cohort_5s_direction": (
                "rise" if delta["risk_5s"] > 0 else ("fall" if delta["risk_5s"] < 0 else "flat")
            ),
            "panel48_5s_delta": panel48["delta_128_minus_64_5s"],
            "panel48_5s_direction": panel48["direction"],
            "direction_match_5s": bool(
                (delta["risk_5s"] > 0) == (panel48["delta_128_minus_64_5s"] > 0)
            ),
            "scope": (
                "direction comparison only: panel48 is a fixed 36-stable cohort; fit297 contains its "
                "own 267 stable events and is not the same cohort"
            ),
        }
        status = "COMPLETED"
    else:
        delta = None
        direction = None
        status = "COMPARABILITY_FAILED"
        step64 = {
            "status": "NOT_RELEASED",
            "reason": "step128 did not reproduce the published 5 s / 8 s risks at atol 1e-9",
        }

    summary = {
        "schema": P4_SUMMARY_SCHEMA,
        "status": status,
        "role": (
            "P4 H3 cheap check: forward-only comparison of frozen A step64 and step128 on the full "
            "fit297 stable cohort; descriptive facts only, no H3 conclusion"
        ),
        "cohort": {
            "fit297_events": len(events),
            "stable": probe.EXPECTED_STABLE,
            "unstable_scored": 0,
        },
        "method": step128["method"],
        "checkpoints": {
            "step64": step64.get("checkpoint") if step64.get("checkpoint") else {
                "path": CHECKPOINTS["step64"].relative_to(ROOT).as_posix(),
                "sha256": _sha(CHECKPOINTS["step64"]),
            },
            "step128": step128["checkpoint"],
            "step64_evaluated": bool(step64.get("checkpoint")),
        },
        "comparability_gate": gate,
        "step64": step64["metrics"] if step64.get("metrics") else step64,
        "step128": step128["metrics"],
        "delta_128_minus_64": delta,
        "panel48_reference": panel48,
        "direction_5s": direction,
        "denominators": {
            "step64": step64.get("cohort") if step64.get("cohort") else None,
            "step128": step128["cohort"],
        },
        "evidence": {
            "observed": [
                "step128 fresh forward 5 s / 8 s risks and pooled RMSE (comparability gate)",
                "step64 fresh forward 5 s / 8 s risks and pooled RMSE",
                "267/267 stable fit297 denominator per checkpoint",
                "panel48 published 64/128 5 s values (separate fixed cohort reference)",
            ],
            "diagnostic": [
                "step128 per-event cross-check against the frozen-prediction decomposition",
                "full-cohort 64 to 128 direction versus panel48 direction (direction only)",
            ],
            "unverified": [
                "any H3 interpretation, mechanism attribution or optimization-sufficiency claim",
                "statistical uncertainty, noise model or cross-cohort generalization",
                "model adoption or checkpoint selection",
            ],
        },
        "runtime": probe._runtime(
            started,
            {
                "step128_wall_seconds": step128["runtime"]["wall_seconds"],
                "step64_wall_seconds": step64["runtime"]["wall_seconds"] if step64.get("runtime") else None,
                "predictions_stored": False,
            },
        ),
        "limits": (
            "Forward-only descriptive evaluation. 64 vs 128 differences are reported as facts; panel48 "
            "is a different cohort and is shown only as a direction reference."
        ),
    }
    probe.write_json(private_dir / "p4_summary.json", summary)
    public = _public_copy(summary)
    probe.write_json(public_path, public)
    return summary


def _curve_references() -> dict:
    """Reference values for the 64/128 comparability gate (preceding P4 artifacts)."""
    references = {}
    for step in (64, 128):
        path = EXISTING_P4[step]
        if path.is_file():
            payload = json.loads(path.read_text(encoding="utf-8"))
            references[step] = {
                "source": path.relative_to(ROOT).as_posix(),
                "sha256": _sha(path),
                "metrics": {key: float(payload["metrics"][key]) for key in FALLBACK_P4[step]},
                "provenance": "preceding P4 forward evaluation",
            }
        else:
            references[step] = {
                "source": "hardcoded fallback from the P4 run",
                "sha256": None,
                "metrics": dict(FALLBACK_P4[step]),
                "provenance": "fallback constant",
            }
    return references


def _curve_row(step: int, payload: dict) -> dict:
    """Scalar-only row; no per-event row or trajectory is retained for the curve."""
    return {
        "step": int(step),
        "checkpoint": payload["checkpoint"],
        "risk_5s": payload["metrics"]["risk_5s"],
        "risk_8s": payload["metrics"]["risk_8s"],
        "common_omega_rmse_8s": payload["metrics"]["common_omega_rmse_8s"],
        "normalized_p_rmse_8s": payload["metrics"]["normalized_p_rmse_8s"],
        "eligible": payload["cohort"]["eligible"],
        "wall_seconds": payload["runtime"]["wall_seconds"],
        "rollout_seconds": payload["runtime"]["rollout_seconds"],
    }


def _curve_public(summary: dict) -> dict:
    gate = summary["comparability_gate"]
    public_gate = {
        "passed": gate["passed"],
        "atol": gate["atol"],
        "rows": gate["rows"],
        "references": {
            f"step{int(step):06d}": value for step, value in gate["references"].items()
        },
        "note": gate["note"],
    }
    public = {
        "schema": CURVE_PUBLIC_SCHEMA,
        "status": summary["status"],
        "role": summary["role"],
        "generated_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "cohort": summary["cohort"],
        "method": summary["method"],
        "checkpoints": summary["checkpoints"],
        "curve": summary["curve"],
        "comparability_gate": public_gate,
        "best_step": summary["best_step"],
        "denominators": summary["denominators"],
        "evidence": summary["evidence"],
        "runtime": summary["runtime"],
        "boundaries": [
            "Forward-only descriptive curve over frozen checkpoints; no training or optimizer.",
            "Only the 267 stable fit297 events are scored; the 30 unstable events are not scored.",
            "The 64/128 comparability gate re-runs the same evaluation path and must match the "
            "preceding P4 values at atol 1e-16.",
            "Argmin is a descriptive location on this cohort only; it is not model selection.",
            "No H3, mechanism or optimization-sufficiency conclusion is asserted.",
        ],
    }
    probe._assert_public(public)
    return public


def publish_curve(private_dir: Path, public_path: Path) -> dict:
    """Rebuild the aggregate-only P4b public file from the completed private curve."""
    private_dir = probe._resolve(private_dir)
    private_dir.relative_to(probe._resolve(probe.PRIVATE_ROOT))
    public_path = probe.guard_output(public_path, probe.RESULTS_ROOT)
    summary = json.loads((private_dir / "curve.json").read_text(encoding="utf-8"))
    public = _curve_public(summary)
    probe.write_json(public_path, public)
    return public


def curve(private_dir: Path, public_path: Path) -> dict:
    """P4b: evaluate all frozen A_30311 checkpoints on the stable fit297 cohort."""
    started = time.perf_counter()
    torch.set_num_threads(1)
    record, events = probe.load_frozen_events()
    identity = probe.verify_frozen_scales()
    private_dir = probe._resolve(private_dir)
    private_dir.relative_to(probe._resolve(probe.PRIVATE_ROOT))
    public_path = probe.guard_output(public_path, probe.RESULTS_ROOT)
    references = _curve_references()

    rows, checkpoint_meta = [], {}
    for step in CURVE_STEPS:
        label = f"step{step:06d}"
        checkpoint = CURVE_CHECKPOINT_DIR / label / "model.json"
        if not checkpoint.is_file():
            raise FileNotFoundError(checkpoint)
        payload = evaluate_checkpoint(label, checkpoint, events, identity)
        rows.append(_curve_row(step, payload))
        checkpoint_meta[label] = {
            "path": payload["checkpoint"]["path"],
            "sha256": payload["checkpoint"]["sha256"],
            "history_dt": payload["checkpoint"]["history_dt"],
            "mass_p_scale_sha256": payload["checkpoint"]["mass_p_scale_sha256"],
        }
        probe.write_json(
            private_dir / "partial.json",
            {
                "schema": CURVE_SCHEMA,
                "status": "RUNNING",
                "completed_steps": [row["step"] for row in rows],
                "curve": rows,
            },
        )

    gate_rows = []
    for step in (64, 128):
        actual_row = next(row for row in rows if row["step"] == step)
        for key in ("risk_5s", "risk_8s", "common_omega_rmse_8s", "normalized_p_rmse_8s"):
            reference = references[step]["metrics"][key]
            actual = actual_row[key]
            error = abs(actual - reference)
            bitwise = actual == reference
            gate_rows.append(
                {
                    "step": step,
                    "metric": key,
                    "reference": reference,
                    "actual": actual,
                    "abs_error": error,
                    "bitwise_equal": bool(bitwise),
                    "passed": bool(bitwise or error <= CURVE_GATE_ATOL),
                    "reference_source": references[step]["source"],
                }
            )
    gate_passed = all(row["passed"] for row in gate_rows)

    best_5s = min(rows, key=lambda row: row["risk_5s"])
    best_8s = min(rows, key=lambda row: row["risk_8s"])
    best_step = {
        "risk_5s": {"step": best_5s["step"], "value": best_5s["risk_5s"]},
        "risk_8s": {"step": best_8s["step"], "value": best_8s["risk_8s"]},
        "note": (
            "descriptive argmin over the 267 stable fit297 events; no composite score and no model "
            "selection"
        ),
    }
    summary = {
        "schema": CURVE_SCHEMA,
        "status": "COMPLETED" if gate_passed else "COMPARABILITY_FAILED",
        "role": (
            "P4b full-cohort risk-versus-step curve of nine frozen A_30311 checkpoints; descriptive "
            "only, no training, no model selection and no H3 conclusion"
        ),
        "cohort": {
            "fit297_events": len(events),
            "stable": probe.EXPECTED_STABLE,
            "unstable_scored": 0,
        },
        "method": {
            "fine_dt": FINE_DT,
            "fine_steps": FINE_STEPS,
            "history_dt_reference": HISTORY_DT_REFERENCE,
            "path": "fine[::2][1:steps+1, :30] versus x[11:11+steps, :30]",
            "horizon_steps": dict(HORIZONS),
            "block_size": BLOCK,
            "rollout": "true RK4, graph cache, CPU float64, torch.set_num_threads(1)",
            "predictions_stored": False,
            "avoided_fine_array_bytes_per_checkpoint": int(
                probe.EXPECTED_STABLE * (FINE_STEPS + 1) * 40 * 8
            ),
        },
        "checkpoints": checkpoint_meta,
        "curve": rows,
        "comparability_gate": {
            "passed": gate_passed,
            "atol": CURVE_GATE_ATOL,
            "rows": gate_rows,
            "references": references,
            "note": "64 and 128 must re-run the same path and match the preceding P4 values",
        },
        "best_step": best_step,
        "denominators": {
            "per_checkpoint": {
                f"step{row['step']:06d}": {
                    "stable": probe.EXPECTED_STABLE,
                    "eligible": row["eligible"],
                }
                for row in rows
            }
        },
        "evidence": {
            "observed": [
                "risk-vs-step curve for nine frozen A_30311 checkpoints on the stable fit297 cohort",
                "64/128 comparability gate re-run on the same evaluation path",
                "267/267 stable denominator for every checkpoint",
            ],
            "diagnostic": [
                "descriptive argmin step for 5 s and 8 s risks",
            ],
            "unverified": [
                "any H3 interpretation or mechanism attribution",
                "statistical uncertainty, noise model or cross-cohort generalization",
                "model selection, adoption or optimization-sufficiency conclusion",
            ],
        },
        "runtime": probe._runtime(
            started,
            {
                "checkpoints_evaluated": len(rows),
                "total_checkpoint_wall_seconds": float(sum(row["wall_seconds"] for row in rows)),
                "predictions_stored": False,
            },
        ),
        "limits": (
            "Forward-only descriptive curve. The argmin is a location on this cohort, not a selection "
            "decision, and no H3 conclusion is drawn."
        ),
    }
    probe.write_json(private_dir / "curve.json", summary)
    probe.write_json(public_path, _curve_public(summary))
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="probe64_20261030",
        description="P4 frozen A step64 vs step128 forward evaluation (read-only, no training).",
    )
    sub = parser.add_subparsers(dest="command", required=True)
    run_parser = sub.add_parser("run", help="evaluate step128 gate then step64 and publish")
    run_parser.add_argument("--private-dir", type=Path, default=P4_PRIVATE)
    run_parser.add_argument("--public", type=Path, default=P4_PUBLIC)
    curve_parser = sub.add_parser("curve", help="P4b nine-checkpoint risk-versus-step curve")
    curve_parser.add_argument("--private-dir", type=Path, default=CURVE_PRIVATE)
    curve_parser.add_argument("--public", type=Path, default=CURVE_PUBLIC)
    publish_parser = sub.add_parser(
        "publish-curve", help="rebuild the aggregate P4b public file from a completed curve"
    )
    publish_parser.add_argument("--private-dir", type=Path, default=CURVE_PRIVATE)
    publish_parser.add_argument("--public", type=Path, default=CURVE_PUBLIC)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "run":
        summary = run(args.private_dir, args.public)
        print(
            json.dumps(
                {
                    "status": summary["status"],
                    "comparability_gate": summary["comparability_gate"]["passed"],
                    "step128_risk_5s": summary["comparability_gate"]["actual"]["risk_5s"],
                    "step64_risk_5s": summary["step64"].get("risk_5s") if isinstance(summary["step64"], dict) else None,
                    "step64_risk_8s": summary["step64"].get("risk_8s") if isinstance(summary["step64"], dict) else None,
                    "wall_seconds": summary["runtime"]["wall_seconds"],
                    "peak_working_set_mb": summary["runtime"]["peak_working_set_mb"],
                }
            )
        )
        return 0 if summary["status"] == "COMPLETED" else 1
    if args.command == "curve":
        summary = curve(args.private_dir, args.public)
        print(
            json.dumps(
                {
                    "status": summary["status"],
                    "comparability_gate": summary["comparability_gate"]["passed"],
                    "best_5s": summary["best_step"]["risk_5s"],
                    "best_8s": summary["best_step"]["risk_8s"],
                    "wall_seconds": summary["runtime"]["wall_seconds"],
                    "peak_working_set_mb": summary["runtime"]["peak_working_set_mb"],
                }
            )
        )
        return 0 if summary["status"] == "COMPLETED" else 1
    if args.command == "publish-curve":
        public = publish_curve(args.private_dir, args.public)
        print(
            json.dumps(
                {
                    "status": public["status"],
                    "comparability_gate": public["comparability_gate"]["passed"],
                    "best_5s": public["best_step"]["risk_5s"],
                    "best_8s": public["best_step"]["risk_8s"],
                }
            )
        )
        return 0 if public["status"] == "COMPLETED" else 1
    parser.error(f"unknown command {args.command}")  # pragma: no cover
    return 2  # pragma: no cover


if __name__ == "__main__":
    raise SystemExit(main())
