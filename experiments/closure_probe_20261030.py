"""Read-only closure probe: P1 four-term risk decomposition and P2 r0 conditioning.

Scope rules
-----------
* No training, no optimizer, no model rollout and no new prediction file.
* The only event payload read is the frozen fit297 preparation (267 stable, 30 unstable).
* P1 consumes the four frozen prediction sets (R_N, A128, S128, P128) and reproduces the
  published risks through the registered metric functions.
* P2 calls the frozen A128 ``history_system`` on the frozen 11x30 causal histories only.
* Every output path is confined to ``private_data/closure_probe_20261030`` or
  ``results/closure_probe_20261030``; val/test are never addressed.

P1 alignment (confirmed against summarize_train_local_memory_0098.event_metrics)
---------------------------------------------------------------------------------
The published stable 5 s / 8 s risks and the unstable actual-interval risks use the
0.005 s fine path decimated by two: ``fine[::2][1:steps+1, :30]`` against
``x[11:11+steps, :30]``.  The 0.01 s coarse file is a cross-check only.  The graded
public values are reproduced at atol 1e-9 before anything is published.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import time
import uuid
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "experiments"), str(ROOT / "src")]

P1_SCHEMA = "closure_probe_20261030_p1_v1"
P2_SCHEMA = "closure_probe_20261030_p2_v1"
PUBLIC_SCHEMA = "closure_probe_20261030_public_v1"

SGM_STUDY = ROOT / "private_data/state_gated_memory_20261028/study01"
SGM_A96_EXPORT = SGM_STUDY / "study/A_28311/checkpoints/step000096/model.json"
RN_SOURCE_EXPORT = ROOT / (
    "private_data/force_graph_20261028/study01/study/N6101_M_28102/checkpoints/step000192/model.json"
)
TRANSPORT_STUDY = ROOT / "private_data/memory_transport_20261029/study01_prelaunch_retry1/study"

MODEL_SPECS = {
    "R_N": {
        "export": RN_SOURCE_EXPORT,
        "predictions": SGM_STUDY / "qualification/R_N_fit297",
        "loader": "force_graph",
        "scope": "frozen R_N fit297 predictions (force-graph N_M192 source)",
    },
    "A": {
        "export": TRANSPORT_STUDY / "A_30311/checkpoints/step000128/model.json",
        "predictions": TRANSPORT_STUDY / "A_30311/fit000128",
        "loader": "transport",
        "scope": "frozen A transport checkpoint step128",
    },
    "S": {
        "export": TRANSPORT_STUDY / "S_30311/checkpoints/step000128/model.json",
        "predictions": TRANSPORT_STUDY / "S_30311/fit000128",
        "loader": "transport",
        "scope": "frozen S transport checkpoint step128",
    },
    "P": {
        "export": TRANSPORT_STUDY / "P_30311/checkpoints/step000128/model.json",
        "predictions": TRANSPORT_STUDY / "P_30311/fit000128",
        "loader": "transport",
        "scope": "frozen P transport checkpoint step128",
    },
}
MODEL_ORDER = ("R_N", "A", "S", "P")
A96_SPEC = {
    "export": SGM_A96_EXPORT,
    "loader": "state_gated",
    "scope": "frozen SGM A96 source checkpoint (initializer-only sensitivity reference)",
}
SCOPES = ("5", "8", "prefix")
SCOPE_STEPS = {"5": 500, "8": 800}
MAX_ACTUAL_STEPS = 1500
FINE_STRIDE = 2
PREDICTION_CHUNKS = 19
FINETIME_POINTS = 3001
COARSE_TIME_POINTS = 1501
HISTORY_ROWS = 11
TERM_NAMES = ("delta_coi", "omega_coi", "common_omega", "p_normalized")
N_TERMS = len(TERM_NAMES)
EXPECTED_EVENTS = 297
EXPECTED_STABLE = 267
EXPECTED_UNSTABLE = 30
ACCEPTANCE_ATOL = 1e-9
TERM_SUM_ATOL = 1e-12
RIDGE_SCAN = (1e-4, 1e-6, 1e-8)
EFFECTIVE_RANK_FACTOR = 1e-8
RIDGE_DOMINANCE_SHARE = 0.5

# Published risk values requested by the task; the acceptance gate is applied to all of them.
PUBLIC_VALUES = {
    ("A", "5"): 0.8675373355146678,
    ("A", "8"): 0.6255711284192054,
    ("A", "prefix"): 3.045603054019912,
    ("S", "5"): 0.8697658541541666,
    ("S", "8"): 0.6260437277394373,
    ("S", "prefix"): 3.0611703963049894,
    ("P", "5"): 0.868234834825832,
    ("P", "8"): 0.6259765602702372,
    ("P", "prefix"): 3.0375187293837564,
    ("R_N", "5"): 0.8304206726680337,
    ("R_N", "8"): 0.5917623840131517,
    ("R_N", "prefix"): 3.512358079792409,
}
PUBLIC_SCOPE_DEFINITION = {
    "5": "stable events, fine[::2][1:501] vs x[11:511]",
    "8": "stable events, fine[::2][1:801] vs x[11:811]",
    "prefix": "unstable events, fine[::2][1:n+1] vs x[11:11+n], n=min(1500,len(x)-11)",
}
# Additional published pooled values; A/R_N are gate values, S/P are reported descriptively.
RMSE_GATES = {
    ("A", "common_omega_rmse_8s"): 0.32140212114056005,
    ("R_N", "common_omega_rmse_8s"): 0.3081470782764491,
    ("A", "normalized_p_rmse_8s"): 0.45687819084414505,
    ("R_N", "normalized_p_rmse_8s"): 0.4426070157702325,
}
EXPECTED_DENOMINATORS = {"5": EXPECTED_STABLE, "8": EXPECTED_STABLE, "prefix": EXPECTED_UNSTABLE}

PRIVATE_ROOT = ROOT / "private_data/closure_probe_20261030"
RESULTS_ROOT = ROOT / "results/closure_probe_20261030"
DATA_SOURCE_PATHS = (
    SGM_STUDY,
    SGM_A96_EXPORT,
    RN_SOURCE_EXPORT,
    TRANSPORT_STUDY,
)


# --------------------------------------------------------------------------- I/O


def sha256_file(path: Path) -> str:
    """Hash file bytes; no interpretation of the payload."""
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _resolve(path: Path | str) -> Path:
    """Resolve CLI paths against the repository root, not the caller's cwd."""
    p = Path(path)
    if not p.is_absolute():
        p = ROOT / p
    return p.resolve()


def guard_output(path: Path | str, root: Path) -> Path:
    """Return the resolved output path only if it is a .json file under ``root``."""
    p = _resolve(path)
    root = _resolve(root)
    try:
        p.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"output path outside the allowed write root {root}: {p}") from exc
    if p.suffix.lower() != ".json":
        raise ValueError(f"output must be a .json file: {p}")
    return p


def guard_input(path: Path | str, root: Path) -> Path:
    """Return the resolved input path only if it is under the probe private root."""
    p = _resolve(path)
    root = _resolve(root)
    try:
        p.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"input path outside the probe private root {root}: {p}") from exc
    if not p.is_file():
        raise FileNotFoundError(p)
    return p


def write_json(path: Path | str, payload: dict) -> Path:
    """Atomic JSON write with NaN/Infinity forbidden."""
    p = _resolve(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(f".{p.name}.{uuid.uuid4().hex}.tmp")
    text = json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False)
    tmp.write_text(text + "\n", encoding="utf-8")
    os.replace(tmp, p)
    return p


def read_json(path: Path | str) -> dict:
    p = _resolve(path)
    if not p.is_file():
        raise FileNotFoundError(p)
    return json.loads(p.read_text(encoding="utf-8"))


# ----------------------------------------------------------------------- deps


def _deps():
    """Lazy import of the frozen modules; importing this probe never reads data."""
    import state_gated_memory_study_20261028 as sgm
    from train_structured_rom import METRIC, _coi, _point_error
    from train_local_memory_0098 import risk

    return sgm, METRIC, _coi, _point_error, risk


def _ridge() -> float:
    from force_graph_model_20261028 import RIDGE

    return float(RIDGE)


# ------------------------------------------------------------------ frozen data


def load_frozen_events():
    """Load and validate the frozen fit297 preparation (read-only)."""
    sgm, _, _, _, _ = _deps()
    record, events = sgm.checked_preparation(SGM_STUDY, fit_only=True)
    if len(events) != EXPECTED_EVENTS:
        raise ValueError(f"fit297 changed: {len(events)} events")
    labels = [event["label"] for event in events]
    if labels.count("stable") != EXPECTED_STABLE or labels.count("unstable") != EXPECTED_UNSTABLE:
        raise ValueError("stable/unstable denominator changed")
    ids = [event["id"] for event in events]
    if len(set(ids)) != EXPECTED_EVENTS:
        raise ValueError("duplicate event id in fit297")
    for event in events:
        if event["x"].shape[1] != 30 or event["x"].dtype != np.float64:
            raise ValueError(f"unexpected event array for {event['id']}")
        if np.asarray(event["x"]).shape[0] != len(event["x"]):
            raise ValueError("event array shape mismatch")
    if record.get("val_read") is not False or record.get("test_read") is not False:
        raise ValueError("frozen preparation does not certify val/test closed")
    return record, events


def _payload_scales(payload: dict, name: str) -> tuple[list[float], list[float]]:
    state = payload.get("state_dict")
    if not isinstance(state, dict) or "mass" not in state or "p_scale" not in state:
        raise ValueError(f"{name}: checkpoint lacks mass/p_scale")
    mass = [float(v) for v in state["mass"]]
    p_scale = [float(v) for v in state["p_scale"]]
    if len(mass) != 10 or len(p_scale) != 10:
        raise ValueError(f"{name}: mass/p_scale must have ten entries")
    if any(not math.isfinite(v) or v <= 0 for v in mass + p_scale):
        raise ValueError(f"{name}: mass/p_scale must be finite and positive")
    return mass, p_scale


def verify_frozen_scales() -> dict:
    """Check mass/p_scale byte-identity across A/S/P, R_N and the SGM A96 source.

    The probe refuses to substitute scales: any mismatch aborts before scoring.
    """
    ridge = _ridge()
    if ridge != 1e-4:
        raise ValueError(f"force_graph RIDGE changed: {ridge}")
    records = {}
    for name, spec in MODEL_SPECS.items():
        path = Path(spec["export"])
        if not path.is_file():
            raise FileNotFoundError(path)
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("ridge") != ridge:
            raise ValueError(f"{name}: checkpoint ridge {payload.get('ridge')} != {ridge}")
        if name != "R_N" and float(payload.get("history_dt", -1)) != 0.0025:
            raise ValueError(f"{name}: history_dt is not the registered 0.0025")
        mass, p_scale = _payload_scales(payload, name)
        records[name] = {
            "path": path.relative_to(ROOT).as_posix(),
            "sha256": sha256_file(path),
            "mass": mass,
            "p_scale": p_scale,
        }
    canonical = records["A"]
    for name, item in records.items():
        if item["mass"] != canonical["mass"] or item["p_scale"] != canonical["p_scale"]:
            raise ValueError(f"{name}: mass/p_scale differs exactly from A step128")
    sgm_payload = json.loads(SGM_A96_EXPORT.read_text(encoding="utf-8"))
    sgm_mass, sgm_p_scale = _payload_scales(sgm_payload, "SGM_A96")
    if sgm_mass != canonical["mass"] or sgm_p_scale != canonical["p_scale"]:
        raise ValueError("SGM A96 source mass/p_scale differs; refusing to substitute scales")
    scale_blob = json.dumps(
        {"mass": canonical["mass"], "p_scale": canonical["p_scale"]},
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return {
        "ridge": ridge,
        "checkpoints": {name: {"path": item["path"], "sha256": item["sha256"]} for name, item in records.items()},
        "scales": {
            "mass": canonical["mass"],
            "p_scale": canonical["p_scale"],
            "sha256": hashlib.sha256(scale_blob).hexdigest(),
        },
        "sgm_source": {
            "path": SGM_A96_EXPORT.relative_to(ROOT).as_posix(),
            "sha256": sha256_file(SGM_A96_EXPORT),
            "mass_p_scale_identical_to_A128": True,
        },
        "check": "byte-exact equality of all fifteen checkpoint mass/p_scale lists; no scale substitution",
    }


def load_model(spec: dict):
    """Load one frozen model; inference is not run here."""
    if spec["loader"] == "force_graph":
        from force_graph_model_20261028 import ForceGraphROM

        model = ForceGraphROM.from_export(Path(spec["export"]))
    elif spec["loader"] == "transport":
        from memory_transport_model_20261029 import MemoryTransportROM

        model = MemoryTransportROM.from_export(Path(spec["export"]))
    elif spec["loader"] == "state_gated":
        from state_gated_memory_model_20261028 import StateGatedMemoryROM

        model = StateGatedMemoryROM.from_export(Path(spec["export"]))
    else:  # pragma: no cover - registry only
        raise ValueError(f"unknown loader {spec['loader']}")
    return model.to("cpu").double().eval()


def _assert_model_scales(model, identity: dict) -> None:
    mass = model.mass.detach().cpu().double().tolist()
    p_scale = model.p_scale.detach().cpu().double().tolist()
    if mass != identity["scales"]["mass"] or p_scale != identity["scales"]["p_scale"]:
        raise ValueError("loaded model mass/p_scale differs from the verified checkpoint payload")


def iter_prediction_blocks(directory: Path, by_id: dict):
    """Yield aligned frozen prediction blocks, validating ids and causal histories."""
    directory = Path(directory)
    files = sorted(directory.glob("predictions_*.npz"))
    if len(files) != PREDICTION_CHUNKS:
        raise ValueError(f"{directory}: expected {PREDICTION_CHUNKS} prediction chunks, got {len(files)}")
    for path in files:
        digest = sha256_file(path)
        with np.load(path, allow_pickle=False) as data:
            required = {"event_ids", "histories", "fine", "coarse"}
            if set(data.files) != required:
                raise ValueError(f"{path}: keys {sorted(data.files)} != {sorted(required)}")
            ids = [str(v) for v in data["event_ids"]]
            fine = np.asarray(data["fine"])
            coarse = np.asarray(data["coarse"])
            histories = np.asarray(data["histories"])
            m = len(ids)
            if not m or len(set(ids)) != m:
                raise ValueError(f"{path}: empty or duplicate event ids")
            if fine.shape != (m, FINETIME_POINTS, 40) or coarse.shape != (m, COARSE_TIME_POINTS, 40):
                raise ValueError(f"{path}: unexpected fine/coarse shapes")
            if histories.shape != (m, HISTORY_ROWS, 30):
                raise ValueError(f"{path}: unexpected history shape")
            if fine.dtype != np.float64 or coarse.dtype != np.float64:
                raise ValueError(f"{path}: predictions must be float64")
            for i, event_id in enumerate(ids):
                if event_id not in by_id:
                    raise ValueError(f"{path}: unknown event id {event_id}")
                if not np.array_equal(histories[i], by_id[event_id]["x"][:HISTORY_ROWS]):
                    raise ValueError(f"{path}: causal history/offset differs for {event_id}")
            yield {
                "path": path,
                "sha256": digest,
                "ids": ids,
                "fine": fine,
                "coarse": coarse,
            }


# ----------------------------------------------------------------------- P1


def point_error_terms(pred: torch.Tensor, truth: torch.Tensor, model):
    """Four additive terms of the registered ``_point_error`` at every time point.

    The last dimension is ``(delta_coi, omega_coi, common_omega, p_normalized)`` and the
    sum equals ``train_structured_rom._point_error(pred, truth, model)`` up to float
    associativity.  ``pred``/``truth`` may carry 40 columns; only the first 30 are used.
    """
    from train_structured_rom import METRIC, _coi

    err = pred[..., :30] - truth[..., :30]
    delta = _coi(err[..., :10], model.mass) / METRIC[0]
    omega = _coi(err[..., 10:20], model.mass) / METRIC[1]
    common = (err[..., 10:20] * model.mass).sum(-1) / model.mass.sum() / METRIC[1]
    p = err[..., 20:30] / model.p_scale
    return torch.stack(
        (
            0.5 * delta.square().mean(-1),
            0.5 * omega.square().mean(-1),
            common.square(),
            p.square().mean(-1),
        ),
        dim=-1,
    )


def scope_steps(event: dict, scope: str) -> int | None:
    """Registered number of future 0.01 s records for one scope, or None if unavailable."""
    if scope not in SCOPES:
        raise ValueError(f"unknown scope {scope}")
    available = len(event["x"]) - HISTORY_ROWS
    if scope in SCOPE_STEPS:
        steps = SCOPE_STEPS[scope]
    elif scope == "prefix":
        steps = min(MAX_ACTUAL_STEPS, available)
    else:  # pragma: no cover - guarded above
        raise ValueError(scope)
    if available < steps or steps < 1:
        return None
    return int(steps)


def fine_slice(fine_row: np.ndarray, steps: int) -> np.ndarray:
    """``fine[::2][1:steps+1, :30]``: the published 0.01 s decimated path."""
    return fine_row[::FINE_STRIDE][1 : steps + 1, :30]


def coarse_slice(coarse_row: np.ndarray, steps: int) -> np.ndarray:
    """``coarse[1:steps+1, :30]``: independent 0.01 s path, cross-check only."""
    return coarse_row[1 : steps + 1, :30]


def truth_slice(event: dict, steps: int) -> np.ndarray:
    return event["x"][HISTORY_ROWS : HISTORY_ROWS + steps, :30]


def cohort(events: list[dict], scope: str) -> list[dict]:
    """Stable cohort for 5 s/8 s; unstable cohort for the actual prefix."""
    label = "stable" if scope in SCOPE_STEPS else "unstable"
    return [event for event in events if event["label"] == label and scope_steps(event, scope) is not None]


def risk_aggregate(values) -> float:
    """Registered 0.8*mean + 0.2*mean(top ceil(10%)) risk on float64 values."""
    _, _, _, _, risk = _deps()
    tensor = torch.tensor([float(v) for v in values], dtype=torch.float64)
    return float(risk(tensor))


def _event_p1_row(model, event: dict, scope: str, fine_row: np.ndarray, coarse_row: np.ndarray):
    """Per-event, per-scope objective value, its four terms, and the coarse cross value."""
    from train_structured_rom import _point_error

    steps = scope_steps(event, scope)
    if steps is None:
        return None
    pred_np = fine_slice(fine_row, steps)
    coarse_np = coarse_slice(coarse_row, steps)
    truth_np = truth_slice(event, steps)
    finite = bool(
        np.isfinite(pred_np).all() and np.isfinite(coarse_np).all() and np.isfinite(truth_np).all()
    )
    if not finite:
        return {"finite": False}
    truth = torch.as_tensor(truth_np, dtype=torch.float64)
    pred = torch.as_tensor(pred_np, dtype=torch.float64)
    coarse_pred = torch.as_tensor(coarse_np, dtype=torch.float64)
    with torch.no_grad():
        point = _point_error(pred, truth, model)
        terms = point_error_terms(pred, truth, model)
        coarse_total = float(_point_error(coarse_pred, truth, model).mean())
    row = {
        "finite": True,
        "total": float(point.mean()),
        "terms": [float(v) for v in terms.mean(0)],
        "coarse_total": coarse_total,
        "term_sum_max_abs_gap": float((terms.sum(-1) - point).abs().max()),
    }
    if scope == "8":
        # Pooled 8 s values used by the published common-frequency and normalized-p RMSE.
        mass = model.mass.detach().cpu().numpy()
        p_scale = model.p_scale.detach().cpu().numpy()
        omega_error = pred_np[:, 10:20] - truth_np[:, 10:20]
        common = omega_error @ mass / mass.sum()
        row["common_omega_sse"] = float(common @ common)
        row["common_omega_n"] = int(common.size)
        p_error = (pred_np[:, 20:30] - truth_np[:, 20:30]) / p_scale[None, :]
        row["normalized_p_sse"] = float(np.sum(p_error**2))
        row["normalized_p_n"] = int(p_error.size)
    return row


def pooled_8s_rmse(rows) -> dict:
    """Pooled sqrt(sum SSE / sum n) for the two published 8 s physical RMSEs."""
    common_sse = common_n = p_sse = p_n = 0.0
    for row in rows:
        common_sse += row["common_omega_sse"]
        common_n += row["common_omega_n"]
        p_sse += row["normalized_p_sse"]
        p_n += row["normalized_p_n"]
    if not common_n or not p_n:
        raise ValueError("pooled 8 s RMSE needs nonempty stable-complete-8s rows")
    return {
        "common_omega_rmse_8s": math.sqrt(common_sse / common_n),
        "normalized_p_rmse_8s": math.sqrt(p_sse / p_n),
        "common_omega_n": int(common_n),
        "normalized_p_n": int(p_n),
        "definition": "pooled sqrt(sum SSE / sum n) over stable complete-8s rows; same fine[::2] path",
    }


def aggregate_scope(rows: list[dict]) -> dict:
    """Risk-level aggregation for one model/scope cohort.

    ``risk_total`` uses the registered ``_point_error`` values; ``risk_term`` aggregates
    each of the four terms separately.  Their sum is not the total risk because the risk
    functional's upper-tail part is nonlinear, so the residual is reported explicitly.
    """
    totals = [row["total"] for row in rows]
    term_matrix = np.asarray([row["terms"] for row in rows], dtype=np.float64)
    coarse = [row["coarse_total"] for row in rows]
    risk_total = risk_aggregate(totals)
    risk_term = [risk_aggregate(term_matrix[:, j]) for j in range(N_TERMS)]
    risk_coarse = risk_aggregate(coarse)
    mean_terms = [float(v) for v in term_matrix.mean(0)]
    term_sum = float(sum(risk_term))
    return {
        "event_count": len(rows),
        "risk_total": risk_total,
        "risk_term": dict(zip(TERM_NAMES, risk_term)),
        "risk_term_sum": term_sum,
        "risk_term_sum_residual": term_sum - risk_total,
        "risk_term_share": {
            name: (value / risk_total if risk_total else None) for name, value in zip(TERM_NAMES, risk_term)
        },
        "risk_coarse": risk_coarse,
        "numerical_perturbation": abs(risk_total - risk_coarse),
        "mean_total": float(np.mean(totals)),
        "mean_term": dict(zip(TERM_NAMES, mean_terms)),
        "mean_term_share": {
            name: (value / float(np.mean(totals)) if np.mean(totals) else None)
            for name, value in zip(TERM_NAMES, mean_terms)
        },
    }


def _gap(per_model: dict, left: str, right: str, events: list[dict], scope: str) -> dict:
    """Event-mean (additive) and risk-level (possibly non-additive) term gaps."""
    ids = [event["id"] for event in cohort(events, scope)]
    left_rows = [per_model[left]["per_event"][scope][event_id] for event_id in ids]
    right_rows = [per_model[right]["per_event"][scope][event_id] for event_id in ids]
    left_terms = np.asarray([row["terms"] for row in left_rows], dtype=np.float64)
    right_terms = np.asarray([row["terms"] for row in right_rows], dtype=np.float64)
    left_totals = np.asarray([row["total"] for row in left_rows], dtype=np.float64)
    right_totals = np.asarray([row["total"] for row in right_rows], dtype=np.float64)
    mean_gap_terms = [float(v) for v in (left_terms - right_terms).mean(0)]
    risk_gap_terms = [risk_aggregate(left_terms[:, j]) - risk_aggregate(right_terms[:, j]) for j in range(N_TERMS)]
    mean_gap_total = float(left_totals.mean() - right_totals.mean())
    risk_gap_total = risk_aggregate(left_totals) - risk_aggregate(right_totals)
    dominant_event = TERM_NAMES[int(np.argmax(np.abs(mean_gap_terms)))]
    dominant_risk = TERM_NAMES[int(np.argmax(np.abs(risk_gap_terms)))]
    return {
        "cohort": len(ids),
        "event_mean_gap_total": mean_gap_total,
        "event_mean_gap_term": dict(zip(TERM_NAMES, mean_gap_terms)),
        "event_mean_gap_term_sum": float(sum(mean_gap_terms)),
        "event_mean_gap_additive_residual": float(sum(mean_gap_terms) - mean_gap_total),
        "risk_gap_total": risk_gap_total,
        "risk_gap_term": dict(zip(TERM_NAMES, risk_gap_terms)),
        "risk_gap_term_sum": float(sum(risk_gap_terms)),
        "risk_gap_residual": float(sum(risk_gap_terms) - risk_gap_total),
        "dominant_event_mean_term": dominant_event,
        "dominant_risk_term": dominant_risk,
        "risk_gap_additivity": "non-additive through the upper-tail 0.2 term; residual reported",
    }


def decompose() -> dict:
    """P1: score the four frozen prediction sets and reproduce the published risks."""
    started = time.perf_counter()
    torch.set_num_threads(1)  # registered CPU float64 scoring mode; no training
    record, events = load_frozen_events()
    identity = verify_frozen_scales()
    by_id = {event["id"]: event for event in events}
    per_model = {}
    prediction_manifest = {}
    for name in MODEL_ORDER:
        spec = MODEL_SPECS[name]
        model = load_model(spec)
        _assert_model_scales(model, identity)
        rows = {scope: {} for scope in SCOPES}
        unresolved = {scope: [] for scope in SCOPES}
        files = []
        seen = set()
        with torch.no_grad():
            for block in iter_prediction_blocks(spec["predictions"], by_id):
                ids = block["ids"]
                if seen.intersection(ids):
                    raise ValueError(f"{name}: duplicate event id across prediction chunks")
                seen.update(ids)
                files.append(
                    {
                        "file": block["path"].relative_to(ROOT).as_posix(),
                        "sha256": block["sha256"],
                        "events": len(ids),
                    }
                )
                for i, event_id in enumerate(ids):
                    event = by_id[event_id]
                    applicable = ("5", "8") if event["label"] == "stable" else ("prefix",)
                    for scope in applicable:
                        row = _event_p1_row(model, event, scope, block["fine"][i], block["coarse"][i])
                        if row is None:
                            continue
                        if not row["finite"]:
                            unresolved[scope].append(event_id)
                            continue
                        rows[scope][event_id] = row
        if seen != set(by_id):
            raise ValueError(f"{name}: prediction chunks do not cover exactly fit297")
        prediction_manifest[name] = files
        per_model[name] = {"per_event": rows, "unresolved": unresolved}

    # Aggregate each cohort and check the public denominators.
    denominators = {
        "events": len(events),
        "stable": sum(event["label"] == "stable" for event in events),
        "unstable": sum(event["label"] == "unstable" for event in events),
        "scopes": {},
    }
    for scope in SCOPES:
        members = cohort(events, scope)
        denominators["scopes"][scope] = {
            "cohort": "stable" if scope in SCOPE_STEPS else "unstable",
            "expected": EXPECTED_DENOMINATORS[scope],
            "eligible": len(members),
            "required_steps": SCOPE_STEPS.get(scope, "min(1500,len(x)-11)"),
            "actual_steps": sorted({scope_steps(event, scope) for event in members}),
        }
    for name in MODEL_ORDER:
        aggregates = {}
        for scope in SCOPES:
            members = cohort(events, scope)
            expected = len(members)
            available = [per_model[name]["per_event"][scope][event["id"]] for event in members if event["id"] in per_model[name]["per_event"][scope]]
            missing = [event["id"] for event in members if event["id"] not in per_model[name]["per_event"][scope]]
            block = {
                "expected_events": expected,
                "eligible": len(available),
                "missing_ids": missing,
                "unresolved_ids": per_model[name]["unresolved"][scope],
            }
            block.update(aggregate_scope(available) if available else {})
            aggregates[scope] = block
        stable_8 = [per_model[name]["per_event"]["8"].get(event["id"]) for event in cohort(events, "8")]
        if any(row is None for row in stable_8):
            raise ValueError(f"{name}: missing stable complete-8s row for pooled RMSE")
        per_model[name]["scopes"] = aggregates
        per_model[name]["pooled_rmse"] = pooled_8s_rmse(stable_8)

    # Acceptance gates: published risks, published pooled RMSE, denominators, term identity.
    gate_rows = []
    for name in MODEL_ORDER:
        for scope in SCOPES:
            expected = PUBLIC_VALUES[(name, scope)]
            aggregate = per_model[name]["scopes"][scope]
            actual = aggregate.get("risk_total")
            eligible = aggregate.get("eligible")
            error = abs(float(actual) - expected) if actual is not None else None
            passed = (
                actual is not None
                and eligible == aggregate.get("expected_events")
                and not aggregate.get("missing_ids")
                and not aggregate.get("unresolved_ids")
                and error is not None
                and error < ACCEPTANCE_ATOL
            )
            gate_rows.append(
                {
                    "model": name,
                    "scope": scope,
                    "expected": expected,
                    "actual": actual,
                    "abs_error": error,
                    "eligible": eligible,
                    "expected_events": aggregate.get("expected_events"),
                    "coarse_value": aggregate.get("risk_coarse"),
                    "coarse_gap": (actual - aggregate["risk_coarse"]) if actual is not None else None,
                    "passed": bool(passed),
                }
            )
    max_error = max((row["abs_error"] for row in gate_rows if row["abs_error"] is not None), default=None)
    risk_passed = all(row["passed"] for row in gate_rows)

    rmse_rows = []
    for (name, metric), expected in RMSE_GATES.items():
        actual = per_model[name]["pooled_rmse"][metric]
        error = abs(float(actual) - expected)
        rmse_rows.append(
            {
                "model": name,
                "metric": metric,
                "expected": expected,
                "actual": float(actual),
                "abs_error": error,
                "passed": bool(error < ACCEPTANCE_ATOL),
            }
        )
    max_rmse_error = max((row["abs_error"] for row in rmse_rows), default=None)
    rmse_passed = all(row["passed"] for row in rmse_rows)

    denominator_rows = []
    for scope, expected in EXPECTED_DENOMINATORS.items():
        actual = len(cohort(events, scope))
        denominator_rows.append(
            {"scope": scope, "expected": expected, "actual": actual, "passed": bool(actual == expected)}
        )
    denominator_passed = all(row["passed"] for row in denominator_rows) and (
        denominators["events"] == EXPECTED_EVENTS
        and denominators["stable"] == EXPECTED_STABLE
        and denominators["unstable"] == EXPECTED_UNSTABLE
    )

    per_event_term_gaps = [
        row["term_sum_max_abs_gap"]
        for name in MODEL_ORDER
        for scope in SCOPES
        for row in per_model[name]["per_event"][scope].values()
    ]
    max_term_gap = max(per_event_term_gaps) if per_event_term_gaps else None
    term_sum_passed = max_term_gap is not None and max_term_gap < TERM_SUM_ATOL
    gates_passed = bool(risk_passed and rmse_passed and denominator_passed and term_sum_passed)

    gaps = {}
    for left, right in (("A", "R_N"), ("A", "S"), ("A", "P")):
        gaps[f"{left}-{right}"] = {scope: _gap(per_model, left, right, events, scope) for scope in SCOPES}

    manifest_blob = json.dumps(
        {name: [entry["sha256"] for entry in files] for name, files in prediction_manifest.items()},
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    per_event = {}
    for name in MODEL_ORDER:
        per_event[name] = {
            scope: {event_id: row for event_id, row in per_model[name]["per_event"][scope].items()}
            for scope in SCOPES
        }
    payload = {
        "schema": P1_SCHEMA,
        "status": "COMPLETED" if gates_passed else "ACCEPTANCE_FAILED",
        "role": "descriptive decomposition; not mechanism discrimination or model comparison",
        "scope": (
            "P1 four-term decomposition of the registered point error on frozen fit297 predictions; "
            "no model rerun, no training, no val/test. Public stable 5/8 s and unstable actual-interval "
            "risks use the fine 0.005 s path decimated by two; coarse is a cross-check. All values are "
            "arithmetic reproductions of frozen artifacts and carry no mechanism attribution."
        ),
        "source": {
            "preparation": "private_data/state_gated_memory_20261028/study01/prepare.json",
            "preparation_schedule_sha256": record.get("schedule_sha256"),
            "checkpoints": identity["checkpoints"],
            "prediction_files": prediction_manifest,
            "prediction_manifest_sha256": hashlib.sha256(manifest_blob).hexdigest(),
            "scales": identity["scales"],
            "sgm_source": identity["sgm_source"],
        },
        "definitions": {
            "point_error": "train_structured_rom._point_error, four terms as registered",
            "risk": "train_local_memory_0098.risk = 0.8*mean + 0.2*mean(worst ceil(10%)), float64",
            "terms": list(TERM_NAMES),
            "fine_path": "fine[::2][1:steps+1, :30] versus x[11:11+steps, :30]",
            "coarse_path": "coarse[1:steps+1, :30] versus the same truth, cross-check only",
            "public_scopes": PUBLIC_SCOPE_DEFINITION,
            "pooled_rmse_8s": "sqrt(sum SSE / sum n) over stable complete-8s rows, same fine path",
            "acceptance_atol": ACCEPTANCE_ATOL,
            "term_sum_atol": TERM_SUM_ATOL,
        },
        "acceptance": {
            "passed": gates_passed,
            "risk_gate": {
                "passed": risk_passed,
                "atol": ACCEPTANCE_ATOL,
                "max_abs_error": max_error,
                "rows": gate_rows,
                "note": "all twelve published risk values (four models x three scopes) at atol 1e-9",
            },
            "rmse_gate": {
                "passed": rmse_passed,
                "atol": ACCEPTANCE_ATOL,
                "max_abs_error": max_rmse_error,
                "rows": rmse_rows,
                "note": "published A/R_N pooled common-omega and normalized-p RMSE at 8 s",
            },
            "term_sum_gate": {
                "passed": term_sum_passed,
                "atol": TERM_SUM_ATOL,
                "max_abs_gap": max_term_gap,
                "checked_event_rows": len(per_event_term_gaps),
                "note": "per-timestep four-term sum versus _point_error, max over all model/scope/event rows",
            },
            "denominator_gate": {
                "passed": denominator_passed,
                "rows": denominator_rows,
                "note": "stable 267 for 5/8 s and unstable 30 for the actual prefix",
            },
        },
        "models": per_model,
        "gaps": gaps,
        "denominators": denominators,
        "per_event": per_event,
        "evidence": {
            "observed": [
                "twelve published risk values (4 models x 5 s / 8 s / unstable actual interval)",
                "published A/R_N pooled common-omega and normalized-p RMSE at 8 s",
                "stable/unstable denominators 267/30 with eligible counts",
                "per-timestep four-term identity against _point_error",
                "coarse-path cross values for every scope",
            ],
            "diagnostic": [
                "four-term risk shares per model/scope (arithmetic decomposition of nonlinear risk)",
                "A-R_N, A-S and A-P event-mean and risk-level gap decomposition",
            ],
            "unverified": [
                "any attribution of differences to common-mode closure, short-window initialization or optimization",
                "any error-bar, noise model or out-of-fit generalization claim",
                "any model selection, adoption or mechanism conclusion",
            ],
        },
        "runtime": _runtime(started, {"prediction_chunks_read": sum(len(v) for v in prediction_manifest.values())}),
        "limits": (
            "Descriptive reproduction of frozen numbers only. The four-term shares and gaps are arithmetic "
            "summaries; they do not identify a mechanism, an optimizer state or an identifiability boundary."
        ),
    }
    return payload


# ----------------------------------------------------------------------- P2


def qr_solve(design: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Reduced-QR least squares, matching MemoryTransportROM.history_system."""
    q, r = torch.linalg.qr(design, mode="reduced")
    return torch.linalg.solve_triangular(r, q.transpose(-1, -2) @ target[..., None], upper=True).squeeze(-1)


def _stats(values) -> dict:
    array = np.asarray([float(v) for v in values], dtype=np.float64)
    array = array[np.isfinite(array)]
    if array.size == 0:
        return {"count": 0, "min": None, "median": None, "p90": None, "max": None, "mean": None}
    return {
        "count": int(array.size),
        "min": float(array.min()),
        "median": float(np.median(array)),
        "p90": float(np.quantile(array, 0.9)),
        "max": float(array.max()),
        "mean": float(array.mean()),
    }


def analyze_history_system(
    system: dict,
    rstar: torch.Tensor,
    *,
    ridge: float,
    scans: tuple[float, ...] = RIDGE_SCAN,
    event_ids: list[str] | None = None,
    labels: list[str] | None = None,
) -> dict:
    """Spectral diagnostics of the frozen r0 ridge problem for a batch of histories.

    The augmented design is ``[A; sqrt(ridge)*F]`` and the target is
    ``[beta; -sqrt(ridge)*q]``; ``beta`` is the returned ``bscaled`` field.  The ridge scan
    is diagnostic only and never changes the registered ridge.
    """
    required = ("A", "F", "bscaled", "q", "r0")
    missing = [key for key in required if key not in system]
    if missing:
        raise ValueError(f"history_system is missing {missing}")
    A = torch.as_tensor(system["A"], dtype=torch.float64)
    F = torch.as_tensor(system["F"], dtype=torch.float64)
    beta = torch.as_tensor(system["bscaled"], dtype=torch.float64)
    q = torch.as_tensor(system["q"], dtype=torch.float64)
    r0 = torch.as_tensor(system["r0"], dtype=torch.float64)
    rstar = torch.as_tensor(rstar, dtype=torch.float64).reshape(10)
    n = A.shape[0]
    if A.shape[1:] != (50, 10) or F.shape[1:] != (10, 10) or beta.shape[1:] != (50,) or q.shape[1:] != (10,):
        raise ValueError("unexpected history_system tensor shapes")
    if r0.shape != (n, 10) or A.shape[0] != n or F.shape[0] != n or beta.shape[0] != n or q.shape[0] != n:
        raise ValueError("history_system batch dimensions differ")
    if not all(torch.isfinite(t).all() for t in (A, F, beta, q, r0, rstar)):
        raise ValueError("nonfinite history_system tensor")

    root_ridge = math.sqrt(ridge)
    augmented = torch.cat((A, root_ridge * F), dim=-2)
    target = torch.cat((beta, -root_ridge * q), dim=-1)
    singular = torch.linalg.svdvals(augmented)
    raw_singular = torch.linalg.svdvals(A)
    _, _, vh = torch.linalg.svd(augmented, full_matrices=False)
    weakest_direction = vh[..., -1, :]
    sigma_max = singular[:, 0]
    sigma_min = singular[:, -1]
    raw_min = raw_singular[:, -1]
    condition = sigma_max / sigma_min
    effective_rank = (singular > EFFECTIVE_RANK_FACTOR * sigma_max[:, None]).sum(-1)
    ridge_contribution = (sigma_min - raw_min) / sigma_min

    coefficients = qr_solve(augmented, target)
    r0_reconstructed = rstar[None, :] * coefficients
    reconstruction_error = float((r0_reconstructed - r0).abs().max())

    r0_norm = torch.linalg.vector_norm(r0, dim=-1)
    direction = rstar[None, :] * weakest_direction
    scaled_singular = torch.linalg.svdvals(augmented * (1.0 / rstar)[None, None, :])
    scaled_condition = scaled_singular[:, 0] / scaled_singular[:, -1]
    relative_by_scan = {}
    cosine_by_scan = {}
    scan = {}
    for scan_ridge in scans:
        scan_root = math.sqrt(scan_ridge)
        scan_design = torch.cat((A, scan_root * F), dim=-2)
        scan_target = torch.cat((beta, -scan_root * q), dim=-1)
        scan_r0 = rstar[None, :] * qr_solve(scan_design, scan_target)
        delta = scan_r0 - r0
        delta_norm = torch.linalg.vector_norm(delta, dim=-1)
        relative = delta_norm / r0_norm
        cosine = (delta * direction).sum(-1) / (delta_norm * torch.linalg.vector_norm(direction, dim=-1))
        absolute_cosine = torch.where(delta_norm > 0, cosine.abs(), torch.full_like(cosine, float("nan")))
        relative_by_scan[f"{scan_ridge:g}"] = relative
        cosine_by_scan[f"{scan_ridge:g}"] = absolute_cosine
        scan[f"{scan_ridge:g}"] = {
            "ridge": scan_ridge,
            "diagnostic_only": True,
            "relative_change": _stats(relative),
            "relative_change_events_over_10pct": int((relative > 0.1).sum()),
            "abs_cosine_with_weakest_direction": _stats(absolute_cosine),
            "abs_cosine_definition": (
                "|cos(Delta r0, diag(rstar) v_min)| with v_min the right singular vector of the "
                "baseline [A; sqrt(1e-4) F] design; null when Delta r0 is zero"
            ),
        }

    design_spread_a = float((A - A[0]).abs().max()) if n else 0.0
    design_spread_f = float((F - F[0]).abs().max()) if n else 0.0
    per_event = []
    for i in range(n):
        per_event.append(
            {
                "event_id": event_ids[i] if event_ids else str(i),
                "label": labels[i] if labels else None,
                "sigma_max": float(sigma_max[i]),
                "sigma_min": float(sigma_min[i]),
                "condition": float(condition[i]),
                "effective_rank": int(effective_rank[i]),
                "raw_sigma_min": float(raw_min[i]),
                "ridge_sigma_min_contribution": float(ridge_contribution[i]),
                "relative_change": {
                    key: (None if not math.isfinite(float(value[i])) else float(value[i]))
                    for key, value in relative_by_scan.items()
                },
                "abs_cosine_with_weakest_direction": {
                    key: (None if not math.isfinite(float(value[i])) else float(value[i]))
                    for key, value in cosine_by_scan.items()
                },
                "r0_norm": float(r0_norm[i]),
            }
        )

    sigma_stats = {f"sigma_{index + 1}": _stats(singular[:, index]) for index in range(singular.shape[1])}
    counts = {
        "events": int(n),
        "effective_rank_lt_full": int((effective_rank < 10).sum()),
        "ridge_dominated": int((ridge_contribution > RIDGE_DOMINANCE_SHARE).sum()),
        "relative_change_gt_10pct": {
            key: int((value > 0.1).sum()) for key, value in relative_by_scan.items()
        },
        "ridge_dominance_threshold": RIDGE_DOMINANCE_SHARE,
        "effective_rank_threshold_factor": EFFECTIVE_RANK_FACTOR,
    }

    grouped = None
    if labels is not None:
        label_array = np.asarray(labels)
        grouped = {}
        for label in ("stable", "unstable"):
            mask = torch.as_tensor(label_array == label)
            if not bool(mask.any()):
                continue
            grouped[label] = {
                "events": int(mask.sum()),
                "condition": _stats(condition[mask]),
                "sigma_min": _stats(sigma_min[mask]),
                "ridge_sigma_min_contribution": _stats(ridge_contribution[mask]),
                "effective_rank_counts": {
                    str(int(rank)): int((effective_rank[mask] == rank).sum())
                    for rank in sorted(set(effective_rank[mask].tolist()))
                },
                "relative_change": {
                    key: _stats(value[mask]) for key, value in relative_by_scan.items()
                },
            }

    return {
        "design": {
            "shape": [int(n), 60, 10],
            "augmented": "[A; sqrt(ridge) * F]",
            "target": "[beta; -sqrt(ridge) * q]",
            "ridge": ridge,
            "beta_source_field": "bscaled",
            "design_spread_across_events": {"max_abs_A": design_spread_a, "max_abs_F": design_spread_f},
            "design_event_independent": bool(design_spread_a == 0.0 and design_spread_f == 0.0),
        },
        "spectrum": {
            "singular_value_stats": sigma_stats,
            "sigma_max": _stats(sigma_max),
            "sigma_min": _stats(sigma_min),
            "condition": _stats(condition),
            "effective_rank": {
                "factor": EFFECTIVE_RANK_FACTOR,
                "counts": {str(int(rank)): int((effective_rank == rank).sum()) for rank in sorted(set(effective_rank.tolist()))},
                "min": int(effective_rank.min()),
                "max": int(effective_rank.max()),
            },
            "raw_A_sigma_min": _stats(raw_min),
            "raw_A_condition": _stats(raw_singular[:, 0] / raw_min),
        },
        "ridge_contribution": {
            "definition": "(sigma_min([A; sqrt(ridge)F]) - sigma_min(A)) / sigma_min([A; sqrt(ridge)F])",
            "stats": _stats(ridge_contribution),
            "dominated_count": int((ridge_contribution > RIDGE_DOMINANCE_SHARE).sum()),
        },
        "r0_coordinate_scaling": {
            "definition": (
                "secondary diagnostic: the registered design acts on the dimensionless coefficients s "
                "(r0 = rstar * s); scaling its columns by 1/rstar expresses the same system in r0 coordinates"
            ),
            "rstar_max_over_min": float(rstar.max() / rstar.min()),
            "condition": _stats(scaled_condition),
        },
        "scan": scan,
        "counts": counts,
        "grouped": grouped,
        "r0": {
            "norm": _stats(torch.linalg.vector_norm(r0, dim=-1)),
            "reconstruction_max_abs_diff": reconstruction_error,
            "reconstruction_passed": reconstruction_error <= 1e-12,
        },
        "per_event": per_event,
        "evidence_class": "observed (spectrum of the frozen A128 design) + diagnostic (ridge scan)",
        "interpretation_scope": (
            "Conditioning and prior sensitivity of the registered ridge least-squares operator only: "
            "singular spectrum, condition number, effective rank, ridge contribution and a diagnostic "
            "ridge scan. This is not a data-identifiability claim under a noise model, and no "
            "model/mechanism conclusion is made."
        ),
    }


def compare_initializers(systems: dict, labels: list[str] | None = None) -> dict:
    """Relative r0 changes produced by different frozen initializers on the same histories.

    ``systems[name]`` must hold ``r0`` (n,10), ``s`` (n,10) and ``rstar`` (10,).  The
    reference is A128.  Both the physical r0 change and the coefficient change
    (``s = r0 / rstar``) are reported, plus the mean-total drift ``Delta(1'r0)/10``.
    This is a prior/model sensitivity diagnostic, not a data-identifiability result.
    """
    names = list(systems)
    if "A128" not in systems:
        raise ValueError("A128 must be the initializer reference")
    base_r0 = torch.as_tensor(systems["A128"]["r0"], dtype=torch.float64)
    base_s = torch.as_tensor(systems["A128"]["s"], dtype=torch.float64)
    base_rstar = torch.as_tensor(systems["A128"]["rstar"], dtype=torch.float64).reshape(10)
    if base_r0.ndim != 2 or base_r0.shape[1] != 10 or base_r0.shape != base_s.shape:
        raise ValueError("initializer r0/s must be (n,10)")
    base_norm = torch.linalg.vector_norm(base_r0, dim=-1)
    label_array = None if labels is None else np.asarray(labels)
    comparisons = {}
    for name in names:
        if name == "A128":
            continue
        r0 = torch.as_tensor(systems[name]["r0"], dtype=torch.float64)
        s = torch.as_tensor(systems[name]["s"], dtype=torch.float64)
        if r0.shape != base_r0.shape or s.shape != base_s.shape:
            raise ValueError(f"{name}: initializer batch differs from A128")
        delta = r0 - base_r0
        relative = torch.linalg.vector_norm(delta, dim=-1) / base_norm
        coefficient_relative = torch.linalg.vector_norm(s - base_s, dim=-1) / torch.linalg.vector_norm(base_s, dim=-1)
        delta_total = (r0.sum(-1) - base_r0.sum(-1)) / 10.0
        entry = {
            "r0_relative_change": _stats(relative),
            "coefficient_relative_change": _stats(coefficient_relative),
            "delta_sum_r0_over_10": _stats(delta_total),
            "abs_delta_sum_r0_over_10": _stats(delta_total.abs()),
            "r0_norm": _stats(torch.linalg.vector_norm(r0, dim=-1)),
        }
        if label_array is not None:
            entry["grouped"] = {}
            for label in ("stable", "unstable"):
                mask = torch.as_tensor(label_array == label)
                if not bool(mask.any()):
                    continue
                entry["grouped"][label] = {
                    "events": int(mask.sum()),
                    "r0_relative_change": _stats(relative[mask]),
                    "coefficient_relative_change": _stats(coefficient_relative[mask]),
                    "abs_delta_sum_r0_over_10": _stats(delta_total[mask].abs()),
                }
        comparisons[name] = entry
    rstar_identical = all(
        torch.equal(torch.as_tensor(systems[name]["rstar"], dtype=torch.float64).reshape(10), base_rstar)
        for name in names
    )
    return {
        "baseline": "A128",
        "systems": names,
        "events": int(base_r0.shape[0]),
        "rstar_identical_across_initializers": bool(rstar_identical),
        "rstar_max_over_min": float(base_rstar.max() / base_rstar.min()),
        "comparisons": comparisons,
        "definition": (
            "per-event ||r0_model - r0_A128|| / ||r0_A128|| with rstar scaling retained; "
            "coefficient change uses s = r0/rstar; mean-total drift is Delta(1'r0)/10"
        ),
        "evidence_class": "diagnostic (initializer sensitivity; no rollout)",
    }


def r0_model_dependence(events: list[dict], identity: dict) -> dict:
    """Run the four frozen initializers on the same 297 histories; no rollout."""
    histories = torch.as_tensor(
        np.stack([event["x"][:HISTORY_ROWS] for event in events]), dtype=torch.float64
    )
    specs = {
        "A128": MODEL_SPECS["A"],
        "S128": MODEL_SPECS["S"],
        "P128": MODEL_SPECS["P"],
        "A96": A96_SPEC,
    }
    systems = {}
    meta = {}
    for name, spec in specs.items():
        model = load_model(spec)
        _assert_model_scales(model, identity)
        if float(model.history_dt) != 0.0025:
            raise ValueError(f"{name}: history_dt is not 0.0025")
        with torch.no_grad():
            system = model.history_system(histories)
        if tuple(system["r0"].shape) != (EXPECTED_EVENTS, 10):
            raise ValueError(f"{name}: unexpected r0 shape")
        systems[name] = {
            "r0": system["r0"].detach().cpu(),
            "s": system["s"].detach().cpu(),
            "rstar": model.rstar.detach().cpu(),
        }
        meta[name] = {
            "checkpoint": Path(spec["export"]).relative_to(ROOT).as_posix(),
            "sha256": (
                identity["checkpoints"][name]["sha256"] if name in identity["checkpoints"]
                else sha256_file(Path(spec["export"]))
            ),
            "history_dt": float(model.history_dt),
        }
    result = compare_initializers(systems, labels=[event["label"] for event in events])
    result["systems"] = meta
    result["history_windows"] = len(events)
    return result


def init_conditioning() -> dict:
    """P2: condition the frozen A128 initial hidden force system on frozen histories."""
    started = time.perf_counter()
    torch.set_num_threads(1)
    record, events = load_frozen_events()
    identity = verify_frozen_scales()
    spec = MODEL_SPECS["A"]
    model = load_model(spec)
    _assert_model_scales(model, identity)
    if float(model.history_dt) != 0.0025:
        raise ValueError("A128 history_dt is not 0.0025")
    histories = torch.as_tensor(
        np.stack([event["x"][:HISTORY_ROWS] for event in events]), dtype=torch.float64
    )
    with torch.no_grad():
        system = model.history_system(histories)
    if tuple(system["A"].shape) != (EXPECTED_EVENTS, 50, 10):
        raise ValueError("history_system returned an unexpected design shape")
    diagnostics = analyze_history_system(
        system,
        model.rstar.detach().cpu().double(),
        ridge=identity["ridge"],
        scans=RIDGE_SCAN,
        event_ids=[event["id"] for event in events],
        labels=[event["label"] for event in events],
    )
    dependence = r0_model_dependence(events, identity)
    return {
        "schema": P2_SCHEMA,
        "status": "COMPLETED",
        "role": "conditioning / prior-sensitivity probe; not a data-identifiability test",
        "scope": (
            "P2 conditioning of the frozen A128 initial hidden force r0 operator and sensitivity of r0 to the "
            "ridge prior and to the frozen initializer (A128/S128/P128/A96) on the same 297 frozen 11x30 "
            "causal histories; history_system only, no training, no rollout, no val/test."
        ),
        "model": {
            "checkpoint": spec["export"].relative_to(ROOT).as_posix(),
            "sha256": identity["checkpoints"]["A"]["sha256"],
            "history_dt": float(model.history_dt),
            "events": len(events),
            "stable": sum(event["label"] == "stable" for event in events),
            "unstable": sum(event["label"] == "unstable" for event in events),
        },
        "ridge": identity["ridge"],
        "diagnostics": diagnostics,
        "model_dependence": dependence,
        "evidence": {
            "observed": [
                "A128 singular spectrum, condition number and effective rank of [A; sqrt(ridge)F]",
                "ridge-block contribution to sigma_min",
                "QR reconstruction of the returned r0 (max abs difference)",
                "rstar vector (identical across A128/S128/P128/A96)",
            ],
            "diagnostic": [
                "ridge scan 1e-4 / 1e-6 / 1e-8 (relative r0 change and weakest-direction cosine)",
                "r0 model dependence across A128/S128/P128/A96 initializers (no rollout)",
            ],
            "unverified": [
                "data identifiability of r0 under measurement noise or model error",
                "downstream effect of ridge/r0 changes on 8 s risk (P2b not run)",
                "any mechanism, optimization-sufficiency or model-selection conclusion",
            ],
        },
        "runtime": _runtime(started, {"history_system_calls": 5}),
        "limits": (
            "The ridge scan and initializer comparison are diagnostics and do not change the registered ridge "
            "or the frozen models. P2 reports conditioning and prior sensitivity, not data identifiability; "
            "no rollout or downstream-risk claim is made."
        ),
    }


# ------------------------------------------------------------------ publish


def _assert_public(payload: dict) -> None:
    """Fail if a public payload contains event-level rows or long arrays."""
    def walk(node, path=""):
        if isinstance(node, dict):
            for key, value in node.items():
                if "per_event" in key.lower() or key.lower() in {"histories", "fine", "coarse"}:
                    raise ValueError(f"public payload contains event-level field {path}/{key}")
                walk(value, f"{path}/{key}")
        elif isinstance(node, list):
            if len(node) > 64:
                raise ValueError(f"public payload contains a long array at {path} (len={len(node)})")
            for index, value in enumerate(node):
                walk(value, f"{path}[{index}]")

    walk(payload)


def public_payload(p1: dict, p2: dict) -> dict:
    """Whitelist aggregate results; never copy per-event predictions or histories."""
    if p1.get("schema") != P1_SCHEMA or p2.get("schema") != P2_SCHEMA:
        raise ValueError("unexpected probe payload schema")
    if p1.get("status") != "COMPLETED" or not p1.get("acceptance", {}).get("passed"):
        raise ValueError("P1 acceptance gate did not pass; refusing public publication")
    if p2.get("status") != "COMPLETED":
        raise ValueError("P2 did not complete")
    diagnostics = p2.get("diagnostics", {})
    payload = {
        "schema": PUBLIC_SCHEMA,
        "status": "COMPLETED",
        "scope": (
            "Read-only scientific probe. P1 descriptively decomposes the registered point-error risk of four "
            "frozen fit297 prediction sets and reproduces published risk/RMSE/denominator gates; P2 reports "
            "conditioning and prior sensitivity of the frozen initial hidden force system. No training, no "
            "model selection, no mechanism or identifiability claim."
        ),
        "generated_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "source": {
            "p1": {
                "checkpoints": p1.get("source", {}).get("checkpoints"),
                "prediction_manifest_sha256": p1.get("source", {}).get("prediction_manifest_sha256"),
                "prediction_files": sum(
                    len(entries) for entries in p1.get("source", {}).get("prediction_files", {}).values()
                ),
                "scales": p1.get("source", {}).get("scales"),
            },
            "p2": p2.get("model"),
        },
        "definitions": p1.get("definitions"),
        "evidence": {
            "observed": p1.get("evidence", {}).get("observed", []) + p2.get("evidence", {}).get("observed", []),
            "diagnostic": p1.get("evidence", {}).get("diagnostic", []) + p2.get("evidence", {}).get("diagnostic", []),
            "unverified": p1.get("evidence", {}).get("unverified", []) + p2.get("evidence", {}).get("unverified", []),
        },
        "p1": {
            "role": p1.get("role"),
            "acceptance": p1.get("acceptance"),
            "models": {
                name: {"scopes": block.get("scopes", {}), "pooled_rmse": block.get("pooled_rmse")}
                for name, block in p1.get("models", {}).items()
            },
            "gaps": p1.get("gaps"),
            "denominators": p1.get("denominators"),
            "evidence": p1.get("evidence"),
        },
        "p2": {
            "role": p2.get("role"),
            "design": diagnostics.get("design"),
            "spectrum": diagnostics.get("spectrum"),
            "ridge_contribution": diagnostics.get("ridge_contribution"),
            "r0_coordinate_scaling": diagnostics.get("r0_coordinate_scaling"),
            "scan": diagnostics.get("scan"),
            "counts": diagnostics.get("counts"),
            "grouped": diagnostics.get("grouped"),
            "r0": diagnostics.get("r0"),
            "model_dependence": p2.get("model_dependence"),
            "evidence": p2.get("evidence"),
            "interpretation_scope": diagnostics.get("interpretation_scope"),
        },
        "boundaries": [
            "P1 reproduces published risks, pooled RMSE, denominators and the four-term identity at the "
            "stated tolerances; it is descriptive and does not discriminate mechanisms.",
            "P2 reports conditioning and prior sensitivity of the frozen initializer; the ridge scan and "
            "r0 model-dependence comparison are diagnostics, not data-identifiability results.",
            "P2b (ridge to 8 s risk) was not run; no downstream-risk claim is made.",
            "No val/test payload is read, no optimizer is created and no model is trained or selected.",
            "No mechanism, optimization-sufficiency or model-adoption conclusion is asserted.",
        ],
    }
    _assert_public(payload)
    return payload


def publish(p1_path: Path | str, p2_path: Path | str, output: Path | str) -> dict:
    p1_path = guard_input(p1_path, PRIVATE_ROOT)
    p2_path = guard_input(p2_path, PRIVATE_ROOT)
    output = guard_output(output, RESULTS_ROOT)
    payload = public_payload(read_json(p1_path), read_json(p2_path))
    write_json(output, payload)
    return payload


# ------------------------------------------------------------------- runtime


def _peak_working_set_bytes() -> int | None:
    """Peak working-set bytes of this process; None when unavailable."""
    try:
        if os.name == "nt":
            import ctypes
            from ctypes import wintypes

            class ProcessMemoryCounters(ctypes.Structure):
                _fields_ = [
                    ("cb", wintypes.DWORD),
                    ("PageFaultCount", wintypes.DWORD),
                    ("PeakWorkingSetSize", ctypes.c_size_t),
                    ("WorkingSetSize", ctypes.c_size_t),
                    ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                    ("PagefileUsage", ctypes.c_size_t),
                    ("PeakPagefileUsage", ctypes.c_size_t),
                ]

            counters = ProcessMemoryCounters()
            counters.cb = ctypes.sizeof(counters)
            handle = ctypes.windll.kernel32.GetCurrentProcess
            handle.restype = wintypes.HANDLE
            get_memory = ctypes.windll.psapi.GetProcessMemoryInfo
            get_memory.argtypes = (
                wintypes.HANDLE,
                ctypes.POINTER(ProcessMemoryCounters),
                wintypes.DWORD,
            )
            if not get_memory(handle(), ctypes.byref(counters), counters.cb):
                return None
            return int(counters.PeakWorkingSetSize)
        import resource

        return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * 1024
    except Exception:
        return None


def _runtime(started: float, extra: dict | None = None) -> dict:
    peak = _peak_working_set_bytes()
    record = {
        "wall_seconds": float(time.perf_counter() - started),
        "peak_working_set_bytes": peak,
        "peak_working_set_mb": (peak / (1024 * 1024)) if peak is not None else None,
        "device": "cpu",
        "precision": "float64",
        "torch_threads": int(torch.get_num_threads()),
        "training": False,
        "optimizer_calls": 0,
        "model_rollout": False,
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "numpy": np.__version__,
    }
    if extra:
        record.update(extra)
    return record


# ----------------------------------------------------------------------- CLI


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="closure_probe_20261030",
        description="Read-only closure probe: P1 risk decomposition, P2 r0 conditioning, publish.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    decompose_parser = sub.add_parser("decompose", help="P1 four-term risk decomposition")
    decompose_parser.add_argument("--output", type=Path, required=True)

    conditioning_parser = sub.add_parser("init-conditioning", help="P2 initial hidden force conditioning")
    conditioning_parser.add_argument("--output", type=Path, required=True)

    publish_parser = sub.add_parser("publish", help="aggregate public probe result")
    publish_parser.add_argument("--p1", type=Path, required=True)
    publish_parser.add_argument("--p2", type=Path, required=True)
    publish_parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "decompose":
        output = guard_output(args.output, PRIVATE_ROOT)
        payload = decompose()
        write_json(output, payload)
        print(
            json.dumps(
                {
                    "output": str(output),
                    "status": payload["status"],
                    "acceptance_passed": payload["acceptance"]["passed"],
                    "risk_max_abs_error": payload["acceptance"]["risk_gate"]["max_abs_error"],
                    "rmse_max_abs_error": payload["acceptance"]["rmse_gate"]["max_abs_error"],
                    "term_sum_max_abs_gap": payload["acceptance"]["term_sum_gate"]["max_abs_gap"],
                    "wall_seconds": payload["runtime"]["wall_seconds"],
                }
            )
        )
        return 0 if payload["acceptance"]["passed"] else 1
    if args.command == "init-conditioning":
        output = guard_output(args.output, PRIVATE_ROOT)
        payload = init_conditioning()
        write_json(output, payload)
        print(
            json.dumps(
                {
                    "output": str(output),
                    "status": payload["status"],
                    "wall_seconds": payload["runtime"]["wall_seconds"],
                    "peak_working_set_mb": payload["runtime"]["peak_working_set_mb"],
                }
            )
        )
        return 0
    if args.command == "publish":
        output = guard_output(args.output, RESULTS_ROOT)
        payload = publish(args.p1, args.p2, output)
        print(json.dumps({"output": str(output), "status": payload["status"], "schema": payload["schema"]}))
        return 0
    parser.error(f"unknown command {args.command}")  # pragma: no cover
    return 2  # pragma: no cover


if __name__ == "__main__":
    raise SystemExit(main())
