"""Read-only COI-driven budget + scalar-closure sufficiency probe (P2, no training).

Scope
-----
* No training, no optimizer step, no model rollout and no new prediction file.
* The only event payload read is the frozen fit297 preparation (267 stable, 30 unstable);
  val/test are never addressed and the frozen preparation must certify them closed.
* Frozen prediction ``*.npz`` and frozen model exports are read.  Writes are confined to
  ``private_data/coi_budget_20261030`` (``run``) and ``results/coi_budget_20261030``
  (``publish``); no existing file is modified.

Objects (frozen mechanical body ``M``, ``D``, ``q_p = jpi_p * storage_base``)
----------------------------------------------------------------------------
Model mechanical equation (``force_graph_model_20261028.ForceGraphROM.field``)::

    M * omega_dot = -grad U(delta) - D * omega + q_p * p - r,   1^T grad U = 0

With ``R1 = 1^T r / 10`` and ``W1 = 1^T omega / 10`` the mechanical inversion of the
collective hidden force under the *frozen* body is

.. math::

    1^T r_est(t) = -1^T(M*omega_dot_true) - 1^T(D*omega_true) + 1^T(q_p*p_true),

where ``omega_dot_true`` is ``np.gradient(omega_true, 0.01, axis=0, edge_order=2)``.
This is a mechanical-inversion estimate under the frozen body; it is **not** the true
hidden force, and any error of the true machine parameters (D, q_p, U) is absorbed into
it.  Only the collective component ``1^T r_est`` is defined by this inversion.

COI budget (model-M-weighted mean; the project COI convention)::

    dw_COI(t) = sum_i M_i (omega_pred_i - omega_true_i) / sum_i M_i
              = (1/sum M) int_0^t [ -1^T dr - 1^T(D*dw) + 1^T(q_p*dp) ] ds,
    dr = r_pred - r_est,  dw = omega_pred - omega_true,  dp = p_pred - p_true.

Because ``r_est`` is defined by inversion of the truth, this identity is closed by
construction up to (i) the initial COI offset ``dw_COI(0)`` (the window starts one
0.01 s step after initialisation) and (ii) finite-difference/trapezoid error.  The
informative content is therefore the three-term share and the residual scale, **not**
"closure holds".

Scalar closure sufficiency
--------------------------
``Q = lambda W + B diag(lambda_ref a_bar) B^T`` with ``B^T 1 = 0`` gives ``1^T Q = lambda 1^T``
exactly, hence ``dR1/dt = kappa W1 - lambda R1`` and ``d(1^T r)/dt = kappa 1^T omega -
lambda 1^T r``.  Driven by ``W1_true`` with the model's own ``kappa``/``lambda`` and
``R1_model(0)`` this yields ``R1_closure(t)``, compared against ``R1_est(t)``.  A
per-event best ``(kappa, lambda, R1(0))`` fit on a fixed lambda grid separates "closure
form insufficient" from "initial value / W1 trajectory wrong".  The same integrator on
the model's own trajectory (``W1_model`` -> ``R1_model``) is the implementation
self-consistency check of the exact family identity.

Acceptance checks (protocol 2026-10-30, gates 4-5)
--------------------------------------------------
* budget closure residual only at finite-difference/discretisation scale (and the
  measured initial COI offset is reported separately, never hidden);
* the scalar integrator recovers a known ``(kappa, lambda, R10)`` trajectory;
* ``r_est`` sign/index alignment is unit-tested;
* a model-trajectory cross-check (inversion of the model's own ``omega``/``p``/``r``)
  reports the finite-difference floor of the whole pipeline.
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

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "experiments"), str(ROOT / "src")]

SCHEMA = "coi_budget_20261030_v1"
PUBLIC_SCHEMA = "coi_budget_20261030_public_v1"

# ----------------------------------------------------------------- frozen constants
OUTPUT_DT = 0.01            # s; the truth/decimated prediction grid
FINE_DT = 0.005             # s; frozen fine prediction grid
FINE_STRIDE = 2             # fine[::2] lands on the 0.01 s output grid
HISTORY_ROWS = 11           # 100 ms causal history
FINE_TIME_POINTS = 3001
COARSE_TIME_POINTS = 1501
PREDICTION_CHUNKS = 19
N_MACHINES = 10
EXPECTED_EVENTS = 297
EXPECTED_STABLE = 267
EXPECTED_UNSTABLE = 30

SCOPES = ("5", "8", "prefix")
SCOPE_STEPS = {"5": 500, "8": 800}
MAX_ACTUAL_STEPS = 1500
MIN_WINDOW_STEPS = 3          # np.gradient(edge_order=2) needs three samples
MIN_FIT_STEPS = 20            # below this a 3-parameter per-event fit is not informative
LAMBDA_GRID = np.geomspace(1e-3, 10.0, 241)   # >= 200 log-spaced points, 1e-3 .. 10 / s
COHORT_LABEL = {"5": "stable", "8": "stable", "prefix": "unstable"}
SCOPE_DEFINITION = {
    "5": "stable events; fine[::2][1:501] vs x[11:511]",
    "8": "stable events; fine[::2][1:801] vs x[11:811]",
    "prefix": "unstable events; n=min(1500,len(x)-11); fine[::2][1:n+1] vs x[11:11+n]",
}

MECHANICAL_FIELDS = ("mass", "damping", "jpi_p", "storage_base", "p_scale")

SGM_STUDY = ROOT / "private_data/state_gated_memory_20261028/study01"
TRANSPORT_STUDY = ROOT / "private_data/memory_transport_20261029/study01_prelaunch_retry1/study"
RN_SOURCE_EXPORT = (
    ROOT
    / "private_data/force_graph_20261028/study01/study/N6101_M_28102/checkpoints/step000192/model.json"
)
MODEL_SPECS = {
    "A": {
        "export": TRANSPORT_STUDY / "A_30311/checkpoints/step000128/model.json",
        "predictions": TRANSPORT_STUDY / "A_30311/fit000128",
        "loader": "transport",
        "scope": "frozen A128 transport checkpoint (fit000128)",
    },
    "S": {
        "export": TRANSPORT_STUDY / "S_30311/checkpoints/step000128/model.json",
        "predictions": TRANSPORT_STUDY / "S_30311/fit000128",
        "loader": "transport",
        "scope": "frozen S128 transport checkpoint (fit000128)",
    },
    "P": {
        "export": TRANSPORT_STUDY / "P_30311/checkpoints/step000128/model.json",
        "predictions": TRANSPORT_STUDY / "P_30311/fit000128",
        "loader": "transport",
        "scope": "frozen P128 transport checkpoint (fit000128)",
    },
    "R_N": {
        "export": RN_SOURCE_EXPORT,
        "predictions": SGM_STUDY / "qualification/R_N_fit297",
        "loader": "force_graph",
        "scope": "frozen R_N force-graph reference (fit297 predictions)",
    },
}
MODEL_ORDER = ("A", "S", "P", "R_N")
HEADLINE_MODELS = ("A", "R_N")

PRIVATE_ROOT = ROOT / "private_data/coi_budget_20261030"
RESULTS_ROOT = ROOT / "results/coi_budget_20261030"
DATA_SOURCE_PATHS = (SGM_STUDY, TRANSPORT_STUDY, RN_SOURCE_EXPORT)

# Calibrated numerical guards; no selection on measured performance.
SHARE_ZERO_TOL = 1e-12          # |sum of terminal integrals| below this -> share null
EIGEN_GUARD = 1e-18             # relative 2x2 determinant guard for the lambda fit
FINITE_GUARD = 1e-300

CLAIM_SCOPE = (
    "Frozen fit297 train evidence only (already consumed; not independent confirmation). "
    "R1_est is a mechanical-inversion estimate under the frozen M/D/q_p, not the true hidden "
    "force. The COI budget is closed by construction up to the initial COI offset and "
    "finite-difference/trapezoid error, so its informative content is the three-term share and "
    "the residual scale. No training, no optimizer call, no rollout, no val/test, no model "
    "promotion. Nothing here identifies a physical mechanism or proves the common-mode closure "
    "hypothesis."
)


# ----------------------------------------------------------------------------- I/O


def _relative_or_absolute(path: Path) -> str:
    """Repository-relative POSIX path when possible, absolute POSIX path otherwise."""
    try:
        return Path(path).resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return Path(path).resolve().as_posix()


def sha256_file(path: Path | str) -> str:
    """Hash file bytes without interpreting the payload."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _resolve(path: Path | str) -> Path:
    """Resolve CLI paths against the repository root, not the caller's cwd."""
    candidate = Path(path)
    if not candidate.is_absolute():
        candidate = ROOT / candidate
    return candidate.resolve()


def guard_output(path: Path | str, root: Path) -> Path:
    """Return the resolved path only if it is a ``.json`` file under ``root``."""
    resolved = _resolve(path)
    root = _resolve(root)
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"output path outside the allowed write root {root}: {resolved}") from exc
    if resolved.suffix.lower() != ".json":
        raise ValueError(f"output must be a .json file: {resolved}")
    return resolved


def guard_input(path: Path | str, root: Path) -> Path:
    """Return the resolved path only if it is an existing file under ``root``."""
    resolved = _resolve(path)
    root = _resolve(root)
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"input path outside the read-only private root {root}: {resolved}") from exc
    if not resolved.is_file():
        raise FileNotFoundError(resolved)
    return resolved


def plain(value):
    """Recursively convert numpy scalars/arrays to plain JSON-compatible values."""
    if isinstance(value, dict):
        return {str(key): plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [plain(item) for item in value]
    if isinstance(value, np.ndarray):
        return [plain(item) for item in value.tolist()]
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        value = float(value)
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("non-finite float in payload")
    return value


def write_json(path: Path | str, payload: dict) -> Path:
    """Atomic JSON write under a caller-guarded path; NaN/Infinity are forbidden."""
    target = _resolve(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_name(f".{target.name}.{uuid.uuid4().hex}.tmp")
    text = json.dumps(plain(payload), ensure_ascii=False, indent=2, allow_nan=False)
    tmp.write_text(text + "\n", encoding="utf-8")
    os.replace(tmp, target)
    return target


def read_json(path: Path | str) -> dict:
    target = _resolve(path)
    if not target.is_file():
        raise FileNotFoundError(target)
    return json.loads(target.read_text(encoding="utf-8"))


# ------------------------------------------------------------------------ frozen data


def load_frozen_events() -> tuple[dict, list[dict]]:
    """Load and validate the frozen fit297 preparation (read-only, fit_only)."""
    import state_gated_memory_study_20261028 as sgm

    record, events = sgm.checked_preparation(SGM_STUDY, fit_only=True)
    if len(events) != EXPECTED_EVENTS:
        raise ValueError(f"fit297 changed: {len(events)} events")
    labels = [event["label"] for event in events]
    if labels.count("stable") != EXPECTED_STABLE or labels.count("unstable") != EXPECTED_UNSTABLE:
        raise ValueError("stable/unstable denominator changed")
    if len({event["id"] for event in events}) != EXPECTED_EVENTS:
        raise ValueError("duplicate fit event id")
    for event in events:
        x = np.asarray(event["x"])
        if x.dtype != np.float64 or x.ndim != 2 or x.shape[1] != 30 or len(x) <= HISTORY_ROWS:
            raise ValueError(f"unexpected event array for {event['id']}")
        if not np.isfinite(x).all():
            raise ValueError(f"non-finite event array for {event['id']}")
    assert_closed_splits(record)
    return record, events


def assert_closed_splits(record: dict) -> None:
    """Refuse any preparation that does not certify val/test closed."""
    if record.get("val_read") is not False or record.get("test_read") is not False:
        raise ValueError("frozen preparation does not certify val/test closed")
    if record.get("status") != "COMPLETED":
        raise ValueError("frozen preparation is not COMPLETED")


def load_model(name: str):
    """Load one frozen model export on CPU float64 in eval mode; no inference is run."""
    if name not in MODEL_SPECS:
        raise ValueError(f"unknown model {name}")
    spec = MODEL_SPECS[name]
    if not Path(spec["export"]).is_file():
        raise FileNotFoundError(spec["export"])
    if spec["loader"] == "transport":
        from memory_transport_model_20261029 import MemoryTransportROM

        model = MemoryTransportROM.from_export(Path(spec["export"]))
    elif spec["loader"] == "force_graph":
        from force_graph_model_20261028 import ForceGraphROM

        model = ForceGraphROM.from_export(Path(spec["export"]))
    else:  # pragma: no cover - registry only
        raise ValueError(f"unknown loader {spec['loader']}")
    return model.to("cpu").double().eval()


def mechanical_buffers(model) -> dict:
    """Extract the frozen mechanical body from a loaded model as float64 arrays."""
    values = {}
    for name in MECHANICAL_FIELDS:
        tensor = getattr(model, name).detach().cpu().double().numpy().copy()
        if tensor.shape != (N_MACHINES,) or not np.isfinite(tensor).all():
            raise ValueError(f"invalid mechanical buffer {name}")
        values[name] = tensor
    values["q_p"] = values["jpi_p"] * values["storage_base"]
    return values


def model_identity(name: str, model) -> dict:
    buffers = mechanical_buffers(model)
    return {
        "name": name,
        "scope": MODEL_SPECS[name]["scope"],
        "export": Path(MODEL_SPECS[name]["export"]).relative_to(ROOT).as_posix(),
        "export_sha256": sha256_file(MODEL_SPECS[name]["export"]),
        "kappa": float(model.kappa),
        "lambda_": float(model.lambda_),
        "variant": getattr(model, "variant", None),
        "buffers": {key: buffers[key].tolist() for key in MECHANICAL_FIELDS},
        "signature": buffer_signature(buffers),
    }


def buffer_signature(buffers: dict) -> str:
    """Exact identity of the five frozen mechanical/product buffers."""
    blob = json.dumps({key: buffers[key].tolist() for key in MECHANICAL_FIELDS},
                      sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(blob).hexdigest()


def check_mechanical_identity(identities: dict, canonical: str = MODEL_ORDER[0]) -> dict:
    """Compare mechanical buffers element-wise; never substitute one model's body."""
    reference = identities[canonical]["buffers"]
    bit_identical, different = {}, {}
    for name, record in identities.items():
        fields = []
        for field in MECHANICAL_FIELDS:
            same = bool(np.array_equal(np.asarray(record["buffers"][field]),
                                       np.asarray(reference[field])))
            if not same:
                fields.append(field)
        bit_identical[name] = not fields
        different[name] = fields
    return {
        "canonical": canonical,
        "fields": list(MECHANICAL_FIELDS),
        "bit_identical_to_canonical": bit_identical,
        "different_fields": different,
        "shared_signature": identities[canonical]["signature"],
        "per_model_signature": {name: record["signature"] for name, record in identities.items()},
        "substitution": "none; each model uses its own frozen buffers",
        "note": (
            "element-wise float64 equality of mass/damping/jpi_p/storage_base/p_scale. "
            "A mismatch would be reported and each model still inverted with its own body."
        ),
    }


# --------------------------------------------------------------------------- windows


def window_steps(event: dict, scope: str) -> int | None:
    """Registered 0.01 s future length for one scope, or None when unavailable."""
    if scope not in SCOPES:
        raise ValueError(f"unknown scope {scope}")
    available = len(event["x"]) - HISTORY_ROWS
    if scope in SCOPE_STEPS:
        steps = SCOPE_STEPS[scope]
    else:
        steps = min(MAX_ACTUAL_STEPS, available)
    if available < steps or steps < MIN_WINDOW_STEPS:
        return None
    return int(steps)


def truth_window(event: dict, steps: int) -> np.ndarray:
    """``x[11:11+steps, :30]``: the observed future on the 0.01 s grid."""
    return np.asarray(event["x"], dtype=np.float64)[HISTORY_ROWS:HISTORY_ROWS + steps, :30]


def prediction_window(fine_row: np.ndarray, steps: int) -> np.ndarray:
    """``fine[::2][1:steps+1]``: the 0.01 s decimated frozen prediction path (40 cols)."""
    return np.asarray(fine_row, dtype=np.float64)[::FINE_STRIDE][1:steps + 1]


def cohort(events: list[dict], scope: str) -> list[dict]:
    """Stable cohort for 5 s/8 s; unstable cohort for the actual interval."""
    label = COHORT_LABEL[scope]
    return [event for event in events
            if event["label"] == label and window_steps(event, scope) is not None]


# ------------------------------------------------------------------- scalar closure


def _pl_weights(lam, dt: float = OUTPUT_DT):
    """Exact piecewise-linear weights of ``dR/dt = -lam R + w(t)`` over one 0.01 s step.

    ``w`` is linear between samples.  ``R(t+dt) = decay*R(t) + w0*alpha + dw*beta`` with
    ``alpha = (1-exp(-lam dt))/lam`` and ``beta = (dt-alpha)/lam``; both limits are stable
    for lam*dt << 1 (``alpha -> dt``, ``beta -> dt**2/2``).
    """
    lam = np.asarray(lam, dtype=np.float64)
    decay = np.exp(-lam * dt)
    alpha = -np.expm1(-lam * dt) / lam
    beta = (dt - alpha) / lam
    return decay, alpha, beta


def integrate_scalar_closure(w1, r10: float, kappa: float, lam: float,
                             dt: float = OUTPUT_DT) -> np.ndarray:
    """Piecewise-linear-exact solution of ``dR1/dt = kappa W1 - lam R1``.

    ``w1`` is the sampled driver (``W1 = mean_i omega_i``); the solution is exact for the
    piecewise-linear interpolant of ``w1``, i.e. the same discrete format as the 0.01 s
    output grid.  No Runge-Kutta truncation is involved.
    """
    w1 = np.asarray(w1, dtype=np.float64)
    if w1.ndim != 1 or len(w1) < 1:
        raise ValueError("one-dimensional driver with at least one sample required")
    decay, alpha, beta = _pl_weights(float(lam), dt)
    out = np.empty_like(w1)
    out[0] = float(r10)
    kappa = float(kappa)
    for index in range(len(w1) - 1):
        rate = kappa * (w1[index] * alpha
                        + (w1[index + 1] - w1[index]) / dt * beta)
        out[index + 1] = decay * out[index] + rate
    return out


def scalar_closure_green(w1, lambda_grid, dt: float = OUTPUT_DT) -> np.ndarray:
    """``G_lambda(t) = int_0^t exp(-lam(t-s)) W1(s) ds`` on the output grid, shape (n, L)."""
    w1 = np.asarray(w1, dtype=np.float64)
    if w1.ndim != 1 or len(w1) < 1:
        raise ValueError("one-dimensional driver required")
    grid = np.atleast_1d(np.asarray(lambda_grid, dtype=np.float64))
    decay, alpha, beta = _pl_weights(grid, dt)
    green = np.zeros((len(w1), len(grid)), dtype=np.float64)
    for index in range(len(w1) - 1):
        rate = w1[index] * alpha + (w1[index + 1] - w1[index]) / dt * beta
        green[index + 1] = decay * green[index] + rate
    return green


def relative_rms(value, reference):
    """``||value|| / ||reference||`` over the window (both RMS, so scale-free)."""
    value = np.asarray(value, dtype=np.float64)
    reference = np.asarray(reference, dtype=np.float64)
    denominator = math.sqrt(float(np.mean(reference * reference)))
    if denominator <= 0.0:
        return None
    return math.sqrt(float(np.mean(value * value))) / denominator


def fit_scalar_closure(target, w1, lambda_grid=None, dt: float = OUTPUT_DT,
                       r10_fixed: float | None = None) -> dict:
    """Best ``lambda`` on the fixed grid for ``R1_est(t) = e^{-lam t} R10 + kappa G_lam(t)``.

    Two linear least-squares variants are returned for one driver/target pair:

    * ``free``: (R10, kappa) free, three effective parameters including lambda;
    * ``fixed``: ``R10 = r10_fixed`` (the model's ``R1_model(0)``), kappa fit per lambda.

    Returns the best grid point plus relative RMS residual.  Grid points whose 2x2 normal
    matrix is numerically singular are skipped and counted.
    """
    target = np.asarray(target, dtype=np.float64)
    w1 = np.asarray(w1, dtype=np.float64)
    if target.ndim != 1 or target.shape != w1.shape:
        raise ValueError("target and driver must be equal one-dimensional arrays")
    if len(target) < MIN_WINDOW_STEPS:
        raise ValueError("window too short for the scalar fit")
    grid = np.atleast_1d(np.asarray(LAMBDA_GRID if lambda_grid is None else lambda_grid,
                                    dtype=np.float64))
    t = dt * np.arange(len(target), dtype=np.float64)
    basis = np.exp(-grid[None, :] * t[:, None])
    green = scalar_closure_green(w1, grid, dt)

    s00 = np.einsum("ij,ij->j", basis, basis)
    s01 = np.einsum("ij,ij->j", basis, green)
    s11 = np.einsum("ij,ij->j", green, green)
    b0 = basis.T @ target
    b1 = green.T @ target
    norm = target @ target
    determinant = s00 * s11 - s01 * s01
    scale = s00 * s11
    valid = (scale > 0.0) & (determinant > EIGEN_GUARD * np.maximum(scale, FINITE_GUARD))
    safe = np.where(valid, determinant, 1.0)
    c0 = np.where(valid, (b0 * s11 - b1 * s01) / safe, np.nan)
    c1 = np.where(valid, (s00 * b1 - s01 * b0) / safe, np.nan)
    rss = (norm
           - 2.0 * c0 * b0 - 2.0 * c1 * b1
           + c0 * c0 * s00 + 2.0 * c0 * c1 * s01 + c1 * c1 * s11)
    rss = np.where(valid, np.clip(rss, 0.0, None), np.inf)
    best = int(np.argmin(rss))
    target_rms = _rms(target)
    free = {
        "lambda": float(grid[best]),
        "kappa": float(c1[best]),
        "r10": float(c0[best]),
        "rss": float(rss[best]),
        "rel_rms": (math.sqrt(max(float(rss[best]), 0.0) / len(target)) / target_rms
                    if target_rms > 0 else None),
        "grid_points": int(len(grid)),
        "skipped_grid_points": int(len(grid) - int(valid.sum())),
    }

    fixed = None
    if r10_fixed is not None:
        residual_basis = target[:, None] - basis * float(r10_fixed)
        denominator = np.maximum(s11, FINITE_GUARD)
        kappa = np.einsum("ij,ij->j", green, residual_basis) / denominator
        fixed_rss = np.einsum("ij,ij->j", residual_basis, residual_basis) \
            - 2.0 * kappa * np.einsum("ij,ij->j", green, residual_basis) \
            + kappa * kappa * s11
        fixed_rss = np.clip(fixed_rss, 0.0, None)
        fixed_valid = s11 > 0.0
        fixed_rss = np.where(fixed_valid, fixed_rss, np.inf)
        fixed_best = int(np.argmin(fixed_rss))
        fixed = {
            "lambda": float(grid[fixed_best]),
            "kappa": float(kappa[fixed_best]),
            "r10": float(r10_fixed),
            "rss": float(fixed_rss[fixed_best]),
            "rel_rms": (math.sqrt(max(float(fixed_rss[fixed_best]), 0.0) / len(target))
                        / target_rms if target_rms > 0 else None),
            "grid_points": int(len(grid)),
        }
    return {"free": free, "fixed": fixed}


def _rms(value) -> float:
    value = np.asarray(value, dtype=np.float64)
    if value.size == 0:
        return float("nan")
    return math.sqrt(float(np.mean(value * value)))


# ------------------------------------------------------------------ budget pieces


def omega_dot(omega: np.ndarray, dt: float = OUTPUT_DT) -> np.ndarray:
    """Second-order central/one-sided time derivative on the 0.01 s output grid."""
    omega = np.asarray(omega, dtype=np.float64)
    if omega.ndim != 2 or omega.shape[0] < MIN_WINDOW_STEPS:
        raise ValueError("omega window must be (steps, 10) with steps >= 3")
    return np.gradient(omega, dt, axis=0, edge_order=2)


def collective_force_sum(omega: np.ndarray, p: np.ndarray, mass: np.ndarray,
                         damping: np.ndarray, q_p: np.ndarray) -> np.ndarray:
    """``1^T r_est = -1^T(M*omega_dot) - 1^T(D*omega) + 1^T(q_p*p)`` per time point."""
    derivative = omega_dot(omega)
    return (-np.einsum("ij,j->i", derivative, mass)
            - np.einsum("ij,j->i", omega, damping)
            + np.einsum("ij,j->i", p, q_p))


def cumulative_trapezoid(values: np.ndarray, dt: float = OUTPUT_DT) -> np.ndarray:
    """Cumulative trapezoid integral from the first sample (0 at index 0)."""
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 1 or len(values) < 1:
        raise ValueError("one-dimensional series required")
    out = np.zeros_like(values)
    out[1:] = np.cumsum(0.5 * dt * (values[1:] + values[:-1]))
    return out


def budget_terms(omega_true, p_true, omega_pred, p_pred, r_pred, mass, damping, q_p,
                 dt: float = OUTPUT_DT, detail: bool = True) -> dict:
    """COI budget terms, cumulative integrals and closure residual for one window.

    ``detail=False`` keeps only the scalar summary (used by the streaming run); the full
    series are kept for unit tests and external recomputation.
    """
    omega_true = np.asarray(omega_true, dtype=np.float64)
    p_true = np.asarray(p_true, dtype=np.float64)
    omega_pred = np.asarray(omega_pred, dtype=np.float64)
    p_pred = np.asarray(p_pred, dtype=np.float64)
    r_pred = np.asarray(r_pred, dtype=np.float64)
    steps = len(omega_true)
    for name, array in (("p_true", p_true), ("omega_pred", omega_pred),
                        ("p_pred", p_pred), ("r_pred", r_pred)):
        if array.shape != (steps, N_MACHINES):
            raise ValueError(f"{name} window shape differs")
    total_mass = float(np.sum(mass))
    sum_r_est = collective_force_sum(omega_true, p_true, mass, damping, q_p)
    delta_omega = omega_pred - omega_true
    delta_p = p_pred - p_true
    delta_r_sum = r_pred.sum(axis=1) - sum_r_est

    term_r = -delta_r_sum
    term_damping = -np.einsum("ij,j->i", delta_omega, damping)
    term_port = np.einsum("ij,j->i", delta_p, q_p)
    integral_r = cumulative_trapezoid(term_r, dt)
    integral_d = cumulative_trapezoid(term_damping, dt)
    integral_p = cumulative_trapezoid(term_port, dt)
    total_integral = integral_r + integral_d + integral_p

    delta_omega_coi = np.einsum("ij,j->i", delta_omega, mass) / total_mass
    closure = delta_omega_coi - total_integral / total_mass
    closure_detrended = (delta_omega_coi - delta_omega_coi[0]) - total_integral / total_mass

    terminal = np.array([integral_r[-1], integral_d[-1], integral_p[-1]], dtype=np.float64)
    terminal_sum = float(terminal.sum())
    if abs(terminal_sum) <= SHARE_ZERO_TOL:
        terminal_share = [None, None, None]
    else:
        terminal_share = (terminal / terminal_sum).tolist()
    mean_abs = np.array([np.mean(np.abs(term_r)),
                         np.mean(np.abs(term_damping)),
                         np.mean(np.abs(term_port))], dtype=np.float64)
    mean_abs_total = float(mean_abs.sum())
    mean_abs_share = (mean_abs / mean_abs_total).tolist() if mean_abs_total > 0 else [None] * 3
    abs_terminal = float(np.abs(terminal).sum())
    cancellation = (abs_terminal / abs(terminal_sum)) if abs(terminal_sum) > SHARE_ZERO_TOL else None

    summary = {
        "steps": steps,
        "terminal": terminal.tolist(),
        "terminal_sum": terminal_sum,
        "terminal_share": terminal_share,
        "mean_abs": mean_abs.tolist(),
        "mean_abs_share": mean_abs_share,
        "cancellation_index": cancellation,
        "delta_omega_coi_first": float(delta_omega_coi[0]),
        "delta_omega_coi_last": float(delta_omega_coi[-1]),
        "closure_max_abs": float(np.max(np.abs(closure))),
        "closure_max_abs_detrended": float(np.max(np.abs(closure_detrended))),
        "closure_rms": float(np.sqrt(np.mean(closure * closure))),
        "closure_rms_detrended": float(np.sqrt(np.mean(closure_detrended ** 2))),
        "r1_est_rms": _rms(sum_r_est / 10.0),
        "r1_est_first": float(sum_r_est[0] / 10.0),
        "r1_est_last": float(sum_r_est[-1] / 10.0),
    }
    result = {"steps": steps, "summary": summary, "sum_r_est": sum_r_est}
    if detail:
        result.update({
            "r1_est": (sum_r_est / 10.0).tolist(),
            "sum_r_est": sum_r_est.tolist(),
            "term_r": term_r.tolist(),
            "term_damping": term_damping.tolist(),
            "term_port": term_port.tolist(),
            "integral_r": integral_r.tolist(),
            "integral_d": integral_d.tolist(),
            "integral_p": integral_p.tolist(),
            "total_integral": total_integral.tolist(),
            "delta_omega_coi": delta_omega_coi.tolist(),
            "closure": closure.tolist(),
            "closure_detrended": closure_detrended.tolist(),
        })
    return result


def cross_check_terms(omega_pred, p_pred, r_pred, mass, damping, q_p,
                      dt: float = OUTPUT_DT) -> dict:
    """Model-trajectory cross-check: invert the model's own path and integrate.

    With ``truth := model`` all differences vanish except the model's own ODE residual
    reconstructed from the finite-difference derivative, so ``max|I_r|/sum M`` measures
    the finite-difference/trapezoid floor of the whole pipeline (should be ~0).
    """
    sum_r_est = collective_force_sum(np.asarray(omega_pred), np.asarray(p_pred),
                                     mass, damping, q_p)
    delta_r_sum = np.asarray(r_pred).sum(axis=1) - sum_r_est
    integral = cumulative_trapezoid(-delta_r_sum, dt)
    total_mass = float(np.sum(mass))
    sum_est = sum_r_est / 10.0
    return {
        "inversion_rms": _rms(sum_est),
        "residual_sum_rms": _rms(delta_r_sum),
        "residual_sum_max_abs": float(np.max(np.abs(delta_r_sum))),
        "closure_max_abs": float(np.max(np.abs(integral)) / total_mass),
        "closure_last_abs": float(abs(integral[-1]) / total_mass),
        "delta_r_sum_first": float(delta_r_sum[0]),
    }


def closure_terms(omega_true, p_true, omega_pred, p_pred, r_pred, sum_r_est,
                  kappa: float, lam: float, lambda_grid=None,
                  dt: float = OUTPUT_DT) -> dict:
    """Scalar-closure sufficiency terms for one window (model kappa/lambda fixed)."""
    omega_true = np.asarray(omega_true, dtype=np.float64)
    omega_pred = np.asarray(omega_pred, dtype=np.float64)
    r1_est = np.asarray(sum_r_est, dtype=np.float64) / 10.0
    r1_model = np.asarray(r_pred, dtype=np.float64).mean(axis=1)
    w1_true = omega_true.mean(axis=1)
    w1_model = omega_pred.mean(axis=1)
    r10_model = float(r1_model[0])
    t = dt * np.arange(len(r1_est), dtype=np.float64)

    forward = integrate_scalar_closure(w1_true, r10_model, kappa, lam, dt)
    oracle_init = integrate_scalar_closure(w1_true, float(r1_est[0]), kappa, lam, dt)
    model_drive = integrate_scalar_closure(w1_model, r10_model, kappa, lam, dt)

    residual = forward - r1_est
    oracle_residual = oracle_init - r1_est
    init_component = np.exp(-float(lam) * t) * (r10_model - float(r1_est[0]))
    w1_component = kappa * scalar_closure_green(w1_true - w1_model,
                                                np.array([float(lam)]), dt)[:, 0]
    residual_norm = float(residual @ residual)
    dW_norm = float(w1_component @ w1_component)
    init_norm = float(init_component @ init_component)
    scale = _rms(r1_est)
    out = {
        "r10_model": r10_model,
        "r1_est_first": float(r1_est[0]),
        "r1_est_rms": scale,
        "forward_rel_rms": relative_rms(residual, r1_est),
        "oracle_r10_rel_rms": relative_rms(oracle_residual, r1_est),
        "init_component_rel": (math.sqrt(init_norm) / (scale * math.sqrt(len(r1_est)))
                               if scale > 0 else None),
        "w1_rel_diff": relative_rms(w1_true - w1_model, w1_true),
        "w1_component_rel": (math.sqrt(dW_norm) / (scale * math.sqrt(len(r1_est)))
                             if scale > 0 else None),
        "amplification": (math.sqrt(residual_norm) / math.sqrt(dW_norm))
        if dW_norm > 0 else None,
        "w1_variance_explained": ((residual_norm - float((residual - w1_component)
                                                        @ (residual - w1_component)))
                                  / residual_norm) if residual_norm > 0 else None,
        "w1_projection_share": (float(w1_component @ residual) / residual_norm)
        if residual_norm > 0 else None,
        "init_projection_share": (float(init_component @ residual) / residual_norm)
        if residual_norm > 0 else None,
        "self_consistency_rel_rms": relative_rms(model_drive - r1_model, r1_model),
        "r1_model_rel_diff": relative_rms(r1_model - r1_est, r1_est),
        "r1_model_rms": _rms(r1_model),
        "fit": None,
        "fit_fixed_r10": None,
    }
    if len(r1_est) >= MIN_FIT_STEPS:
        fitted = fit_scalar_closure(r1_est, w1_true, lambda_grid, dt, r10_fixed=r10_model)
        out["fit"] = fitted["free"]
        out["fit_fixed_r10"] = fitted["fixed"]
    return out


# -------------------------------------------------------------------------- uniformity


def uniformity_block(buffers: dict) -> dict:
    """min/max/mean/std of the frozen D and q_p (uniformity is not assumed)."""
    block = {}
    for name in ("damping", "q_p"):
        values = np.asarray(buffers[name], dtype=np.float64)
        mean = float(values.mean())
        block[name] = {
            "min": float(values.min()),
            "max": float(values.max()),
            "mean": mean,
            "std": float(values.std()),
            "coefficient_of_variation": float(values.std() / mean) if mean != 0 else None,
            "distinct_values": int(np.unique(values).size),
            "uniform": bool(np.allclose(values, mean, rtol=0.0, atol=1e-12)),
        }
    return block


def uniformity_projection(events: list[dict], scope: str, buffers: dict) -> dict:
    """Magnitude of the non-uniform vs uniform parts of the COI driving sums.

    ``1^T(D*omega) = d_bar 1^T omega + 1^T((D-d_bar)*omega)`` and likewise for q_p; the
    second term is the差模 leakage the brief warns about (H4).
    """
    damping = np.asarray(buffers["damping"], dtype=np.float64)
    q_p = np.asarray(buffers["q_p"], dtype=np.float64)
    d_bar = float(damping.mean())
    q_bar = float(q_p.mean())
    nonuniform_d, uniform_d = [], []
    nonuniform_q, uniform_q = [], []
    for event in cohort(events, scope):
        steps = window_steps(event, scope)
        truth = truth_window(event, steps)
        omega = truth[:, 10:20]
        p = truth[:, 20:30]
        nonuniform_d.append(np.einsum("ij,j->i", omega, damping - d_bar))
        uniform_d.append(d_bar * omega.sum(axis=1))
        nonuniform_q.append(np.einsum("ij,j->i", p, q_p - q_bar))
        uniform_q.append(q_bar * p.sum(axis=1))

    def describe(nonuniform, uniform):
        nu = np.concatenate(nonuniform)
        un = np.concatenate(uniform)
        nu_rms = _rms(nu)
        un_rms = _rms(un)
        return {
            "window_count": len(nonuniform),
            "nonuniform_rms": nu_rms,
            "nonuniform_max_abs": float(np.max(np.abs(nu))),
            "uniform_rms": un_rms,
            "nonuniform_over_uniform": (nu_rms / un_rms) if un_rms > 0 else None,
        }

    return {"scope": scope, "cohort": COHORT_LABEL[scope],
            "D": describe(nonuniform_d, uniform_d),
            "q_p": describe(nonuniform_q, uniform_q)}


# ---------------------------------------------------------------------- aggregates


def _quantiles(values, probs=(10.0, 50.0, 90.0)) -> dict | None:
    values = [float(value) for value in values
              if value is not None and math.isfinite(float(value))]
    if not values:
        return None
    points = np.percentile(np.asarray(values, dtype=np.float64), list(probs))
    return {f"p{int(prob)}": float(point) for prob, point in zip(probs, points)}


def _max_median(values) -> dict | None:
    values = [float(value) for value in values
              if value is not None and math.isfinite(float(value))]
    if not values:
        return None
    array = np.asarray(values, dtype=np.float64)
    return {"max": float(array.max()), "median": float(np.median(array)),
            "count": int(len(values))}


def _share_stats(rows, key: str, index: int) -> dict:
    """Per-event share quantiles and the pooled ratio for one budget term."""
    per_event = [row["summary"][key][index] for row in rows
                 if row["summary"][key] is not None
                 and row["summary"][key][index] is not None]
    numerator_key = "terminal" if key == "terminal_share" else "mean_abs"
    pooled_numerator = sum(row["summary"][numerator_key][index] for row in rows)
    pooled_denominator = sum(float(np.sum(row["summary"][numerator_key])) for row in rows)
    return {
        "median": float(np.median(per_event)) if per_event else None,
        "p10": float(np.percentile(per_event, 10)) if per_event else None,
        "p90": float(np.percentile(per_event, 90)) if per_event else None,
        "count": len(per_event),
        "pooled": (pooled_numerator / pooled_denominator)
        if abs(pooled_denominator) > SHARE_ZERO_TOL else None,
    }


def aggregate_scope(rows: list[dict], model_kappa: float, model_lambda: float) -> dict:
    """Aggregate one model x scope over its events (numeric definitions in the report)."""
    if not rows:
        raise ValueError("empty cohort")
    summary = [row["summary"] for row in rows]
    closure = [row["closure"] for row in rows]
    cross = [row["cross_check"] for row in rows]
    steps = [row["steps"] for row in rows]

    terminal = np.asarray([row["summary"]["terminal"] for row in rows], dtype=np.float64)
    mean_abs = np.asarray([row["summary"]["mean_abs"] for row in rows], dtype=np.float64)
    pooled_terminal = terminal.sum(axis=0)
    pooled_mean_abs = mean_abs.sum(axis=0)
    pooled_terminal_sum = float(pooled_terminal.sum())
    pooled_mean_abs_sum = float(pooled_mean_abs.sum())

    term_names = ("r", "D", "p")
    shares = {name: _share_stats(rows, "terminal_share", index)
              for index, name in enumerate(term_names)}
    abs_shares = {name: _share_stats(rows, "mean_abs_share", index)
                  for index, name in enumerate(term_names)}
    mean_abs_share = [
        [row["summary"]["mean_abs_share"][index] if row["summary"]["mean_abs_share"] else None
         for index in range(3)] for row in rows]
    dominant_r = sum(1 for entry in mean_abs_share
                     if entry[0] is not None and entry[0] >= max(
                         entry[1] if entry[1] is not None else -math.inf,
                         entry[2] if entry[2] is not None else -math.inf))

    fit_rows = [row["closure"]["fit"] for row in rows if row["closure"]["fit"]]
    fit_fixed_rows = [row["closure"]["fit_fixed_r10"] for row in rows
                      if row["closure"]["fit_fixed_r10"]]
    fit_block = {
        "count": len(fit_rows),
        "skipped_short_window": len(rows) - len(fit_rows),
        "rel_rms": _quantiles([entry["rel_rms"] for entry in fit_rows]),
        "lambda": _quantiles([entry["lambda"] for entry in fit_rows]),
        "kappa": _quantiles([entry["kappa"] for entry in fit_rows]),
        "r10": _quantiles([entry["r10"] for entry in fit_rows]),
        "lambda_over_model": _quantiles([entry["lambda"] / model_lambda for entry in fit_rows])
        if model_lambda else None,
        "kappa_over_model": _quantiles([entry["kappa"] / model_kappa for entry in fit_rows])
        if model_kappa else None,
        "lambda_at_grid_floor": sum(1 for entry in fit_rows
                                    if entry["lambda"] <= float(LAMBDA_GRID[0]) * (1 + 1e-9)),
        "lambda_at_grid_ceiling": sum(1 for entry in fit_rows
                                      if entry["lambda"] >= float(LAMBDA_GRID[-1]) * (1 - 1e-9)),
        "nonpositive_kappa": sum(1 for entry in fit_rows if entry["kappa"] <= 0.0),
        "model_kappa": model_kappa,
        "model_lambda": model_lambda,
    }
    fit_fixed_block = {
        "count": len(fit_fixed_rows),
        "rel_rms": _quantiles([entry["rel_rms"] for entry in fit_fixed_rows]),
        "lambda": _quantiles([entry["lambda"] for entry in fit_fixed_rows]),
        "kappa": _quantiles([entry["kappa"] for entry in fit_fixed_rows]),
        "kappa_over_model": _quantiles([entry["kappa"] / model_kappa for entry in fit_fixed_rows])
        if model_kappa else None,
        "nonpositive_kappa": sum(1 for entry in fit_fixed_rows if entry["kappa"] <= 0.0),
    }

    return {
        "events": len(rows),
        "steps": {"min": int(min(steps)), "median": int(np.median(steps)),
                  "max": int(max(steps))},
        "budget": {
            "closure_max_abs": _max_median([entry["closure_max_abs"] for entry in summary]),
            "closure_max_abs_detrended": _max_median(
                [entry["closure_max_abs_detrended"] for entry in summary]),
            "closure_rms": _max_median([entry["closure_rms"] for entry in summary]),
            "initial_offset_abs": _max_median(
                [abs(entry["delta_omega_coi_first"]) for entry in summary]),
            "delta_omega_coi_last_abs": _max_median(
                [abs(entry["delta_omega_coi_last"]) for entry in summary]),
            "r1_est_rms": _max_median([entry["r1_est_rms"] for entry in summary]),
            "terminal_share": shares,
            "mean_abs_share": abs_shares,
            "mean_abs": {name: _quantiles(mean_abs[:, index].tolist())
                         for index, name in enumerate(term_names)},
            "cancellation_index": _quantiles(
                [entry["cancellation_index"] for entry in summary
                 if entry["cancellation_index"] is not None]),
            "pooled_cancellation_index": (
                float(np.abs(pooled_terminal).sum() / abs(pooled_terminal_sum))
                if abs(pooled_terminal_sum) > SHARE_ZERO_TOL else None),
            "r_dominant_fraction": (dominant_r / len(rows)) if rows else None,
            "pooled_terminal": pooled_terminal.tolist(),
            "pooled_terminal_sum": pooled_terminal_sum,
            "pooled_mean_abs": pooled_mean_abs.tolist(),
            "share_definitions": {
                "terminal_share": "I_x(T) / (I_r+I_D+I_p)(T) for x=r,D,p; terminal-value share",
                "mean_abs_share": "mean_t|f_x(t)| / sum_y mean_t|f_y(t)|; mean-absolute share",
                "pooled_terminal": "sum over events of I_x(T) / sum over events of the total",
                "pooled_mean_abs": "sum over events and times of |f_x| / the same sum over terms",
            },
        },
        "closure": {
            "forward_rel_rms": _quantiles([entry["forward_rel_rms"] for entry in closure]),
            "oracle_r10_rel_rms": _quantiles([entry["oracle_r10_rel_rms"] for entry in closure]),
            "init_component_rel": _quantiles([entry["init_component_rel"] for entry in closure]),
            "w1_rel_diff": _quantiles([entry["w1_rel_diff"] for entry in closure]),
            "w1_component_rel": _quantiles([entry["w1_component_rel"] for entry in closure]),
            "amplification": _quantiles([entry["amplification"] for entry in closure
                                         if entry["amplification"] is not None]),
            "w1_variance_explained": _quantiles(
                [entry["w1_variance_explained"] for entry in closure
                 if entry["w1_variance_explained"] is not None]),
            "w1_projection_share": _quantiles([entry["w1_projection_share"] for entry in closure
                                               if entry["w1_projection_share"] is not None]),
            "init_projection_share": _quantiles(
                [entry["init_projection_share"] for entry in closure
                 if entry["init_projection_share"] is not None]),
            "self_consistency_rel_rms": _max_median(
                [entry["self_consistency_rel_rms"] for entry in closure]),
            "r1_model_rel_diff": _quantiles([entry["r1_model_rel_diff"] for entry in closure]),
            "r1_model_rms": _max_median([entry["r1_model_rms"] for entry in closure]),
            "model_kappa": model_kappa,
            "model_lambda": model_lambda,
            "fit": fit_block,
            "fit_fixed_r10": fit_fixed_block,
        },
        "cross_check": {
            "closure_max_abs": _max_median([entry["closure_max_abs"] for entry in cross]),
            "residual_sum_rms": _max_median([entry["residual_sum_rms"] for entry in cross]),
            "inversion_rms": _max_median([entry["inversion_rms"] for entry in cross]),
            "relative_to_r1_est": _quantiles(
                [entry["closure_max_abs"] / row["closure"]["r1_est_rms"]
                 for entry, row in zip(cross, rows) if row["closure"]["r1_est_rms"] > 0]),
        },
    }


# ------------------------------------------------------------------------- the run


def _event_row(event: dict, scope: str, fine_row: np.ndarray, buffers: dict,
               kappa: float, lam: float) -> dict:
    """All P2 quantities for one model/event/scope window."""
    steps = window_steps(event, scope)
    if steps is None:
        raise ValueError("window unavailable")
    truth = truth_window(event, steps)
    predicted = prediction_window(fine_row, steps)
    if predicted.shape != (steps, 40):
        raise ValueError(f"prediction window shape {predicted.shape} != ({steps}, 40)")
    omega_true = truth[:, 10:20]
    p_true = truth[:, 20:30]
    omega_pred = predicted[:, 10:20]
    p_pred = predicted[:, 20:30]
    r_pred = predicted[:, 30:40]
    mass = buffers["mass"]
    damping = buffers["damping"]
    q_p = buffers["q_p"]

    budget = budget_terms(omega_true, p_true, omega_pred, p_pred, r_pred,
                          mass, damping, q_p, detail=False)
    closure = closure_terms(omega_true, p_true, omega_pred, p_pred, r_pred,
                            budget["sum_r_est"], kappa, lam)
    cross = cross_check_terms(omega_pred, p_pred, r_pred, mass, damping, q_p)
    summary = budget["summary"]
    return {
        "event_id": event["id"],
        "label": event["label"],
        "scope": scope,
        "steps": steps,
        "summary": summary,
        "closure": closure,
        "cross_check": cross,
    }


def process_model(name: str, events: list[dict], model, buffers: dict,
                  max_events: int | None = None) -> list[dict]:
    """Stream one frozen prediction directory and measure every registered window."""
    directory = Path(MODEL_SPECS[name]["predictions"])
    files = sorted(directory.glob("predictions_*.npz"))
    if len(files) != PREDICTION_CHUNKS:
        raise ValueError(f"{directory}: expected {PREDICTION_CHUNKS} prediction chunks, "
                         f"got {len(files)}")
    by_id = {event["id"]: event for event in events}
    kappa = float(model.kappa)
    lam = float(model.lambda_)
    rows, seen, chunk_identity = [], set(), []
    for path in files:
        with np.load(path, allow_pickle=False) as data:
            required = {"event_ids", "histories", "fine", "coarse"}
            if set(data.files) != required:
                raise ValueError(f"{path}: keys {sorted(data.files)}")
            ids = [str(value) for value in data["event_ids"]]
            fine = np.asarray(data["fine"])
            histories = np.asarray(data["histories"])
            coarse = np.asarray(data["coarse"])
            count = len(ids)
            if fine.shape != (count, FINE_TIME_POINTS, 40) or coarse.shape != \
                    (count, COARSE_TIME_POINTS, 40) or histories.shape != \
                    (count, HISTORY_ROWS, 30):
                raise ValueError(f"{path}: unexpected shapes")
            if fine.dtype != np.float64 or coarse.dtype != np.float64:
                raise ValueError(f"{path}: predictions must be float64")
            for index, event_id in enumerate(ids):
                if event_id not in by_id or event_id in seen:
                    raise ValueError(f"{path}: unknown or duplicate event id {event_id}")
                seen.add(event_id)
                if not np.array_equal(histories[index],
                                      np.asarray(by_id[event_id]["x"])[:HISTORY_ROWS]):
                    raise ValueError(f"{path}: causal history/offset differs for {event_id}")
                event = by_id[event_id]
                if not np.isfinite(fine[index]).all():
                    raise ValueError(f"{path}: non-finite prediction for {event_id}")
                scopes = [scope for scope in ("5", "8")] if event["label"] == "stable" \
                    else ["prefix"]
                for scope in scopes:
                    if window_steps(event, scope) is None:
                        continue
                    rows.append(_event_row(event, scope, fine[index], buffers, kappa, lam))
                if max_events is not None and len(rows) >= max_events:
                    chunk_identity.append({"path": path.relative_to(ROOT).as_posix(),
                                           "sha256": sha256_file(path)})
                    return rows, chunk_identity
        chunk_identity.append({"path": path.relative_to(ROOT).as_posix(),
                               "sha256": sha256_file(path)})
    if seen != set(by_id):
        raise ValueError(f"{name}: prediction event set differs from fit297")
    return rows, chunk_identity


def paired_model_gap(left_rows: list[dict], right_rows: list[dict],
                     left: str, right: str) -> dict:
    """Per-event paired difference ``left - right`` on the same event/scope windows."""
    index = {(row["event_id"], row["scope"]): row for row in left_rows}
    getters = {
        "forward_rel_rms": lambda row: row["closure"]["forward_rel_rms"],
        "oracle_r10_rel_rms": lambda row: row["closure"]["oracle_r10_rel_rms"],
        "w1_rel_diff": lambda row: row["closure"]["w1_rel_diff"],
        "w1_component_rel": lambda row: row["closure"]["w1_component_rel"],
        "amplification": lambda row: row["closure"]["amplification"],
        "w1_variance_explained": lambda row: row["closure"]["w1_variance_explained"],
        "r1_model_rel_diff": lambda row: row["closure"]["r1_model_rel_diff"],
        "mean_abs_share_r": lambda row: row["summary"]["mean_abs_share"][0],
        "closure_max_abs": lambda row: row["summary"]["closure_max_abs"],
    }
    result = {"left": left, "right": right,
              "definition": "paired per-event differences (left - right) on the same windows"}
    for scope in SCOPES:
        pairs = [(index[(row["event_id"], scope)], row) for row in right_rows
                 if row["scope"] == scope and (row["event_id"], scope) in index]
        block = {"pairs": len(pairs)}
        for name, getter in getters.items():
            deltas = []
            for left_row, right_row in pairs:
                try:
                    left_value, right_value = getter(left_row), getter(right_row)
                except (KeyError, TypeError, IndexError):
                    continue
                if left_value is None or right_value is None:
                    continue
                deltas.append(float(left_value) - float(right_value))
            if deltas:
                block[name] = {
                    **_quantiles(deltas),
                    "positive_fraction": sum(1 for value in deltas if value > 0) / len(deltas),
                    "count": len(deltas),
                }
        result[scope] = block
    return result


def run_probe(models: tuple[str, ...] = MODEL_ORDER, max_events: int | None = None) -> dict:
    """Execute the read-only P2 probe and return the private payload."""
    started = time.perf_counter()
    timings = {}
    tick = time.perf_counter()
    record, events = load_frozen_events()
    timings["load_events_seconds"] = time.perf_counter() - tick

    tick = time.perf_counter()
    loaded, identities, buffers = {}, {}, {}
    for name in models:
        model = load_model(name)
        loaded[name] = model
        identities[name] = model_identity(name, model)
        buffers[name] = mechanical_buffers(model)
    timings["load_models_seconds"] = time.perf_counter() - tick

    identity_check = check_mechanical_identity(identities, canonical=models[0])

    tick = time.perf_counter()
    variants: dict[str, dict] = {}
    for name in models:
        signature = identities[name]["signature"]
        if signature in variants:
            continue
        body = buffers[name]
        variants[signature] = {
            "models": [key for key in models if identities[key]["signature"] == signature],
            "uniformity": uniformity_block(body),
            "projection": [uniformity_projection(events, scope, body) for scope in SCOPES],
        }
    timings["uniformity_seconds"] = time.perf_counter() - tick

    results, per_event = {}, {}
    model_blocks = {}
    for name in models:
        tick = time.perf_counter()
        rows, chunk_identity = process_model(name, events, loaded[name], buffers[name],
                                             max_events=max_events)
        timings[f"process_{name}_seconds"] = time.perf_counter() - tick
        per_event[name] = rows
        model_blocks[name] = {
            "identity": identities[name],
            "prediction_chunks": chunk_identity,
            "scopes": {},
        }
        for scope in SCOPES:
            subset = [row for row in rows if row["scope"] == scope]
            expected = EXPECTED_STABLE if scope in SCOPE_STEPS else EXPECTED_UNSTABLE
            if max_events is None and len(subset) != expected:
                raise ValueError(f"{name}/{scope}: {len(subset)} events != {expected}")
            if subset:
                model_blocks[name]["scopes"][scope] = aggregate_scope(
                    subset, identities[name]["kappa"], identities[name]["lambda_"])
        results[name] = model_blocks[name]["scopes"]

    denominators = {scope: sum(1 for event in cohort(events, scope)) for scope in SCOPES}
    if max_events is None and (denominators["5"] != EXPECTED_STABLE
                               or denominators["8"] != EXPECTED_STABLE
                               or denominators["prefix"] != EXPECTED_UNSTABLE):
        raise ValueError(f"registered denominators changed: {denominators}")

    headline_gap = None
    if "A" in models and "R_N" in models and max_events is None:
        headline_gap = paired_model_gap(per_event["A"], per_event["R_N"], "A", "R_N")

    acceptance = {
        "denominators": denominators,
        "denominator_expected": {"5": EXPECTED_STABLE, "8": EXPECTED_STABLE,
                                 "prefix": EXPECTED_UNSTABLE},
        "budget_closure_scale": {
            name: {scope: {
                "closure_max_abs": block["budget"]["closure_max_abs"],
                "closure_max_abs_detrended": block["budget"]["closure_max_abs_detrended"],
                "initial_offset_abs": block["budget"]["initial_offset_abs"],
            } for scope, block in results[name].items() if "budget" in block}
            for name in models
        },
        "scalar_closure_self_consistency": {
            name: {scope: block["closure"]["self_consistency_rel_rms"]
                   for scope, block in results[name].items()}
            for name in models
        },
        "model_trajectory_cross_check": {
            name: {scope: block["cross_check"] for scope, block in results[name].items()}
            for name in models
        },
    }

    payload = {
        "schema": SCHEMA,
        "status": "COMPLETED",
        "created_unix": time.time(),
        "root": str(ROOT),
        "command": "run",
        "optimizer_calls": 0,
        "model_rollouts": 0,
        "splits": {"val_read": False, "test_read": False},
        "claim_scope": CLAIM_SCOPE,
        "method": {
            "mechanical_equation": "M*omega_dot = -grad U - D*omega + q_p*p - r; 1^T grad U = 0",
            "inversion": "1^T r_est = -1^T(M*omega_dot_true) - 1^T(D*omega_true) + 1^T(q_p*p_true)",
            "omega_dot": "np.gradient(omega_true, 0.01, axis=0, edge_order=2)",
            "r1_definition": "R1 = 1^T r / 10; W1 = 1^T omega / 10 (unweighted collective mean)",
            "coi_definition": "dw_COI = sum_i M_i*(omega_pred_i-omega_true_i) / sum_i M_i (model-M weighted)",
            "budget_integral": "cumulative trapezoid, dt=0.01, from the first compared sample",
            "budget_identity": "dw_COI(t) = (1/sum M) int_0^t [-1^T dr - 1^T(D*dw) + 1^T(q_p*dp)] ds",
            "budget_closure": "max|dw_COI - (I_r+I_D+I_p)/sum M|; also after removing dw_COI(0)",
            "scalar_closure": "dR1/dt = kappa W1 - lambda R1; exact identity of 1^T Q = lambda 1^T",
            "scalar_integration": "piecewise-linear-exact: W1 linear between 0.01 s samples; no RK4 truncation",
            "scalar_fit": f"fixed lambda grid geomspace(1e-3,10,{len(LAMBDA_GRID)}); "
                          "(R10,kappa) linear least squares per lambda",
            "scalar_fit_short_window": f"fit skipped below {MIN_FIT_STEPS} samples "
                                       "(3 free parameters would be degenerate); short "
                                       "windows remain an optimistic in-sample lower bound "
                                       "and nonpositive fitted kappa is counted, not clipped",
            "prediction_alignment": "fine[::2][1:steps+1] vs x[11:11+steps] (verified against histories)",
            "cross_check": "same inversion/integration on the model's own (omega,p,r) path",
        },
        "frozen_preparation": {
            "path": SGM_STUDY.relative_to(ROOT).as_posix(),
            "prepare_sha256": sha256_file(SGM_STUDY / "prepare.json"),
            "schedule_sha256": sha256_file(SGM_STUDY / "schedule.json"),
            "event_count": len(events),
            "stable": sum(event["label"] == "stable" for event in events),
            "unstable": sum(event["label"] == "unstable" for event in events),
            "val_read": record["val_read"],
            "test_read": record["test_read"],
            "already_consumed": True,
        },
        "mechanical_identity": identity_check,
        "mechanical_variants": variants,
        "cohorts": {
            scope: {
                "label": COHORT_LABEL[scope],
                "events": denominators[scope],
                "definition": SCOPE_DEFINITION[scope],
            } for scope in SCOPES
        },
        "models": results,
        "headline_gap": headline_gap,
        "model_blocks": model_blocks,
        "per_event": per_event if max_events is None else {},
        "acceptance": acceptance,
        "timings": timings,
        "seconds": time.perf_counter() - started,
        "code": {
            "experiments/coi_budget_20261030.py": sha256_file(Path(__file__)),
        },
    }
    return payload


# ------------------------------------------------------------------------- publish


PUBLIC_SCOPES = ("5", "8", "prefix")


def build_public(budget: dict, budget_path: Path) -> dict:
    """Aggregate-only public projection of the private budget."""
    if budget.get("schema") != SCHEMA or budget.get("status") != "COMPLETED":
        raise ValueError("private budget is not a completed coi_budget payload")
    denominators = budget["acceptance"]["denominators"]
    if denominators != {"5": EXPECTED_STABLE, "8": EXPECTED_STABLE,
                        "prefix": EXPECTED_UNSTABLE}:
        raise ValueError(f"unexpected registered denominators: {denominators}")
    models = {}
    for name in MODEL_ORDER:
        if name not in budget["models"]:
            continue
        block = budget["models"][name]
        models[name] = {
            "kappa": budget["model_blocks"][name]["identity"]["kappa"],
            "lambda_": budget["model_blocks"][name]["identity"]["lambda_"],
            "variant": budget["model_blocks"][name]["identity"]["variant"],
            "export": budget["model_blocks"][name]["identity"]["export"],
            "export_sha256": budget["model_blocks"][name]["identity"]["export_sha256"],
            "scopes": {scope: block[scope] for scope in PUBLIC_SCOPES if scope in block},
        }
    return {
        "schema": PUBLIC_SCHEMA,
        "status": "COMPLETED",
        "created_unix": time.time(),
        "probe": "P2 COI-driven budget + scalar-closure sufficiency (read-only, no training)",
        "private_budget": {
            "path": _relative_or_absolute(budget_path),
            "sha256": sha256_file(budget_path),
            "schema": budget["schema"],
        },
        "code": budget["code"],
        "frozen_preparation": budget["frozen_preparation"],
        "method": budget["method"],
        "mechanical_identity": {
            "canonical": budget["mechanical_identity"]["canonical"],
            "fields": budget["mechanical_identity"]["fields"],
            "bit_identical_to_canonical": budget["mechanical_identity"]["bit_identical_to_canonical"],
            "different_fields": budget["mechanical_identity"]["different_fields"],
            "shared_signature": budget["mechanical_identity"]["shared_signature"],
            "substitution": budget["mechanical_identity"]["substitution"],
        },
        "mechanical_variants": budget["mechanical_variants"],
        "cohorts": budget["cohorts"],
        "models": models,
        "headline_gap": budget.get("headline_gap"),
        "acceptance": budget["acceptance"],
        "timings": {
            "process_seconds": {key: value for key, value in budget["timings"].items()},
            "total_seconds": budget["seconds"],
        },
        "limitations": [
            "Frozen fit297 train evidence only; already consumed, not independent confirmation.",
            "R1_est is a mechanical-inversion estimate under the frozen M/D/q_p, not the true hidden force.",
            "The COI budget is closed by construction up to the initial COI offset and finite-difference/trapezoid error.",
            "The three terminal-value shares can be ill-conditioned because the terms cancel; mean-absolute shares are reported alongside.",
            "A per-event best (kappa,lambda,R10) fit is an optimistic in-sample bound on a single global closure; short unstable windows are excluded below 20 samples and nonpositive fitted kappa is counted, not clipped.",
            "Unstable-prefix windows are short (min 3, median 26 samples), so prefix fit/amplification ratios are numerically fragile and are not evidence about closure form.",
            "No training, no optimizer call, no rollout, no val/test, no model promotion, no mechanism identification.",
        ],
        "splits": {"val_read": False, "test_read": False},
    }


# ------------------------------------------------------------------------------ CLI


def _parse_models(value: str) -> tuple[str, ...]:
    names = tuple(part.strip() for part in value.split(",") if part.strip())
    if not names or any(name not in MODEL_SPECS for name in names):
        raise ValueError(f"--models must be a non-empty subset of {MODEL_ORDER}")
    return names


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="coi_budget_20261030",
        description="Read-only COI-driven budget + scalar-closure sufficiency probe (P2).",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    run_parser = subparsers.add_parser("run", help="measure the frozen predictions")
    run_parser.add_argument("--output", required=True,
                            help="private JSON path under private_data/coi_budget_20261030")
    run_parser.add_argument("--models", default=",".join(MODEL_ORDER),
                            help="comma-separated subset of A,S,P,R_N (default: all)")
    run_parser.add_argument("--max-events", type=int, default=None,
                            help="debug-only cap on measured windows; disables full validation")
    publish_parser = subparsers.add_parser("publish", help="write the aggregate public payload")
    publish_parser.add_argument("--input", required=True,
                                help="private budget JSON under private_data/coi_budget_20261030")
    publish_parser.add_argument("--output", required=True,
                                help="public JSON path under results/coi_budget_20261030")
    args = parser.parse_args(argv)

    if args.command == "run":
        output = guard_output(args.output, PRIVATE_ROOT)
        payload = run_probe(_parse_models(args.models), max_events=args.max_events)
        write_json(output, payload)
        print(json.dumps({
            "status": payload["status"],
            "output": str(output),
            "seconds": round(payload["seconds"], 3),
            "denominators": payload["acceptance"]["denominators"],
            "models": list(payload["models"]),
        }, ensure_ascii=False))
        return 0

    source = guard_input(args.input, PRIVATE_ROOT)
    output = guard_output(args.output, RESULTS_ROOT)
    budget = read_json(source)
    public = build_public(budget, source)
    write_json(output, public)
    print(json.dumps({
        "status": public["status"],
        "output": str(output),
        "private_budget_sha256": public["private_budget"]["sha256"],
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
