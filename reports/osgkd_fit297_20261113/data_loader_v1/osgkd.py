"""Motion-only data packets and train-only calibration for the OSGKD pilot."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

RUN = "osgkd_fit297_20261113"
NODE_COUNT = 10
DT = .01
PORTS = ("theta", "omega", "tau_m", "tau_e")
ROOT = Path(__file__).resolve().parents[3]


def canonical_json(value):
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n").encode()


def sha_bytes(raw):
    return hashlib.sha256(raw).hexdigest()


def group_order(events):
    groups = sorted({e["group"] for e in events},
                    key=lambda group: hashlib.sha256(f"{RUN}|{group}".encode()).digest())
    if len(groups) < 2 or not all(isinstance(g, str) and g for g in groups):
        raise ValueError("two or more verified fault groups required")
    return groups


def split_events(events):
    order = group_order(events)
    check_groups = set(order[::5])
    return ([e for e in events if e["group"] not in check_groups],
            [e for e in events if e["group"] in check_groups])


def motion_packet(event, constants_id):
    history = event["history"]
    return {"history": {key: history[key].copy() for key in ("t", "theta", "omega")},
            "constants_id": constants_id}


def event_record(event_id, group, label, t, theta, omega, mechanical_input, tau_m, tau_e, metadata, mass):
    """Separate the legal history from fitting/scoring-only arrays."""
    if label not in ("stable", "unstable"):
        raise ValueError("invalid recorded label")
    t = np.asarray(t, dtype=np.float64)
    if (t.ndim != 1 or t.size < 11 or not np.isfinite(t).all() or abs(t[0]) > 1e-9
            or not np.allclose(np.diff(t), DT, atol=1e-9, rtol=0)):
        raise ValueError("an explicit original 11-point uniform record is required")
    raw = {k: np.asarray(v, dtype=np.float64) for k,v in zip(
        (*PORTS,"mechanical_input"), (theta,omega,tau_m,tau_e,mechanical_input), strict=True)}
    if any(value.shape != (t.size,NODE_COUNT) for value in raw.values()):
        raise ValueError("invalid ten-machine array shapes")
    if not all(np.isfinite(raw[key]).all() for key in ("theta","omega")):
        raise ValueError("nonfinite original motion record")
    # One constant angular gauge is shared by the complete history and future.
    mass = np.asarray(mass, dtype=np.float64)
    if mass.shape != (NODE_COUNT,) or not np.all(np.isfinite(mass) & (mass > 0)):
        raise ValueError("ten positive physical masses required")
    angular_offset = float(raw["theta"][0] @ (mass / mass.sum()))
    raw["theta"] = raw["theta"] - angular_offset
    mask = {key: np.isfinite(value) for key,value in raw.items()}
    return dict(id=event_id,group=group,label=label,
                history={"t": np.linspace(-.1,0.,11), "theta": raw["theta"][:11].copy(),
                         "omega": raw["omega"][:11].copy()},
                t=(t[10:]-t[10]).copy(),
                targets={key: raw[key][10:].copy() for key in PORTS},
                masks={key: mask[key][10:].copy() for key in PORTS},
                metadata={**metadata,"angle_gauge_offset":angular_offset},
                calibration={key: raw[key][t <= 2.1+1e-9].copy() for key in
                             ("omega","mechanical_input","tau_m","tau_e")})


def calibrate(fit_events):
    scales = {}
    counts = {}
    for source, name, minimum in (
        ("omega","omega_scale",.01), ("mechanical_input","Sp",.001),
        ("tau_m","S_m",.001), ("tau_e","Se",.001),
    ):
        means = []
        for event in fit_events:
            x = event["calibration"][source]
            valid = np.isfinite(x)
            denominator = np.count_nonzero(valid,axis=0)
            squared = np.sum(np.where(valid,x,0.)**2,axis=0)
            means.append(np.divide(squared,denominator,out=np.full(NODE_COUNT,np.nan),where=denominator>0))
        means = np.asarray(means)
        denominator = np.count_nonzero(np.isfinite(means),axis=0)
        if np.any(denominator == 0):
            raise ValueError(f"missing fit-only calibration for {source}")
        scales[name] = np.maximum(np.sqrt(np.nansum(means,axis=0)/denominator),minimum)
        counts[name] = denominator.tolist()
    scales["s_coi"] = float(np.sqrt(np.mean(scales["omega_scale"]**2)))
    scales["k_ref"] = float(np.median(scales["Se"]))
    scales["theta_scale"] = 1.
    return scales,counts


def balance_window(t, omega, tau_m, tau_e, mass, damping, start, end):
    # Use the two predefined trapezoidal grids on the same recorded endpoints.
    indices = np.flatnonzero((t >= start-1e-9)&(t <= end+1e-9))
    if len(indices)<3:
        return dict(status="not_applicable")
    if not all(np.isfinite(v[indices]).all() for v in (omega,tau_m,tau_e)):
        return dict(status="missing")
    coarse = indices[::2]
    if coarse[-1] != indices[-1]:
        coarse = np.r_[coarse, indices[-1]]
    duration = t[indices[-1]] - t[indices[0]]
    normalizer = duration*np.maximum(np.maximum(np.sqrt(np.mean(tau_m[indices]**2,axis=0)),
                                                np.sqrt(np.mean(tau_e[indices]**2,axis=0))),.001)
    impulse = mass*(omega[indices[-1]]-omega[indices[0]])
    force = tau_m-tau_e-damping*omega
    fine, rough = ((impulse-np.trapezoid(force[i],t[i],axis=0))/normalizer for i in (indices,coarse))
    rms = lambda x: float(np.sqrt(np.mean(x*x)))
    gap = rms(fine-rough)
    fine_rms = rms(fine)
    return dict(status="passed" if fine_rms <= .05 and gap <= .02 else "failed",
                points=len(indices), duration_s=float(duration), fine_rms=fine_rms,
                coarse_rms=rms(rough), mesh_difference_rms=gap)


def _bound_record(ref, path):
    if ref["path"] != path.relative_to(ROOT).as_posix():
        raise ValueError("unexpected audit path")
    raw = path.read_bytes()
    if sha_bytes(raw) != ref["sha256"] or len(raw) != ref["bytes"]:
        raise ValueError("audit identity differs")
    return json.loads(raw)


def load_fit297(audit_ref):
    """Load the separately released fit297 bundle, without opening source stores."""
    results = ROOT/"results"/RUN
    release = _bound_record(audit_ref, results/"port_audit_release.json")
    if release.get("run") != RUN or release.get("status") != "passed":
        raise ValueError("port audit release did not pass")
    original_ref, recheck_ref, ref = (release[key] for key in ("original_audit","recheck","bundle"))
    original = _bound_record(original_ref, results/"port_audit.json")
    recheck = _bound_record(recheck_ref, results/"port_audit_recheck.json")
    if (original.get("run") != RUN or original.get("status") != "blocked"
            or recheck.get("run") != RUN or recheck.get("status") != "passed"
            or recheck.get("original_audit") != original_ref or recheck.get("bundle") != ref
            or recheck.get("ordered_ids") != original.get("ordered_ids")
            or recheck.get("fit_ids") != original.get("fit_ids")
            or recheck.get("check_ids") != original.get("check_ids")):
        raise ValueError("audit release does not bind the reviewed original cohort and bundle")
    path = ROOT/"artifacts"/RUN/"port_audit"/"data.npz"
    if ref["path"] != path.relative_to(ROOT).as_posix():
        raise ValueError("unexpected audit bundle path")
    with path.open("rb") as stream:
        if hashlib.file_digest(stream,"sha256").hexdigest() != ref["sha256"] or stream.tell() != ref["bytes"]:
            raise ValueError("audit bundle identity differs")
        stream.seek(0)
        with np.load(stream,allow_pickle=False) as cache:
            constants = json.loads(str(cache["constants_json"]))
            descriptors = json.loads(str(cache["events_json"]))
            if len(descriptors) != 297:
                raise ValueError("fit297 bundle denominator changed")
            events = []
            for index, desc in enumerate(descriptors):
                prefix = f"event_{index:04d}_"
                events.append(dict(**desc,history={key:cache[prefix+"history_"+key] for key in
                                                   ("t","theta","omega")},
                                   t=cache[prefix+"t"],targets={key:cache[prefix+key] for key in PORTS},
                                   masks={key:cache[prefix+"mask_"+key] for key in PORTS}))
    fit, check = split_events(events)
    if ([e["id"] for e in events] != recheck["ordered_ids"]
            or constants["constants_id"] != recheck["constants_id"]
            or [e["id"] for e in fit] != recheck["fit_ids"]
            or [e["id"] for e in check] != recheck["check_ids"]):
        raise ValueError("audit data identities, split or constants differ")
    return events, {**constants,"fit_ids":recheck["fit_ids"],"check_ids":recheck["check_ids"]}
