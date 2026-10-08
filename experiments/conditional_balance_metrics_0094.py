"""Pure preregistered arithmetic for Loop 0094; no data/model readers."""
from __future__ import annotations
import numpy as np

SOMEGA = 1.0033667249602924
HORIZONS = (1, 2, 5)
ARMS = ("base", "k", "d", "full")


def _weighted(r, mass):
    common = (r * mass).sum(-1, keepdims=True) / mass.sum()
    return np.concatenate((np.sqrt(.5 / 10) * (r-common) / SOMEGA,
                           common / SOMEGA), axis=-1)


def _energy(a, b=None):
    """Mean over time of the inner product, not mean over coordinates."""
    return np.sum(a * (a if b is None else b), axis=-1).mean(-1)


def _risk(values):
    mean, tail = float(values.mean()), float(np.sort(values)[-35:].mean())
    return {"mean": mean, "worst35": tail, "risk": .8*mean+.2*tail}


def validate_inputs(fine, coarse, truth, lengths, mass, event_ids, labels):
    fine, coarse, truth, lengths, mass = map(np.asarray, (fine, coarse, truth, lengths, mass))
    if (fine.ndim != 3 or coarse.ndim != 3 or fine.shape[0::2] != (346,33)
            or coarse.shape[0::2] != (346,33) or truth.shape != (346,501,30)):
        raise ValueError("packed/physical shapes")
    if (lengths.shape != (346,) or not np.isfinite(lengths).all()
            or not np.equal(lengths, lengths.astype(int)).all()
            or np.any(lengths < 1) or np.any(lengths > 500)):
        raise ValueError("genuine lengths")
    if mass.shape != (10,) or not np.isfinite(mass).all() or np.any(mass <= 0):
        raise ValueError("mass")
    if np.asarray(event_ids).shape != (346,) or len(set(map(str,event_ids))) != 346:
        raise ValueError("event identity")
    labels = np.asarray(labels).astype(str)
    if labels.shape != (346,) or np.sum(labels=="stable") != 311 or np.sum(labels=="unstable") != 35:
        raise ValueError("label identity")
    for i,n in enumerate(lengths.astype(int)):
        if fine.shape[1] <= 2*n or coarse.shape[1] <= n:
            raise ValueError("grid shorter than genuine prefix")
        if not all(np.isfinite(a).all() for a in
                   (fine[i,:2*n+1],coarse[i,:n+1],truth[i,:n+1])):
            raise FloatingPointError("nonfinite genuine prefix")
    return fine, coarse, truth, lengths.astype(int), mass, labels=="unstable"


def compute_metrics(fine, coarse, truth, lengths, mass, *, event_ids, labels):
    fine,coarse,truth,lengths,mass,severe = validate_inputs(
        fine,coarse,truth,lengths,mass,event_ids,labels)
    scores = np.empty((2,4,3,346))
    terms = np.empty((2,3,346,6))
    numeric_energy = np.empty((4,3,346))
    for i,n in enumerate(lengths):
        omega = truth[i,1:n+1,10:20] - truth[i,0,10:20]
        ws = []
        for gi,packed in enumerate((fine[i,2:2*n+1:2],coarse[i,1:n+1])):
            w0 = _weighted(packed[:,3:13]-omega,mass)
            wk = _weighted(packed[:,13:23],mass)
            wd = _weighted(packed[:,23:33],mass)
            arms = np.stack((w0,w0+wk,w0+wd,w0+wk+wd))
            ws.append(arms)
            for hi,h in enumerate(HORIZONS):
                end = min(100*h,n)
                a,b,c = w0[:end],wk[:end],wd[:end]
                scores[gi,:,hi,i] = _energy(arms[:,:end])
                terms[gi,hi,i] = (_energy(a),_energy(b),_energy(c),
                    2*_energy(a,b),2*_energy(a,c),2*_energy(b,c))
        for hi,h in enumerate(HORIZONS):
            end = min(100*h,n)
            numeric_energy[:,hi,i] = _energy(ws[0][:,:end]-ws[1][:,:end])
    # Five-second prefix contains the whole genuine interval (length <= 500).
    fullprefixrms = np.sqrt(numeric_energy[:,2])
    if not all(np.isfinite(a).all() for a in (scores,terms,numeric_energy)):
        raise FloatingPointError("nonfinite metric arithmetic")
    operand_scale = np.maximum(1.,np.maximum(np.max(np.abs(terms),axis=-1),
                                            np.abs(scores[:,3])))
    term_relative = float(np.max(np.abs(terms.sum(-1)-scores[:,3])/operand_scale))
    numeric_gate = bool(np.max(fullprefixrms) <= .001)
    valid_numeric = numeric_gate and term_relative <= 1e-10
    summary = {"grids":{}, "denominator":{}, "ratios":{},
               "term_identity_max_relative":term_relative,
               "numeric_gate":numeric_gate,
               "max_fullprefixrms":float(np.max(fullprefixrms)),
               "grid_support":{}}
    noise_avg = float(np.mean([_risk(numeric_energy[0,h])["risk"] for h in range(3)]))
    noise_severe = float(numeric_energy[0,2,severe].mean())
    noise_worst = float(np.sort(numeric_energy[0,2,severe])[-4:].mean())
    for gi,gname in enumerate(("fine","coarse")):
        gs = {}
        for ai,arm in enumerate(ARMS):
            desc = {str(h):_risk(scores[gi,ai,hi]) for hi,h in enumerate(HORIZONS)}
            desc["average_risk"] = float(np.mean([desc[str(h)]["risk"] for h in HORIZONS]))
            sev = scores[gi,ai,2,severe]
            desc["severe_mean"] = float(sev.mean())
            desc["severe_worst4"] = float(np.sort(sev)[-4:].mean())
            gs[arm] = desc
        # Fixed cohorts only, never a decomposition of arm-specific sorted tails.
        gs["fixed_terms_all"] = terms[gi].mean(axis=1).tolist()
        gs["fixed_terms_severe"] = terms[gi][:,severe].mean(axis=1).tolist()
        summary["grids"][gname] = gs
        denominator = {}
        for key,metric,noise in (("average","average_risk",noise_avg),
                                ("severe","severe_mean",noise_severe),
                                ("worst4","severe_worst4",noise_worst)):
            value = gs["base"][metric]
            denominator[key] = {"value":value,"numeric":noise,
                                "passes":bool(value>1e-12 and value>100*noise)}
        denominator["primary_passes"] = all(denominator[k]["passes"] for k in ("average","severe"))
        summary["denominator"][gname] = denominator
        ratios = {}
        for key,metric,ratio_key in (
                ("average","average_risk","average_full_over_base"),
                ("severe","severe_mean","severe_full_over_base"),
                ("worst4","severe_worst4","worst4_full_over_base")):
            ratios[ratio_key] = (gs["full"][metric]/gs["base"][metric]
                                if valid_numeric and denominator[key]["passes"] else None)
        summary["ratios"][gname] = ratios
        summary["grid_support"][gname] = (
            bool(ratios["average_full_over_base"] >= 1.25 and
                 ratios["severe_full_over_base"] >= 1.25)
            if valid_numeric and denominator["primary_passes"] else None)
    support = list(summary["grid_support"].values())
    summary["grid_classification"] = (
        "inconclusive" if None in support or support[0] != support[1]
        else "supported" if all(support) else "rejected")
    summary["primary_joint_gate"] = summary["grid_classification"] == "supported"
    return dict(summary=summary,scores=scores,terms=terms,
                numeric_energy=numeric_energy,fullprefixrms=fullprefixrms)
