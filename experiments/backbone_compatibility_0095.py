"""Pure physical-path basis, Gram, and score arithmetic for Loop 0095."""
from __future__ import annotations
from typing import Any
import numpy as np

SOMEGA = 1.0033667249602924
HORIZONS = (1.0, 2.0, 5.0)


def _arr(name, x):
    a = np.asarray(x)
    if not np.issubdtype(a.dtype, np.number): raise TypeError(f"{name} numeric")
    return a


def known_basis(truthprefix, mass, D, phi, dt):
    """Return cumulative per-unit k/d integrals, sampled on the .01 grid.

    Each .01 physical interval uses Simpson with a linearly interpolated
    midpoint; ``dt`` selects the registered fine (.005) or coarse (.01)
    quadrature convention while the returned samples remain common-grid.
    """
    x = _arr("truthprefix", truthprefix); mass = _arr("mass", mass); D = _arr("D", D); phi = _arr("phi", phi)
    if x.ndim != 2 or x.shape[1] != 30 or len(x)<2 or mass.shape != (10,) or D.shape != (26,10) or phi.shape != (26,): raise ValueError("basis shapes")
    if dt not in (.005, .01): raise ValueError("basis dt")
    if not all(np.isfinite(a).all() for a in (x, mass, D, phi)): raise FloatingPointError("nonfinite basis inputs")
    if np.any(mass<=0): raise ValueError("positive mass required")
    def rate_at(frac):
        xx = x[:-1] * (1. - frac) + x[1:] * frac
        delta = xx[:, :10] @ D.T
        sk = np.sin(phi[None, :] + delta) - np.sin(phi[None, :])
        bk = -(sk[:, :, None] * D[None, :, :]).transpose(0, 2, 1) / mass[None, :, None]
        bd = np.zeros((len(xx), 10, 10)); bd[:, np.arange(10), np.arange(10)] = -xx[:, 10:20] / mass[None, :]
        return np.concatenate((bk, bd), axis=2)
    out = np.zeros((len(x), 10, 36), dtype=float)
    left, mid, right = rate_at(0.), rate_at(.5), rate_at(1.)
    if dt == .005:
        quarter, threequarter = rate_at(.25), rate_at(.75)
        increments = (.005 / 6.) * (left + 4.*quarter + 2.*mid + 4.*threequarter + right)
    else:
        increments = (.01 / 6.) * (left + 4.*mid + right)
    out[1:] = np.cumsum(increments, axis=0)
    if not np.isfinite(out).all(): raise FloatingPointError("nonfinite basis")
    return out


def _W(r, mass):
    mw = mass[None, :, None]
    coi = r - (r * mw).sum(-2, keepdims=True) / mass.sum()
    common = (r * mw).sum(-2, keepdims=True) / mass.sum()
    return np.concatenate((np.sqrt(.5 / 10.) * coi / SOMEGA, common / SOMEGA), axis=-2)


def _features(packed, truth, basis, indices, truth_indices, mass, scales):
    z = packed[indices]; td = truth[truth_indices, 10:20] - truth[:1, 10:20]
    r0 = z[:, 3:13] - td
    B = basis[truth_indices]
    wb = _W(B, mass) * np.asarray(scales)[None, None, :]
    wr = _W(r0[:, :, None], mass)
    return np.concatenate((wb, wr), axis=2)


def build_grams(data94: dict[str, Any], scales, *, deadline=None, progress=None, partial_path=None) -> dict[str, Any]:
    """Build both fixed-grid Grams; require the original 0094 increment anchors."""
    from time import perf_counter
    fine,coarse,truth = (np.asarray(data94[k]) for k in ("fine","coarse","truth"))
    lengths,mass,D,phi,dk,dd = (np.asarray(data94[k]) for k in
                              ("lengths","mass","D","phi","delta_k","delta_d"))
    scales=np.asarray(scales)
    if (fine.shape!=(346,1001,33) or coarse.shape!=(346,501,33)
            or truth.shape!=(346,501,30) or lengths.shape!=(346,)
            or scales.shape!=(36,) or dk.shape!=(26,) or dd.shape!=(10,)):
        raise ValueError("0095 input shapes")
    if (not np.isfinite(lengths).all() or np.any(lengths<1) or np.any(lengths>500)
            or not np.equal(lengths,lengths.astype(int)).all()):
        raise ValueError("genuine lengths")
    if not all(np.isfinite(a).all() for a in (scales,dk,dd)) or np.any(scales<=0):
        raise ValueError("scales/increments")
    grams=np.full((2,3,346,37,37),np.nan)
    diff=np.full((3,346,37,37),np.nan)
    increment_error=0.; completed=0
    try:
        for i,n0 in enumerate(lengths):
            n=int(n0)
            if deadline is not None and perf_counter()>deadline: raise TimeoutError("Gram deadline")
            if progress is not None: progress.update(stage="Grams",event=i,completed_events=completed)
            for a,end in ((fine[i],2*n+1),(coarse[i],n+1),(truth[i],n+1)):
                if not np.isfinite(a[:end]).all() or not np.isnan(a[end:]).all():
                    raise FloatingPointError("genuine prefix / NaN padding")
            bfs=(known_basis(truth[i,:n+1],mass,D,phi,.005),
                 known_basis(truth[i,:n+1],mass,D,phi,.01))
            sampled=(fine[i,:2*n+1:2],coarse[i,:n+1])
            features=[]
            for B,z in zip(bfs,sampled):
                for predicted,reference in ((B[:,:,:26]@dk,z[:,13:23]),
                                            (B[:,:,26:]@dd,z[:,23:33])):
                    err=float(np.max(np.abs(predicted-reference))/max(
                        1.,float(np.max(np.abs(predicted))),float(np.max(np.abs(reference)))))
                    increment_error=max(increment_error,err)
                    if err>1e-10: raise AssertionError("0094 known-increment identity")
                r0=z[1:,3:13]-(truth[i,1:n+1,10:20]-truth[i,0,10:20])
                features.append(np.concatenate((_W(B[1:],mass)*scales[None,None,:],
                                                _W(r0[:,:,None],mass)),axis=2))
            for hi,h in enumerate(HORIZONS):
                end=min(int(100*h),n)
                for gi,F in enumerate(features):
                    flat=F[:end].reshape(-1,37)
                    grams[gi,hi,i]=flat.T@flat/end
                delta=(features[0][:end]-features[1][:end]).reshape(-1,37)
                diff[hi,i]=delta.T@delta/end
            completed=i+1
    except Exception:
        if partial_path is not None:
            np.savez_compressed(partial_path,grams=grams,difference_grams=diff,
                                completed_events=completed)
        raise
    grams=.5*(grams+grams.swapaxes(-1,-2)); diff=.5*(diff+diff.swapaxes(-1,-2))
    if not np.isfinite(grams).all() or not np.isfinite(diff).all():
        raise FloatingPointError("nonfinite Grams")
    return {"grams":grams,"difference_grams":diff,
            "anchors":{"increment_reconstruction_max_relative":increment_error}}


def factors(grams_fine, *, deadline=None, progress=None, counters=None):
    from time import perf_counter
    G = _arr("grams_fine", grams_fine)
    if G.shape != (3, 346, 37, 37): raise ValueError("factor shape")
    if not np.isfinite(G).all(): raise FloatingPointError("nonfinite Gram")
    F = np.zeros_like(G); eigmins = np.empty((3, 346)); errors = np.empty((3, 346))
    for h in range(3):
        for i in range(346):
            if deadline is not None and perf_counter()>deadline: raise TimeoutError("factor deadline")
            if progress is not None: progress.update(stage="fine factors",horizon=h,event=i)
            if counters is not None: counters['eighs']=counters.get('eighs',0)+1
            A = .5 * (G[h, i] + G[h, i].T); val, vec = np.linalg.eigh(A); scale = max(1., float(np.max(np.abs(val))))
            if val.min() < -1e-10 * scale: raise FloatingPointError("Gram not PSD")
            eigmins[h, i] = val.min(); F[h, i] = np.diag(np.sqrt(np.maximum(val, 0.))) @ vec.T
            errors[h, i] = np.linalg.norm(F[h, i].T @ F[h, i] - A, "fro") / max(1., np.linalg.norm(A, "fro"))
            if not np.isfinite(errors[h,i]) or errors[h,i]>1e-10: raise FloatingPointError("factor reconstruction")
    return F, eigmins, errors


def score(grams, v):
    G = _arr("grams", grams); v = _arr("v", v)
    if G.shape != (2, 3, 346, 37, 37) or v.shape != (36,): raise ValueError("score shapes")
    if not np.isfinite(G).all() or not np.isfinite(v).all(): raise FloatingPointError("nonfinite score inputs")
    u = np.r_[v, 1.]; return np.einsum("...ij,j,i->...", G, u, u)


def risk_summary(scores, labels):
    S = _arr("scores", scores); labels = np.asarray(labels).astype(str)
    if S.shape != (2, 3, 346) or labels.shape != (346,): raise ValueError("summary shapes")
    if not np.isfinite(S).all(): raise FloatingPointError("nonfinite scores")
    if np.sum(labels=='stable')!=311 or np.sum(labels=='unstable')!=35: raise ValueError("label cohorts")
    result = {"grids": {}}
    for gi, g in enumerate(("fine", "coarse")):
        result["grids"][g] = {}
        for hi, h in enumerate(HORIZONS):
            x = S[gi, hi]; tail = np.sort(x)[-35:]; sev = x[labels == "unstable"]; wt = np.sort(sev)[-4:]
            result["grids"][g][str(int(h))] = {"mean": float(x.mean()), "worst35": float(tail.mean()), "risk": float(.8*x.mean()+.2*tail.mean()), "severe_mean": float(sev.mean()), "severe_worst4": float(wt.mean()), "tail_indices": np.argsort(x,kind='stable')[-35:].tolist(), "severe_worst4_indices": np.flatnonzero(labels == "unstable")[np.argsort(sev,kind='stable')[-4:]].tolist()}
        gs=result['grids'][g]
        gs['average_risk']=float(np.mean([gs[str(int(h))]['risk'] for h in HORIZONS]))
        gs['severe_mean']=gs['5']['severe_mean']; gs['severe_worst4']=gs['5']['severe_worst4']
    return result
