"""Weak elimination of hidden force for fixed A/C ROMs (preparation only).

No event reader, training loop, data authorization or checkpoint selection lives here.
Callers supply already-authorized train delta/omega/p observations. The weak least
squares objective is NOT the autonomous J. X and y share observation error; this
module makes no unbiasedness, noise robustness or true-structure claim.
"""
from __future__ import annotations

from dataclasses import dataclass
import argparse
import importlib.metadata
import json
import math
import os
from pathlib import Path
import time
import numpy as np

PAIRS = tuple((i, j) for i in range(10) for j in range(i + 1, 10))
MAXITER = 4700
SCIPY_VERSION = '1.18.0'


def incidence():
    B = np.zeros((10, 45), dtype=np.float64)
    for e, (i, j) in enumerate(PAIRS):
        B[i, e], B[j, e] = 1., -1.
    return B


@dataclass(frozen=True)
class Skeleton:
    mass: np.ndarray
    damping: np.ndarray
    G: np.ndarray
    Sp: np.ndarray
    rpp: np.ndarray
    coupling: np.ndarray
    phi: np.ndarray
    scale_omega: np.ndarray

    def __post_init__(self):
        for name in self.__dataclass_fields__:
            a = np.array(getattr(self, name), dtype=np.float64, copy=True)
            expected = (45,) if name in ('coupling', 'phi') else (10,)
            if a.shape != expected or not np.isfinite(a).all():
                raise ValueError('invalid skeleton ' + name)
            if name != 'phi' and (a < 0).any():
                raise ValueError('negative skeleton ' + name)
            if name in ('mass', 'G', 'Sp', 'scale_omega') and (a <= 0).any():
                raise ValueError('nonpositive skeleton ' + name)
            a.setflags(write=False)
            object.__setattr__(self, name, a)

    @classmethod
    def from_model(cls, model):
        """Extract only fixed buffers from an independently authorized model."""
        def value(name):
            return getattr(model, name).detach().cpu().numpy()
        return cls(*(value(n) for n in ('mass', 'damping', 'jpi_p', 'storage_base',
                   'rpp', 'reference_coupling45', 'phi', 'input_scale_omega')))

    @property
    def references(self):
        # T_ref=1 s; kappa_ref=m_ref/T_ref**2, lambda_ref=alpha_ref=1/T_ref.
        mref = np.exp(np.log(self.mass).mean())
        return np.r_[mref, np.ones(46)], mref * self.scale_omega

    def force(self, x):
        d, w, p = np.split(x, 3, axis=-1)
        B = incidence()
        potential = ((np.sin(d @ B + self.phi) - np.sin(self.phi)) * self.coupling) @ B.T
        return -potential - self.damping * w + self.G * self.Sp * p


@dataclass(frozen=True)
class Observations:
    key: str
    unstable: bool
    x: np.ndarray  # exactly [delta, omega, p]; no r/RHS/torque accepted

    def __post_init__(self):
        a = np.asarray(self.x, dtype=np.float64)
        if a.ndim != 2 or a.shape[1] != 30 or not np.isfinite(a).all():
            raise ValueError('finite delta/omega/p rows of width 30 required')
        object.__setattr__(self, 'x', a)


def fixed_windows(npoints):
    """Inclusive node endpoints; interval ranges are [start,end)."""
    N = min(800, max(0, npoints - 11))
    end = 10 + N
    windows, start = [], 10
    while end - start >= 100:
        windows.append((start, start + 100)); start += 100
    tail = 4 * ((end - start) // 4)
    if tail >= 20:
        windows.append((start, start + tail)); start += tail
    return windows, {'eligible_intervals': N, 'eligible_start': 10,
                     'eligible_end': end, 'unused_interval_ranges':
                     [[start, end]] if start < end else [],
                     'reason': None if windows else 'fewer_than_20_intervals'}


def test_functions(nintervals):
    if nintervals < 20 or nintervals % 4:
        raise ValueError('window requires >=20 intervals divisible by four')
    s = np.arange(nintervals + 1, dtype=np.float64) / nintervals
    T = nintervals * .01
    # Polynomial coefficients in ascending order; derivatives are analytic.
    values = []
    for c in ([0., 0., 1., -2., 1.], [0., 0., -1., 4., -5., 2.]):
        p = np.polynomial.polynomial
        values.append((p.polyval(s, c), p.polyval(s, p.polyder(c)) / T,
                       p.polyval(s, p.polyder(c, 2)) / T**2))
    return values


def simpson(values, stride=1):
    if stride not in (1, 2):
        raise ValueError('only original or same-endpoint decimated grid')
    a = values[::stride]
    if len(a) < 3 or (len(a) - 1) % 2:
        raise ValueError('composite Simpson requires an even number of intervals')
    return (.01 * stride / 3) * (a[0] + a[-1] + 4*a[1:-1:2].sum(axis=0)
                                      + 2*a[2:-1:2].sum(axis=0))


def weak_window(x, skeleton, family, stride=1):
    """Physical X,y. r=f-M*omega_dot is eliminated, never differentiated."""
    if family not in ('A', 'C'):
        raise ValueError('only A/C admit this linear weak problem')
    if x.ndim != 2 or x.shape[1] != 30 or not np.isfinite(x).all():
        raise ValueError('finite width-30 observations required')
    B = incidence()
    d, w, _ = np.split(x, 3, axis=-1)
    f, v = skeleton.force(x), skeleton.mass * w
    eta = d @ B
    gate = np.ones_like(eta) if family == 'A' else (1 + np.cos(eta))/2
    gd = np.zeros_like(eta) if family == 'A' else -.5*np.sin(eta)*(w @ B)
    # e actions are b_e b_e^T M omega, NOT M b_e b_e^T omega.
    Lf = (f @ B)[:, None, :] * B[None, :, :]
    Lv = (v @ B)[:, None, :] * B[None, :, :]
    rows, targets = [], []
    for psi, dp, ddp in test_functions(len(x)-1):
        y = -simpson(dp[:, None]*f + ddp[:, None]*v, stride)
        k = simpson(psi[:, None]*w, stride)
        lam = -simpson(psi[:, None]*f + dp[:, None]*v, stride)
        edge = -simpson(psi[:, None, None]*gate[:, None, :]*Lf
                       + (dp[:, None, None]*gate[:, None, :]
                          + psi[:, None, None]*gd[:, None, :])*Lv, stride)
        rows.append(np.column_stack((k, lam, edge))); targets.append(y)
    return np.concatenate(rows), np.concatenate(targets)


@dataclass
class WeakMatrices:
    A: np.ndarray
    b: np.ndarray
    A2: np.ndarray
    b2: np.ndarray
    unstable_rows: np.ndarray
    coverage: list


def build_matrices(events, skeleton, family='A'):
    """Retain no-window events in coverage; never fabricate their zero rows."""
    events = list(events)
    if len({e.key for e in events}) != len(events):
        raise ValueError('event keys must be unique')
    specs = [fixed_windows(len(e.x)) for e in events]
    count = sum(bool(w) for w, _ in specs)
    refs, force_scale = skeleton.references
    arrays = [[], [], [], []]; labels, coverage = [], []
    for e, (windows, cov) in zip(events, specs):
        used_nodes = set()
        for a, b in windows:
            used_nodes.update(range(a, b+1))
        coverage.append(dict(key=e.key, unstable=bool(e.unstable), **cov,
                             windows=[list(w) for w in windows],
                             unique_used_nodes=len(used_nodes),
                             used_intervals=sum(b-a for a,b in windows),
                             used_duration_s=.01*sum(b-a for a,b in windows)))
        if not windows:
            continue
        weight = math.sqrt(1 / (count * len(windows) * 2 * 10))
        scale = np.tile(force_scale, 2)
        for a, b in windows:
            for j, stride in enumerate((1, 2)):
                X, y = weak_window(e.x[a:b+1], skeleton, family, stride)
                arrays[2*j].append(X / scale[:, None] * refs[None, :] * weight)
                arrays[2*j+1].append(y / scale * weight)
            labels.extend([e.unstable]*20)
    joined = [np.concatenate(a) if a else np.empty((0,47) if j%2 == 0 else (0,))
              for j,a in enumerate(arrays)]
    if not all(np.isfinite(a).all() for a in joined):
        raise ValueError('nonfinite weak matrix')
    return WeakMatrices(*joined, np.asarray(labels, dtype=bool), coverage)


def sensitivity(matrices):
    m = matrices
    result = {}
    for name, mask in [('all', np.ones(len(m.b), bool)), ('stable', ~m.unstable_rows),
                       ('unstable', m.unstable_rows)]:
        if not mask.any():
            result[name] = dict(rows=0, E_X=None, E_y=None, passed=None, information='no_weak_rows')
            continue
        A, b = m.A[mask], m.b[mask]
        ex = float(np.linalg.norm(A-m.A2[mask]) / max(1., np.linalg.norm(A)))
        ey = float(np.linalg.norm(b-m.b2[mask]) / max(1., np.linalg.norm(b)))
        result[name] = dict(rows=int(mask.sum()), E_X=ex, E_y=ey, passed=ex<=1e-3 and ey<=1e-3)
    result['passed'] = bool(len(m.b)) and all(r['passed'] is not False for r in result.values())
    return result


def kkt_report(A, b, z):
    finite = bool(np.isfinite(z).all())
    nonnegative = finite and bool((z >= 0).all())
    if not finite:
        return dict(finite=False, nonnegative=False, residual=None, passed=False)
    residual = A @ z - b
    g = A.T @ residual
    af = np.linalg.norm(A)
    S = 1 + af*(af*np.linalg.norm(z) + np.linalg.norm(b))
    value = float(np.max(np.abs(np.minimum(z, g/S))))
    ok = np.isfinite(S) and np.isfinite(g).all() and np.isfinite(value)
    return dict(finite=finite, nonnegative=nonnegative, residual=value,
                residual_l2=float(np.linalg.norm(residual)), scale=float(S),
                passed=bool(nonnegative and ok and value <= 1e-10))


def raw_coordinates(z):
    """Only mapping eligibility; original numerical/causal gates still required."""
    z = np.asarray(z, dtype=np.float64)
    if z.shape != (47,) or not np.isfinite(z).all() or (z<0).any():
        raise ValueError('invalid nonnegative weak solution')
    if z[0] <= 0 or z[1] <= 0:
        raise ValueError('kappa/lambda boundary: preserve weak solution, refuse conversion')
    raw = np.log(z[:2])
    if not np.isfinite(raw).all() or not np.isfinite(np.exp(raw)).all() or (np.exp(raw)<=0).any():
        raise ValueError('invalid exponential mapping')
    return dict(kappa_raw=float(raw[0]), lambda_raw=float(raw[1]), a_bar=z[2:].tolist())


def solve_once(matrices, *, nnls_event=None):
    """Exactly one public scipy.optimize.nnls call, only on the primary matrix.

    Public SciPy returns (x,rnorm), no actual iteration count or internal active
    set. Zero-coordinate indices are explicitly inferred from returned x.
    Caller must freeze the reported runtime version/API before authorizing use.
    """
    from scipy.optimize import nnls
    if importlib.metadata.version('scipy') != SCIPY_VERSION:
        raise RuntimeError('frozen scipy version required: ' + SCIPY_VERSION)
    check = sensitivity(matrices)
    if not check['passed']:
        return dict(success=False, reason='quadrature_or_no_coverage', sensitivity=check)
    A, b = matrices.A, matrices.b
    sv = np.linalg.svd(A, compute_uv=False)  # exactly one rank decomposition
    threshold = float(sv[0]*max(A.shape)*np.finfo(np.float64).eps)
    rank = int((sv>threshold).sum())
    report = dict(sensitivity=check, scipy_version=importlib.metadata.version('scipy'),
                  solver='scipy.optimize.nnls', maxiter=MAXITER,
                  iterations=dict(available=False, value=None,
                                  reason='public API does not return actual iterations'),
                  singular_values=sv.tolist(), rank=rank, rank_threshold=threshold,
                  condition=float(sv[0]/sv[-1]) if sv[-1]>0 else None,
                  parameter_identity='full_numerical_column_rank' if rank==47 else 'nonunique_weak_start_unresolved')
    if nnls_event is not None:
        nnls_event('attempt', dict(maxiter=MAXITER, scipy_version=SCIPY_VERSION))
    try:
        z, rnorm = nnls(A, b, maxiter=MAXITER)
    except BaseException as exc:
        if nnls_event is not None:
            nnls_event('failed', dict(type=type(exc).__name__, message=str(exc)))
        if not isinstance(exc, (RuntimeError, ValueError, np.linalg.LinAlgError)):
            raise
        return dict(report, success=False, reason=type(exc).__name__, message=str(exc))
    if nnls_event is not None:
        nnls_event('completed', dict(solver_rnorm=float(rnorm)))
    kkt = kkt_report(A,b,z)
    report.update(z=z.tolist(), solver_success=True, solver_rnorm=float(rnorm),
                  inferred_zero_coordinates=np.flatnonzero(z==0).tolist(),
                  active_set_source='exact zeros inferred from public returned vector', kkt=kkt)
    try:
        raw = raw_coordinates(z)
    except ValueError as exc:
        return dict(report, success=False, reason=str(exc), conversion_eligible=False)
    qualified = kkt['passed'] and bool(np.isfinite(rnorm))
    return dict(report, success=qualified, conversion_eligible=qualified,
                raw_coordinates=raw, original_numerical_qualification='still_required')


def fixture_spec():
    """Frozen artificial ROM; never derives parameters from real assets."""
    i = np.arange(10, dtype=np.float64)
    s = Skeleton(1+i/10, .1+i/100, 1+i/30, 1+i/20, .2+i/100,
                 np.full(45,.1), np.zeros(45), np.ones(10))
    theta = np.r_[1., .5, .01*(1+np.arange(45)/45)]
    initials = []
    for j in range(10):
        x = np.zeros(40)
        x[:10] = -.15/10; x[j] += .15
        x[10+(j+1)%10] = .1; x[20+(j+2)%10] = .05; x[30+(j+3)%10] = .1
        initials.append(x)
    return s, theta, initials


def fixture_rhs(skeleton, theta, family):
    B = incidence()
    def rhs(t, state):
        d,w,p,r = np.split(state,4)
        gate = np.ones(45) if family=='A' else (1+np.cos(d @ B))/2
        rdot = theta[0]*w-theta[1]*r-((r @ B)*theta[2:]*gate) @ B.T
        return np.r_[w,(skeleton.force(state[:30])-r)/skeleton.mass,
                     -skeleton.G*w-skeleton.rpp*skeleton.Sp*p,rdot]
    return rhs


def causal_fixture_model(skeleton, theta, family):
    # Reuse the original estimator; it alone reconstructs past p from terminal p.
    import torch
    from state_gated_memory_model_20261028 import StateGatedMemoryROM
    s = skeleton
    cfg = dict(mass=s.mass.tolist(), damping=s.damping.tolist(), jpi_p=s.G.tolist(),
               storage_base=s.Sp.tolist(), rpp=s.rpp.tolist(), D=incidence().T.tolist(),
               coupling=s.coupling.tolist(), delta_star=[0.]*10,
               input_scale_delta=[1.]*10,input_scale_omega=[1.]*10,
               p_scale=[1.]*10,gain=[1.]*10,rate=[1.]*10)
    model = StateGatedMemoryROM(cfg,family,history_dt=.0025)
    refs,_ = s.references
    raw = raw_coordinates(theta/refs)
    with torch.no_grad():
        model.kappa_raw.copy_(torch.tensor(raw['kappa_raw'],dtype=torch.float64))
        model.lambda_raw.copy_(torch.tensor(raw['lambda_raw'],dtype=torch.float64))
        model.a_bar.copy_(torch.tensor(raw['a_bar'],dtype=torch.float64))
    return model


def json_safe(value):
    if isinstance(value, float) and not math.isfinite(value):
        return str(value)
    if isinstance(value, dict):
        return {k:json_safe(v) for k,v in value.items()}
    if isinstance(value, list):
        return [json_safe(v) for v in value]
    return value


def write_new_json(path, value):
    with Path(path).open('x',encoding='utf-8') as stream:
        json.dump(json_safe(value),stream,ensure_ascii=False,indent=2,allow_nan=False)
        stream.write('\n'); stream.flush(); os.fsync(stream.fileno())


def qualify_fixtures(output_directory, *, repair_of=None, nnls_event=None):
    """Explicit opt-in command AFTER root protocol freeze. No real data access.

    Four fixed NNLS calls total: one nonstationary and one stationary per family,
    as explicitly frozen by the root protocol. No alternate solves or searches.
    """
    directory = Path(output_directory)
    directory.mkdir(parents=True,exist_ok=False)
    prior_attempts = 0
    if repair_of is not None:
        original = Path(repair_of)
        manifest = json.loads((original/'manifest.json').read_text(encoding='utf-8'))
        if manifest.get('repair_of') is not None:
            raise ValueError('only one complete fixture repair is registered')
        prior_attempts = sum(json.loads(line).get('event')=='nnls_attempt'
            for line in (original/'journal.jsonl').read_text(encoding='utf-8').splitlines())
        if prior_attempts > 4:
            raise ValueError('original fixture ledger exceeds four attempts')
        # Exclusive claim prevents two repair directories from resetting the ledger.
        # The caller/root, never this function, classifies a genuine implementation bug.
        write_new_json(original/'repair_claim.json',dict(output_directory=str(directory.resolve())))
    write_new_json(directory/'manifest.json',dict(repair_of=None if repair_of is None else str(Path(repair_of).resolve()),
        normal_suite_nnls_limit=4,total_artificial_nnls_limit=8,prior_attempts=prior_attempts,
        scipy_version=SCIPY_VERSION,causal_prediction='StateGatedMemoryROM.rollout RK4 .01 history .0025',
        scientific_authorization='external_root_required'))
    stages = {}
    for family in ('A','C'):
        names = ([f'{family}_reference_{j}' for j in range(10)]
                 + [f'{family}_matrices',f'{family}_nonstationary_nnls',f'{family}_stationary_nnls']
                 + [f'{family}_{kind}_prediction_{j}' for kind in ('true','weak') for j in range(10)])
        stages.update({name:'not_run' for name in names})
    output = dict(passed=False,stages=stages,artificial_nnls_attempts=0,
                  prior_attempts=prior_attempts,completed_nnls=0,failed_nnls=0)
    def journal(event, **detail):
        with (directory/'journal.jsonl').open('a',encoding='utf-8') as stream:
            stream.write(json.dumps(json_safe(dict(event=event,time_unix=time.time(),**detail)),allow_nan=False)+'\n')
            stream.flush(); os.fsync(stream.fileno())
    def execute(name, action):
        stages[name]='started'; journal('stage_started',stage=name)
        try:
            value=action()
        except BaseException as exc:
            stages[name]='failed'; journal('stage_failed',stage=name,type=type(exc).__name__,message=str(exc))
            raise
        stages[name]='completed'; journal('stage_completed',stage=name)
        return value
    def solve(name, matrices):
        def event(status, detail):
            if status=='attempt':
                if output['artificial_nnls_attempts']>=4 or prior_attempts+output['artificial_nnls_attempts']>=8:
                    raise RuntimeError('fixed artificial NNLS allowance exhausted')
                output['artificial_nnls_attempts']+=1
            elif status=='completed': output['completed_nnls']+=1
            elif status=='failed': output['failed_nnls']+=1
            journal('nnls_'+status,stage=name,**detail)
            if nnls_event is not None: nnls_event(status,dict(stage=name,**detail))
        result=execute(name,lambda:solve_once(matrices,nnls_event=event))
        write_new_json(directory/(name+'.json'),result)
        if 'solver_success' not in result and 'singular_values' not in result:
            journal('nnls_not_run',stage=name,reason=result.get('reason'))
        return result
    def save_arrays(name, **arrays):
        with (directory/(name+'.npz')).open('xb') as stream:
            np.savez(stream,**arrays); stream.flush(); os.fsync(stream.fileno())
    journal('suite_started',prior_attempts=prior_attempts)
    try:
        from scipy.integrate import solve_ivp
        import torch
        s,theta,initials=fixture_spec(); refs,rstar=s.references
        times=np.arange(411,dtype=np.float64)*.01
        for family in ('A','C'):
            result=dict(passed=False,reconstruction=[]); output[family]=result
            rhs=fixture_rhs(s,theta,family); trajectories=[]
            for j,initial in enumerate(initials):
                name=f'{family}_reference_{j}'
                def reference(initial=initial,name=name):
                    sol=solve_ivp(rhs,(0.,4.1),initial,method='DOP853',t_eval=times,rtol=1e-12,atol=1e-13)
                    save_arrays(name,time=sol.t,state=sol.y.T)
                    write_new_json(directory/(name+'.json'),dict(success=bool(sol.success),message=sol.message))
                    if not sol.success or sol.y.shape!=(40,411) or not np.isfinite(sol.y).all():
                        raise RuntimeError('fixed fixture reference integration failed')
                    return sol.y.T
                trajectories.append(execute(name,reference))
            events=[Observations(str(j),False,x[:211,:30]) for j,x in enumerate(trajectories)]
            def matrices_action():
                m=build_matrices(events,s,family)
                save_arrays(f'{family}_matrices',A=m.A,b=m.b,A2=m.A2,b2=m.b2)
                write_new_json(directory/f'{family}_coverage.json',m.coverage)
                return m
            matrices=execute(f'{family}_matrices',matrices_action)
            fit=solve(f'{family}_nonstationary_nnls',matrices); result['fit']=fit
            stationary=build_matrices([Observations('equilibrium',False,np.zeros((211,30)))],s,family)
            stationary_fit=solve(f'{family}_stationary_nnls',stationary)
            equilibrium=dict(X_zero=bool((stationary.A==0).all()),y_zero=bool((stationary.b==0).all()),
                             all_parameters_unidentifiable=True,fit=stationary_fit,conversion_eligible=False)
            result['equilibrium']=equilibrium
            write_new_json(directory/f'{family}_equilibrium.json',equilibrium)
            if 'z' in fit:
                z=np.asarray(fit['z'])
                error=float(np.max(np.abs(z-theta/refs)/np.maximum(1,np.abs(theta/refs))))
                result.update(coefficient_error=error,coefficient_pass=error<=1e-3,
                    conditional_recovery_resolved=fit['rank']==47 and error<=1e-3 and fit['success'])
            predictions=[('true',theta)]
            if fit['success']: predictions.append(('weak',np.asarray(fit['z'])*refs))
            for kind,param in predictions:
                model=causal_fixture_model(s,param,family)
                for j,x in enumerate(trajectories):
                    name=f'{family}_{kind}_prediction_{j}'
                    def prediction(x=x,j=j,name=name,kind=kind):
                        history=np.array(x[200:211,:30],copy=True); history[:-1,20:]=np.nan
                        with torch.no_grad():
                            path=model.rollout(torch.tensor(history,dtype=torch.float64),200,.01,
                                               method='rk4',history_dt=.0025).detach().cpu().numpy()
                        save_arrays(name,time=times[210:],state=path)
                        if path.shape!=(201,40) or not np.isfinite(path).all():
                            raise RuntimeError('fixed causal RK4 prediction failed')
                        rerr=float(np.max(np.abs(path[0,30:]-x[210,30:])/rstar))
                        err=np.max(np.abs(path[:,:30]-x[210:,:30]),axis=0)
                        record=dict(parameters=kind,initial=j,r0_error=rerr,r0_pass=rerr<=1e-2,
                                    component_max_errors=err.tolist(),trajectory_pass=bool((err<=2e-2).all()))
                        write_new_json(directory/(name+'.json'),record)
                        return record
                    result['reconstruction'].append(execute(name,prediction))
            result['passed']=bool(fit['success'] and result.get('conditional_recovery_resolved',False)
                and equilibrium['X_zero'] and equilibrium['y_zero']
                and stationary_fit.get('kkt',{}).get('passed',False)
                and stationary_fit.get('solver_rnorm')==0.
                and not stationary_fit.get('conversion_eligible',True)
                and len(result['reconstruction'])==20
                and all(r['r0_pass'] and r['trajectory_pass'] for r in result['reconstruction']))
            write_new_json(directory/(family+'_result.json'),result)
            if not result['passed']:
                journal('qualification_failed',family=family)
                break
        output['passed']=all(output.get(f,{}).get('passed',False) for f in ('A','C'))
    except BaseException as exc:
        output['exception']=dict(type=type(exc).__name__,message=str(exc))
        journal('suite_exception',**output['exception'])
        if not isinstance(exc,Exception):
            write_new_json(directory/'result.json',output)
            raise
    journal('suite_completed',passed=output['passed'],attempts=output['artificial_nnls_attempts'])
    write_new_json(directory/'result.json',output)
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--qualify-fixed-fixtures', action='store_true')
    parser.add_argument('--output', type=Path,help='new dedicated directory; existing path is rejected')
    parser.add_argument('--repair-of',type=Path,help='root-classified implementation repair only; one original suite')
    args = parser.parse_args()
    if not args.qualify_fixed_fixtures or args.output is None:
        parser.error('explicit --qualify-fixed-fixtures --output PATH required; root authorization external')
    result=qualify_fixtures(args.output,repair_of=args.repair_of)
    if not result['passed']: raise SystemExit(1)


if __name__ == '__main__':
    main()
