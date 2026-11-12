"""Pure NumPy L1 adapter for the frozen train fit297 Run3 protocol.

Only the source-level pure numerical/classification helpers are reused; importing
this module does not read any event, checkpoint or dataset configuration.
"""
from __future__ import annotations
import math
import numpy as np
from evaluate_structured_rom import _pair_row, _scored_numerics
from frequency_dissipation_20261021 import actual15_qualified
from evaluate_l1_latent import apply_numerical_gate

S_DELTA = .47675314588683515
S_OMEGA = 1.0033667249602924
HORIZONS = (1, 2, 5, 8)
DENOMINATORS = {'all': 297, 'stable': 267, 'unstable': 30, 'initializations': 594}


def risk(values):
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 1 or not values.size or not np.isfinite(values).all():
        return None
    k = math.ceil(.1 * len(values))
    return float(.8 * values.mean() + .2 * np.sort(values)[-k:].mean())


def score_coordinates(path, delta_star):
    path = np.asarray(path, dtype=np.float64)
    if path.ndim != 2 or path.shape[1] != 50:
        raise ValueError('expected a complete 50-state path')
    result = path.copy()
    result[:, :10] -= np.asarray(delta_star, dtype=np.float64).reshape(10)
    return result


def error_metrics(pred, truth, mass, p_scale):
    pred, truth = np.asarray(pred), np.asarray(truth)
    mass, p_scale = np.asarray(mass), np.asarray(p_scale)
    if (pred.ndim != 2 or pred.shape[1] < 30 or pred.shape != truth.shape
            or len(pred) == 0 or mass.shape != (10,) or p_scale.shape != (10,)
            or not np.isfinite(mass).all() or not np.isfinite(p_scale).all()
            or np.any(mass <= 0) or np.any(p_scale <= 0)):
        raise ValueError('invalid aligned metric input or scales')
    if not np.isfinite(pred).all() or not np.isfinite(truth).all():
        raise ValueError('nonfinite metric input')
    e = pred - truth
    weights = mass / mass.sum()
    dc, wc = e[:, :10] @ weights, e[:, 10:20] @ weights
    d = e[:, :10] - dc[:, None]
    w = e[:, 10:20] - wc[:, None]
    p = e[:, 20:30] / p_scale
    q = (.5 * np.mean((d / S_DELTA)**2, axis=1)
         + .5 * np.mean((w / S_OMEGA)**2, axis=1)
         + (wc / S_OMEGA)**2 + np.mean(p*p, axis=1))
    delta_sse, omega_sse = float(np.sum(d*d)), float(np.sum(w*w))
    normalized_sse = .5 * (delta_sse / S_DELTA**2 + omega_sse / S_OMEGA**2)
    common_sse, p_sse = float(wc @ wc), float(np.sum(p*p))
    n = int(pred.shape[0])
    return dict(mean_point_error=float(q.mean()), n=n, component_n=10*n,
        delta_sse=delta_sse, omega_sse=omega_sse, normalized_sse=normalized_sse,
        coi_normalized_rmse=float(np.sqrt(normalized_sse/(10*n))),
        common_omega_sse=common_sse, common_omega_rmse=float(np.sqrt(common_sse/n)),
        normalized_p_sse=p_sse, normalized_p_rmse=float(np.sqrt(p_sse/(10*n))))


def admissible(row):
    return (row.get('finite_to_endpoint') is True and row.get('los_pair_pass') is True
        and all(isinstance(row.get(key), (float, int)) and not isinstance(row[key], bool)
                and np.isfinite(row[key]) and row[key] <= .001
                for key in ('coi_normalized_rmse', 'common_normalized_rmse')))


def numeric_scope(fine, coarse, n, mass):
    if n <= 0 or len(fine) < 2*n+1 or len(coarse) < n+1:
        return dict(available=False, finite=False, numerically_admissible=False)
    finite = bool(np.isfinite(fine[:2*n+1]).all() and np.isfinite(coarse[:n+1]).all())
    row = dict(available=True, finite=finite, finite_to_endpoint=finite, los_pair_pass=True,
               coi_normalized_rmse=None, common_normalized_rmse=None)
    if finite:
        e = fine[::2][:n+1] - coarse[:n+1]
        b = mass / mass.sum()
        common = e[:, 10:20] @ b
        d = e[:, :10] - (e[:, :10] @ b)[:, None]
        w = e[:, 10:20] - common[:, None]
        row.update(coi_normalized_rmse=float(np.sqrt(.5*np.mean((d/S_DELTA)**2+(w/S_OMEGA)**2))),
                   common_normalized_rmse=float(np.sqrt(np.mean((common/S_OMEGA)**2))))
    row['numerically_admissible'] = admissible(row)
    return row


def _classification(pair, event, ok, future_steps):
    raw = dict(pair['fine_classification'])
    c = apply_numerical_gate(raw, ok)
    meta = event['classification_meta']
    hit = c.get('trigger_index')
    record = None if hit is None else float(hit) * .01
    true = meta.get('event_time_s')
    delay = float(meta.get('record_start_delay_s') or 0.)
    clearing = meta.get('clearing_time_s')
    endpoint = min((len(event['x'])-1)*.01, .1+future_steps*.01)
    timing_comparable = (record is not None and true is not None and
                         record <= endpoint+1e-12 and float(true) <= endpoint+1e-12)
    return dict(raw_classification=raw, classification=c, numerically_admissible=bool(ok),
        predicted_los_time_record_s=record,
        predicted_los_time_decision_s=None if record is None else record-.1,
        predicted_los_time_since_clearing_s=None if record is None else record+delay,
        predicted_los_time_absolute_s=None if record is None or clearing is None else record+delay+float(clearing),
        true_event_time_record_s=true,
        timing_comparable=bool(timing_comparable),
        trigger_time_error_s=record-float(true) if timing_comparable else None)


def score_event(event, fine, coarse, *, delta_star, mass, p_scale):
    """Both grids must be the complete 15 s forecasts from one returned point."""
    fine, coarse = score_coordinates(fine, delta_star), score_coordinates(coarse, delta_star)
    if fine.shape != (3001, 50) or coarse.shape != (1501, 50):
        raise ValueError('15 s prediction shape mismatch; short paths cannot be silently scored')
    mass = np.asarray(mass, dtype=np.float64)
    x = np.asarray(event['x'], dtype=np.float64)
    if x.ndim != 2 or x.shape[1] != 30 or len(x) < 11:
        raise ValueError('expected canonical [time,30] event')
    meta = event['classification_meta']
    shift = np.asarray(meta['equilibrium_delta']) - np.asarray(delta_star)
    if shift.shape != (10,) or not np.isfinite(shift).all() or np.max(np.abs(shift-shift.mean())) > 1e-12:
        raise ValueError('classification equilibrium differs beyond the common angle gauge')
    meta = dict(meta, equilibrium_delta=np.asarray(delta_star))
    history = x[:11]
    pair8 = _pair_row(fine[:1601], coarse[:801], history, event, meta, mass)
    # Retain the exact legacy horizon rows, including checks of all 50 states.
    scored = _scored_numerics(fine, coarse, event, mass)
    n = min(1500, len(x)-11)
    pair15 = _pair_row(fine[:2*n+1], coarse[:n+1], history, event, meta, mass)
    real15 = numeric_scope(fine, coarse, n, mass)
    actual_ok = actual15_qualified(real15, pair15)
    scopes = {}
    for h in (*HORIZONS, 'actual15'):
        steps = n if h == 'actual15' else h*100
        available = steps > 0 and len(x)-11 >= steps
        numerical = real15 if h == 'actual15' else next(r for r in scored if r['horizon'] == str(h))
        # Original scientific gate combines pair8 and available scored rows for 5/8 s.
        ok = actual_ok if h == 'actual15' else admissible(numerical)
        if h in (5, 8):
            ok = ok and admissible(pair8) and all(admissible(r) for r in scored if r.get('available'))
        row = dict(available=available, numerically_admissible=bool(ok), numerical=numerical,
                   fine=None, coarse=None, finite=False, score_perturbation=None)
        if available:
            finite = bool(np.isfinite(fine[:2*steps+1]).all() and np.isfinite(coarse[:steps+1]).all()
                          and np.isfinite(x[11:11+steps]).all())
            row['finite'] = finite
            if finite:
                row['fine'] = error_metrics(fine[::2][1:steps+1, :30], x[11:11+steps], mass, p_scale)
                row['coarse'] = error_metrics(coarse[1:steps+1, :30], x[11:11+steps], mass, p_scale)
                row['score_perturbation'] = {key: abs(row['fine'][key]-row['coarse'][key]) for key in
                    ('mean_point_error', 'common_omega_rmse', 'normalized_p_rmse')}
        scopes[str(h)] = row
    ok8 = admissible(pair8) and all(admissible(r) for r in scored if r.get('available'))
    return dict(event_id=str(event['id']), group=event['group'], label=event['label'],
        scopes=scopes, pair8=pair8, pair15=pair15, numerical_scored=scored,
        classification_8s=_classification(pair8, event, ok8, 800),
        classification_actual15=_classification(pair15, event, actual_ok, n), actual_future_points=n,
        failure=None)


def failure_event(event, stage, error):
    return dict(event_id=str(event['id']), group=event['group'], label=event['label'], scopes={},
                failure=dict(stage=stage, type=type(error).__name__, message=str(error)))


def _aggregate(rows, grid):
    values = [r[grid] for r in rows]
    if not values:
        return None
    return dict(risk=risk([v['mean_point_error'] for v in values]),
        common_omega_rmse=float(np.sqrt(sum(v['common_omega_sse'] for v in values)/sum(v['n'] for v in values))),
        normalized_p_rmse=float(np.sqrt(sum(v['normalized_p_sse'] for v in values)/sum(v['component_n'] for v in values))))


def summarize(records):
    """Official summaries preserve the fixed original denominators on any failure."""
    ids = [r['event_id'] for r in records]
    if len(ids) != len(set(ids)):
        raise ValueError('duplicate scoring ID')
    result = dict(denominators=DENOMINATORS.copy(), scopes={}, classifications={})
    for label, scopes in (('stable', ('1','2','5','8')), ('unstable', ('actual15',))):
        cohort = [r for r in records if r['label'] == label]
        for scope in scopes:
            good = [r['scopes'][scope] for r in cohort if scope in r.get('scopes', {})
                    and r['scopes'][scope]['available'] and r['scopes'][scope]['finite']
                    and r['scopes'][scope]['numerically_admissible']]
            unresolved = [r['event_id'] for r in cohort if scope not in r.get('scopes', {})
                or not all(r['scopes'][scope].get(k) for k in ('available','finite','numerically_admissible'))]
            official = len(cohort) == DENOMINATORS[label] and len(good) == DENOMINATORS[label]
            f, c = _aggregate(good, 'fine'), _aggregate(good, 'coarse')
            result['scopes'][label+'_'+scope] = dict(denominator=DENOMINATORS[label], covered=len(good),
                unresolved_ids=unresolved, covered_ids=[r['event_id'] for r in cohort if r['event_id'] not in unresolved], official_fine=f if official else None,
                official_coarse=c if official else None, covered_only_fine=f, covered_only_coarse=c,
                perturbation=None if not official else {k: abs(f[k]-c[k]) for k in f})
    for scope in ('8s','actual15'):
        field='classification_'+scope
        counts = dict(missed_unstable_ids=[], false_los_ids=[], undetermined_ids=[], uncovered_ids=[],
                      failure_ids=[], detected_unstable_ids=[], settled_ids=[], numerical_unresolved_ids=[], unsettled_ids=[])
        for row in records:
            item = row.get(field)
            eid = row['event_id']
            if item is None:
                counts['failure_ids'].append(eid)
                continue
            c = item['classification']
            if c['category'] == 'latency_uncovered':
                counts['uncovered_ids'].append(eid)
                continue
            if not item['numerically_admissible'] or c['category'] == 'numerical_unresolved':
                counts['numerical_unresolved_ids'].append(eid)
                counts['undetermined_ids'].append(eid)
                continue
            if not(c['predicted_los'] or c['window_settled']):
                counts['unsettled_ids'].append(eid)
                counts['undetermined_ids'].append(eid)
            if c['predicted_los']:
                counts['detected_unstable_ids' if row['label']=='unstable' else 'false_los_ids'].append(eid)
            elif row['label']=='unstable': counts['missed_unstable_ids'].append(eid)
            if c['window_settled']: counts['settled_ids'].append(eid)
        counts.update(raw_denominator=297, unstable_denominator=30, stable_denominator=267,
                      missing_record_count=297-len(records))
        result['classifications'][scope]=counts
    return result


def paired_summary(static, modulated):
    """Paired score differences use both models' discretization perturbations."""
    a, b = summarize(static), summarize(modulated)
    if {r['event_id'] for r in static} != {r['event_id'] for r in modulated}:
        raise ValueError('paired model identities differ')
    out = dict(static=a, modulated=b, differences={}, timing={}, common_subset_diagnostic={})
    aa,bb={r['event_id']:r for r in static},{r['event_id']:r for r in modulated}
    for name in a['scopes']:
        label,scope=name.split('_',1)
        ids=sorted(set(a['scopes'][name]['covered_ids']) & set(b['scopes'][name]['covered_ids']))
        out['common_subset_diagnostic'][name]=dict(event_ids=ids, n=len(ids), original_denominator=DENOMINATORS[label],
            static_fine=_aggregate([aa[eid]['scopes'][scope] for eid in ids],'fine'),
            modulated_fine=_aggregate([bb[eid]['scopes'][scope] for eid in ids],'fine'),
            scope='auxiliary common subset; does not replace original denominator')
    for scope in a['scopes']:
        x,y=a['scopes'][scope],b['scopes'][scope]
        if x['official_fine'] is None or y['official_fine'] is None:
            out['differences'][scope]=None
            continue
        out['differences'][scope]={key: dict(static_minus_modulated=x['official_fine'][key]-y['official_fine'][key],
            perturbation_sum=x['perturbation'][key]+y['perturbation'][key],
            distinguishable=abs(x['official_fine'][key]-y['official_fine'][key])>x['perturbation'][key]+y['perturbation'][key])
            for key in x['official_fine']}
    for scope in ('8s','actual15'):
        aa,bb={r['event_id']:r for r in static},{r['event_id']:r for r in modulated}
        pairs=[]
        for eid in sorted(aa):
            x,y=aa[eid].get('classification_'+scope),bb[eid].get('classification_'+scope)
            if x and y and x['numerically_admissible'] and y['numerically_admissible']:
                tx,ty=x['trigger_time_error_s'],y['trigger_time_error_s']
                if tx is not None and ty is not None:
                    pairs.append(dict(event_id=eid, static_error_s=tx, modulated_error_s=ty,
                                      static_minus_modulated_s=tx-ty))
        out['timing'][scope]=dict(common_comparable_ids=[r['event_id'] for r in pairs], pairs=pairs)
    return out
