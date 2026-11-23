"""Frozen electrical decomposition: saved states only, no training or integration."""
from __future__ import annotations
import argparse
from collections import Counter
from contextlib import ExitStack
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys
from time import perf_counter, process_time

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT)]
import numpy as np
import torch
from experiments.osgkd_v2_early_closure_20261123.run import identity, verified, read, write, require, repo_path
from experiments.osgkd_v2_p0_20261123.run import spectrum, validate_rows, distribution
from kiki.models.osgkd import OSGKD
from kiki.models.osgkd_anchored_potential import OSGKDAnchoredPotential

RUN = 'osgkd_v2_electrical_decomposition_20261123'
ANALYSIS = 'results/osgkd_v2_p1b_electrical_unlock_20261123/analysis.json'
AUDIT = 'results/osgkd_fit297_20261113/port_audit_release.json'
CONFLICT = 'results/osgkd_v2_conflict_profile_20261123/probe01/summary.json'
PROTOCOL = 'campaigns/osgkd_v2/electrical_decomposition_protocol.md'
ALIASES = {'parent_1860':'parent', 'control_2020':'control', 'candidate_2020':'candidate'}
WINDOWS = {'0.03':3, 'capped_0.25':25}
SOURCES = {f'experiments/{RUN}/run.py', PROTOCOL, ANALYSIS,
 'experiments/osgkd_v2_early_closure_20261123/run.py', 'experiments/osgkd_v2_p0_20261123/run.py',
 'src/kiki/models/osgkd.py', 'src/kiki/models/osgkd_anchored_potential.py', 'src/kiki/data/osgkd.py',
 'src/kiki/evaluation/gftd_analysis.py'}
EPS = 1e-12


def validate(release):
    require(set(release) == {'run_id','job_id','protocol','sources','inputs','port_audit','conflict_summary','status'},
            'exact release keys required')
    require((release['run_id'],release['job_id'],release['status']) == (RUN,'probe01','released'), 'released probe01 required')
    require(release['protocol']['path'] == PROTOCOL, 'canonical protocol required')
    verified(release['protocol'])
    require(SOURCES <= release['sources'].keys(), 'source coverage incomplete')
    for name,digest in release['sources'].items():
        require(identity(repo_path(name))['sha256'] == digest, 'source differs: '+name)
    require(release['port_audit']['path'] == AUDIT and release['conflict_summary']['path'] == CONFLICT, 'canonical audit/conflict required')
    verified(release['port_audit'])
    analysis = read(identity(repo_path(ANALYSIS)))
    job = read(analysis['job'])
    require(job['status'] == 'completed' and job['result']['fixed_endpoint'] == 2020 and
            not job['result']['stopped_early'], 'completed fixed P1b analysis required')
    require(set(release['inputs']) == set(ALIASES), 'exact endpoint aliases required')
    for alias,name in ALIASES.items():
        refs = release['inputs'][alias]
        require(set(refs) == {'checkpoint','rows','predictions','summary'} and refs == analysis['inputs'][name], 'analysis binding differs')
        for ref in refs.values(): verified(ref)
    conflict = read(release['conflict_summary'])
    require(conflict['status'] == 'completed' and conflict['release']['input_refs']['audit'] == release['port_audit'], 'conflict audit differs')
    require(job['release']['input_refs']['audit'] == release['port_audit'], 'P1b audit differs')
    return read(conflict['result']['rows_ref'])


def decompose(d, se, mass, selected=None):
    """Time means of pointwise quantities; aggregation subsequently gives each event one vote."""
    require(d.ndim == 2 and d.shape[1] == 10 and len(d) > 0 and np.isfinite(d).all(), 'complete ten-node errors required')
    net = np.repeat(d.mean(axis=1, keepdims=True),10,axis=1)
    zero = d-net
    u, un, uz = d/se, net/se, zero/se
    irreducible = se[None,:]*d.sum(axis=1,keepdims=True)/np.sum(se**2)
    removable = u-irreducible
    require(np.max(abs(np.sum(removable*irreducible,axis=1))) <= EPS*max(1,float(np.max(np.sum(u*u,axis=1)))), 'irreducible normalized orthogonality failed')
    accel = -d/mass
    common = np.repeat((accel@mass/mass.sum())[:,None],10,axis=1)
    relative = accel-common
    require(np.max(abs(net+zero-d)) <= EPS and np.max(abs(zero.sum(axis=1))) <= EPS, 'physical decomposition identity failed')
    energy = lambda x: float(np.mean(np.sum(x*x,axis=1)))
    out = {'physical_total':energy(d), 'physical_net':energy(net), 'physical_zero':energy(zero),
           'normalized_total':energy(u), 'normalized_net':energy(un), 'normalized_zero':energy(uz),
           'normalized_cross':float(2*np.mean(np.sum(un*uz,axis=1))),
           'projection_net':float(np.mean(np.sum(u*un,axis=1))),
           'projection_zero':float(np.mean(np.sum(u*uz,axis=1))),
           'irreducible_normalized':energy(irreducible),
           'accel_total_M':float(np.mean(np.sum(mass*accel**2,axis=1))),
           'accel_common_M':float(np.mean(np.sum(mass*common**2,axis=1))),
           'accel_relative_M':float(np.mean(np.sum(mass*relative**2,axis=1)))}
    require(abs(out['projection_net']+out['projection_zero']-out['normalized_total']) <= EPS*max(1,out['normalized_total']), 'normalized projection identity failed')
    require(abs(out['physical_net']+out['physical_zero']-out['physical_total']) <= EPS*max(1,out['physical_total']), 'physical orthogonality failed')
    require(abs(out['accel_common_M']+out['accel_relative_M']-out['accel_total_M']) <= EPS*max(1,out['accel_total_M']), 'M orthogonality failed')
    keep = np.arange(10) != int(np.argmax(se))
    local = d[:,keep]; local_net = np.repeat(local.mean(axis=1,keepdims=True),9,axis=1)
    out.update(sensitivity_without_largest_Se_total=energy(local), sensitivity_without_largest_Se_net=energy(local_net))
    if selected is not None:
        out.update(selected_total=float(u[:,selected].mean()), selected_net=float(un[:,selected].mean()), selected_zero=float(uz[:,selected].mean()))
    return out


def amplitude_profile(ratio):
    return {'absolute_ratio':distribution(abs(ratio).ravel()),
            'same_absolute_initial_bound_scale_not_ode_constraint':True,
            'thresholds':{str(t):{'node_time_count':int((abs(ratio)>=t).sum()),
              'nodes_any':int((abs(ratio)>=t).any(axis=0).sum()),
              'event_any':bool((abs(ratio)>=t).any())} for t in (.90,.95,.98,.99)}}


def components(model, state):
    with torch.no_grad():
        x = torch.as_tensor(state,dtype=torch.float64)
        p = model._parts(x)
        base = (p.kernel*p.phase.sin()-p.c*model.sin_ref) @ model.incidence.T
        memory = model.Se*p.e
        static = model.environment_coeff[0]*p.tanh_a + model.environment_coeff[1]*p.tanh_a.square()
        edge_terms = ((p.cos_phase-model.cos_ref).unsqueeze(-1)*model.edge_coeff[:,:,4]).flatten(-2,-1)
        env_edge = torch.zeros_like(p.e).index_add(-1,model.edge_nodes,edge_terms)
        result = {'Fc':p.ports.Fc, 'tau_env':p.ports.tau_env, 'Se_e':memory, 'amplitude_static':static,
                  'environment_edge':env_edge,
                  'base_Fc':base, 'anchored_added_Fc':p.ports.Fc-base}
        result = {k:v.numpy() for k,v in result.items()}
        kernel,c,beta = p.kernel,p.c,p.beta
    require(np.max(abs(result['Fc'].sum(axis=1))) <= EPS, 'internal coupling not zero sum')
    require(np.max(abs(result['Se_e']+result['amplitude_static']+result['environment_edge']-result['tau_env'])) <= EPS, 'environment component identity failed')
    require(np.isfinite(np.stack(list(result.values()))).all() and np.allclose(result['Fc']+result['tau_env'],p.ports.tau_e.numpy(),atol=EPS,rtol=0), 'finite component/output comparison failed')
    return result,kernel.numpy(),c.numpy(),beta.numpy()


def edge_profile(kernel,c,beta,edges,node):
    active = c > 1e-4
    adjacent = active & np.any(edges == node,axis=0) if node is not None else active
    tunable = adjacent & (beta/np.maximum(c,EPS) >= .05)
    ratio = beta[adjacent]/c[adjacent]
    gap = (kernel[:,tunable]-(c[tunable]+beta[tunable]*(np.exp(-1)-1)))/(beta[tunable]*(np.e-np.exp(-1)))
    coverage = bool(tunable.any())
    return {'active_edges':int(active.sum()), 'adjacent_active_edges':int(adjacent.sum()),
            'adjacent_tunable_edges':int(tunable.sum()), 'tunable_coverage':coverage,
            'K_over_c':distribution((kernel[:,adjacent]/c[adjacent]).ravel()),
            'K_over_c_q10':float(np.quantile(kernel[:,adjacent]/c[adjacent],.10)) if adjacent.any() else None,
            'K_over_c_le_045_fraction':float((kernel[:,adjacent]/c[adjacent]<=.45).mean()) if adjacent.any() else None,
            'beta_over_c':distribution(ratio), 'weak_modulation_edges':int((ratio<.05).sum()),
            'exact_floor_over_c':distribution(1+ratio*(np.exp(-1)-1)),
            'normalized_headroom_gap':distribution(gap.ravel()),
            'near_floor_time_fraction':float((gap<=.05).any(axis=1).mean()) if coverage else None,
            'far_floor_edge_time_fraction':float((gap>.10).mean()) if coverage else None,
            'raw_K_over_c':(kernel[:,adjacent]/c[adjacent]).tolist(), 'raw_beta_over_c':ratio.tolist(),
            'raw_gap':gap.tolist(), 'adjacent_edge_indices':np.flatnonzero(adjacent).tolist()}


def evaluate(events,constants,models,archives,selected,work):
    rows = []
    for index,event in enumerate(events):
        stop = min(25,len(event['t'])-1)
        require(stop>=1 and np.allclose(event['t'][:stop+1],np.arange(stop+1)*.01,atol=EPS,rtol=0), 'saved output grid differs')
        row = {'event_id':event['id'],'label':event['label'],'selected_machine':selected.get(event['id']),
               'real_horizon_s':float(event['t'][-1]),'models':{}}
        for name,model in models.items():
            state = np.asarray(archives[name][f'event_{index:04d}_fine'][:stop+1],dtype=np.float64)
            require(state.shape == (stop+1,50) and np.isfinite(state).all(), 'saved fine state prefix differs')
            anchor = np.r_[event['history']['theta'][-1],event['history']['omega'][-1]]
            require(np.array_equal(state[0,:20],anchor), 'saved fine initial motion differs')
            require(work['port_state_points']+len(state)<=23166, 'saved readout point cap would be exceeded')
            values,kernel,c,beta = components(model,state)
            work['port_state_points'] += len(state); work['port_calls'] += 1
            require(work['port_state_points']<=23166, 'saved readout point cap exceeded')
            bound = model.init_bounds[10:20].numpy()
            ratio = state[:,30:40]/bound
            require(np.all(abs(ratio[0])<=1+EPS), 'saved initial amplitude outside initial bounds')
            item = {'a0_over_bound':ratio[0].tolist(),'a0_tanh_derivative':(1-ratio[0]**2).tolist(),
                    'initial_amplitude':amplitude_profile(ratio[:1]),'short_path_amplitude':amplitude_profile(ratio),'windows':{}}
            for window,requested in WINDOWS.items():
                steps = min(requested,stop) if window=='capped_0.25' else requested
                if steps>stop:
                    item['windows'][window] = {'complete':False,'reason':'short_record','point_count':0}; continue
                mask = np.asarray(event['masks']['tau_e'][1:steps+1],bool) & np.isfinite(event['targets']['tau_e'][1:steps+1])
                mask = mask.all(axis=1)
                valid = np.flatnonzero(mask)+1
                w = {'complete':bool(mask.all()),'requested_point_count':steps,'point_count':len(valid),
                     'node_time_count':10*len(valid),'time_indices':valid.tolist(), 'end_s':float(event['t'][steps])}
                if w['complete']:
                    d = values['Fc'][valid]+values['tau_env'][valid]-event['targets']['tau_e'][valid]
                    w['metrics'] = decompose(d,np.asarray(constants['Se']),np.asarray(constants['mass']),selected.get(event['id']))
                    w['components'] = {k:{'normalized_signed_mean':float((v[valid]/constants['Se']).mean()),
                       'normalized_mean_square':float(((v[valid]/constants['Se'])**2).mean()),
                       'selected_normalized_signed_mean':float((v[valid]/constants['Se'])[:,selected[event['id']]].mean()) if event['id'] in selected else None,
                       'selected_normalized_mean_square':float((((v[valid]/constants['Se'])[:,selected[event['id']]])**2).mean()) if event['id'] in selected else None}
                       for k,v in values.items()}
                    w['edges'] = edge_profile(kernel[valid],c,beta,model.edge_index.numpy(),selected.get(event['id']))
                item['windows'][window] = w
            row['models'][name] = item
        rows.append(row)
    return rows


def ratios(means):
    result = {}
    for key,num,den in (('R_net_physical','physical_net','physical_total'),('C_net','projection_net','normalized_total'),
        ('C_zero','projection_zero','normalized_total'),('R_star','irreducible_normalized','normalized_total'),
        ('normalized_cross_fraction','normalized_cross','normalized_total'),('common_acceleration_fraction_M','accel_common_M','accel_total_M'),
        ('relative_acceleration_fraction_M','accel_relative_M','accel_total_M'),
        ('sensitivity_without_largest_Se_net_fraction','sensitivity_without_largest_Se_net','sensitivity_without_largest_Se_total'),
        ('selected_net_fraction','selected_net','selected_total'),('selected_zero_fraction','selected_zero','selected_total')):
        result[key] = means[num]/means[den] if means.get(den) is not None and means[den]>EPS else None
    return result


def aggregate(rows,ids,name,window):
    chosen = [r for r in rows if r['event_id'] in ids]
    eligible = [r['models'][name]['windows'][window] for r in chosen if r['models'][name]['windows'][window].get('metrics') and
                r['models'][name]['windows'][window]['complete']]
    keys = sorted({k for w in eligible for k in w['metrics']})
    dist = {k:distribution([w['metrics'][k] for w in eligible if k in w['metrics']]) for k in keys}
    means = {k:v['mean'] for k,v in dist.items()}
    components = {}
    for component in ('Fc', 'tau_env', 'Se_e', 'amplitude_static', 'environment_edge', 'base_Fc', 'anchored_added_Fc'):
        components[component] = {}
        for metric in ('normalized_signed_mean', 'normalized_mean_square',
                       'selected_normalized_signed_mean', 'selected_normalized_mean_square'):
            values = [w['components'][component][metric] for w in eligible
                      if w['components'][component][metric] is not None]
            components[component][metric] = {'count': len(values),
                                             'mean': float(np.mean(values)) if values else None}
    return {'denominator':len(ids),'covered_events':len(eligible),'complete_events':sum(w['complete'] for w in eligible),
            'point_count':sum(w['point_count'] for w in eligible), 'event_equal_means':means,
            'distributions':dist,'fractions':ratios(means),
            'edges':{'covered_tunable_events':sum(w['edges']['tunable_coverage'] for w in eligible),
                     'event_K_over_c_q05':distribution([w['edges']['K_over_c']['quantiles']['q05'] for w in eligible if w['edges']['K_over_c']['count']]),
                     'event_K_over_c_q10':distribution([w['edges']['K_over_c_q10'] for w in eligible if w['edges']['K_over_c_q10'] is not None]),
                     'event_K_over_c_median':distribution([w['edges']['K_over_c']['quantiles']['median'] for w in eligible if w['edges']['K_over_c']['count']]),
                     'event_K_over_c_le_045_fraction':distribution([w['edges']['K_over_c_le_045_fraction'] for w in eligible if w['edges']['K_over_c_le_045_fraction'] is not None]),
                     'event_beta_over_c_mean':distribution([w['edges']['beta_over_c']['mean'] for w in eligible if w['edges']['beta_over_c']['count']]),
                     'event_exact_floor_over_c_mean':distribution([w['edges']['exact_floor_over_c']['mean'] for w in eligible if w['edges']['exact_floor_over_c']['count']]),
                     'near_floor_time_fraction':distribution([w['edges']['near_floor_time_fraction'] for w in eligible if w['edges']['tunable_coverage']]),
                     'far_floor_edge_time_fraction':distribution([w['edges']['far_floor_edge_time_fraction'] for w in eligible if w['edges']['tunable_coverage']])},
            'components':components,
            'amplitude':{scope:{'thresholds':{str(t):{'events_any':sum(r['models'][name][scope]['thresholds'][str(t)]['event_any'] for r in chosen),
                       'node_time_count':sum(r['models'][name][scope]['thresholds'][str(t)]['node_time_count'] for r in chosen)} for t in (.90,.95,.98,.99)}}
                       for scope in ('initial_amplitude','short_path_amplitude')}}


def gate(rows,groups):
    name = 'parent_1860'
    complete = {r['event_id'] for r in rows if all(r['models'][name]['windows'][w]['complete'] for w in WINDOWS)}
    missed = set(groups['parent_misses']) & complete
    selected = set(groups['persistent_22']) & complete
    coverage = len(missed)>=22 and len(selected)>=18
    chosen_common = [r for r in rows if r['event_id'] in selected]
    joint_B = [r for r in chosen_common if r['models'][name]['a0_over_bound'][r['selected_machine']]<=-.95 and
               all(r['models'][name]['windows'][w]['edges']['tunable_coverage'] and
                   r['models'][name]['windows'][w]['edges']['near_floor_time_fraction']>=.30 for w in WINDOWS)]
    joint_B_fraction = len(joint_B)/len(chosen_common) if chosen_common else None
    checks = {}
    for window in WINDOWS:
        m = aggregate(rows,missed,name,window); s = aggregate(rows,selected,name,window)
        f,g = m['fractions'],s['fractions']
        valid = coverage and all(x is not None for x in (f['R_star'],f['C_zero'],g['selected_net_fraction'],g['selected_zero_fraction']))
        chosen = [r for r in rows if r['event_id'] in selected]
        edges = [r['models'][name]['windows'][window]['edges'] for r in chosen]
        edge_coverage = bool(chosen) and all(e['tunable_coverage'] for e in edges)
        neg = np.mean([r['models'][name]['a0_over_bound'][r['selected_machine']]<=-.95 for r in chosen]) if chosen else None
        unsat = np.mean([max(abs(np.asarray(r['models'][name]['a0_over_bound'])))<.90 for r in chosen]) if chosen else None
        near = float(np.mean([e['near_floor_time_fraction'] for e in edges])) if edge_coverage else None
        far = float(np.mean([e['far_floor_edge_time_fraction'] for e in edges])) if edge_coverage else None
        zero = valid and f['R_star']<=.35 and f['C_zero']>=.65 and g['selected_zero_fraction']>=.60
        checks[window] = {'missed':m,'selected':s,'eligible':bool(valid),'tunable_edge_coverage':edge_coverage,
          'selected_negative_095_event_fraction':float(neg) if neg is not None else None,
          'all_nodes_below_090_event_fraction':float(unsat) if unsat is not None else None,
          'near_floor_time_fraction_event_equal':near,'far_floor_edge_time_fraction_event_equal':far,
          'A':bool(valid and f['R_star']>=.45 and g['selected_net_fraction']>=.40),
          'joint_B_event_fraction_both_windows':joint_B_fraction,
          'B':bool(zero and joint_B_fraction>=.30) if edge_coverage else None,
          'C':bool(zero and unsat>=.90 and far>=.90) if edge_coverage else None}
    passed = [branch for branch in ('A','B','C') if all(c[branch] is True for c in checks.values())]
    return {'selected':passed[0] if len(passed)==1 else 'unresolved','checks':checks,
            'common_complete_missed_ids':sorted(missed),'common_complete_selected_ids':sorted(selected),
            'coverage_pass':coverage,'interpretation':'fixed structural contrast priority only; components have no separate observed labels'}


def execute(path):
    start,cpu = perf_counter(),process_time()
    path = path.resolve(); require(path.is_relative_to(ROOT),'release outside repository')
    release = json.loads(path.read_bytes()); profile = validate(release)
    local,public = [ROOT/side/RUN/'probe01' for side in ('artifacts','results')]
    require(not local.exists() and not public.exists(),'exclusive probe already claimed')
    local.mkdir(parents=True,exist_ok=False)
    receipt = {'run_id':RUN,'job_id':'probe01','status':'started','pid':os.getpid(),'release':release,
               'release_ref':identity(path),'started_utc':datetime.now(timezone.utc).isoformat(),'command':[sys.executable,*sys.argv]}
    write(local/'started.json',receipt)
    try:
        public.mkdir(parents=True,exist_ok=False); torch.set_num_threads(1)
        from kiki.data.osgkd import load_fit297
        events,constants = load_fit297(release['port_audit'])
        require(len(events)==297 and len({e['id'] for e in events})==297 and Counter(e['label'] for e in events)=={'stable':267,'unstable':30},'fit297 identities/labels differ')
        selected = {r['event_id']:r['selected_machine'] for r in profile['signed_electrical'] if all(v['persistent_80pct'] is True for v in r['endpoints'].values())}
        require(len(profile['signed_electrical'])==27 and all(set(r['endpoints'])=={'parent','control','candidate'} for r in profile['signed_electrical']), 'frozen three-endpoint sign profile required')
        require(len(selected)==22 and all(type(i) is int and 0<=i<10 for i in selected.values()), 'persistent selected machines differ')
        groups = profile['cohort_event_ids'].copy(); groups['persistent_22'] = list(selected)
        require(set(groups)=={'parent_misses','worsened','remaining','persistent_22'}, 'exact four reporting groups required')
        require({k:len(groups[k]) for k in ('parent_misses','worsened','remaining')}=={'parent_misses':27,'worsened':24,'remaining':243},'frozen group denominators differ')
        require(all(len(set(ids))==len(ids) for ids in groups.values()), 'duplicate frozen group identity')
        stable = {e['id'] for e in events if e['label']=='stable'}
        require(set(groups['worsened']).isdisjoint(groups['remaining']) and set(groups['worsened'])|set(groups['remaining'])==stable and
                set(selected)<=set(groups['parent_misses']) and set(groups['parent_misses'])<={e['id'] for e in events if e['label']=='unstable'},'frozen group identities differ')
        models = {}; work = Counter()
        with ExitStack() as stack:
            archives = {}
            for name,refs in release['inputs'].items():
                saved = torch.load(verified(refs['checkpoint']),map_location='cpu',weights_only=True)
                require(saved['attempt']==(1860 if name=='parent_1860' else 2020) and saved['provenance']['audit_ref']==release['port_audit'],'checkpoint identity differs')
                cls = OSGKDAnchoredPotential if name=='candidate_2020' else OSGKD
                model = cls.from_checkpoint(saved['model']).cpu().double().eval().requires_grad_(False)
                require(model.n==10 and model.config.variant=='C1','ten-node C1 required')
                for key in ('mass','damping','F','Sp','Se','omega_scale','theta_ref'):
                    expected = np.asarray(constants[key]); actual = getattr(model,key).numpy()
                    if key in ('mass','damping'): expected = np.diag(expected)
                    require(np.array_equal(actual,expected),'frozen constants differ: '+key)
                require(np.array_equal(model.edge_index.numpy(),np.asarray(np.triu_indices(10,1))), 'frozen complete ten-node candidate graph differs')
                require(np.all(np.asarray(constants['mass'])>0) and np.all(np.asarray(constants['Se'])>0),'positive mass/Se required')
                scores = read(refs['rows']); validate_rows(scores,events)
                require([r['event_id'] for r in scores]==[e['id'] for e in events],'saved path ordering differs')
                read(refs['summary'])
                models[name] = model
                archives[name] = stack.enter_context(np.load(verified(refs['predictions']),allow_pickle=False))
            rows = evaluate(events,constants,models,archives,selected,work)
        from unittest.mock import patch
        parent = models['parent_1860']
        original_rhs = parent.rhs
        def counted_rhs(*args,**kwargs):
            work['spectrum_rhs_calls'] += 1
            return original_rhs(*args,**kwargs)
        with patch.object(parent,'rhs',side_effect=counted_rhs):
            spec,detail = spectrum(parent)
        work['reference_spectra'] += 1
        require(spec['qualified'] and spec['dimension']==49 and all(v['relative_frobenius']<=1e-6 for v in spec['finite_difference']), 'reference spectrum numerical check failed')
        tolerance = max(1e-7,spec['max_real_fd_difference'])
        spec['unstable_count_real_tolerance_per_s'] = tolerance
        spec['unstable_mode_count'] = sum(v['real']>tolerance for v in detail['eigenvalues'])
        spec['near_zero_mode_count'] = sum(abs(v['real'])<=tolerance for v in detail['eigenvalues'])
        spec['historical_max_real_per_s'] = {'v1':1.475,'epoch1732':.997}
        rr = write(local/'rows.json',{'events':rows,'group_event_ids':groups,'spectrum_detail':detail})
        receipt.update(status='completed',completed_utc=datetime.now(timezone.utc).isoformat(),result={
          'rows_ref':rr,'groups':{g:{w:{m:aggregate(rows,set(ids),m,w) for m in ALIASES} for w in WINDOWS} for g,ids in groups.items()},
          'gate':gate(rows,groups),'spectrum':spec,'resource_counts':{**dict(work),'parameter_updates':0,'integrations':0,'encoder_calls':0,'new_fom':0,
              'spectrum_autodiff_jacobians':1,'spectrum_finite_difference_directions':50*len(spec['finite_difference'])},
          'component_scope':'output magnitudes and signs only; no separately observed internal port labels',
          'sensitivity_removed_generator':{'index':int(np.argmax(constants['Se'])),
              'name':constants['generator_order'][int(np.argmax(constants['Se']))],
              'bus':constants['generator_buses'][int(np.argmax(constants['Se']))]},
          'window_mean':'t>0; complete ten-node label points; time mean per event followed by event-equal group mean'},
          cost={'wall_s':perf_counter()-start,'cpu_s':process_time()-cpu},environment={'device':'cpu','dtype':'float64','threads':torch.get_num_threads(),'torch':str(torch.__version__),'numpy':np.__version__})
        require(len((json.dumps(receipt,ensure_ascii=False,indent=2,allow_nan=False)+'\n').encode('utf-8'))<512*1024, 'public summary exceeds 512 KiB')
        sr = write(public/'summary.json',receipt); write(local/'completed.json',{'status':'completed','summary_ref':sr})
        return 0
    except BaseException as error:
        write(local/'failed.json',{'status':'failed','error':f'{type(error).__name__}: {error}','wall_s':perf_counter()-start,'cpu_s':process_time()-cpu})
        raise


def self_check():
    import copy
    from kiki.models.osgkd import OSGKDConfig
    torch.set_num_threads(1)
    se = np.arange(1.,11.); mass = np.arange(2.,12.)
    d = np.tile(np.arange(1.,11.),(3,1))
    x = decompose(d,se,mass,0)
    assert abs(x['projection_net']+x['projection_zero']-x['normalized_total'])<EPS
    assert abs(x['normalized_net']+x['normalized_zero']+x['normalized_cross']-x['normalized_total'])<EPS
    assert abs(x['selected_net']+x['selected_zero']-x['selected_total'])<EPS
    assert ratios(x)['R_star']<=1+EPS
    z = decompose(np.tile(np.r_[1.,-1.,np.zeros(8)],(3,1)),se,mass)
    assert z['physical_net']==0 and z['irreducible_normalized']==0
    n = decompose(np.ones((3,10)),se,mass)
    assert n['physical_zero']==0 and n['projection_zero']==0
    assert ratios(decompose(np.zeros((1,10)),se,mass))['R_star'] is None
    p = amplitude_profile(np.full((2,10),-.96))
    assert p['thresholds']['0.95']['node_time_count']==20 and p['thresholds']['0.98']['node_time_count']==0
    c=np.array([1.,2.]); b=np.array([.5,0.]); edges=np.array([[0,0],[1,2]])
    e=edge_profile(np.tile(c+b*(np.exp(-1)-1),(3,1)),c,b,edges,0)
    assert e['near_floor_time_fraction']==1 and e['adjacent_tunable_edges']==1 and e['weak_modulation_edges']==1
    assert edge_profile(c[None,:],c,np.zeros(2),edges,0)['near_floor_time_fraction'] is None
    model = OSGKD(OSGKDConfig('C1'),mass=np.ones(10),damping=np.zeros(10),F=np.eye(10),
                  Sp=np.ones(10),Se=np.ones(10),omega_scale=np.ones(10),theta_ref=np.zeros(10),edge_index=edges)
    states = np.zeros((4,50)); states[:,0]=.2; states[:,30:40]=-.96
    comp,*_ = components(model,states)
    assert np.max(abs(comp['anchored_added_Fc']))==0
    event = {'id':'short','label':'unstable','t':np.arange(4)*.01,
             'history':{'theta':np.tile(states[0,:10],(11,1)),'omega':np.zeros((11,10))},
             'targets':{'tau_e':np.zeros((4,10))},'masks':{'tau_e':np.ones((4,10),bool)}}
    constants={'mass':np.ones(10),'Se':np.ones(10)}
    counts=Counter()
    short=evaluate([event],constants,{'parent_1860':model},{'parent_1860':{'event_0000_fine':states}}, {'short':0},counts)
    assert counts['port_state_points']==4 and short[0]['models']['parent_1860']['windows']['capped_0.25']['point_count']==3
    event['masks']['tau_e'][2,0]=False
    missing=evaluate([event],constants,{'parent_1860':model},{'parent_1860':{'event_0000_fine':states}}, {'short':0},Counter())
    assert missing[0]['models']['parent_1860']['windows']['0.03']['point_count']==2
    assert aggregate(missing,{'short'},'parent_1860','0.03')['covered_events']==0
    metric = decompose(np.ones((3,10)),np.ones(10),np.ones(10),0)
    template = {'event_id':'x','selected_machine':0,'models':{'parent_1860':{
       'a0_over_bound':[-.96]*10,'initial_amplitude':amplitude_profile(states[:1,30:40]),
       'short_path_amplitude':amplitude_profile(states[:,30:40]),'windows':{}}}}
    for window in WINDOWS:
        template['models']['parent_1860']['windows'][window] = {'complete':True,'point_count':3,'metrics':metric.copy(),
          'components':{k:{'normalized_signed_mean':0.,'normalized_mean_square':0.,'selected_normalized_signed_mean':0.,'selected_normalized_mean_square':0.} for k in comp},
          'edges':copy.deepcopy(e)}
    toy = [copy.deepcopy(template) for _ in range(27)]
    for i,row in enumerate(toy): row['event_id']=str(i)
    groups={'parent_misses':[str(i) for i in range(27)],'persistent_22':[str(i) for i in range(22)]}
    assert gate(toy,groups)['selected']=='A'
    for row in toy:
        for w in row['models']['parent_1860']['windows'].values():
            w['metrics'].update(irreducible_normalized=0.,projection_net=0.,projection_zero=10.,selected_net=0.,selected_zero=1.)
    assert gate(toy,groups)['selected']=='B'
    for row in toy:
        row['models']['parent_1860']['windows']['0.03']['edges']['near_floor_time_fraction']=0.
    assert gate(toy,groups)['selected']=='unresolved'
    for row in toy:
        row['models']['parent_1860']['windows']['0.03']['edges']['near_floor_time_fraction']=1.
    for row in toy:
        row['models']['parent_1860']['a0_over_bound']=[0.]*10
        for w in row['models']['parent_1860']['windows'].values(): w['edges'].update(near_floor_time_fraction=0.,far_floor_edge_time_fraction=1.)
    assert gate(toy,groups)['selected']=='C'
    toy[0]['models']['parent_1860']['windows']['0.03']['edges']['tunable_coverage']=False
    assert gate(toy,groups)['selected']=='unresolved'
    for row in toy[:10]: row['models']['parent_1860']['windows']['0.03']['complete']=False
    assert not gate(toy,groups)['coverage_pass']
    try: validate({})
    except ValueError: pass
    else: raise AssertionError('unreleased probe accepted')
    print('synthetic self-check passed: physical/normalized/M identities, signed selection, irreducible share, zero denominators, saturation, exact edge floors and missing tunability; no private assets read')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command',required=True)
    sub.add_parser('self-check')
    probe = sub.add_parser('probe'); probe.add_argument('--release',type=Path,required=True)
    args = parser.parse_args()
    if args.command=='self-check': self_check()
    else: raise SystemExit(execute(args.release))
