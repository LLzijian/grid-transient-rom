"""Evaluate frozen synthetic endpoints on independent observations and true structure."""

import argparse
from functools import lru_cache
from pathlib import Path
import sys
from time import perf_counter, process_time

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT/'src')]

import numpy as np
import torch

from experiments.osgkd_v2_sprint2a_20261125 import resume_host as io
from experiments.osgkd_v2_wp2_layer1_20261127 import engine, models, recovery, run, synthetic, training
from kiki.data.osgkd_synthetic import model_constants, motion_packet


@lru_cache(maxsize=2)
def held_out(n, condition):
    system = synthetic.make_system(n=n, condition=condition)
    levels = synthetic.NOISE_LEVELS if n == 8 else (.01,)
    events = {synthetic.noise_key(level): [] for level in levels}
    truth, checks = [], []
    for j in range(24):
        seed, excitation = synthetic.VAL_SEED+j, synthetic.EXCITATIONS[j//6]
        coarse, t1 = synthetic.generate_episode(system, seed, excitation, noise_levels=levels)
        _, t2 = synthetic.generate_episode(system, seed, excitation, noise_levels=(0.,), internal_dt=.00125)
        checks.append(float(np.max(np.abs(t1['state']-t2['state']))))
        for key in events:
            events[key].append(coarse[key])
        truth.append(t1)
    io.require(max(checks) <= 1e-6, 'held-out synthetic truth failed integration qualification')
    constants = model_constants(motion_packet(next(iter(events.values()))[0]))
    return events, constants, truth, system, max(checks)


def check_stage(endpoint, binding, stage, attempts, directory, variant=None, parent=None):
    declared = {**binding, 'stage': stage}
    if variant is not None:
        declared.update(variant=variant, parent=parent)
    elif stage == 'ode_adapt' and parent is not None:
        declared['parent'] = parent
    saved_binding = endpoint['binding']
    io.require(all(saved_binding.get(k) == v for k, v in declared.items()), 'terminal stage binding differs')
    if endpoint['status'] == 'completed':
        name = 'frozen.pt' if not attempts else f'checkpoint_{attempts:04d}.pt'
        expected = (directory/name).relative_to(ROOT).as_posix()
        io.require(endpoint['attempts'] == endpoint['accepted'] == attempts
                   and endpoint['checkpoint']['path'] == expected, 'fixed stage budget or checkpoint differs')
        io.verified(endpoint['checkpoint'])
    else:
        io.require(endpoint['status'] in {'qualification_failed','scientific_stopped','source_endpoint_unavailable'}
                   and endpoint['checkpoint'] is None and 0 <= endpoint['accepted'] <= endpoint['attempts'] <= attempts,
                   'invalid terminal stage exclusion')


def compare_endpoint(trajectory_model, endpoint, observations, truth, system, b, event_ids, rows_path):
    result = dict(fit=endpoint, evaluated=False)
    if endpoint['status'] != 'completed':
        return result
    saved = training.restore(endpoint['checkpoint'], trajectory_model)
    support = run.support_from_summary(endpoint['support'])
    io.require(saved['binding'] == endpoint['binding'] and len(saved.get('rows', [])) == endpoint['accepted'],
               'frozen endpoint binding differs')
    if support is not None:
        io.require(support.base.numel() == trajectory_model.edge_index.shape[1] and
                   not bool((support.modulation & ~support.base).any()), 'frozen support invalid')
    model = trajectory_model.eval().requires_grad_(False)
    with torch.inference_mode():
        engine.trajectory(model, observations.select(slice(0, 1)), 2, .005)
    prediction = engine.score_events(model, observations, (100, 200, 400), b, event_ids=event_ids)
    io.write(rows_path, prediction.pop('rows'))
    prediction['rows_ref'] = io.identity(rows_path)
    explanation = recovery.score_structure(model, support, truth, system)
    nonzero = {name:int(getattr(model,name).detach().ne(0).sum()) for name in ('c','beta') if hasattr(model,name)}
    if hasattr(model, 'edge_coeff'):
        nonzero['incident_dictionary_edges'] = int(model.edge_coeff.detach().ne(0).any(dim=(1, 2)).sum())
    result.update(evaluated=True, forecast=prediction, structure=explanation,
                  nonzero_ode_edge_counts=nonzero or None,
                  parameter_count=sum(p.numel() for p in model.parameters()),
                  encoder_parameters=sum(p.numel() for p in model.encoder.parameters()),
                  evolved_state_dimension=getattr(model, 'effective_state_dimension', 5*model.n))
    return result


def evaluate(path):
    release = io.read(path)
    io.require(path == run.PUBLIC/'evaluate_release.json' and release['run_id'] == run.RUN and release['released'] is True,
               'canonical Layer 1 score release required')
    registered = run.validate(io.verified(release['fit_release']))
    fit = io.read(io.verified(release['fit_summary']))
    io.require(fit['release'] == release['fit_release'] and fit['status'] == 'completed'
               and fit['ieee39_event_access'] is False and fit['synthetic_val_access'] is False,
               'frozen training book differs')
    output = run.PUBLIC/'evaluate01'
    local = ROOT/'artifacts'/run.RUN/'evaluate01'
    io.require(not output.exists() and not local.exists(), 'Layer 1 score already claimed')
    output.mkdir(parents=True)
    local.mkdir(parents=True)
    torch.set_num_threads(1)
    started, cpu = perf_counter(), process_time()
    all_refs = {}
    for job in registered['jobs']:
        n, condition, noise, seed = (job[k] for k in ('n', 'condition', 'noise', 'seed'))
        key = run.cell_key(n, condition, noise, seed)
        source = fit['cells'][key]
        io.require(source['job'] == job and set(source['methods']) == set(job['methods']), 'terminal cell differs')
        by_noise, constants, truth, system, gap = held_out(n, condition)
        events = engine.Episodes.from_events(by_noise[noise])
        event_ids = [sample['id'] for sample in by_noise[noise]]
        batch = registered['microbatch_by_n'][str(n)]
        result = dict(job=job, truth_mesh_gap=gap, val_count=len(events), event_sha256=training.data_digest(events), methods={})
        for method in job['methods']:
            ref = source['methods'][method]
            directory = run.LOCAL/key/method
            io.require(ref['path'] == (directory.parent/(method+'_summary.json')).relative_to(ROOT).as_posix(),
                       'terminal method path differs')
            terminal = io.read(io.verified(ref))
            binding = dict(release=release['fit_release'], cell=job, cell_key=key, method=method)
            io.require(terminal['binding'] == binding, 'method/source identity differs')
            if n == 8:
                parent = terminal['dense']
                check_stage(parent, binding, 'dense', run.DENSE, directory/'dense')
                variants = terminal['variant_results']
                expected = {'full','sparse'} if method == 'osgkd' else {'full'}
                io.require(set(variants) == (expected if parent['status']=='completed' else set()), 'refit branches differ')
                for variant, item in variants.items():
                    check_stage(item, binding, 'refit', run.REFIT, directory/variant, variant, parent['checkpoint'])
                result['methods'][method] = {'dense':parent, 'weak_refit':terminal['weak_refit'], 'variants':{}}
            else:
                item = terminal['result']
                source_ref = fit['cells'][run.cell_key(8, 'open', '0.01', seed)]['methods']['osgkd']
                transfer_parent = io.read(io.verified(source_ref))['variant_results'].get('full', {})
                parent_checkpoint = transfer_parent.get('checkpoint')
                span = {'same_template':run.DENSE+run.REFIT, 'ode_adapt':run.ADAPT, 'frozen_lift':0}[method]
                if method != 'same_template' and item['status'] == 'completed':
                    io.require(item['source_checkpoint'] == parent_checkpoint, 'transfer parent differs')
                check_stage(item, binding, method, span, directory, parent=parent_checkpoint)
                variants = {'full':item}
                result['methods'][method] = {'variants':{}}
            for variant, endpoint in variants.items():
                model = models.make_model(method if n == 8 else 'osgkd', constants, seed)
                rows_file = local/(key+'_'+method+'_'+variant+'_rows.json')
                result['methods'][method]['variants'][variant] = compare_endpoint(model, endpoint, events, truth, system,
                                                                                 batch, event_ids, rows_file)
            print(key, method, 'scored', flush=True)
        # Candidate support lists are small but private here. Publish the results and aggregate counts.
        detailed = local/(key+'.json')
        io.write(detailed, result)
        for item in result['methods'].values():
            for endpoint in item['variants'].values():
                fit_info = endpoint['fit']
                for name in ('support','active_support','end_function_rms'):
                    fit_info.pop(name, None)
        result['detail_ref'] = io.identity(detailed)
        public = output/(key+'.json')
        io.write(public, result)
        io.require(public.stat().st_size < 512*1024, 'public cell exceeds size budget')
        all_refs[key] = io.identity(public)
    final = dict(run_id=run.RUN, status='completed', release=io.identity(path), controls=fit['controls'],
                 cells=all_refs, ieee39_event_access=False, test_trajectory_access=False, new_fom=0,
                 wall_s=perf_counter()-started, cpu_s=process_time()-cpu, completed_utc=io.utc())
    io.write(output/'summary.json', final)
    print(io.identity(output/'summary.json'), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--release', type=Path, required=True)
    evaluate(parser.parse_args().release.resolve())
