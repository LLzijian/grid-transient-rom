"""Released zero-update graph endpoint spectrum, val event, and train effect audit."""

import argparse
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

from experiments.osgkd_v2_p0_20261123 import run as p0
from experiments.osgkd_v2_p2a_equivariant_graph_20261124 import run as graph_run
from experiments.osgkd_v2_p2a_post_audit_20261124 import contributions
from kiki.data.osgkd_train_val import load_train, load_val
from kiki.evaluation.osgkd import motion_metrics, score_event
from kiki.evaluation.osgkd_autonomous import autonomous_outcome
from kiki.models.osgkd_fitting import IntegrationWork
from kiki.models.osgkd_graph import OSGKDGraph

baseline, scoring = graph_run.baseline, graph_run.scoring
identity, read, require, verified, repo_path = (graph_run.identity, graph_run.read, graph_run.require,
                                               graph_run.verified, graph_run.repo_path)
RUN = 'osgkd_v2_p2a_post_audit_20261124'
JOB = 'probe01'
PROTOCOL = 'campaigns/osgkd_v2/p2a_post_audit_protocol.md'
FIT = 'results/osgkd_v2_p2a_equivariant_graph_20261124/fit01/summary.json'
EVAL = 'results/osgkd_v2_p2a_equivariant_graph_20261124/evaluate01/summary.json'
INPUTS = {
    'fit': (FIT, 'e6b534b8b785e9fe275d156ebd0edbe9c83ba25adba1500bb2322984f61e9a38'),
    'evaluate': (EVAL, '25922f88cfb7c8352d4313d5886ca17422b2c7d2a6e72beb4dd74b760f5eba43'),
    'anchor': (graph_run.INPUTS['anchor_analysis'], graph_run.ANCHOR_SHA),
}
SOURCES = graph_run.SOURCES | set(p0.SOURCES) | {PROTOCOL, f'experiments/{RUN}/run.py', f'experiments/{RUN}/contributions.py'}
RELEASE_KEYS = {'run_id', 'job_id', 'released', 'device', 'threads', 'source_hashes', 'input_refs'}


def validate(release):
    require(set(release) == RELEASE_KEYS and tuple(release[key] for key in
            ('run_id', 'job_id', 'released', 'device', 'threads')) == (RUN, JOB, True, 'cpu', 1), 'release identity differs')
    require(set(release['source_hashes']) == SOURCES and set(release['input_refs']) == set(INPUTS), 'release coverage differs')
    for name, sha in release['source_hashes'].items():
        require(identity(repo_path(name))['sha256'] == sha, 'source identity differs: ' + name)
    loaded = {}
    for key, (path, sha) in INPUTS.items():
        ref = release['input_refs'][key]
        require((ref['path'], ref['sha256']) == (path, sha), 'input binding differs')
        loaded[key] = read(ref)
    require(loaded['evaluate']['release']['fit_ref'] == release['input_refs']['fit'] and
            loaded['fit']['result']['valid_endpoint'] and loaded['fit']['status'] == loaded['evaluate']['status'] == 'completed',
            'source jobs incomplete or mismatched')
    return loaded


def read_models(fit):
    refs = fit['result']['checkpoints']
    require(set(refs) == {'1792', '2048'}, 'fixed graph endpoints required')
    models = {}
    for step, ref in refs.items():
        saved = torch.load(verified(ref), map_location='cpu', weights_only=True)
        require(saved['attempt'] == int(step) and saved['release'] == fit['release'], 'checkpoint binding differs')
        models[step] = OSGKDGraph.from_checkpoint(saved['model']).eval().requires_grad_(False)
    return models, refs


def read_spectra(models, local):
    summaries = {}
    detail = None
    for step in ('1792', '2048'):
        summary, detail = p0.spectrum(models[step])
        require(summary['qualified'] and summary.get('dimension') == 49, 'fixed-reference spectrum not qualified')
        poles = detail['eigenvalues']
        summary['positive_count'] = sum(value['real'] > 1e-7 for value in poles)
        summary['leading_eigenvalue'] = poles[0]
        summary['spectrum_ref'] = baseline.atomic_write(local / f'spectrum_{step}.json', detail)
        summaries[step] = summary
    return summaries, detail


def false_alarm_rows(evaluation):
    found = {}
    for step in ('1792', '2048'):
        rows = read(evaluation['result']['val'][step]['rows_ref'])
        matches = [row for row in rows if row['label'] == 'stable' and autonomous_outcome(row) == 'autonomous_trigger']
        require(len(rows) == 117 and len(matches) == 1, 'saved val false alarm population differs')
        found[step] = matches[0]
    require(found['1792']['event_id'] == found['2048']['event_id'], 'false alarm identities differ')
    return found['2048']


@torch.no_grad()
def inspect_event(model, event, archived, constants, spectrum, work, local):
    require(event['label'] == 'stable' and len(event['t']) >= 801, 'fixed event has incomplete stable truth')
    theta, omega = (torch.as_tensor(event['history'][key], dtype=torch.float64) for key in ('theta', 'omega'))
    initial = model.initialize(theta, omega)
    work.encoder_events += 1
    main = scoring.integrate(model, initial, 800, work=work)
    fine = scoring.integrate(model, initial, 800, internal_step=.0025, work=work)
    ports = model.ports(fine)
    evaluated = score_event(event, main.numpy(), fine.numpy(), {key: getattr(ports, key).numpy() for key in ('tau_m', 'tau_e')}, constants)
    match = evaluated['classification_8s']
    predicted_time = match['predicted_time_record_s']
    require(autonomous_outcome(evaluated) == 'autonomous_trigger' and predicted_time == archived['classification_8s']['predicted_time_record_s']
            and abs(predicted_time - .10 - 5.51) < 1e-12 and work.rhs['forward'] == 19200,
            'saved false alarm time or numerical qualification differs')
    for key in ('1', '2', '5', '8'):
        actual = evaluated['scopes'][key]['metrics']['mean_motion_error']
        reference = archived['scopes'][key]['metrics']['mean_motion_error']
        require(np.isclose(actual, reference, rtol=1e-9, atol=1e-10), 'saved motion metric differs')
    target = np.column_stack((event['targets']['theta'][:801], event['targets']['omega'][:801]))
    pred = fine.numpy()
    weights = np.asarray(constants['mass'])
    weights = weights / weights.sum()
    prefix = {str(h): motion_metrics(pred[1:1+100*h, :20], target[1:1+100*h], weights,
                                     constants['omega_scale'], constants['s_coi']) for h in (1,2,4,5)}
    basis, scales = np.asarray(spectrum['basis']), np.asarray(spectrum['scales'])
    y = ((pred - np.asarray(spectrum['reference'])) / scales) @ basis
    rhs = (model.rhs(None, fine).numpy() / scales) @ basis
    work.rhs['forward'] += len(fine)
    linear = y @ np.asarray(spectrum['quotient_jacobian']).T
    windows = {}
    for left, right in ((0,1),(1,2),(2,4),(4,5),(5,6),(6,8)):
        interval = slice(left*100, right*100+1)
        def envelope(path):
            angle = np.ptp(path[interval, :10], axis=1)
            speed = path[interval, 10:20]
            return dict(angle_min_rad=float(angle.min()), angle_max_rad=float(angle.max()),
                        angle_end_rad=float(angle[-1]),
                        relative_speed_max_rad_s=float(np.abs(speed-(speed@weights)[:, None]).max()))
        norm = float(np.linalg.norm(rhs[interval]))
        windows[f'{left}-{right}'] = dict(predicted=envelope(pred), observed=envelope(target),
            quotient_state_rms=float(np.sqrt(np.mean(y[interval]**2))),
            linearization_relative_residual=float(np.linalg.norm((rhs-linear)[interval])/norm) if norm > 0 else None)
    trace = local / 'fixed_val_trace.npz'
    with trace.open('xb') as stream:
        np.savez_compressed(stream, t=event['t'][:801], graph_primary=main.numpy(), graph_fine=pred, observed=target)
    return dict(source_rows_checked=True, trace_ref=identity(trace), trigger_after_forecast_s=predicted_time-.10,
                trigger_pair_gap_s=match['main_fine_trigger_gap_s'], numerical=match['numerical'],
                prefix_motion=prefix, windows=windows, diagnostic_rhs=801)


def self_check():
    torch.set_num_threads(1)
    constants = {key: np.ones(10) for key in ('mass','damping','Sp','Se','omega_scale','S_m')}
    constants.update(F=np.eye(10), theta_ref=np.linspace(-.1,.1,10), edge_index=np.stack(np.triu_indices(10,k=1)))
    model = graph_run.fit.new_model(constants).requires_grad_(False)
    spectral, detail = p0.spectrum(model)
    assert spectral['qualified'] and spectral['dimension'] == len(detail['eigenvalues']) == 49
    contributions.self_check()
    print('synthetic self-check passed')


def execute(release_path):
    wall, cpu = perf_counter(), process_time()
    release_path = release_path.resolve()
    release_ref = identity(release_path)
    release = read(release_ref)
    inputs = validate(release)
    local, public = (ROOT / group / RUN / JOB for group in ('artifacts','results'))
    require(not local.exists() and not public.exists(), 'probe already claimed')
    local.mkdir(parents=True)
    public.mkdir(parents=True)
    receipt = dict(run_id=RUN, job_id=JOB, pid=os.getpid(), status='started', release=release,
                   release_ref=release_ref, started_utc=datetime.now(timezone.utc).isoformat(),
                   environment=dict(python=sys.version, torch=str(torch.__version__), device='cpu', dtype='float64', threads=1))
    baseline.atomic_write(local/'started.json', receipt)
    phase = 'spectra'
    val_work, train_work = IntegrationWork(), IntegrationWork()
    try:
        torch.set_num_threads(1)
        models, refs = read_models(inputs['fit'])
        spectra, detail = read_spectra(models, local)
        phase = 'data_identity'
        train, constants = load_train()
        val = load_val(constants)
        for split, events in (('train',train),('val',val)):
            require(baseline.data_identity(events) == inputs['anchor']['evaluation'][split]['data_identity'], 'data identity differs')
        phase = 'single_val_trajectory'
        saved = false_alarm_rows(inputs['evaluate'])
        event, = [e for e in val if e['id'] == saved['event_id']]
        diagnostic = inspect_event(models['2048'], event, saved, constants, detail, val_work, local)
        phase = 'train_functional_effects'
        effects = contributions.measure(models['2048'], train, train_work, local)
        require(train_work.rhs['forward'] <= 2464000 and effects['diagnostic_rhs'] <= 308385 and
                val_work.rhs['forward'] == 20001, 'probe work differs')
        choice = {'positive':'enable_spectral_hinge', 'negative':'use_unsupplemented_recipe'}.get(spectra['2048']['direction'], 'unresolved')
        result = dict(checkpoints=refs, spectra=spectra, spectral_branch=choice, fixed_val_false_alarm=diagnostic,
                      train_functional_effects=effects, optimizer_updates=0, new_fom=0,
                      work=dict(train=train_work.snapshot(), val=val_work.snapshot(), val_rollout_rhs=19200,
                                val_instantaneous_rhs=801, spectrum_calls_performed=2))
        receipt.update(status='completed', result=result, completed_utc=datetime.now(timezone.utc).isoformat(),
                       cost=dict(wall_s=perf_counter()-wall, cpu_s=process_time()-cpu))
        ref = baseline.atomic_write(public/'summary.json', receipt, public=True)
        baseline.atomic_write(local/'completed.json', dict(status='completed', summary_ref=ref))
        print(json.dumps(ref), flush=True)
    except BaseException as exc:
        baseline.atomic_write(local/'failed.json', dict(status='failed', phase=phase, error=repr(exc),
            work=dict(train=train_work.snapshot(), val=val_work.snapshot()),
            cost=dict(wall_s=perf_counter()-wall, cpu_s=process_time()-cpu)))
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    choices = parser.add_subparsers(dest='command', required=True)
    choices.add_parser('self-check')
    choices.add_parser('probe').add_argument('--release', type=Path, required=True)
    args = parser.parse_args()
    self_check() if args.command == 'self-check' else execute(args.release)
