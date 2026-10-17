"""Bounded TRAIN-only E1 joint fits; historical trainers remain immutable."""
from __future__ import annotations
import argparse
import contextlib
import csv
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

for _key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[_key] = '1'
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'experiments'), str(ROOT / 'src')]
import h5py
import numpy as np
import torch
from kiki.data.private_dataset import canonical_arrays
from kiki.models.private_pm import MechanicalROM
from train_joint_campaign import sample_batch
from train_structured_rom import _event_rollout_errors, _point_error

BASE = ROOT / 'private_data/task2_m3_0020/model.npz'
NORM = ROOT / 'private_data/task2_long_0060_5s/normalization.npz'
ARMS = ('G3-S', 'L10-S', 'HL11-S')
SEED = 98


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save_json(path, obj):
    Path(path).write_text(json.dumps(obj, indent=2, allow_nan=False), encoding='utf-8')


def sources():
    paths = {'base': BASE, 'normalization': NORM,
             'trainer': Path(__file__), 'model_code': ROOT/'experiments/local_memory_rom_0098.py',
             'structured_core': ROOT/'experiments/structured_rom.py',
             'objective_helpers': ROOT/'experiments/train_structured_rom.py',
             'sampling_helpers': ROOT/'experiments/train_joint_campaign.py',
             'preregistration': ROOT/'reports/task2_local_memory_0098.md',
             'roadmap': ROOT/'docs/TASK2_after_loop_97.md'}
    for directory in ('ieee39_postfault', 'ieee39_expansion'):
        for name in ('dataset_index.csv', 'splits.json'):
            paths[f'{directory}/{name}'] = ROOT/'private_data'/directory/name
    return {key: {'path': str(path), 'sha256': digest(path)} for key, path in paths.items()}


def read_train():
    """Open only explicit TRAIN payloads, keeping original lengths and te units."""
    base = MechanicalROM.load(BASE)
    events, excluded = [], []
    for directory in ('ieee39_postfault', 'ieee39_expansion'):
        root = ROOT/'private_data'/directory
        splits = json.loads((root/'splits.json').read_text())
        assert not set(splits['train']) & (set(splits['validation']) | set(splits['test']))
        with (root/'dataset_index.csv').open() as stream:
            rows = {row['event_id']: row for row in csv.DictReader(stream)}
        with h5py.File(root/'dataset.h5', 'r') as file:
            eq = {key: file['equilibrium'][key][:] for key in ('delta', 'omega', 'Pm', 'Pe')}
            inertia, omega_s = file['inertia'][:], float(file.attrs['omega_s'])
            for eid in splits['train']:
                row = rows[eid]
                if float(row['trajectory_end_s']) <= .1+1e-9:
                    excluded.append({'id': eid, 'root': directory, 'reason': 'latency_uncovered'})
                    continue
                raw = file['events'][eid]
                arrays = {key: raw[key][:] for key in ('t', 'delta', 'omega', 'Pm', 'Pe')}
                can = canonical_arrays(arrays, eq, inertia, omega_s)
                x = np.column_stack((can['delta_sync'], can['omega_sync'], can['Pm_raw']-base.pm_star))
                torque = raw['te'][:]-base.pm_star
                if len(x) <= 11 or not np.isfinite(x).all() or not np.isfinite(torque).all():
                    raise ValueError(f'invalid original TRAIN event: {eid}')
                if not np.allclose(np.diff(can['t']), .01, atol=1e-9):
                    raise ValueError(f'nonuniform original interval: {eid}')
                events.append({'id': eid, 'group': row['fault_bus'], 'label': row['label'],
                               'x': x, 'torque': torque, 'root': directory})
    assert len(events) == 346 and len(excluded) == 23
    assert sum(e['label'] == 'unstable' for e in events) == 35
    assert len(set(e['group'] for e in events)) == 24
    return events, excluded


def panel_events(events):
    selected = []
    for label, count in (('unstable', 3), ('stable', 9)):
        groups = sorted(set(e['group'] for e in events if e['label'] == label))
        pools = [sorted([e for e in events if e['label'] == label and e['group'] == group],
                        key=lambda e: e['id']) for group in groups]
        for offset in range(max(map(len, pools))):
            for pool in pools:
                if offset < len(pool):
                    selected.append(pool[offset])
                    count -= 1
                    if count == 0:
                        break
            if count == 0:
                break
    assert len(selected) == 12 and sum(e['label']=='unstable' for e in selected) == 3
    return selected


def make_model(arm):
    from local_memory_rom_0098 import LocalMemoryROM
    norm = np.load(NORM, allow_pickle=False)
    torch.manual_seed(SEED)
    return LocalMemoryROM(MechanicalROM.load(BASE), norm['input_scale'],
                          norm['feature_scale'][10:20], arm=arm, seed=SEED).double()


def load_model(directory):
    from local_memory_rom_0098 import LocalMemoryROM
    directory = Path(directory)
    meta = json.loads((directory/'checkpoint.json').read_text())
    model = LocalMemoryROM.from_config(meta['config']) if 'config' in meta else make_model(meta['arm'])
    model.load_state_dict(torch.load(directory/'model.pt', map_location='cpu', weights_only=True), strict=True)
    model.eval()
    return model


@contextlib.contextmanager
def graph(model):
    model.begin_graph_cache()
    try:
        yield
    finally:
        model.end_graph_cache()


@contextlib.contextmanager
def preserved_template(model, index, params):
    old = [p.detach().clone() for p in params]
    mask = model.template_mask.detach().clone()
    try:
        yield old
    finally:
        model.template_mask.requires_grad_(False)
        with torch.no_grad():
            for p, value in zip(params, old):
                p.copy_(value)
            model.template_mask.copy_(mask)


def process_memory_bytes():
    """Windows working set, avoiding an extra dependency in the research env."""
    if os.name != 'nt':
        import resource
        return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)*1024
    import ctypes
    from ctypes import wintypes
    class Counters(ctypes.Structure):
        _fields_ = [('cb', wintypes.DWORD), ('faults', wintypes.DWORD)]+[
            (name, ctypes.c_size_t) for name in ('peak', 'working', 'peak_paged',
            'paged', 'peak_nonpaged', 'nonpaged', 'pagefile', 'peak_pagefile')]
    counters = Counters(); counters.cb = ctypes.sizeof(counters)
    handle = ctypes.windll.kernel32.GetCurrentProcess
    handle.restype = wintypes.HANDLE
    get_memory = ctypes.windll.psapi.GetProcessMemoryInfo
    get_memory.argtypes = (wintypes.HANDLE, ctypes.POINTER(Counters), wintypes.DWORD)
    if not get_memory(handle(), ctypes.byref(counters), counters.cb):
        raise OSError('cannot enforce process memory cap')
    return int(counters.working)


def histories_for(samples):
    return torch.as_tensor(np.stack([e['x'][i-10:i+1] for e, i in samples]), dtype=torch.float64)


def risk(values):
    return .8*values.mean()+.2*values.topk(max(1, int(np.ceil(.1*len(values))))).values.mean()


def autonomous(model, events, horizon):
    samples = [(e, 10) for e in events]
    return _event_rollout_errors(model, samples, histories_for(samples), horizon)


def objective(model, samples, history, horizon, force_scale):
    events = [e for e, _ in samples]
    long = autonomous(model, events, horizon)
    local = _event_rollout_errors(model, samples, history, .2).mean()
    q = model.encode(history)
    state = torch.cat((history[:, -1], q), -1)
    adjacent = torch.as_tensor(np.stack([e['x'][i-9:i+2] for e,i in samples]), dtype=torch.float64)
    latent = ((model.rk4(state, .01)[:, 30:]-model.encode(adjacent))/.1).square().mean()
    predicted_torque = state[:, 20:30]-model.mass*model.field(state)[:, 10:20]
    truth_torque = torch.as_tensor(np.stack([e['torque'][i] for e,i in samples]), dtype=torch.float64)
    torque = ((predicted_torque-truth_torque)/force_scale).square().mean()
    complexity = model.complexity(state)
    terms = {'risk': risk(long), 'mean': long.mean(), 'local': local,
             'latent_consistency': latent, 'torque': torque, 'complexity': complexity}
    total = 10*terms['risk']+local+.1*latent+.1*torque+1e-4*complexity
    return total, terms


def optimizer_for(model):
    blocks = {}
    positive_templates = {id(p) for index in (5, 6) for p in template_params(model, index)}
    for name, value in model.named_parameters():
        if not value.requires_grad:
            continue
        if any(part in name for part in ('encoder', 'gru', 'head')):
            lr = 3e-4
        elif id(value) in positive_templates or '_q_quadratic_raw' in name or name.startswith('structured.'):
            lr = 1e-2
        else:
            lr = 1e-3
        blocks.setdefault(lr, []).append(value)
    return torch.optim.Adam([{'params': values, 'lr': lr} for lr, values in blocks.items()])


def monitor(model, panel):
    out = {}
    with torch.no_grad(), graph(model):
        for horizon in (1., 2., 5.):
            values = autonomous(model, panel, horizon)
            if not torch.isfinite(values).all():
                raise FloatingPointError('nonfinite fixed-panel monitor')
            unstable = torch.tensor([e['label']=='unstable' for e in panel], dtype=torch.bool)
            out[str(int(horizon))] = {'risk': float(risk(values)), 'mean': float(values.mean()),
                                      'severe_mean': float(values[unstable].mean()),
                                      'values': values.tolist()}
    return {'average_risk': float(np.mean([v['risk'] for v in out.values()])),
            'severe_mean': float(np.mean([v['severe_mean'] for v in out.values()])),
            'horizons': out, 'event_ids': [e['id'] for e in panel]}


def check_structure(model):
    zero = torch.zeros((2, model.dim), dtype=torch.float64)
    probe = torch.linspace(-.02, .02, 2*model.dim, dtype=torch.float64).reshape(2, model.dim)
    with torch.no_grad():
        f0 = model.field(zero)
        f = model.field(probe)
        shift = probe.clone(); shift[:, :10] += .37
        qa = {'f0': float(f0.abs().max()), 'kinematic': float((f[:, :10]-probe[:, 10:20]).abs().max()),
              'gauge': float((model.field(shift)-f).abs().max())}
    if max(qa.values()) > 1e-10 or not torch.isfinite(f).all():
        raise AssertionError(qa)
    return qa


def checkpoint(output, name, model, optimizer, rng, arm, step):
    directory = output/name
    directory.mkdir(exist_ok=False)
    torch.save(model.state_dict(), directory/'model.pt')
    torch.save(optimizer.state_dict(), directory/'optimizer.pt')
    torch.save({'numpy': rng.bit_generator.state, 'torch': torch.get_rng_state()}, directory/'rng.pt')
    meta = {'arm': arm, 'step': step, 'seed': SEED, 'dimension': model.dim,
            'trainable_parameters': sum(p.numel() for p in model.parameters() if p.requires_grad),
            'active_templates': model.template_mask.detach().tolist(),
            'optimizer_rates': [g['lr'] for g in optimizer.param_groups]}
    if hasattr(model, 'config'):
        meta['config'] = model.config()
    save_json(directory/'checkpoint.json', meta)
    if hasattr(model, 'export'):
        model.export(directory/'structured_numeric_export.json')
    before = torch.get_rng_state()
    recovered = load_model(directory)
    torch.set_rng_state(before)
    probe = torch.linspace(-.02, .02, 2*model.dim, dtype=torch.float64).reshape(2, model.dim)
    with torch.no_grad():
        if not torch.allclose(model.field(probe), recovered.field(probe), atol=1e-12, rtol=0):
            raise AssertionError('checkpoint reconstruction changed field')
    return directory


def template_params(model, index):
    values = model.template_parameters(index)
    return [values] if isinstance(values, torch.Tensor) else list(values)


def grow(model, panel):
    """Differentiate a genuine autonomous multistep panel, not conditional RHS."""
    candidates = []
    for index, name in enumerate(model.template_names):
        if bool(model.template_mask[index]):
            continue
        params = template_params(model, index)
        with preserved_template(model, index, params) as old:
            signed = index not in (5, 6)
            if signed:
                model.set_active(index, True)
                with torch.no_grad():
                    for p in params:
                        p.zero_()
                with graph(model):
                    loss = risk(autonomous(model, panel, .2))
                    grads = torch.autograd.grad(loss, params, allow_unused=False)
            else:
                # Physical rho derivative at zero, without softplus scaling.
                model.template_mask.requires_grad_(True)
                with graph(model):
                    loss = risk(autonomous(model, panel, .2))
                    gate_gradient = torch.autograd.grad(loss, model.template_mask)[0][index]
                rho = torch.nn.functional.softplus(params[0].detach())
                grads = (gate_gradient / rho,)
        flat = torch.cat([g.reshape(-1) for g in grads])
        if not torch.isfinite(flat).all():
            raise FloatingPointError(f'nonfinite growth gradient:{name}')
        score = float(torch.linalg.vector_norm(flat)) if signed else max(0., -float(flat.sum()))
        proposals = []
        for value, gradient in zip(old, grads):
            if signed:
                proposal = -1e-3*gradient/(torch.linalg.vector_norm(flat).clamp_min(1e-30))
            else:
                proposal = value
            proposals.append(proposal.detach())
        candidates.append({'index': index, 'name': name, 'score': score, 'proposals': proposals,
                           'gradient': flat.tolist()})
    ranked = sorted(candidates, key=lambda row: (-row['score'], row['index']))
    chosen = {row['index'] for row in ranked[:3] if row['score'] > 1e-12}
    rows = []
    for row in candidates:
        chosen_here = row['index'] in chosen
        if chosen_here:
            with torch.no_grad():
                for p, proposed in zip(template_params(model, row['index']), row['proposals']):
                    p.copy_(proposed)
            model.set_active(row['index'], True)
        rows.append({key: value for key, value in row.items() if key != 'proposals'} | {'activated': chosen_here})
    return rows


def prune(model, panel, optimizer):
    with torch.no_grad(), graph(model):
        baseline = float(risk(autonomous(model, panel, .2)))
    rows = []
    for index, name in enumerate(model.template_names):
        if index in (3, 5) or not bool(model.template_mask[index]):
            continue
        model.set_active(index, False)
        with torch.no_grad(), graph(model):
            ablated = float(risk(autonomous(model, panel, .2)))
        remove = (ablated-baseline)/max(baseline, 1e-30) < .001
        model.set_active(index, not remove)
        if remove:
            for parameter in template_params(model, index):
                parameter.requires_grad_(False)
                optimizer.state.pop(parameter, None)
        rows.append({'name': name, 'baseline_loss': baseline, 'ablated_loss': ablated, 'removed': remove})
        if remove:
            baseline = ablated
    return rows


def horizon(step):
    if step <= 100:
        return 1.
    if step <= 300:
        return (1., 2.)[(step-101) % 2]
    return (1., 2., 5.)[(step-301) % 3]


def run(arm, output, phase='smoke', cap=None):
    start = time.perf_counter()
    output = Path(output); output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1); torch.manual_seed(SEED)
    source_before = sources()
    events, excluded = read_train()
    panel = panel_events(events)
    model = make_model(arm)
    optimizer = optimizer_for(model)
    rng = np.random.default_rng(SEED)
    force_scale = float(np.load(NORM)['force_scale'])
    phase_cap = {'smoke': 180., 'pilot': 180., 'formal': 2400.}[phase]
    if cap is not None:
        if cap > phase_cap:
            raise ValueError('cannot exceed registered phase cap')
        phase_cap = cap
    evidence = {'arm': arm, 'phase': phase, 'test_read': False, 'validation_read': False,
                'source_before': source_before, 'excluded': excluded,
                'git_head': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
                'panel': [{'id': e['id'], 'group': e['group'], 'label': e['label'], 'length': len(e['x'])} for e in panel],
                'train_counts': {'eligible': len(events), 'unstable': 35, 'groups': 24},
                'phase_cap_seconds': phase_cap, 'seed': SEED, 'monitors': {}, 'traces': []}
    step = 0; selected = None; last = None; covered = set(); stop = None
    try:
        evidence['initial_qa'] = check_structure(model)
        with torch.no_grad():
            hist = histories_for([(e, 10) for e in panel])
            q = model.encode(hist)
            z = torch.cat((hist[:, -1], q), -1)
            z0 = torch.cat((hist[:, -1], torch.zeros_like(q)), -1)
            evidence['initial_memory'] = {'q_rms': float(q.square().mean().sqrt()),
                'physical_memory_field_rms': float((model.field(z)[:, :30]-model.field(z0)[:, :30]).square().mean().sqrt())}
            if hasattr(model, 'W'):
                evidence['initial_memory']['v_rms'] = float((q@model.W.T).square().mean().sqrt())
        save_json(output/'started.json', evidence)
        if phase == 'smoke':
            samples, histories = sample_batch(events, rng, 12)
            t0 = time.perf_counter()
            before = {key: value.detach().clone() for key, value in model.state_dict().items()}
            with graph(model):
                loss, terms = objective(model, samples, histories, 1., force_scale)
                loss.backward()
            gradient = {name: float(p.grad.norm()) if p.grad is not None else None
                        for name, p in model.named_parameters() if p.requires_grad}
            if not torch.isfinite(loss) or any(p.grad is not None and not torch.isfinite(p.grad).all()
                                             for p in model.parameters()):
                raise FloatingPointError('nonfinite smoke')
            assert all(torch.equal(value, model.state_dict()[key]) for key, value in before.items())
            evidence.update(smoke_step_seconds=time.perf_counter()-t0, loss=float(loss.detach()),
                            terms={k: float(v.detach()) for k,v in terms.items()}, gradients=gradient,
                            no_optimizer_update=True, status='smoke_complete')
        else:
            last = checkpoint(output, 'step0000', model, optimizer, rng, arm, 0)
            evidence['monitors']['0'] = monitor(model, panel)
            initial_risk = evidence['monitors']['0']['average_risk']
            initial_severe = evidence['monitors']['0']['severe_mean']
            print(json.dumps({'event': 'initial_monitor', 'arm': arm, 'phase': phase,
                              'risk': initial_risk, 'seconds': time.perf_counter()-start}), flush=True)
            pool = panel if phase == 'pilot' else events
            cover_order = rng.permutation(len(pool)); cursor = 0
            maxstep = 60 if phase == 'pilot' else 6000
            for step in range(1, maxstep+1):
                elapsed = time.perf_counter()-start
                # Keep an explicit reserve for final serialization, panel QA
                # and any just-triggered growth/monitor work inside this cap.
                reserve = 30. if phase == 'pilot' else 60.
                if elapsed+reserve >= phase_cap:
                    step -= 1; stop = 'wall_cap'; break
                # Prevent starting a step when the observed cost cannot fit.
                recent = [r['step_seconds'] for r in evidence['traces'][-10:] if 'step_seconds' in r]
                if recent and elapsed+max(recent)*1.5+reserve >= phase_cap:
                    step -= 1; stop = 'wall_cap'; break
                t0 = time.perf_counter()
                samples, _ = sample_batch(pool, rng, 10)
                for _ in range(2):
                    if cursor >= len(pool):
                        cover_order = rng.permutation(len(pool)); cursor = 0
                    event = pool[int(cover_order[cursor])]; cursor += 1
                    endpoint = int(rng.integers(10, len(event['x'])-1))
                    samples.append((event, endpoint))
                histories = histories_for(samples)
                covered.update(e['id'] for e, _ in samples)
                optimizer.zero_grad(set_to_none=True)
                h = (.2, 1.)[(step-1)%2] if phase == 'pilot' else horizon(step)
                with graph(model):
                    loss, terms = objective(model, samples, histories, h, force_scale)
                    if not torch.isfinite(loss):
                        raise FloatingPointError('nonfinite autonomous loss')
                    loss.backward()
                active = [p for p in model.parameters() if p.requires_grad and p.grad is not None]
                if any(not torch.isfinite(p.grad).all() for p in active):
                    raise FloatingPointError('nonfinite gradient')
                norm = float(torch.nn.utils.clip_grad_norm_(active, 100.))
                optimizer.step()
                if any(not torch.isfinite(p).all() for p in model.parameters()):
                    raise FloatingPointError('nonfinite parameters')
                qa = check_structure(model)
                memory = process_memory_bytes()
                if memory > 8*1024**3:
                    raise MemoryError('registered 8 GiB/process cap exceeded')
                row = {'step': step, 'horizon': h, 'loss': float(loss.detach()),
                       'terms': {k: float(v.detach()) for k,v in terms.items()}, 'gradient_norm': norm,
                       'sample_ids': [e['id'] for e, _ in samples], 'endpoints': [i for _,i in samples],
                       'step_seconds': time.perf_counter()-t0, 'elapsed': time.perf_counter()-start,
                       'qa': qa, 'covered_events': len(covered), 'working_set_bytes': memory}
                evidence['traces'].append(row)
                with (output/'trace.jsonl').open('a', encoding='utf-8') as stream:
                    stream.write(json.dumps(row, allow_nan=False)+'\n')
                if step % 10 == 0:
                    print(json.dumps({key: row[key] for key in ('step','loss','elapsed','covered_events')}), flush=True)
                if phase == 'formal' and step in (100, 400):
                    checkpoint(output, f'step{step:04d}_before_growth', model, optimizer, rng, arm, step)
                    proposals = grow(model, panel)
                    save_json(output/f'growth_{step}.json', proposals)
                    checkpoint(output, f'step{step:04d}_after_growth', model, optimizer, rng, arm, step)
                if phase == 'formal' and step == 800:
                    checkpoint(output, 'step0800_before_prune', model, optimizer, rng, arm, step)
                    save_json(output/'prune_800.json', prune(model, panel, optimizer))
                if phase == 'formal' and step % 500 == 0:
                    evidence['monitors'][str(step)] = monitor(model, panel)
                    last = checkpoint(output, f'step{step:04d}', model, optimizer, rng, arm, step)
                    save_json(output/f'monitor_{step}.json', evidence['monitors'][str(step)])
                    if step >= 1000 and all(evidence['monitors'][str(s)]['severe_mean'] > 10*initial_severe
                                           for s in (step-500, step)):
                        stop = 'severe_stop'; break
                    if step in (1000, 3000):
                        keys = (0, 500, 1000) if step == 1000 else (2000, 2500, 3000)
                        risks = [evidence['monitors'][str(s)]['average_risk'] for s in keys]
                        if not all((a-b)/max(a, 1e-30) >= .01 for a,b in zip(risks, risks[1:])):
                            stop = 'extension_not_triggered'; break
            if stop is None:
                stop = 'step_cap'
            last = checkpoint(output, f'last_step{step:04d}', model, optimizer, rng, arm, step)
            evidence['fit_and_selection_seconds'] = time.perf_counter()-start
            endpoint_start = time.perf_counter()
            evidence['last_monitor'] = monitor(model, panel)
            evidence['endpoint_assessment_seconds'] = time.perf_counter()-endpoint_start
            eligible = [(int(k), v) for k,v in evidence['monitors'].items() if int(k) in (1000,3000,6000)]
            if phase == 'formal' and eligible:
                best = min(eligible, key=lambda item: (item[1]['average_risk'], item[0]))[0]
                selected = output/f'step{best:04d}'
            else:
                selected = last
            evidence.update(status=stop, steps=step, selected_checkpoint=str(selected),
                            last_checkpoint=str(last), covered_event_ids=sorted(covered),
                            optimization='incomplete_milestone_resource_limited' if phase=='formal' and step<1000
                            else 'evaluated_at_declared_budget')
    except Exception as error:
        torch.save(model.state_dict(), output/'failure_model.pt')
        evidence.update(status='implementation_or_numerical_failure', steps=step,
                        error_type=type(error).__name__, error=str(error))
        raise
    finally:
        evidence['seconds'] = time.perf_counter()-start
        evidence['sources_after'] = sources()
        evidence['sources_unchanged'] = evidence['sources_after'] == source_before
        if not evidence['sources_unchanged']:
            evidence['status_before_quarantine'] = evidence.get('status')
            evidence['status'] = 'quarantined_source_drift'
        evidence['within_phase_cap'] = evidence['seconds'] <= phase_cap
        if not evidence['within_phase_cap']:
            evidence['budget_exception'] = 'mandatory endpoint/evidence work crossed soft cap; no extension authorized'
        save_json(output/'evidence.json', evidence)
    return evidence


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--arm', choices=ARMS, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--phase', choices=('smoke','pilot','formal'), default='smoke')
    parser.add_argument('--cap', type=float)
    args = parser.parse_args()
    run(args.arm, args.output, args.phase, args.cap)


if __name__ == '__main__':
    main()
