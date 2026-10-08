"""Loop0058: event-tail-aware joint sparse electrical-state discovery."""
import argparse
import csv
import hashlib
import json
import subprocess
import time
from pathlib import Path

import h5py
import numpy as np
import torch
from run_private_m2 import aggregate
from torch import nn

from kiki.data.private_dataset import canonical_arrays
from kiki.models.private_pm import MechanicalROM
from kiki.physics import coi_project

DT = .01
SCALES = np.array([.47675314588683515, 1.0033667249602924])
ROOTS = [Path('private_data/ieee39_postfault'), Path('private_data/ieee39_expansion')]
BASE = Path('private_data/task2_m3_0020/model.npz')
NETWORK = Path('private_data/task2_electrical_ports_0048/physical_network.npz')


def tensor(x):
    return torch.as_tensor(x, dtype=torch.float64)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_development(base, network, full_train=False):
    """Read only explicit train/validation IDs, never enumerate TEST payloads."""
    events, excluded, identities = {'train': [], 'validation': []}, {}, {}
    for root in ROOTS:
        splits = json.loads((root/'splits.json').read_text())
        assert not set(splits['train']) & set(splits['validation'])
        assert not (set(splits['train']) | set(splits['validation'])) & set(splits['test'])
        identities[str(root)] = {name: digest(root/name) for name in ('dataset_index.csv', 'splits.json')}
        with (root/'dataset_index.csv').open() as stream:
            rows = {row['event_id']: row for row in csv.DictReader(stream)}
        with h5py.File(root/'dataset.h5', 'r') as file:
            eq = {key: file['equilibrium'][key][:] for key in ('delta', 'omega', 'Pm', 'Pe')}
            names = json.loads(file.attrs['algebraic_names'])
            positions = [names.index(f'v Bus {bus}') for bus in network['bus_ids']]
            voltage_star = file['equilibrium/algebraic_state'][:][positions][network['observed_buses']]
            if 'voltage_star' in network:
                assert np.array_equal(voltage_star, network['voltage_star'])
            network['voltage_star'] = voltage_star
            for split, selected in events.items():
                for eid in splits[split]:
                    if float(rows[eid]['trajectory_end_s']) <= .1+1e-9:
                        excluded[eid] = split
                        continue
                    raw = file['events'][eid]
                    # The historical 811-point cap is retained by default.  The
                    # long-horizon campaign explicitly opts into the complete
                    # stored TRAIN trajectory; validation remains capped.
                    cap = None if (full_train and split == 'train') else 811
                    arrays = {key: raw[key][:cap] for key in ('t', 'delta', 'omega', 'Pm', 'Pe')}
                    e = canonical_arrays(arrays, eq, file['inertia'][:], file.attrs['omega_s'])
                    state = np.column_stack((e['delta_sync'], e['omega_sync'], e['Pm_raw']-base.pm_star))
                    eta = (state[:, :10]+base.delta_star)@base.incidence
                    electrical = ((np.sin(eta)-np.sin(base.delta_star@base.incidence))*base.coupling)@base.incidence.T
                    # Exact M3-minus-FOM force, to be SUBTRACTED from swing force.
                    force = raw['te'][:cap]-base.pm_star-electrical-state[:, 10:20]*base.damping
                    voltage = raw['bus_v'][:cap][:, network['load_buses']]
                    load = voltage**2@network['load_g']-network['equilibrium_components'][0]
                    observed_v = raw['bus_v'][:cap][:, network['observed_buses']]
                    z = np.column_stack((arrays['Pe']-eq['Pe'], np.abs(observed_v)-voltage_star))
                    assert len(state) > 11 and np.isfinite(force).all()
                    assert np.allclose(np.diff(e['t']), DT, atol=1e-9)
                    selected.append({'id': eid, 'group': rows[eid]['fault_bus'],
                        'label': rows[eid]['label'], 'x': state, 'force': force, 'load': load, 'z': z})
    assert not {e['group'] for e in events['train']} & {e['group'] for e in events['validation']}
    return events, excluded, identities


class JointROM(nn.Module):
    def __init__(self, base, ports, input_scale, feature_scale, force_scale, load_scale, electrical=None, latent_dim=1):
        super().__init__()
        self.rank = latent_dim
        for name in ('mass', 'incidence', 'delta_star', 'coupling', 'damping', 'rate', 'gain'):
            self.register_buffer(name, tensor(getattr(base, name)))
        active = np.flatnonzero(base.coupling > 0)
        self.register_buffer('angle_incidence', tensor(base.incidence[:, active]))
        self.register_buffer('ports', tensor(ports))
        self.register_buffer('input_scale', tensor(input_scale))
        self.register_buffer('feature_scale', tensor(feature_scale))
        self.register_buffer('force_scale', tensor(force_scale))
        self.register_buffer('load_scale', tensor(load_scale))
        self.register_buffer('metric_scale', tensor(SCALES))
        self.gru = nn.GRU(20, 8, batch_first=True).double()
        self.head = nn.Linear(8, self.rank).double()
        self.register_buffer('q_center', torch.zeros(self.rank, dtype=torch.float64))
        self.register_buffer('q_transform', torch.eye(self.rank, dtype=torch.float64))
        self.c = nn.Parameter(torch.randn(ports.shape[1], self.rank, dtype=torch.float64))
        self.amplitude = nn.Parameter(tensor(np.full(self.rank, force_scale*np.sqrt(10))))
        self.xi = nn.Parameter(torch.zeros(73+self.rank, self.rank, dtype=torch.float64))
        self.register_buffer('support', torch.ones_like(self.xi))
        self.alpha = nn.Parameter(torch.zeros(52, self.rank, dtype=torch.float64))
        self.register_buffer('alpha_support', torch.ones_like(self.alpha))
        self.register_buffer('memory_enabled', torch.ones(self.rank, dtype=torch.float64))
        if electrical is None:  # Synthetic unit fixtures only.
            electrical = (np.ones(20), np.ones(10), np.arange(10), np.ones(10), 10.)
        z_scale, voltage_star, load_positions, load_g, load_star = electrical
        self.register_buffer('z_scale', tensor(z_scale))
        self.register_buffer('voltage_star', tensor(voltage_star))
        self.register_buffer('load_positions', torch.as_tensor(load_positions, dtype=torch.long))
        self.register_buffer('load_g', tensor(load_g))
        self.register_buffer('load_star', tensor(load_star))
        self.decoder = nn.Linear(10+self.rank, len(z_scale)).double()

    def coi(self, x):
        return x-(x*self.mass).sum(dim=-1, keepdim=True)/self.mass.sum()

    def observations(self, state):
        return torch.cat((self.coi(state[..., :10]), state[..., 10:20]), dim=-1)/self.input_scale

    def raw_encode(self, history):
        assert history.shape[-2:] == (11, 30)
        return self.head(self.gru(self.observations(history))[0][:, -1])

    def encode(self, history):
        return (self.raw_encode(history)-self.q_center)@self.q_transform

    def condition_coordinates(self, reference, weights):
        raw = self.raw_encode(reference)
        mean = (raw*weights[:, None]).sum(dim=0)
        centered = raw-mean
        covariance = centered.T@(centered*weights[:, None])
        eye = torch.eye(self.rank, dtype=raw.dtype)
        lower = torch.linalg.cholesky(covariance+1e-6*eye)
        self.q_center = mean
        self.q_transform = torch.linalg.solve_triangular(lower.T, eye, upper=True)

    def spatial(self):
        # Normalize a direction, not the evolving physical state.
        return self.ports@(self.c/torch.linalg.vector_norm(self.c, dim=0, keepdim=True))

    def memory(self, q):
        return (self.amplitude*q)@self.spatial().T

    def algebraic_coordinate(self, state):
        return self.features(state)[..., 21+self.rank:]@(self.alpha*self.alpha_support)

    def features(self, state):
        d, w, p, q = state[..., :10], state[..., 10:20], state[..., 20:30], state[..., 30:]
        phase = self.delta_star@self.angle_incidence
        eta = d@self.angle_incidence+phase
        physical = torch.cat((w, p, torch.sin(eta)-torch.sin(phase),
                              torch.cos(eta)-torch.cos(phase)), dim=-1)/self.feature_scale
        return torch.cat((torch.ones_like(q[..., :1]), q, physical), dim=-1)

    def field(self, state):
        d, w, p, q = state[..., :10], state[..., 10:20], state[..., 20:30], state[..., 30:]
        phase = self.delta_star@self.incidence
        e0 = ((torch.sin(d@self.incidence+phase)-torch.sin(phase))*self.coupling)@self.incidence.T
        action = self.memory(self.memory_enabled*q+self.algebraic_coordinate(state))
        return torch.cat((w, (p-e0-self.damping*w-action)/self.mass,
                          -self.rate*p-self.gain*w, self.features(state)@(self.xi*self.support)), dim=-1)

    def rk4(self, x, dt):
        a = self.field(x)
        b = self.field(x+dt/2*a)
        c = self.field(x+dt/2*b)
        d = self.field(x+dt*c)
        return x+dt/6*(a+2*b+2*c+d)

    def rollout(self, history, count, dt):
        state = torch.cat((history[:, -1], self.encode(history)), dim=-1)
        result = [state]
        for _ in range(count):
            state = self.rk4(state, dt)
            result.append(state)
        return torch.stack(result, dim=1)

    def electrical_output(self, state, q):
        raw = self.decoder(torch.cat((self.coi(state[..., :10])/self.input_scale[:10], q), dim=-1))
        voltage = torch.abs(self.voltage_star+raw[..., 10:]*self.z_scale[10:])
        z = torch.cat((raw[..., :10], (voltage-self.voltage_star)/self.z_scale[10:]), dim=-1)
        load = (voltage[..., self.load_positions].square()@self.load_g-self.load_star)/self.load_scale
        return z, load.unsqueeze(-1)


def train_scales(events, base):
    obs, feat, force, load, electrical = [], [], [], [], []
    b = base.incidence[:, base.coupling > 0]
    phase = base.delta_star@b
    for e in events:
        x = e['x'][10:]
        obs.append(np.mean(np.column_stack((coi_project(x[:, :10], base.mass), x[:, 10:20]))**2, axis=0))
        eta = x[:, :10]@b+phase
        f = np.column_stack((x[:, 10:30], np.sin(eta)-np.sin(phase), np.cos(eta)-np.cos(phase)))
        feat.append(np.mean(f*f, axis=0))
        force.append(np.mean(e['force'][10:]**2))
        load.append(np.mean(e['load'][10:]**2))
        electrical.append(np.mean(e['z'][10:]**2, axis=0))
    return (np.maximum(np.sqrt(np.mean(obs, axis=0)), .01),
            np.maximum(np.sqrt(np.mean(feat, axis=0)), .01),
            float(np.sqrt(np.mean(force))), float(np.sqrt(np.mean(load))),
            np.maximum(np.sqrt(np.mean(electrical, axis=0)), 1e-4))


def batch(events, rng, count=32):
    samples = []
    for i in range(count):
        e = events[int(rng.integers(len(events)))]
        endpoint = 10 if i % 2 == 0 else int(rng.integers(10, min(700, len(e['x'])-2)+1))
        samples.append((e, endpoint))
    history = tensor(np.stack([e['x'][i-10:i+1] for e, i in samples]))
    adjacent = tensor(np.stack([e['x'][i-9:i+2] for e, i in samples]))
    force = tensor(np.stack([e['force'][i] for e, i in samples]))
    load = tensor(np.array([e['load'][i] for e, i in samples])[:, None])
    electrical = tensor(np.stack([e['z'][i] for e, i in samples]))
    return samples, history, adjacent, force, load, electrical


def training_reference(events):
    histories, weights = [], []
    for e in events:
        last = min(110, len(e['x'])-1)
        indices = np.unique(np.linspace(10, last, min(11, last-9), dtype=int))
        histories.extend(e['x'][i-10:i+1] for i in indices)
        weights.extend([1/(len(events)*len(indices))]*len(indices))
    return tensor(np.stack(histories)), tensor(np.array(weights))


def loss_terms(model, samples, history, adjacent, force, load, electrical, horizon):
    q = model.encode(history)
    state = torch.cat((history[:, -1], q), dim=-1)
    dyn = ((model.rk4(state, DT)[:, 30:]-model.encode(adjacent))/.1).square().mean()
    force_loss = ((model.memory(q+model.algebraic_coordinate(state))-force)/model.force_scale).square().mean()
    z, load_prediction = model.electrical_output(history[:, -1], q)
    zerror = z-electrical/model.z_scale
    elec = (zerror[:, :10].square().mean()+zerror[:, 10:].square().mean()+
            (load_prediction-load/model.load_scale).square().mean())/3
    norm = (q.mean(dim=0).square()+(q.var(dim=0, unbiased=False)-1).square()).mean()
    centered = q-q.mean(dim=0)
    cov = centered.T@centered/len(q)
    decorrelation = (cov-torch.diag(torch.diag(cov))).square().mean()
    roll, roll_mean, roll_tail = tensor(0.), tensor(0.), tensor(0.)
    if horizon:
        pred, targets, masks = [], [], []
        indices = [1, *range(2, round(horizon/DT)+1, 2)]
        previous = 0
        for offset in indices:
            state = model.rk4(state, (offset-previous)*DT)
            previous = offset
            pred.append(state[:, :20])
            masks.append([i+offset < len(e['x']) for e, i in samples])
            # Unobserved targets are placeholders strictly excluded by mask.
            targets.append([e['x'][i+offset, :20] if i+offset < len(e['x']) else np.zeros(20) for e, i in samples])
        error = torch.stack(pred, dim=1)-tensor(np.swapaxes(np.array(targets), 0, 1))
        delta = model.coi(error[..., :10])/model.metric_scale[0]
        omega = model.coi(error[..., 10:])/model.metric_scale[1]
        mask = tensor(np.array(masks).T)
        mse = .5*(delta.square().mean(dim=-1)+omega.square().mean(dim=-1))
        event_loss = (mse*mask).sum(dim=1)/mask.sum(dim=1).clamp_min(1)
        roll_mean, roll_tail = event_loss.mean(), event_loss.topk(int(np.ceil(.1*len(event_loss)))).values.mean()
        roll = .8*roll_mean+.2*roll_tail
    return {'roll': roll, 'roll_mean': roll_mean, 'roll_tail': roll_tail, 'dyn': dyn,
            'force': force_loss, 'load': elec, 'norm': norm, 'decorrelation': decorrelation}


def metrics(predictions, events, base):
    rows = []
    for pred, e in zip(predictions, events, strict=True):
        available = min(100, len(e['x'])-11)
        err = pred[1:available+1, :20]-e['x'][11:11+available, :20]
        de = coi_project(err[:, :10], base.mass)
        we = coi_project(err[:, 10:20], base.mass)
        for h in (.2, .5, 1., 'prefix'):
            count = available if h == 'prefix' else round(h/.01)
            if count > available:
                continue
            ds, ws = float(np.sum(de[:count]**2)), float(np.sum(we[:count]**2))
            sse = .5*(ds/SCALES[0]**2+ws/SCALES[1]**2)
            rows.append({'event_id': e['id'], 'group': e['group'], 'label': e['label'],
                'horizon': str(h), 'n': count*10, 'delta_sse': ds, 'omega_sse': ws,
                'normalized_sse': sse, 'rmse': float(np.sqrt(sse/(count*10)))})
    return {h: aggregate([r for r in rows if r['horizon'] == h]) for h in ('0.2', '0.5', '1.0', 'prefix')}, rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--smoke', action='store_true', help='contract execution only, not a scientific fit')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    torch.set_num_threads(1)
    torch.manual_seed(53)
    rng = np.random.default_rng(53)
    base, network = MechanicalROM.load(BASE), dict(np.load(NETWORK))
    network['observed_buses'] = np.unique(np.r_[network['load_buses'], network['generator_buses']])
    network['load_positions'] = np.searchsorted(network['observed_buses'], network['load_buses'])
    events, excluded, identities = read_development(base, network)
    ports = network['port_incidence']
    port_ids = np.flatnonzero(np.any(ports != 0, axis=0))
    scales = train_scales(events['train'], base)
    electrical = (scales[4], network['voltage_star'], network['load_positions'],
                  network['load_g'], network['equilibrium_components'][0])
    model = JointROM(base, ports[:, port_ids], *scales[:4], electrical=electrical, latent_dim=3)
    reference, reference_weights = training_reference(events['train'])
    optimizer = torch.optim.Adam(model.parameters(), lr=.003)
    stages = [('warm', 300, 0.), ('short', 200, .2), ('medium', 200, .5),
              ('one_second', 400, 1.), ('sparse', 200, 1.), ('refit', 200, 1.)]
    losses = []
    print(json.dumps({'eligible': {s: len(v) for s, v in events.items()}, 'excluded': len(excluded),
                      'scales': [np.asarray(s).tolist() for s in scales]}), flush=True)
    for stage, steps, horizon in stages:
        if args.smoke:
            steps = 1
        for step in range(steps):
            optimizer.zero_grad()
            model.condition_coordinates(reference, reference_weights)
            samples = batch(events['train'], rng)
            terms = loss_terms(model, *samples, horizon)
            sparse = .001 if stage == 'sparse' else .0001
            loss = (10*terms['roll']+terms['dyn']+.2*terms['force']+.1*terms['load']+
                    .1*terms['norm']+.01*terms['decorrelation']+sparse*((model.xi*model.support).abs().mean()+
                                            (model.alpha*model.alpha_support).abs().mean()))
            if not torch.isfinite(loss):
                raise FloatingPointError(f'nonfinite loss at {stage}/{step}')
            loss.backward()
            optimizer.step()
            if stage == 'sparse' and (step+1) % 100 == 0:
                with torch.no_grad():
                    for weights, mask, cap in ((model.xi, model.support, 12), (model.alpha, model.alpha_support, 6)):
                        keep = weights.abs() >= .03
                        for column in range(model.rank):
                            order = torch.argsort(weights[:, column].abs(), descending=True)
                            keep[order[cap:], column] = False
                        mask.mul_(keep)
                        weights.mul_(mask)
            if step % 50 == 0 or step+1 == steps:
                row = {'stage': stage, 'step': step+1, 'loss': float(loss.detach()),
                       **{k: float(v.detach()) for k, v in terms.items()},
                       'seconds': time.perf_counter()-started}
                losses.append(row)
                print(json.dumps(row), flush=True)
    with torch.no_grad():
        model.condition_coordinates(reference, reference_weights)
    torch.save(model.state_dict(), args.output/'model.pt')
    np.savez_compressed(args.output/'normalization.npz', input_scale=scales[0], feature_scale=scales[1],
                        force_scale=scales[2], load_scale=scales[3], port_ids=port_ids,
                        z_scale=scales[4], voltage_star=network['voltage_star'],
                        load_positions=network['load_positions'], q_center=model.q_center.numpy(),
                        q_transform=model.q_transform.numpy())
    if args.smoke:
        print('SMOKE ONLY: all six stages executed; no scientific evaluation', flush=True)
        return
    model.eval()
    with torch.no_grad():
        histories = tensor(np.stack([e['x'][:11] for e in events['validation']]))
        fine = model.rollout(histories, 200, .005)[:, ::2].numpy()
        coarse = model.rollout(histories, 100, .01).numpy()
        model.memory_enabled.zero_()
        ablated = model.rollout(histories, 200, .005)[:, ::2].numpy()
        model.memory_enabled.fill_(1.)
        mode_ablations = []
        for coordinate in range(model.rank):
            model.memory_enabled[coordinate] = 0.
            mode_ablations.append(model.rollout(histories, 200, .005)[:, ::2].numpy())
            model.memory_enabled[coordinate] = 1.
        initial = np.stack([e['x'][10] for e in events['validation']])
        initial[:, 20:30] += base.pm_star
        baseline = base.rollout(initial, None, np.arange(101)*DT)
        baseline[:, :, 20:30] -= base.pm_star
        latent, discrepancies, latent_arrays, electrical_rows, latent_cross = [], [], {}, [], []
        for split in ('train', 'validation'):
            for j, e in enumerate(events[split]):
                endpoints = np.arange(10, min(len(e['x']), 111))
                h = tensor(np.stack([e['x'][i-10:i+1] for i in endpoints]))
                encoded = model.encode(h).numpy()
                latent_arrays[e['id']+'_encoded_q'] = encoded
                state_q = tensor(np.column_stack((e['x'][endpoints], encoded)))
                latent_arrays[e['id']+'_algebraic_a'] = model.algebraic_coordinate(state_q).numpy()
                if split == 'train':
                    latent.append([encoded.mean(axis=0), np.mean(encoded**2, axis=0)])
                    latent_cross.append(encoded.T@encoded/len(encoded))
                else:
                    discrepancies.append(np.mean((fine[j, :len(encoded), 30:]-encoded)**2, axis=0))
                    zpred, lpred = model.electrical_output(tensor(e['x'][endpoints]), tensor(encoded))
                    ztrue = e['z'][endpoints]/scales[4]
                    ltrue = e['load'][endpoints]/scales[3]
                    latent_arrays[e['id']+'_electrical_prediction'] = np.column_stack((zpred.numpy(), lpred.numpy()))
                    latent_arrays[e['id']+'_electrical_truth'] = np.column_stack((ztrue, ltrue))
                    error = np.column_stack((zpred.numpy()-ztrue, lpred.numpy().ravel()-ltrue))
                    true = np.column_stack((ztrue, ltrue))
                    electrical_rows.append({'event_id': e['id'], 'error': [float(np.mean(error[:, :10]**2)),
                        float(np.mean(error[:, 10:-1]**2)), float(np.mean(error[:, -1]**2))],
                        'equilibrium_error': [float(np.mean(true[:, :10]**2)),
                        float(np.mean(true[:, 10:-1]**2)), float(np.mean(true[:, -1]**2))]})
    train_mean, second = np.mean(latent, axis=0)
    train_var = second-train_mean**2
    train_cov = np.mean(latent_cross, axis=0)-np.outer(train_mean, train_mean)
    covariance_eigenvalues = np.linalg.eigvalsh(train_cov)
    qgap = np.sqrt(np.mean(discrepancies, axis=0))
    step_error = np.column_stack((coi_project((fine-coarse)[..., :10], base.mass).reshape(-1, 10),
                                 coi_project((fine-coarse)[..., 10:20], base.mass).reshape(-1, 10)))
    step_error = step_error.reshape(len(fine), 101, 20)
    numerical = np.sqrt(.5*np.mean((step_error[..., :10]/SCALES[0])**2+
                                  (step_error[..., 10:20]/SCALES[1])**2, axis=(1, 2)))
    summary, records = metrics(fine, events['validation'], base)
    baseline_summary, baseline_records = metrics(baseline, events['validation'], base)
    ablated_summary, ablated_records = metrics(ablated, events['validation'], base)
    mode_summaries = [metrics(pred, events['validation'], base)[0] for pred in mode_ablations]
    groups = sorted({r['group'] for r in records if r['horizon'] == '1.0'})
    group_results = {g: {name: float(np.mean([r['rmse'] for r in rows if r['group'] == g and r['horizon'] == '1.0']))
                         for name, rows in [('joint', records), ('M3', baseline_records)]} for g in groups}
    gates = {'mean': summary['1.0']['mean'] <= .95*baseline_summary['1.0']['mean'],
             'pooled': summary['1.0']['pooled'] <= .95*baseline_summary['1.0']['pooled'],
             'tail': summary['1.0']['cvar10'] <= 1.05*baseline_summary['1.0']['cvar10'],
             'groups': sum(v['joint'] < v['M3'] for v in group_results.values()) > len(groups)/2,
             'latent': bool(np.all((train_var >= .1) & (train_var <= 10) & (abs(train_mean) <= .5))),
             'support': bool(torch.all((model.support.sum(dim=0) >= 1) & (model.support.sum(dim=0) <= 12))),
             'latent_consistency': bool(np.all(qgap <= np.sqrt(train_var))),
             'finite': bool(np.isfinite(fine).all()), 'numerical': float(numerical.max()) < .001}
    electrical_error = np.mean([r['error'] for r in electrical_rows], axis=0)
    equilibrium_error = np.mean([r['equilibrium_error'] for r in electrical_rows], axis=0)
    singular_values = np.linalg.svd((model.spatial()*model.amplitude).detach().numpy(), compute_uv=False)
    gates.update({'algebraic_support': bool(torch.all((model.alpha_support.sum(dim=0) >= 1) & (model.alpha_support.sum(dim=0) <= 6))),
        'improve_lower_rank_mean': summary['1.0']['mean'] <= .95*.3838849635678801,
        'improve_lower_rank_pooled': summary['1.0']['pooled'] <= .95*.5215964842500675,
        'rank_one_tail': summary['1.0']['cvar10'] <= 1.05*1.0985225742969793,
        'independent_force_modes': singular_values[-1]/singular_values[0] >= .1,
        'independent_latent_coordinates': covariance_eigenvalues[0]/covariance_eigenvalues[-1] >= .05,
        'preserve_best_mean': summary['1.0']['mean'] <= 1.02*.3059317619353877,
        'preserve_best_pooled': summary['1.0']['pooled'] <= 1.02*.40097654416431927,
        'preserve_best_tail': summary['1.0']['cvar10'] <= 1.05*.8245004711545303,
        'each_mode_contributes': all(summary['1.0'][metric] <= .99*arm['1.0'][metric]
                                    for arm in mode_summaries for metric in ('mean', 'pooled')),
        'memory_mean': summary['1.0']['mean'] <= .98*ablated_summary['1.0']['mean'],
        'memory_pooled': summary['1.0']['pooled'] <= .98*ablated_summary['1.0']['pooled'],
        'electrical': bool(np.all(electrical_error < equilibrium_error))})
    arrays = {**latent_arrays, 'predictions': fine, 'coarse_predictions': coarse, 'M3': baseline, 'memory_off': ablated,
              'event_ids': np.array([e['id'] for e in events['validation']])}
    for e in events['validation']:
        arrays[e['id']+'_truth'] = e['x'][:111]
    arrays['individual_mode_off'] = np.stack(mode_ablations)
    np.savez_compressed(args.output/'trajectories.npz', **arrays)
    result = {'code_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
        'baseline_sha256': digest(BASE), 'network_sha256': digest(NETWORK), 'data': identities,
        'model_sha256': digest(args.output/'model.pt'),
        'torch': torch.__version__, 'numpy': np.__version__, 'seed': 53,
        'seconds': time.perf_counter()-started, 'eligible': {s: len(v) for s, v in events.items()},
        'excluded_predecision': excluded, 'history_s': .1, 'test_read': False,
        'losses': losses, 'summary': summary, 'baseline_summary': baseline_summary,
        'records': records, 'baseline_records': baseline_records, 'group_results': group_results,
        'ablated_summary': ablated_summary, 'ablated_records': ablated_records,
        'electrical_rows': electrical_rows, 'electrical_error': electrical_error.tolist(),
        'equilibrium_electrical_error': equilibrium_error.tolist(),
        'unstable_prefix': {name: aggregate([r for r in rows if r['horizon'] == 'prefix' and r['label'] == 'unstable'])
                            for name, rows in [('joint', records), ('M3', baseline_records)]},
        'latent_dimension': model.rank, 'q_mean': train_mean.tolist(), 'q_variance': train_var.tolist(),
        'reference_history_count': len(reference), 'reference_train_events': len(events['train']),
        'individual_mode_off_summary': mode_summaries,
        'q_covariance': train_cov.tolist(), 'q_covariance_eigenvalues': covariance_eigenvalues.tolist(),
        'q_autonomous_encoder_rms': qgap.tolist(), 'force_singular_values': singular_values.tolist(),
        'numerical_max_event_rmse': float(numerical.max()), 'gates': {k: bool(v) for k, v in gates.items()},
        'screen_pass': bool(all(gates.values())), 'xi': (model.xi*model.support).detach().tolist(),
        'C': (model.c/torch.linalg.vector_norm(model.c, dim=0, keepdim=True)).detach().tolist(),
        'alpha': (model.alpha*model.alpha_support).detach().tolist(),
        'amplitude': model.amplitude.detach().tolist(), 'scope': '1s development prototype only; no stability/M4 acceptance'}
    (args.output/'evidence.json').write_text(json.dumps(result, indent=2, allow_nan=False), encoding='utf-8')
    print(json.dumps({k: v for k, v in result.items() if k not in ('records', 'baseline_records', 'ablated_records',
                                                                 'electrical_rows', 'data', 'losses')}, indent=2), flush=True)


if __name__ == '__main__':
    main()
