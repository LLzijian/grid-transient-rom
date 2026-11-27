"""Shared fixed-grid fitting, qualification and scoring for synthetic systems."""

from dataclasses import dataclass
import math
from time import perf_counter

import numpy as np
import torch

from kiki.models.osgkd_fitting import integrate

from .models import edge_features, mask_gradients, project


METRICS = ('theta', 'omega', 'tau_m', 'tau_e')


@dataclass
class Episodes:
    theta_history: torch.Tensor
    omega_history: torch.Tensor
    targets: torch.Tensor

    @classmethod
    def from_events(cls, events):
        def tensor(values):
            return torch.tensor(np.stack(values), dtype=torch.float64)
        return cls(tensor([e['history']['theta'] for e in events]),
                   tensor([e['history']['omega'] for e in events]),
                   tensor([[e['targets'][name] for name in METRICS] for e in events]))

    def __len__(self):
        return len(self.targets)

    def select(self, ids):
        return Episodes(self.theta_history[ids], self.omega_history[ids], self.targets[ids])


def rollout(model, initial, steps, dt):
    if dt != .01:
        return integrate(model, initial, steps, internal_step=dt)
    outputs = [initial]
    for _ in range(steps):
        initial = model.rk4_step(initial, dt)
        outputs.append(initial)
    return torch.stack(outputs, dim=-2)


def trajectory(model, events, steps, dt):
    initial = model.initialize(events.theta_history, events.omega_history)
    return rollout(model, initial, steps, dt)


def residuals(model, path, events):
    states = path[:, 1:]
    readout = model.ports(states)
    predicted = torch.stack((states[..., :model.n], states[..., model.n:2 * model.n],
                             readout.tau_m, readout.tau_e), dim=1)
    return predicted - events.targets[:, :, 1:path.shape[1]]


def motion_objective(model, path, events, weights):
    steps = path.shape[1] - 1
    theta, omega = path[:, 1:, :2 * model.n].split(model.n, -1)
    sq = (torch.stack((theta, omega), dim=1) - events.targets[:, :2, 1:steps + 1]).square()
    active = [(a, min(steps, b), weight) for a, b, weight in weights if a < steps]
    return sum(weight * sq[:, :, a:b].mean() for a, b, weight in active) / sum(w for _, _, w in active)


def objective(model, path, events, weights):
    return (residuals(model, path, events).square().mean() if weights is None else
            motion_objective(model, path, events, weights))


def loss(model, events, steps, dt, weights=None):
    return objective(model, trajectory(model, events, steps, dt), events, weights)


def plan(n, batch_size, attempts, seed):
    if n < batch_size or n % batch_size:
        raise ValueError('whole batches required to preserve exact event balancing')
    rng = np.random.default_rng(np.random.SeedSequence([seed, 27322]))
    order = [rng.permutation(n).reshape(-1, batch_size) for _ in range(math.ceil(attempts * batch_size / n))]
    return np.concatenate(order)[:attempts].tolist()


def fit_step(model, optimizer, events, steps, dt, microbatch, masks, weights=None):
    optimizer.zero_grad(set_to_none=True)
    total = 0.
    components = events.targets.new_zeros(len(METRICS)) if weights is None else None
    for start in range(0, len(events), microbatch):
        batch = events.select(slice(start, start + microbatch))
        path = trajectory(model, batch, steps, dt)
        weight = len(batch) / len(events)
        if weights is None:
            sq = residuals(model, path, batch).square()
            value = sq.mean()
            components += weight * sq.detach().mean(dim=(0, 2, 3))
        else:
            value = motion_objective(model, path, batch, weights)
        if not bool(value.detach().isfinite()):
            raise FloatingPointError('nonfinite trajectory objective')
        total += float(value.detach()) * weight
        (value * weight).backward()
    mask_gradients(model, masks)
    norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 5.))
    if not math.isfinite(norm):
        raise FloatingPointError('nonfinite gradient')
    optimizer.step()
    project(model, masks)
    adam = (value for state in optimizer.state.values() for value in state.values() if isinstance(value, torch.Tensor))
    if not (all(p.isfinite().all() for p in model.parameters()) and all(v.isfinite().all() for v in adam)):
        raise FloatingPointError('nonfinite projected parameter or Adam state')
    return total, norm, dict(zip(METRICS, components.tolist(), strict=True)) if components is not None else None


def optimizer(model):
    encoder = list(model.encoder.parameters())
    ids = {id(p) for p in encoder}
    groups = ([p for p in encoder if p.requires_grad],
              [p for p in model.parameters() if p.requires_grad and id(p) not in ids])
    return torch.optim.Adam([dict(params=group, lr=lr) for group, lr in zip(groups, (.003, .01), strict=True)])


def set_schedule(optimizer, step, attempts):
    multiplier = .2 + .4 * (1 + math.cos(math.pi * step / max(1, attempts - 1)))
    for group, lr in zip(optimizer.param_groups, (.003, .01), strict=True):
        group['lr'] = lr * multiplier


def qualification(model, events, steps, dt=.01, fine=.005, weights=None, masks=None):
    """Test full registered horizon on one fixed batch, without parameters changing."""
    paths, gradients, losses = [], [], []
    for mesh in (dt, fine):
        model.zero_grad(set_to_none=True)
        path = trajectory(model, events, steps, mesh)
        value = objective(model, path, events, weights)
        if not (bool(path.detach().isfinite().all()) and bool(value.detach().isfinite())):
            model.zero_grad(set_to_none=True)
            return {'passed': False, 'nonfinite_mesh': mesh}
        value.backward()
        mask_gradients(model, masks or {})
        gradients.append(torch.cat([(p.grad if p.grad is not None else torch.zeros_like(p)).flatten()
                                    for p in model.parameters() if p.requires_grad]))
        losses.append(float(value.detach()))
        paths.append(path.detach())
        del path, value
    model.zero_grad(set_to_none=True)
    path_gap = float((paths[0] - paths[1]).abs().max())
    g, g_fine = gradients
    if not (g.isfinite().all() & g_fine.isfinite().all()):
        return {'passed': False, 'reason': 'nonfinite_gradient', 'state_max_gap': path_gap}
    grad_gap = float((g - g_fine).norm())
    scale = float(g_fine.norm())
    rel = grad_gap / max(scale, 1e-8)
    rel_loss = abs(losses[0] - losses[1]) / max(abs(losses[1]), 1e-8)
    return {'passed': path_gap <= 1e-3 and rel <= .05 and rel_loss <= .02,
            'state_max_gap': path_gap, 'gradient_norm': float(g.norm()),
            'gradient_relative_gap': rel, 'loss_relative_gap': rel_loss, 'main': dt, 'fine': fine}


@torch.no_grad()
def function_rms(model, events, steps, batch_size):
    if not hasattr(model, 'c'):
        return None
    sums = model.c.new_zeros((2, model.c.numel()))
    count = 0
    for start in range(0, len(events), batch_size):
        batch = events.select(slice(start, start + batch_size))
        state = trajectory(model, batch, steps, .01)[:, 1:]
        phase, envelope = edge_features(model, state[..., :model.n], state[..., 3*model.n:4*model.n])
        sums[0] += (phase * model.c).square().sum(dim=(0, 1))
        if hasattr(model, 'beta'):
            sums[1] += (envelope * model.beta).square().sum(dim=(0, 1))
        count += len(batch) * steps
    rms = (sums / count).sqrt()
    if not rms.isfinite().all():
        raise FloatingPointError('nonfinite train-only edge function')
    return rms


@torch.inference_mode()
def score_events(model, events, windows, batch_size, dt=.005, fine=.0025, event_ids=None):
    """Both meshes from the same causal input; timers include initialization and RK4 only."""
    weight = model.mass.diag() / model.mass.diag().sum()
    event_ids = list(range(len(events))) if event_ids is None else event_ids
    if len(event_ids) != len(events) or len(set(event_ids)) != len(event_ids):
        raise ValueError('one unique ID per held-out event required')
    rows, timing = [], 0.
    for start in range(0, len(events), batch_size):
        batch = events.select(slice(start, start + batch_size))
        begin = perf_counter()
        coarse = trajectory(model, batch, max(windows), dt)
        timing += perf_counter() - begin
        reference = trajectory(model, batch, max(windows), fine)
        res = residuals(model, coarse, batch)
        finer = residuals(model, reference, batch)
        sq, sq_fine = res.square(), finer.square()
        ok = (sq.isfinite().all(dim=(1, 2, 3)) & sq_fine.isfinite().all(dim=(1, 2, 3)) &
              coarse.isfinite().all(dim=(1, 2)) & reference.isfinite().all(dim=(1, 2)))
        rmse = sq[:, :2].mean(dim=(1, 2, 3)).sqrt()
        error_gap = (rmse - sq_fine[:, :2].mean(dim=(1, 2, 3)).sqrt()).abs()
        path_gap = (coarse[..., :2*model.n] - reference[..., :2*model.n]).square().mean(dim=(1, 2)).sqrt()
        ok &= rmse.isfinite() & path_gap.isfinite() & error_gap.isfinite()
        error_limit = torch.maximum(rmse * .01, rmse.new_tensor(1e-4))
        accepted = ok & (path_gap <= 1e-3) & (error_gap <= error_limit)
        for i in range(len(batch)):
            finite = bool(ok[i])
            measures = {}
            if finite:
                for step in windows:
                    mse = sq[i, :, :step].mean(dim=(1, 2)).tolist()
                    measures[str(step)] = dict(mse=dict(zip(METRICS, mse, strict=True)),
                                               common_speed_mse=float((res[i, 1, :step] @ weight).square().mean()))
            rows.append(dict(id=event_ids[start+i], finite=finite, qualified=bool(accepted[i]), scopes=measures,
                             mesh_motion_rms=float(path_gap[i]) if finite else None,
                             mesh_rmse_gap=float(error_gap[i]) if finite else None,
                             mesh_rmse_limit=float(error_limit[i]) if finite else None))
    passed = [row for row in rows if row['qualified']]
    coverage = len(passed)
    output = {}
    for step in windows:
        if not coverage:
            output[str(step)] = None
            continue
        measures = [row['scopes'][str(step)] for row in passed]
        sq = torch.tensor([[item['mse'][name] for name in METRICS] for item in measures], dtype=torch.float64)
        risk = sq[:, :2].mean(dim=1)
        output[str(step)] = {'rmse': dict(zip(METRICS, sq.mean(dim=0).sqrt().tolist(), strict=True)),
                             'risk_mean': float(risk.mean()),
                             'risk_mean_tail': float(.8 * risk.mean() + .2 * risk.topk(math.ceil(.1 * coverage)).values.mean()),
                             'common_speed_rmse': math.sqrt(sum(m['common_speed_mse'] for m in measures) / coverage)}
    qualified = coverage == len(events)
    return dict(coverage=coverage, raw_count=len(events), whole_window_qualified=qualified,
                windows=output if qualified else dict.fromkeys(output), covered_only_windows=output,
                max_mesh_motion_rms=max((r['mesh_motion_rms'] for r in rows if r['finite']), default=None),
                inference_wall_s=timing, inference_per_event_s=timing/len(events), main_dt=dt, fine_dt=fine, rows=rows)
