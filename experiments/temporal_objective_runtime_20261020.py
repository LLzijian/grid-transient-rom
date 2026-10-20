"""Registered true-interval temporal objective; original equations stay intact."""
from __future__ import annotations

import math
import torch

from closure_training_runtime import rollout_errors
from train_local_memory_0098 import graph, risk


def temporal_rollout_errors(model, prepared, initial=None):
    """Return per-event (half mean + half tail, mean, tail) on genuine points.

    Segment masks use global future indices, including tails spanning segments.
    Every event is encoded once and its state survives each ragged boundary.
    All masks/counts are prepared before stepping; no host reads inside RK4.
    """
    from train_structured_rom import _point_error
    h, counts = prepared.history, prepared.counts
    if h.ndim != 3 or h.shape[1:] != (11, 30) or not len(h):
        raise ValueError('nonempty 11x30 causal histories required')
    if counts.ndim != 1 or len(counts) != len(h) or counts.device != h.device:
        raise ValueError('counts must match histories and device')
    if not bool((torch.isfinite(counts) & (counts >= 1) & (counts == counts.floor())).all()):
        raise ValueError('positive integral genuine counts required')
    n = counts.to(torch.long)
    m = (n + 4) // 5
    segments = []
    coverage = torch.zeros_like(n)
    start = 0
    for ix, truth in prepared.segments:
        if ix.ndim != 1 or ix.dtype != torch.long or truth.ndim != 3 or truth.shape[1:] != (len(ix), 30) or truth.shape[0] < 1:
            raise ValueError('invalid genuine rollout segment')
        if ix.device != h.device or truth.device != h.device or not bool(((ix >= 0) & (ix < len(h))).all()):
            raise ValueError('segment indices/targets must match the histories')
        expected = torch.nonzero(n >= start + truth.shape[0], as_tuple=False).flatten()
        if not torch.equal(ix, expected):
            raise ValueError('segments must retain exactly the ascending active rows for their genuine interval')
        steps = torch.arange(start + 1, start + 1 + truth.shape[0], device=h.device)
        mask = (steps[:, None] > (n - m).index_select(0, ix)[None, :]).to(h.dtype)
        segments.append((ix, truth, mask))
        coverage = coverage.index_add(0, ix, torch.full_like(ix, truth.shape[0]))
        start += truth.shape[0]
    if not segments or not torch.equal(coverage, n):
        raise ValueError('genuine future segments required')
    state = torch.cat((h[:, -1], model.encode(h)), -1) if initial is None else initial
    sums, tail_sums = h.new_zeros(len(h)), h.new_zeros(len(h))
    for ix, truth, mask in segments:
        current = state.index_select(0, ix)
        subtotal, tail = h.new_zeros(len(ix)), h.new_zeros(len(ix))
        for target, tail_weight in zip(truth.unbind(0), mask.unbind(0)):
            current = model.rk4(current, .01)
            error = _point_error(current, target, model)
            subtotal = subtotal + error
            tail = tail + error * tail_weight
        state = state.index_copy(0, ix, current)
        sums = sums.index_add(0, ix, subtotal)
        tail_sums = tail_sums.index_add(0, ix, tail)
    time_mean, tail_mean = sums / counts, tail_sums / m
    return .5 * time_mean + .5 * tail_mean, time_mean, tail_mean


def objective(model, store, samples, horizon, force, tail_weighted=False):
    if not isinstance(tail_weighted, bool):
        raise ValueError('tail_weighted must be an explicit bool from the frozen phase schedule')
    if not math.isfinite(horizon) or horizon <= 0 or not math.isfinite(force) or force <= 0:
        raise ValueError('positive finite horizon and force required')
    long_batch = store.prepare([(e, 10) for e, _ in samples], horizon)
    local_batch = store.prepare(samples, .2)
    weighted, time_mean, tail_mean = temporal_rollout_errors(model, long_batch)
    long = weighted if tail_weighted else time_mean
    history = local_batch.history
    state = torch.cat((history[:, -1], model.encode(history)), -1)
    local = rollout_errors(model, local_batch, initial=state).mean()
    adjacent = store.histories(samples, adjacent=True)
    latent = ((model.rk4(state, .01)[:, 30:] - model.encode(adjacent)) / .1).square().mean()
    predicted = state[:, 20:30] - model.mass * model.field(state)[:, 10:20]
    target = torch.stack([store.events[e['id']][1][i] for e, i in samples])
    torque = ((predicted - target) / force).square().mean()
    terms = {'risk': risk(long), 'mean': long.mean(), 'local': local,
             'latent_consistency': latent, 'torque': torque, 'complexity': model.complexity(state),
             'time_mean': time_mean.mean(), 'tail_mean': tail_mean.mean()}
    return 10 * terms['risk'] + local + .1 * latent + .1 * torque, terms


def _tensors(value):
    if torch.is_tensor(value):
        yield value
    elif isinstance(value, dict):
        for v in value.values(): yield from _tensors(v)
    elif isinstance(value, (tuple, list)):
        for v in value: yield from _tensors(v)


def _finite(value):
    # Adam step scalars can remain on CPU; checks occur outside integration.
    return all(bool(torch.isfinite(t).all()) for t in _tensors(value))


def one_update(model, opt, store, samples, horizon, force, tail_weighted=False):
    opt.zero_grad(set_to_none=True)
    try:
        with graph(model):
            value, components = objective(model, store, samples, horizon, force, tail_weighted)
            if not _finite([value, components]):
                raise FloatingPointError('nonfinite complete objective')
            value.backward()
        params = [p for p in model.parameters() if p.requires_grad]
        if not _finite([p.grad for p in params]):
            raise FloatingPointError('nonfinite complete objective gradient')
        norm = torch.nn.utils.clip_grad_norm_(params, 100.)
        if not _finite(norm):
            raise FloatingPointError('nonfinite clipped gradient norm')
        opt.step()
        model.end_graph_cache()
        if not _finite([model.state_dict(), opt.state_dict()]):
            raise FloatingPointError('nonfinite Adam update/state')
        return {'loss': float(value.detach()), 'gradient_norm': float(norm.detach()),
                'components': {k: float(v.detach()) for k, v in components.items()}}
    finally:
        model.end_graph_cache()
