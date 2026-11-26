"""Prospective data-only restricted refits; no loading, release, or scheduling."""

import math

import torch

from experiments.osgkd_v2_discovery import prepare
from experiments.osgkd_v2_refits.optimizer import VARIANTS, new_optimizer
from kiki.models.osgkd_restricted import RestrictedOSGKDGraph


ATTEMPTS = 128
HORIZON = 800
INTERNAL_STEP = .01
previous = prepare.fit


def new_model(parent, spec):
    """Copy the supplied frozen parent and impose its registered restriction."""
    return RestrictedOSGKDGraph.from_parent(parent, spec).train()


def qualify(model, first_two_events, constants, work):
    """Qualify data-only gradients on the first batch's first stable/last unstable.

    The worker selects these two tensor events from the frozen batch plan;
    tensor_events intentionally carries no scientific label metadata.
    """
    if len(first_two_events) != 2 or first_two_events[0]['id'] == first_two_events[1]['id']:
        raise ValueError('two distinct fixed qualification events required')
    result = previous.guard(model, first_two_events, constants, work, limits=(HORIZON,))
    return {**result, 'event_ids': [event['id'] for event in first_two_events],
            'official_updates': 0, 'objective': 'data_only', 'spectral_jacobian_calls': 0}


def _number(value):
    value = float(value.detach())
    return value if math.isfinite(value) else None


def _loss_row(comp, loss, batch):
    means = comp.detach().mean(0)
    return {'event_ids': [event['id'] for event in batch], 'horizon_s': 8.,
            'internal_step': INTERNAL_STEP, 'loss': _number(loss),
            'motion': _number(means[0]), 'port': _number(means[1]),
            'latent': _number(means[2])}


def attempt(model, opt, batch, k, constants, work, multipliers):
    """One fixed-8s data-only update, indexed 0..127, with external LR course."""
    if type(k) is not int or not 0 <= k < ATTEMPTS:
        raise ValueError('attempt index must be in 0..127')
    if len(multipliers) != ATTEMPTS or any(
            not math.isfinite(float(value)) or float(value) < 0 for value in multipliers):
        raise ValueError('128 finite nonnegative prospective multipliers required')
    multiplier = float(multipliers[k])
    for group, base in zip(opt.param_groups, (.001, .0003), strict=True):
        group['lr'] = base * multiplier
    opt.zero_grad(set_to_none=True)
    paths = {}
    comp = previous.components(model, batch, constants, work, INTERNAL_STEP, HORIZON, paths)
    loss = previous.objective(comp)
    row = {**_loss_row(comp, loss, batch), 'index': k, 'attempt': k+1,
           'lr_multiplier': multiplier, 'accepted': False,
           'encoder_lr': opt.param_groups[0]['lr'], 'ode_lr': opt.param_groups[1]['lr'],
           'gradient_norm': None, 'update_norm': None, 'clipped': None}
    if not bool(torch.isfinite(comp).all() & torch.isfinite(loss)) or not all(
            bool(torch.isfinite(path).all()) for path in paths.values()):
        return {**row, 'rejection': 'nonfinite_forward'}
    loss.backward()
    parameters = [p for p in model.parameters() if p.requires_grad]
    norm = float(torch.nn.utils.clip_grad_norm_(parameters, 5.))
    if not math.isfinite(norm):
        opt.zero_grad(set_to_none=True)
        return {**row, 'rejection': 'nonfinite_gradient'}
    before = [parameter.detach().clone() for parameter in parameters]
    opt.step()
    if not previous.old.prior.finite_tree(model.state_dict()) or not previous.old.prior.finite_tree(opt.state_dict()):
        raise ValueError('nonfinite parameter or Adam: immediate stop')
    model.project_parameters_()
    if not previous.old.prior.finite_tree(model.state_dict()):
        raise ValueError('nonfinite projection: immediate stop')
    update_norm = float(sum((parameter.detach()-old).square().sum()
                            for parameter,old in zip(parameters,before,strict=True)).sqrt())
    return {**row, 'accepted': True, 'gradient_norm': norm, 'update_norm': update_norm,
            'clipped': norm > 5.}


@torch.no_grad()
def diagnostic(model, fixed_batch, constants, work):
    """Call on the identical worker-frozen batch at start and end, without grads."""
    paths = {}
    comp = previous.components(model, fixed_batch, constants, work, INTERNAL_STEP, HORIZON, paths)
    loss = previous.objective(comp)
    finite = bool(torch.isfinite(comp).all() & torch.isfinite(loss)) and all(
        bool(torch.isfinite(path).all()) for path in paths.values())
    return {**_loss_row(comp, loss, fixed_batch), 'finite': finite}
