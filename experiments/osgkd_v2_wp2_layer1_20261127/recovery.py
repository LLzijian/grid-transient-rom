"""Scoring-only access to synthetic truth after support is frozen."""

import numpy as np
import torch

from kiki.data.osgkd_synthetic import recovery_metrics

from .models import edge_features


def _relative(value, reference):
    scale = float(reference.norm())
    return float(value.norm()) / scale if scale else None


@torch.no_grad()
def score_structure(model, support, truth_events, true_system):
    if support is None:
        return None
    edges = model.edge_index.T
    base = edges[support.base].tolist()
    modulation = edges[support.modulation].tolist()
    scores = recovery_metrics(base, modulation, true_system)
    c = model.c.detach()
    beta = getattr(model, 'beta', torch.zeros_like(c)).detach()
    true_index = {tuple(edge): i for i, edge in enumerate(edges.tolist())}
    index = [true_index[tuple(edge)] for edge in true_system['edges']]
    c_true, beta_true = torch.zeros_like(c), torch.zeros_like(beta)
    c_true[index] = torch.as_tensor(true_system['c'])
    beta_true[index] = torch.as_tensor(true_system['beta'])
    states = torch.as_tensor(np.stack([item['state'][10:] for item in truth_events]))
    sine, envelope = edge_features(model, states[:, :, 0], states[:, :, 3])
    excited_c = sine.square().mean(dim=(0, 1)).sqrt() >= .01
    excited_beta = envelope.square().mean(dim=(0, 1)).sqrt() >= .01
    contribution = (c - c_true)*sine + (beta - beta_true)*envelope
    known = c_true*sine + beta_true*envelope
    scores['resolved_by_excitation'] = {
        'base_true_edges': int((excited_c & (c_true > 0)).sum()),
        'modulated_true_edges': int((excited_beta & (beta_true > 0)).sum()),
        'candidate_base_features': int(excited_c.sum()),
        'candidate_modulated_features': int(excited_beta.sum()),
        'c_relative_l2': _relative((c-c_true)[excited_c], c_true[excited_c]),
        'beta_relative_l2_in_true_a_coordinate': _relative((beta-beta_true)[excited_beta], beta_true[excited_beta]),
        'same_state_normalized_edge_rmse': float(contribution.square().mean().sqrt()),
        'same_state_normalized_node_force_rmse': float((contribution @ model.incidence.T).square().mean().sqrt()),
        'same_state_node_force_relative_l2': _relative(contribution @ model.incidence.T, known @ model.incidence.T),
    }
    return scores
