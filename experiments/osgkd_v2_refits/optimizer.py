"""Leaf-parameter optimizer for prospective restricted graph refits."""

import torch

from experiments.osgkd_v2_discovery.prepare import VARIANTS


def new_optimizer(model):
    """Keep the registered encoder/ODE rates, including parametrized leaves."""
    encoder_ids = {id(parameter) for parameter in model.encoder.parameters()}
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    return torch.optim.Adam(
        [{'params': [p for p in trainable if id(p) in encoder_ids], 'lr': .001},
         {'params': [p for p in trainable if id(p) not in encoder_ids], 'lr': .0003}],
        betas=(.9, .999), eps=1e-8, weight_decay=0,
    )
