"""Hard group restrictions for matched OSGKD graph refits."""

from dataclasses import asdict, dataclass

import torch
from torch import nn
from torch.nn.utils import parametrize

from .osgkd import OSGKDConfig
from .osgkd_graph import OSGKDGraph


_CONSTANTS = ('mass', 'damping', 'F', 'Sp', 'Se', 'omega_scale', 'theta_ref', 'edge_index')


@dataclass(frozen=True)
class Restrictions:
    edge_ids: tuple[int, ...] | None = None
    static_environment_rows: tuple[int, ...] = ()
    remove_beta: bool = False
    remove_memory: bool = False
    remove_coupling: bool = False
    freeze_mechanical: bool = False

    def __post_init__(self):
        for key in ('edge_ids', 'static_environment_rows'):
            indices = getattr(self, key)
            if key == 'edge_ids' and indices is None:
                continue
            if any(type(i) is not int or i < 0 for i in indices) or len(indices) != len(set(indices)):
                raise ValueError(f'{key} requires distinct nonnegative indices')
            object.__setattr__(self, key, tuple(sorted(indices)))
        if any(row not in (0, 1) for row in self.static_environment_rows):
            raise ValueError('static environment rows are first and second powers of amplitude')


class _ZeroMaskedEntries(nn.Module):
    def __init__(self, keep):
        super().__init__()
        self.register_buffer('keep', keep.clone())

    def forward(self, value):
        return value.masked_fill(~self.keep, 0.)


class RestrictedOSGKDGraph(OSGKDGraph):
    checkpoint_version = 8

    def __init__(self, config, *, restrictions=Restrictions(), **constants):
        super().__init__(config, **constants)
        self.restrictions = restrictions
        edge_keep = torch.ones_like(self.c, dtype=torch.bool)
        if restrictions.edge_ids is not None:
            if any(i >= edge_keep.numel() for i in restrictions.edge_ids):
                raise ValueError('retained edge index is out of range')
            edge_keep.zero_()
            edge_keep[list(restrictions.edge_ids)] = True

        coupling_keep = edge_keep & (not restrictions.remove_coupling)
        masks = {
            'c': coupling_keep,
            'beta': coupling_keep & (not restrictions.remove_beta),
            'edge_coeff': edge_keep[:, None, None].expand_as(self.edge_coeff).clone(),
            'environment_coeff': torch.ones_like(self.environment_coeff, dtype=torch.bool),
            'memory_coeff': torch.full_like(self.memory_coeff, not restrictions.remove_memory, dtype=torch.bool),
            'mechanical_coeff': torch.full_like(self.mechanical_coeff, not restrictions.freeze_mechanical, dtype=torch.bool),
        }
        masks['environment_coeff'][list(restrictions.static_environment_rows)] = False
        if restrictions.remove_memory:
            masks['edge_coeff'][..., 2:4] = False
        for name, keep in masks.items():
            if not bool(keep.all()):
                parametrize.register_parametrization(self, name, _ZeroMaskedEntries(keep))
        if restrictions.remove_memory:
            readout_keep = torch.ones_like(self.encoder.readout.weight, dtype=torch.bool)
            readout_keep[2] = False
            parametrize.register_parametrization(self.encoder.readout, 'weight', _ZeroMaskedEntries(readout_keep))
        self.project_parameters_()

    def _base_parameter(self, name):
        if parametrize.is_parametrized(self, name):
            return self.parametrizations[name].original
        return getattr(self, name)

    @torch.no_grad()
    def project_parameters_(self):
        c, beta = (self._base_parameter(name) for name in ('c', 'beta'))
        c.clamp_(min=0.)
        beta.copy_(torch.minimum(beta.clamp_min(0.), c))
        for module in (self, self.encoder.readout):
            if parametrize.is_parametrized(module):
                for mapping in module.parametrizations.values():
                    mapping.original.masked_fill_(~mapping[0].keep, 0.)

    def _parts(self, state):
        if self.restrictions.remove_memory and state.ndim and state.shape[-1] == 5 * self.n:
            state = torch.cat((state[..., :4 * self.n], torch.zeros_like(state[..., 4 * self.n:])), dim=-1)
        return super()._parts(state)

    def complexity(self):
        inactive = sum(
            int((~mapping[0].keep).sum())
            for module in (self, self.encoder.readout)
            if parametrize.is_parametrized(module)
            for mapping in module.parametrizations.values()
        )
        effective_edges = (self.c > 0) | self.edge_coeff.ne(0).any(dim=(1, 2))
        groups = {
            'baseline_exchange': bool((self.c > 0).any()),
            'dynamic_modulation': bool((self.beta > 0).any()),
            'amplitude_dynamics': bool(self.amplitude_coeff.ne(0).any() | self.edge_coeff[..., :2].ne(0).any()),
            'mechanical_dynamics': bool(self.mechanical_coeff.ne(0).any()),
            'environment_memory': not self.restrictions.remove_memory,
            'environment_amplitude': bool(self.environment_coeff[0].ne(0).any()),
            'environment_amplitude_squared': bool(self.environment_coeff[1].ne(0).any()),
            'environment_edge_cosine': bool(self.edge_coeff[..., 4].ne(0).any()),
        }
        return {
            'parameters': sum(p.numel() for p in self.parameters()) - inactive,
            'embedded_state_dimension': 5 * self.n,
            'evolved_state_dimension': self.n * (5 - self.restrictions.remove_memory - self.restrictions.freeze_mechanical),
            'constant_mechanical_coordinates': self.n if self.restrictions.freeze_mechanical else 0,
            'active_edge_paths': int(effective_edges.sum()),
            'active_module_groups': sum(groups.values()),
            'exchange_edges': int((self.c > 0).sum()),
            'modulated_edges': int((self.beta > 0).sum()),
        }

    def checkpoint(self):
        return {**super().checkpoint(), 'restrictions': asdict(self.restrictions)}

    @classmethod
    def from_parent(cls, parent, restrictions):
        if type(parent) is not OSGKDGraph:
            raise ValueError('a dense OSGKDGraph parent is required')
        model = cls(parent.config, restrictions=restrictions, **{key: getattr(parent, key) for key in _CONSTANTS})
        with torch.no_grad():
            for name, parameter in parent.named_parameters():
                if not name.startswith('encoder.'):
                    model._base_parameter(name).copy_(parameter)
            weights = parent.encoder.state_dict()
            if restrictions.remove_memory:
                model.encoder.readout.parametrizations.weight.original.copy_(weights['readout.weight'])
                model.encoder.gru.load_state_dict(parent.encoder.gru.state_dict())
                model.encoder.edge_mlp.load_state_dict(parent.encoder.edge_mlp.state_dict())
            else:
                model.encoder.load_state_dict(weights)
        model.project_parameters_()
        return model

    @classmethod
    def from_checkpoint(cls, payload):
        if (payload.get('contract'), payload.get('version')) != (cls.contract, cls.checkpoint_version):
            raise ValueError('unsupported restricted OSGKD checkpoint')
        weights = payload['state_dict']
        model = cls(OSGKDConfig(**payload['config']), restrictions=Restrictions(**payload['restrictions']),
                    **{key: weights[key] for key in _CONSTANTS})
        for name, mask in model.named_buffers():
            if name.endswith('.keep') and not torch.equal(weights[name], mask):
                raise ValueError('checkpoint restriction mask differs from its specification')
        model.load_state_dict(weights)
        if any(not torch.isfinite(p).all() for p in model.parameters()):
            raise ValueError('checkpoint parameters must be finite')
        model.project_parameters_()
        return model
