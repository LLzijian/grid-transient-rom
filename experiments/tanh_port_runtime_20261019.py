"""Explicit, narrow equivalent tanh(2q) field; reference equations remain authoritative."""
from __future__ import annotations
import copy
import torch
from closure_training_runtime import TrainingClosureROM
from structured_rom import StructuredROM

TERM = {'channel': 'port', 'expression': {'op': 'tanh', 'scale': 2., 'arg': {'op': 'var', 'name': 'q'}}}

class TanhPortTrainingROM(TrainingClosureROM):
    """Only specialize the registered single-port L10-S skeleton.

    The reference already evaluates port values without analytic derivatives.
    This path avoids unrelated generic feature construction and energy work.
    """
    def supports_fast_field(self):
        s = self.structured
        return (self.arm == 'L10-S' and self.n == self.rank == 10 and self.dim == 40
                and self.mass.dtype == torch.float64 and type(s) is StructuredROM
                and s.n == s.n_p == s.n_q == 10 and not s.storage_specs
                and self.terms == [TERM] and self.coefficients.shape == (1,))

    def field(self, z, u=None):
        if not self.supports_fast_field():
            return super().field(z, u)
        z = torch.as_tensor(z, dtype=self.mass.dtype, device=self.mass.device)
        cache = getattr(self, '_runtime_cache', None)
        A = (cache if cache is not None else self._constants())[0]
        d, w, p, q = z.split(10, -1)
        s = self.structured
        # Use the actual reference skeleton buffers, including its mass map.
        gd = ((torch.sin(s.phi + d @ s.D.T) - torch.sin(s.phi)) * s._pos('coupling')) @ s.D
        storage = s._pos('storage_base')
        gp = storage[:10] * p
        gq = storage[10:] * q + self.q_quadratic * q
        gradient = torch.cat((gd, (w * self.mass) / s.mass, gp, gq), -1)
        # Under external input the coefficient must remain disconnected (grad=None).
        pv = self.coefficients[0] * torch.tanh(2. * q) if u is None else torch.as_tensor(u, dtype=z.dtype, device=z.device)
        if pv.shape[-1] != 10:
            raise ValueError('external port input must have width 10')
        f = gradient @ A
        return torch.cat((f[..., :10], (f[..., 10:20] + pv) / self.mass, f[..., 20:]), -1)


def select_runtime(source, *, fast=False):
    """Clone exact state, parameter order and training identity; opt-in only."""
    if type(source) not in (TrainingClosureROM, TanhPortTrainingROM):
        raise TypeError('requires an unwrapped reference training runtime')
    if getattr(source, '_runtime_cache', None) is not None:
        raise RuntimeError('close graph before cloning')
    result = copy.deepcopy(source)
    result.__class__ = TanhPortTrainingROM if fast else TrainingClosureROM
    result._clear_runtime()
    return result
