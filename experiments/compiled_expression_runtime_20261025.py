"""Opt-in expression execution; historical runners and exports stay unchanged.

Compile immutable AST metadata once, reuse constant feature derivatives within
a rollout, and construct only the storage features actually consumed. Tensor
arithmetic retains the reference order. This is not polynomial expansion.
"""
from __future__ import annotations

import copy
from collections import OrderedDict

import torch

from closure_training_runtime import TrainingClosureROM
from symbolic_closure_rom import _VARIABLES, _json_key, _origin_expression


class ExpressionKernel:
    def __init__(self, expression, channel):
        self.expression = copy.deepcopy(expression)
        self.channel = channel
        self.variables = _VARIABLES[channel]
        self.origin, self.origin_gradient = _origin_expression(_json_key(expression), channel)
        self._constants = OrderedDict()
        self._origins = {}
        self._evaluate = self._compile(self.expression)

    def __deepcopy__(self, memo):
        # A copied Python function otherwise retains its original closure/cache.
        result = type(self)(self.expression, self.channel)
        memo[id(self)] = result
        return result

    def clear(self):
        self._constants.clear()
        self._origins.clear()

    def _constant(self, features, index):
        key = (tuple(features.shape), tuple(features.stride()), features.dtype, features.device, index)
        if key not in self._constants:
            # Evaluation can precede training on the same instance. Keep
            # reusable constants normal tensors even inside inference_mode.
            with torch.inference_mode(False):
                gradient = torch.zeros_like(features)
                if index is not None:
                    gradient[..., index] = 1.0
            self._constants[key] = gradient
            # Bound even standalone use with changing batch shapes.
            if len(self._constants) > 24:
                self._constants.popitem(last=False)
        return self._constants[key]

    def _compile(self, ast):
        op = ast['op']
        if op == 'var':
            index = self.variables.index(ast['name'])
            return lambda f: (f[..., index], self._constant(f, index))
        if op == 'const':
            number = ast['value']
            return lambda f: (f[..., 0] * 0 + number, self._constant(f, None))
        if op in ('add', 'mul'):
            children = tuple(self._compile(child) for child in ast['args'])
            def combine(features):
                # Same evaluation and reduction order as the reference AST.
                args = [child(features) for child in children]
                value, grad = args[0]
                for other, other_grad in args[1:]:
                    if op == 'add':
                        value, grad = value + other, grad + other_grad
                    else:
                        grad, value = (grad * other.unsqueeze(-1) + other_grad * value.unsqueeze(-1),
                                       value * other)
                return value, grad
            return combine
        child = self._compile(ast['arg'])
        if op.startswith('pow'):
            power = int(op[-1])
            def powered(features):
                value, grad = child(features)
                return value.pow(power), grad * (power * value.pow(power - 1)).unsqueeze(-1)
            return powered
        if op != 'tanh':
            raise ValueError('unsupported validated expression operator: ' + op)
        scale = ast['scale']
        def saturated(features):
            value, grad = child(features)
            result = torch.tanh(scale * value)
            return result, grad * (scale * (1 - result.square())).unsqueeze(-1)
        return saturated

    def __call__(self, features):
        value, grad = self._evaluate(features)
        if self.channel == 'edge':
            reverse = torch.stack((features[..., 1], features[..., 0], features[..., 3],
                                   features[..., 2], -features[..., 4], features[..., 5]), -1)
            other, other_grad = self._evaluate(reverse)
            other_grad = torch.stack((other_grad[..., 1], other_grad[..., 0], other_grad[..., 3],
                                      other_grad[..., 2], -other_grad[..., 4], other_grad[..., 5]), -1)
            value = .5 * (value + other)
            grad = .5 * (grad + other_grad)
        # Six/two scalar constants, no state or trainable-parameter graph.
        origin_key = (features.dtype, features.device)
        if origin_key not in self._origins:
            with torch.inference_mode(False):
                self._origins[origin_key] = features.new_tensor(self.origin_gradient)
        origin_grad = self._origins[origin_key]
        if self.channel == 'port':
            return value - self.origin, grad.clone()
        return value - self.origin - (origin_grad * features).sum(-1), grad - origin_grad


class CompiledExpressionROM(TrainingClosureROM):
    """Use compiled generic node/edge storage; all other models fall back."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._prepare_expressions()

    def _prepare_expressions(self):
        self._expression_enabled = (not self.structured.storage_specs and bool(self.terms)
            and all(t['channel'] in ('node', 'edge') and k.startswith('generic_')
                    for t, k in zip(self.terms, self._term_kinds)))
        self._expression_kernels = ([ExpressionKernel(t['expression'], t['channel']) for t in self.terms]
                                    if self._expression_enabled else [])

    def _clear_runtime(self):
        super()._clear_runtime()
        for kernel in getattr(self, '_expression_kernels', ()):
            kernel.clear()

    def with_terms(self, terms):
        result = super().with_terms(terms)
        result._prepare_expressions()
        return result

    def energy_and_gradient(self, internal):
        if not self._expression_enabled:
            return super().energy_and_gradient(internal)
        x = torch.as_tensor(internal, dtype=self.mass.dtype, device=self.mass.device)
        base_energy, base_gradient = self.structured.energy_and_gradient(x)
        d, pi, p, q = self._split(x)
        quadratic = self.q_quadratic
        base_energy = base_energy + .5 * quadratic * q.square().sum(-1)
        base_gradient = base_gradient + torch.cat((torch.zeros_like(x[..., :30]), quadratic * q), -1)
        extra_energy = x.new_zeros(x.shape[:-1])
        gd, gp, gq = torch.zeros_like(d), torch.zeros_like(p), torch.zeros_like(q)
        pnorm = p / self.p_scale
        features = {}
        if any(t['channel'] == 'node' for t in self.terms):
            features['node'] = torch.stack((pnorm, q), -1)
        if any(t['channel'] == 'edge' for t in self.terms):
            eta = d @ self.base_D.T
            angle = self.phi + eta
            sine = torch.sin(angle) - torch.sin(self.phi)
            cosine = torch.cos(angle) - torch.cos(self.phi)
            ds = torch.cos(angle)[..., :, None] * self.base_D
            dc = -torch.sin(angle)[..., :, None] * self.base_D
            positive, negative = self.base_D.clamp_min(0), (-self.base_D).clamp_min(0)
            ep = torch.einsum('...n,en->...e', pnorm, positive)
            ej = torch.einsum('...n,en->...e', pnorm, negative)
            eqi = torch.einsum('...n,en->...e', q, positive)
            eqj = torch.einsum('...n,en->...e', q, negative)
            features['edge'] = torch.stack((ep, ej, eqi, eqj, sine, cosine), -1)
        for coefficient, term, kernel in zip(self.coefficients, self.terms, self._expression_kernels):
            channel = term['channel']
            value, grad = kernel(features[channel])
            if channel == 'node':
                contribution = value.sum(-1)
                term_gp, term_gq = grad[..., :, 0] / self.p_scale, grad[..., :, 1]
                term_gd = torch.zeros_like(d)
            else:
                slope = kernel.origin_gradient[5]
                # The reference recomputes eta here; retain its arithmetic.
                tangent_eta = d @ self.base_D.T
                if slope:
                    value = value + slope * (features['edge'][..., 5] + torch.sin(self.phi) * tangent_eta)
                weights = self.base_coupling / self.base_coupling.mean().clamp_min(1e-15)
                contribution = (weights * value).sum(-1)
                weighted = weights[..., None] * grad
                term_gp = (torch.einsum('...e,en->...n', weighted[..., 0], positive)
                           + torch.einsum('...e,en->...n', weighted[..., 1], negative)) / self.p_scale
                term_gq = (torch.einsum('...e,en->...n', weighted[..., 2], positive)
                           + torch.einsum('...e,en->...n', weighted[..., 3], negative))
                term_gd = (torch.einsum('...e,...en->...n', weighted[..., 4], ds)
                           + torch.einsum('...e,...en->...n', weighted[..., 5], dc))
                if slope:
                    term_gd = term_gd + (weights * slope * (-torch.sin(self.phi + tangent_eta)
                                                          + torch.sin(self.phi))) @ self.base_D
            # Retain the full-gradient concatenation/slicing order as well.
            full_grad = torch.cat((term_gd, torch.zeros_like(pi), term_gp, term_gq), -1)
            extra_energy = extra_energy + coefficient * contribution
            gd = gd + coefficient * full_grad[..., :10]
            gp = gp + coefficient * full_grad[..., 20:30]
            gq = gq + coefficient * full_grad[..., 30:]
        extra_gradient = torch.cat((gd, torch.zeros_like(pi), gp, gq), -1)
        return base_energy + extra_energy, base_gradient + extra_gradient


def adapt_base(reference):
    """Independent model for explicit opt-in, preserving parameters and RNG."""
    if getattr(reference, '_runtime_cache', None) is not None:
        raise RuntimeError('close graph before adapting an expression model')
    cpu_rng = torch.get_rng_state()
    cuda_rng = torch.cuda.get_rng_state_all() if reference.mass.is_cuda else None
    try:
        snapshot = copy.deepcopy(reference).cpu()
        payload = snapshot._export_payload()
        result = CompiledExpressionROM.from_config(payload['config'], terms=payload['terms'])
        result.load_state_dict(snapshot.state_dict(), strict=True)
        flags = {name: p.requires_grad for name, p in reference.named_parameters()}
        for name, parameter in result.named_parameters():
            parameter.requires_grad_(flags[name])
        result.provenance = copy.deepcopy(reference.provenance)
        return result.to(device=reference.mass.device, dtype=reference.mass.dtype)
    finally:
        torch.set_rng_state(cpu_rng)
        if cuda_rng is not None:
            torch.cuda.set_rng_state_all(cuda_rng)
