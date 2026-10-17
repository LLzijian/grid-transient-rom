"""Device-aware execution of the existing closure equations and training loss.

Historical discovery runners are deliberately unchanged. This module accelerates
execution, retaining float64, RK4, causal windows and actual event intervals.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from symbolic_closure_rom import SymbolicClosureROM


def resolve_device(name="cuda"):
    if name == "auto":
        name = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(name)
    if device.type not in ("cpu", "cuda"):
        raise ValueError("supported devices: cpu, cuda, cuda:N, auto")
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable; use the GPU environment or explicitly select --device cpu")
        if device.index is not None and device.index >= torch.cuda.device_count():
            raise ValueError("CUDA device index is unavailable")
        if device.index is None:
            device = torch.device("cuda", torch.cuda.current_device())
    return device


def synchronize(device):
    if torch.device(device).type == "cuda":
        torch.cuda.synchronize(device)


class TrainingClosureROM(SymbolicClosureROM):
    """Same model; collect parameter expressions once per optimizer graph.

    Seeded closures share angle and port features. Other legal expressions use
    the original analytic compiler. Cached tensors retain their gradients and
    are discarded after backward, before any parameter update.
    """
    def _clear_runtime(self):
        self._graph_cache = None
        self._runtime_cache = None

    def _apply(self, fn, recurse=True):
        self._clear_runtime()
        return super()._apply(fn, recurse=recurse)

    def load_state_dict(self, *args, **kwargs):
        self._clear_runtime()
        return super().load_state_dict(*args, **kwargs)

    def _constants(self):
        J, R = self.matrices()
        return ((J - R).T, self.structured._pos("coupling"),
                self.structured._pos("storage_base"), self.q_quadratic,
                self.phi.sin(), self.phi.cos(), self.base_D.abs(),
                self.base_coupling / self.base_coupling.mean().clamp_min(1e-15))

    def begin_graph_cache(self):
        if getattr(self, "_runtime_cache", None) is not None:
            raise RuntimeError("nested training graph cache")
        self._runtime_cache = self._constants()

    def end_graph_cache(self):
        self._clear_runtime()

    def field(self, z, u=None):
        z = torch.as_tensor(z, dtype=self.mass.dtype, device=self.mass.device)
        cache = getattr(self, "_runtime_cache", None)
        A, coupling, storage, quadratic, sin0, cos0, absD, weights = cache if cache is not None else self._constants()
        if not self.structured.storage_specs and all(not k.startswith("generic_") for k in self._term_kinds):
            d, w, p, q = z.split(10, dim=-1)
            angle = self.phi + d @ self.base_D.T
            sin, cos = angle.sin(), angle.cos()
            s, c = sin - sin0, cos - cos0
            gd = ((s * coupling) @ self.base_D)
            gp = storage[:10] * p
            gq = (storage[10:] + quadratic) * q
            pn = p / self.p_scale
            pv = torch.zeros_like(p)
            for a, kind in zip(self.coefficients, self._term_kinds):
                if kind == "node_pq":
                    gp = gp + a * q / self.p_scale
                    gq = gq + a * pn
                elif kind == "edge_sine":
                    gd = gd + a * ((weights * (q @ self.base_D.T) * cos) @ self.base_D)
                    gq = gq + a * ((weights * s) @ self.base_D)
                elif kind == "edge_cosine":
                    gd = gd + a * ((-weights * (q @ absD.T) * sin) @ self.base_D)
                    gq = gq + a * ((weights * c) @ absD)
                elif kind == "port_w":
                    pv = pv + a * (w / self.input_scale_omega)
                elif kind == "port_p":
                    pv = pv + a * pn
                elif kind == "port_q":
                    pv = pv + a * q
                elif kind == "port_es":
                    pv = pv + a * (s @ self.port_s_weight)
                elif kind == "port_ec":
                    pv = pv + a * (c @ self.port_c_weight)
                elif kind == "port_nq":
                    pv = pv + a * (q @ self.neighbor_A.T - q)
            gradient = torch.cat((gd, w, gp, gq), -1)
        else:
            x = self.interface_to_state(z)
            gradient = self.energy_and_gradient(x)[1]
            pv = self.port_values(x)
        if u is not None:
            pv = torch.as_tensor(u, dtype=z.dtype, device=z.device)
        if pv.shape[-1] != 10:
            raise ValueError("external port input must have width 10")
        f = gradient @ A
        # The registered B has identity on the momentum rows only.
        return torch.cat((f[..., :10], (f[..., 10:20] + pv) / self.mass, f[..., 20:]), -1)


@dataclass
class PreparedRollout:
    history: torch.Tensor
    segments: list
    counts: torch.Tensor


class TensorEventStore:
    """Immutable ragged event tensors, transferred once, with no invented tails."""
    def __init__(self, events, device="cpu"):
        self.device = resolve_device(device)
        self.events = {}
        self.lengths = {}
        for event in events:
            key = event["id"]
            if key in self.events:
                raise ValueError("duplicate event identity")
            x = np.asarray(event["x"], dtype=np.float64)
            torque = np.asarray(event["torque"], dtype=np.float64)
            if x.ndim != 2 or x.shape[1] != 30 or torque.shape != (len(x), 10):
                raise ValueError("expected genuine Nx30 observations and Nx10 torque")
            if len(x) <= 11 or not np.isfinite(x).all() or not np.isfinite(torque).all():
                raise ValueError("nonfinite event or insufficient causal history/future")
            self.events[key] = (torch.tensor(x, device=self.device), torch.tensor(torque, device=self.device))
            self.lengths[key] = len(x)

    def histories(self, samples, adjacent=False):
        shift = int(adjacent)
        return torch.stack([self.events[e["id"]][0][i-10+shift:i+1+shift] for e, i in samples])

    def prepare(self, samples, horizon):
        if horizon <= 0 or not samples:
            raise ValueError("positive horizon and nonempty samples required")
        available = []
        for e, endpoint in samples:
            length = self.lengths[e["id"]]
            if not 10 <= endpoint < length - 1:
                raise ValueError("every window needs 100ms history and a real future")
            n = min(round(horizon / .01), length - 1 - endpoint)
            if n <= 0:
                raise ValueError("horizon must contain a real integration step")
            available.append(n)
        segments = []
        start = 0
        for end in sorted(set(available)):
            rows = [j for j, n in enumerate(available) if n >= end]
            # Targets and indices are prepared once outside the integration loop.
            truth = torch.stack([self.events[samples[j][0]["id"]][0][samples[j][1]+1+start:samples[j][1]+1+end]
                                 for j in rows], dim=1)
            ix = torch.tensor(rows, dtype=torch.long, device=self.device)
            segments.append((ix, truth))
            start = end
        return PreparedRollout(self.histories(samples), segments,
                               torch.tensor(available, dtype=torch.float64, device=self.device))


def rollout_errors(model, prepared, initial=None):
    from train_structured_rom import _point_error
    h = prepared.history
    current = torch.cat((h[:, -1], model.encode(h)), -1) if initial is None else initial
    sums = h.new_zeros(len(h))
    for ix, truth in prepared.segments:
        state = current.index_select(0, ix)
        subtotal = h.new_zeros(len(ix))
        for target in truth.unbind(0):
            state = model.rk4(state, .01)
            subtotal = subtotal + _point_error(state, target, model)
        current = current.index_copy(0, ix, state)
        sums = sums.index_add(0, ix, subtotal)
    return sums / prepared.counts


def objective(model, store, samples, horizon, force_scale):
    from train_local_memory_0098 import risk
    if not np.isfinite(force_scale) or force_scale <= 0:
        raise ValueError("positive finite train force scale required")
    long_batch = store.prepare([(e, 10) for e, _ in samples], horizon)
    local_batch = store.prepare(samples, .2)
    history = local_batch.history
    long = rollout_errors(model, long_batch)
    state = torch.cat((history[:, -1], model.encode(history)), -1)
    local = rollout_errors(model, local_batch, initial=state).mean()
    adjacent = store.histories(samples, adjacent=True)
    latent = ((model.rk4(state, .01)[:, 30:] - model.encode(adjacent)) / .1).square().mean()
    predicted = state[:, 20:30] - model.mass * model.field(state)[:, 10:20]
    target = torch.stack([store.events[e["id"]][1][i] for e, i in samples])
    torque = ((predicted - target) / force_scale).square().mean()
    terms = {"risk": risk(long), "mean": long.mean(), "local": local,
             "latent_consistency": latent, "torque": torque, "complexity": model.complexity(state)}
    return 10 * terms["risk"] + local + .1 * latent + .1 * torque, terms


def export_cpu(model, path: str | Path):
    # Legacy complexity reporting intentionally runs on CPU; never move the
    # live training model or optimizer, nor leave a detached cache on the copy.
    if getattr(model, "_runtime_cache", None) is not None:
        raise RuntimeError("export after closing the optimizer graph")
    return copy.deepcopy(model).cpu().export(path)


def all_finite(tensors):
    values = [torch.isfinite(t).all() for t in tensors if t is not None]
    return bool(torch.stack(values).all()) if values else True
