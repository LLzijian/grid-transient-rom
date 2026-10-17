"""Loop0078 matched C-free / shared-gradient structure screen.

This file deliberately contains the small campaign-specific wrappers rather
than changing the accepted JointROM trainer.  Importing it is model-only: the
development reader is called only by :func:`run`.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import shutil
import time
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
from torch import nn

from evaluate_long_latent import _construct
from structure_grammar import PrimitiveSpec, primitive_value_and_gradient
from structured_rom import DissipationSpec, ExchangeSpec, StructuredROM
from train_joint_campaign import sample_batch
from train_joint_latent import BASE, DT, NETWORK, JointROM, read_development, tensor
from kiki.models.private_pm import MechanicalROM
from kiki.physics import coi_project

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "private_data/task2_long_0060_5s"
SEED = 78
SELECTION_SEED = 780
BATCH = 32
STEPS = 1200
LR = 1e-4
METRIC = (0.47675314588683515, 1.0033667249602924)
PORT_V = (0, 3, 6, 10, 13, 20, 21, 22)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _spinv(value: torch.Tensor) -> torch.Tensor:
    """Stable inverse-softplus for positive empirical coordinates."""
    return torch.where(value > 20, value, torch.log(torch.expm1(value.clamp_min(1e-14))))


def horizon_for_step(step: int) -> float:
    if not 1 <= step <= STEPS:
        raise ValueError("step is outside the fixed 1200-update screen")
    if step <= 100:
        return 1.0
    if step <= 300:
        return (1.0, 2.0)[(step - 101) % 2]
    return (1.0, 2.0, 5.0)[(step - 301) % 3]


def _coi(x: torch.Tensor, mass: torch.Tensor) -> torch.Tensor:
    return x - (x * mass).sum(-1, keepdim=True) / mass.sum()


def _rms(value: torch.Tensor) -> torch.Tensor:
    """Zero-safe RMS: norm has a finite zero subgradient, sqrt(mean(x²)) does not."""
    return torch.linalg.vector_norm(value) / np.sqrt(value.numel())


def _internal_normalized(interface_field: torch.Tensor, model: _ObserverDecoder) -> torch.Tensor:
    return interface_field / torch.cat((interface_field.new_full((10,), METRIC[0]),
                                        interface_field.new_full((10,), METRIC[1]),
                                        model.p_scale, interface_field.new_ones(3)))


def _clone_linear(old: nn.Linear) -> nn.Linear:
    out = nn.Linear(old.in_features, old.out_features, old.bias is not None, dtype=torch.float64)
    out.load_state_dict(old.state_dict())
    return out


class _ObserverDecoder(nn.Module):
    """The frozen coordinate translation required by the 0078 interface."""
    def __init__(self, source: JointROM):
        super().__init__()
        self.register_buffer("mass", source.mass.detach().clone())
        self.register_buffer("input_scale", source.input_scale.detach().clone())
        self.register_buffer("q_transform", source.q_transform.detach().clone())
        self.register_buffer("z_scale", source.z_scale.detach().clone())
        self.register_buffer("voltage_star", source.voltage_star.detach().clone())
        self.register_buffer("load_positions", source.load_positions.detach().clone())
        self.register_buffer("load_g", source.load_g.detach().clone())
        self.register_buffer("load_star", source.load_star.detach().clone())
        self.register_buffer("load_scale", source.load_scale.detach().clone())
        self.register_buffer("p_scale", source.feature_scale.detach().clone()[10:20])
        self.register_buffer("frozen_incidence", source.incidence.detach().clone())
        self.register_buffer("frozen_coupling", source.coupling.detach().clone())
        self.register_buffer("frozen_damping", source.damping.detach().clone())
        self.register_buffer("frozen_delta_star", source.delta_star.detach().clone())
        self.gru = copy.deepcopy(source.gru)
        self.head = _clone_linear(source.head)
        # The raw-zero subtraction cancels this bias exactly.  Keeping it
        # frozen prevents an unidentifiable coordinate shift during fitting.
        if self.head.bias is not None:
            self.head.bias.requires_grad_(False)
        self.decoder = _clone_linear(source.decoder)
        with torch.no_grad():
            zero = torch.zeros((1, 11, 30), dtype=torch.float64)
            old_zero = source.encode(zero).reshape(-1)
            self.decoder.bias.add_(self.decoder.weight[:, 10:] @ old_zero)

    @property
    def rank(self) -> int:
        return int(self.q_transform.shape[0])

    def observations(self, state: torch.Tensor) -> torch.Tensor:
        delta = _coi(state[..., :10], self.mass)
        return torch.cat((delta, state[..., 10:20]), -1) / self.input_scale

    def raw_encode(self, history: torch.Tensor) -> torch.Tensor:
        if history.shape[-2:] != (11, 30):
            raise ValueError("encoder requires [batch,11,30] histories")
        return self.head(self.gru(self.observations(history))[0][:, -1])

    def encode(self, history: torch.Tensor) -> torch.Tensor:
        return (self.raw_encode(history) - self.raw_encode(torch.zeros_like(history))) @ self.q_transform

    def electrical_output(self, state: torch.Tensor, q: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        raw = self.decoder(torch.cat((_coi(state[..., :10], self.mass) / self.input_scale[:10], q), -1))
        voltage = torch.abs(self.voltage_star + raw[..., 10:] * self.z_scale[10:])
        z = torch.cat((raw[..., :10], (voltage - self.voltage_star) / self.z_scale[10:]), -1)
        load = (voltage[..., self.load_positions].square() @ self.load_g - self.load_star) / self.load_scale
        return z, load.unsqueeze(-1)


class CFreeROM(_ObserverDecoder):
    """Ordinary sparse closure, with the three empirical mechanical vectors free."""
    def __init__(self, source: JointROM):
        super().__init__(source)
        self.core = copy.deepcopy(source)
        # The wrapper owns its observer/decoder; do not retain a second trainable
        # copy of them in the historical core.
        for name, parameter in self.core.named_parameters():
            parameter.requires_grad_(name in {"c", "amplitude", "xi", "alpha"})
        # These have the same positive/raw parameterization and learning-rate
        # class as G's empirical nonnegative base coordinates.
        self.damping_raw = nn.Parameter(_spinv(source.damping.detach().clone().clamp_min(1e-4)))
        self.rate_raw = nn.Parameter(_spinv(source.rate.detach().clone().clamp_min(1e-14)))
        self.gain_raw = nn.Parameter(_spinv(source.gain.detach().clone().clamp_min(1e-14)))
        active = source.coupling.detach() > 0
        self.register_buffer("active_incidence", source.incidence[:, active].detach().clone())
        self.register_buffer("force_incidence", source.incidence.detach().clone())
        self.register_buffer("force_coupling", source.coupling.detach().clone())
        self.register_buffer("force_delta_star", source.delta_star.detach().clone())
        self.register_buffer("force_damping", source.damping.detach().clone())
        self.coupling_raw = nn.Parameter(_spinv(source.coupling.detach()[active].clone()))
        self.register_buffer("force_scale", source.force_scale.detach().clone())

    def field(self, z: torch.Tensor) -> torch.Tensor:
        d, w, p, q = z[..., :10], z[..., 10:20], z[..., 20:30], z[..., 30:]
        phase = self.core.delta_star @ self.active_incidence
        coupling = torch.nn.functional.softplus(self.coupling_raw)
        e0 = ((torch.sin(d @ self.active_incidence + phase) - torch.sin(phase)) * coupling) @ self.active_incidence.T
        state = torch.cat((d, w, p, q), -1)
        action = self.core.memory(q + self.core.algebraic_coordinate(state))
        damping, rate, gain = (torch.nn.functional.softplus(self.damping_raw),
                                torch.nn.functional.softplus(self.rate_raw),
                                torch.nn.functional.softplus(self.gain_raw))
        return torch.cat((w, (p - e0 - damping * w - action) / self.mass,
                          -rate * p - gain * w,
                          self.core.features(state) @ (self.core.xi * self.core.support)), -1)

    def rk4(self, state: torch.Tensor, dt: float) -> torch.Tensor:
        a = self.field(state); b = self.field(state + dt * a / 2)
        c = self.field(state + dt * b / 2); d = self.field(state + dt * c)
        return state + dt * (a + 2*b + 2*c + d) / 6

    def rollout(self, history: torch.Tensor, steps: int, dt: float) -> torch.Tensor:
        state = torch.cat((history[:, -1], self.encode(history)), -1); out = [state]
        for _ in range(steps): state = self.rk4(state, dt); out.append(state)
        return torch.stack(out, 1)

    def complexity(self, states: torch.Tensor | None = None) -> torch.Tensor:
        if states is None: raise ValueError("0078 complexity requires its fitted states")
        theta=self.core.features(states)
        terms=[]; xi=self.core.xi*self.core.support
        for row,col in zip(*torch.nonzero(self.core.support,as_tuple=True)):
            field=torch.zeros_like(states); field[:,30+col]=theta[:,row]*xi[row,col]
            terms.append(_rms(_internal_normalized(field,self)))
        # Each alpha scalar contributes a physical force through the matching
        # memory column; direct q-memory columns are separately counted.
        af=theta[:,21+self.rank:]; alpha=self.core.alpha*self.core.alpha_support
        spatial=self.core.spatial()*self.core.amplitude
        for row,col in zip(*torch.nonzero(self.core.alpha_support,as_tuple=True)):
            field=torch.zeros_like(states); field[:,10:20]=-af[:,row,None]*alpha[row,col]*spatial[:,col]/self.mass
            terms.append(_rms(_internal_normalized(field,self)))
        for col in range(self.rank):
            field=torch.zeros_like(states); field[:,10:20]=-states[:,30+col,None]*spatial[:,col]/self.mass
            terms.append(5*_rms(_internal_normalized(field,self)))
        return torch.stack(terms).mean() if terms else states.new_zeros(())

    def prune(self, windows: torch.Tensor, optimizer: torch.optim.Optimizer) -> dict[str, Any]:
        """Whole sparse terms only; clear Adam state so masked terms stay zero."""
        state = torch.cat((windows[:, -1], self.encode(windows)), -1)
        theta = self.core.features(state)
        xi = self.core.xi * self.core.support
        alpha = self.core.alpha * self.core.alpha_support
        alpha_features = theta[..., 21+self.rank:]; spatial=self.core.spatial()*self.core.amplitude
        xi_fields=[]; alpha_fields=[]
        for row,col in zip(*torch.nonzero(self.core.support,as_tuple=True)):
            field=torch.zeros_like(state); field[:,30+col]=theta[:,row]*xi[row,col]; xi_fields.append((row,col,field))
        for row,col in zip(*torch.nonzero(self.core.alpha_support,as_tuple=True)):
            field=torch.zeros_like(state); field[:,10:20]=-alpha_features[:,row,None]*alpha[row,col]*spatial[:,col]/self.mass; alpha_fields.append((row,col,field))
        total=self.field(state); threshold=1e-3*_rms(_internal_normalized(total,self))
        keep_xi=torch.zeros_like(self.core.support); keep_alpha=torch.zeros_like(self.core.alpha_support)
        for row,col,field in xi_fields:
            if _rms(_internal_normalized(field,self)) >= threshold: keep_xi[row,col]=1.
        for row,col,field in alpha_fields:
            if _rms(_internal_normalized(field,self)) >= threshold: keep_alpha[row,col]=1.
        before = {"xi": self.core.support.detach().clone(), "alpha": self.core.alpha_support.detach().clone()}
        with torch.no_grad():
            self.core.support.mul_(keep_xi); self.core.alpha_support.mul_(keep_alpha)
            self.core.xi.mul_(self.core.support); self.core.alpha.mul_(self.core.alpha_support)
        for parameter, mask in ((self.core.xi, self.core.support), (self.core.alpha, self.core.alpha_support)):
            state_adam = optimizer.state.get(parameter, {})
            for value in state_adam.values():
                if torch.is_tensor(value) and value.shape == parameter.shape: value.mul_(mask)
        return {"before": {k: int(v.sum()) for k,v in before.items()}, "threshold":float(threshold),
                "after": {"xi": int(self.core.support.sum()), "alpha": int(self.core.alpha_support.sum())}}


def _basis(n: int, index: int) -> tuple[float, ...]:
    x = np.zeros(n); x[index] = 1.; return tuple(x)


def _active_template(base: MechanicalROM, source: JointROM) -> StructuredROM:
    active = np.flatnonzero(base.coupling > 0)
    D = base.incidence[:, active].T
    # These are the frozen empirical coordinates; their positive amplitudes
    # are represented by StructuredROM softplus parameters.
    pscale = source.feature_scale.detach().cpu().numpy()[10:20]
    sv = np.r_[pscale, np.ones(3)]
    mapping = (source.spatial() * source.amplitude).detach().cpu().numpy()
    u, _, _ = np.linalg.svd(mapping, full_matrices=False)
    directions = []
    for col in range(2):
        v = u[:, col].copy()
        if v[np.argmax(np.abs(v))] < 0: v *= -1
        directions.append(np.r_[v, np.zeros(13)])
    scales_v = np.r_[base.mass * source.feature_scale.detach().cpu().numpy()[:10], pscale, np.ones(3)]
    ports = tuple({"direction": tuple(directions[j]), "terms": tuple({"kind":"linear", "index":i, "coefficient":1e-3, "scale":float(scales_v[i])} for i in PORT_V)} for j in range(2))
    return StructuredROM(mass=base.mass, D=D, coupling=base.coupling[active], delta_star=base.delta_star,
        damping=np.maximum(base.damping, 1e-4), rate=base.rate, gain=base.gain, n_q=3,
        scales=sv, storage=(), exchange=(), dissipation=(), ports=ports, n_p=10,
        jpi_p=base.gain, rpp=base.rate*base.gain, storage_base=np.r_[1/base.gain, np.ones(3)]).double()


class GROM(_ObserverDecoder):
    """G has a StructuredROM generator only: it never carries legacy Xi."""
    def __init__(self, base: MechanicalROM, source: JointROM, arm: str):
        super().__init__(source)
        self.arm = arm
        self.structured = _active_template(base, source)
        self.extra_groups: list[tuple[str,int,str]] = []
        self.register_buffer("force_scale", source.force_scale.detach().clone())
        self.register_buffer("force_incidence", torch.as_tensor(base.incidence[:, base.coupling > 0], dtype=torch.float64))
        self.register_buffer("force_coupling", torch.as_tensor(base.coupling[base.coupling > 0], dtype=torch.float64))
        self.register_buffer("force_delta_star", torch.as_tensor(base.delta_star, dtype=torch.float64))
        self.register_buffer("force_damping", torch.as_tensor(base.damping, dtype=torch.float64))
        self._graph_cache: tuple[torch.Tensor,torch.Tensor,torch.Tensor] | None = None
        if arm == "G-fixed":
            self._install_fixed_groups()
        elif arm != "G-auto":
            raise ValueError("G arm must be G-fixed or G-auto")

    def field(self, z: torch.Tensor) -> torch.Tensor:
        if self._graph_cache is None:
            return self.structured.interface_field(z)
        x=self.structured.interface_to_state(z); _,e=self.structured.energy_and_gradient(x); J,R,B=self._graph_cache
        pv=self.structured.port_values(x); f=e@(J-R).T
        if pv.shape[-1]: f=f+pv@B.T
        return self.structured.state_to_interface(f)

    def begin_graph_cache(self) -> None:
        # J/R/B are state-independent but parameter-dependent; cache them only
        # inside one loss graph, so gradients remain exact and never stale.
        J,R=self.structured.matrices(); self._graph_cache=(J,R,self.structured.port_matrix())

    def end_graph_cache(self) -> None: self._graph_cache=None

    def rk4(self, state: torch.Tensor, dt: float) -> torch.Tensor:
        a=self.field(state); b=self.field(state+dt*a/2); c=self.field(state+dt*b/2); d=self.field(state+dt*c)
        return state+dt*(a+2*b+2*c+d)/6

    def rollout(self, history: torch.Tensor, steps: int, dt: float) -> torch.Tensor:
        state=torch.cat((history[:,-1],self.encode(history)),-1); out=[state]
        for _ in range(steps): state=self.rk4(state,dt); out.append(state)
        return torch.stack(out,1)

    def _install_fixed_groups(self) -> None:
        # v indices are pi[0:10], p[10:20], q[20:23]; storage s is p,q.
        for a,b in ((10,20),(13,21),(16,22),(20,21),(21,22)):
            self.add_storage(PrimitiveSpec("product", (a-10,b-10), 1e-3,
                (float(self.structured.scales[a-10]), float(self.structured.scales[b-10]))), normalized=1.)
        for q,e in ((20,0),(21,8),(22,16)):
            self.add_storage(PrimitiveSpec("state_sin", (q-10,e), 1e-3,
                (float(self.structured.scales[q-10]),), float(self.structured.phi[e]), "fixed_state_sin"), normalized=1.)
        for a,b in ((0,20),(3,21),(6,22),(19,20)):
            self.add_exchange(a,b,1e-3)
        for a in (20,21,22): self.add_dissipation(a,1e-3)
        common = np.r_[base_mass := self.mass.detach().cpu().numpy(), np.zeros(13)]
        self.add_dissipation_vector(common,1e-3)

    def add_storage(self, spec: PrimitiveSpec, normalized: float) -> list[nn.Parameter]:
        self.structured.storage_specs = self.structured.storage_specs + (spec,)
        # 86c625a: RMS normalization is a ranking-only quantity.  The stored
        # signed gamma is the preregistered raw +/-1e-3 coefficient.
        p=nn.Parameter(torch.as_tensor(spec.coefficient,dtype=torch.float64)); self.structured.storage_gamma.append(p)
        self.extra_groups.append(("H",len(self.structured.storage_gamma)-1,spec.name or spec.kind))
        return [p]

    def add_exchange(self, a: int, b: int, coefficient: float) -> list[nn.Parameter]:
        n=23; spec=ExchangeSpec(_basis(n,a),_basis(n,b),coefficient, name=f"J:{a},{b}")
        self.structured.exchange_specs=self.structured.exchange_specs+(spec,)
        ps=[nn.Parameter(torch.tensor(coefficient,dtype=torch.float64)), nn.Parameter(torch.tensor(spec.a,dtype=torch.float64),requires_grad=False), nn.Parameter(torch.tensor(spec.b,dtype=torch.float64),requires_grad=False)]
        self.structured.exchange_kappa.append(ps[0]); self.structured.exchange_a.append(ps[1]); self.structured.exchange_b.append(ps[2]); self.extra_groups.append(("J",len(self.structured.exchange_kappa)-1,spec.name)); return ps

    def add_dissipation(self, a: int, coefficient: float) -> list[nn.Parameter]: return self.add_dissipation_vector(_basis(23,a),coefficient)

    def add_dissipation_vector(self, vector: Iterable[float], coefficient: float) -> list[nn.Parameter]:
        spec=DissipationSpec(tuple(vector),coefficient,name="R")
        self.structured.dissipation_specs=self.structured.dissipation_specs+(spec,)
        mask=torch.cat((self.structured.dissipation_mask,torch.ones(1,dtype=torch.float64))); self.structured.dissipation_mask=mask
        rho=nn.Parameter(torch.log(torch.expm1(torch.tensor(coefficient,dtype=torch.float64))))
        direction=nn.Parameter(torch.tensor(spec.a,dtype=torch.float64),requires_grad=False)
        self.structured.dissipation_rho.append(rho); self.structured.dissipation_a.append(direction); self.extra_groups.append(("R",len(self.structured.dissipation_rho)-1,spec.name)); return [rho,direction]

    def complexity(self, states: torch.Tensor | None = None) -> torch.Tensor:
        if states is None: raise ValueError("0078 complexity requires its fitted states")
        x=self.structured.interface_to_state(states); _,energy_gradient=self.structured.energy_and_gradient(x); J,R=self.structured.matrices(); n=self.structured.n
        terms=[]
        for gamma,spec in zip(self.structured.storage_gamma,self.structured.storage_specs):
            _,ge,gs=primitive_value_and_gradient(spec,x[...,:n]@self.structured.D.T,x[...,2*n:])
            grad=torch.cat((ge@self.structured.D,torch.zeros_like(x[...,n:2*n]),gs),-1)*gamma
            terms.append(_rms(_internal_normalized(self.structured.state_to_interface(grad@(J-R).T),self)))
        for k,a,b in zip(self.structured.exchange_kappa,self.structured.exchange_a,self.structured.exchange_b):
            aa=torch.zeros(self.structured.dim,dtype=x.dtype,device=x.device); bb=aa.clone(); aa[n:]=a/torch.linalg.vector_norm(a); bb[n:]=b/torch.linalg.vector_norm(b)
            terms.append(_rms(_internal_normalized(self.structured.state_to_interface(energy_gradient@(k*(torch.outer(aa,bb)-torch.outer(bb,aa))).T),self)))
        for index,(rho,a) in enumerate(zip(self.structured.dissipation_rho,self.structured.dissipation_a)):
            aa=torch.zeros(self.structured.dim,dtype=x.dtype,device=x.device); aa[n:]=a/torch.linalg.vector_norm(a)
            field=-(energy_gradient@aa)[...,None]*aa*torch.nn.functional.softplus(rho)*self.structured.dissipation_mask[index]
            terms.append(_rms(_internal_normalized(self.structured.state_to_interface(field),self)))
        pv=self.structured.port_values(x); B=self.structured.port_matrix()
        for col in range(pv.shape[-1]): terms.append(5*_rms(_internal_normalized(self.structured.state_to_interface(pv[:,col,None]*B[:,col]),self)))
        return torch.stack(terms).mean() if terms else states.new_zeros(())

    def _group_parameters(self, kind: str, index: int) -> list[nn.Parameter]:
        if kind == "H": return [self.structured.storage_gamma[index]]
        if kind == "J": return [self.structured.exchange_kappa[index]]
        if kind == "R": return [self.structured.dissipation_rho[index]]
        raise ValueError(kind)


def _point_error(pred: torch.Tensor, truth: torch.Tensor, model: _ObserverDecoder) -> torch.Tensor:
    err = pred[..., :30] - truth[..., :30]
    d = _coi(err[..., :10], model.mass) / METRIC[0]
    w = _coi(err[..., 10:20], model.mass) / METRIC[1]
    common = (err[..., 10:20] * model.mass).sum(-1) / model.mass.sum() / METRIC[1]
    p = err[..., 20:30] / model.p_scale
    return .5 * (d.square().mean(-1) + w.square().mean(-1)) + common.square() + p.square().mean(-1)


def _event_rollout_errors(model, samples, history: torch.Tensor, horizon: float) -> torch.Tensor:
    """One encode per supplied causal window; targets follow its OWN endpoint."""
    if horizon <= 0 or not samples or len(history) != len(samples):
        raise ValueError("invalid rollout batch")
    current = torch.cat((history[:, -1], model.encode(history)), -1)
    maxsteps = round(horizon / DT)
    available = np.asarray([min(maxsteps, len(e['x']) - 1 - endpoint)
                            for e, endpoint in samples], dtype=int)
    if np.any(available <= 0) or any(endpoint < 10 for _, endpoint in samples):
        raise ValueError("every window needs causal history and a real future")
    sums = history.new_zeros(len(samples))
    nums = history.new_zeros(len(samples))
    for offset in range(1, int(available.max()) + 1):
        rows = np.flatnonzero(available >= offset)
        ix = torch.as_tensor(rows, dtype=torch.long, device=history.device)
        advanced = model.rk4(current.index_select(0, ix), DT)
        current = current.index_copy(0, ix, advanced)
        truth = torch.as_tensor(np.stack([samples[j][0]['x'][samples[j][1] + offset]
                                         for j in rows]), dtype=history.dtype, device=history.device)
        value = _point_error(advanced, truth, model)
        sums = sums.index_add(0, ix, value)
        nums = nums.index_add(0, ix, torch.ones_like(value))
    return sums / nums


def _advance_loss(model, samples, history: torch.Tensor, horizon: float):
    event = _event_rollout_errors(model, samples, history, horizon)
    mean = event.mean()
    tail = event.topk(max(1, int(np.ceil(.1 * len(samples))))).values.mean()
    return mean, .8 * mean + .2 * tail


def objective(model, samples, histories: torch.Tensor, horizon: float, *, complexity_weight=1e-4):
    to = lambda data: torch.as_tensor(data, dtype=histories.dtype, device=histories.device)
    # Long rollout always starts at the event's first 100ms decision window.
    initial = to(np.stack([e['x'][:11] for e, _ in samples]))
    long_mean, risk = _advance_loss(model, [(e, 10) for e, _ in samples], initial, horizon)
    # Local integral windows instead start at their sampled TRAIN endpoints.
    local, _ = _advance_loss(model, samples, histories, .2)
    states = torch.cat((histories[:, -1], model.encode(histories)), -1)
    adjacent = to(np.stack([e['x'][i - 9:i + 2] for e, i in samples]))
    latent = ((model.rk4(states, DT)[:, 30:] - model.encode(adjacent)) / .1).square().mean()
    force_pred = states[:, 20:30] - model.mass * model.field(states)[:, 10:20]
    old_force = to(np.stack([e['force'][i] for e, i in samples]))
    d, w = histories[:, -1, :10], histories[:, -1, 10:20]
    incidence, k = model.frozen_incidence, model.frozen_coupling
    phase = model.frozen_delta_star @ incidence
    e0 = ((torch.sin(d @ incidence + phase) - torch.sin(phase)) * k) @ incidence.T
    force_truth = old_force + e0 + model.frozen_damping * w
    force = ((force_pred - force_truth) / model.force_scale).square().mean()
    z, load = model.electrical_output(histories[:, -1], states[:, 30:])
    ztruth = to(np.stack([e['z'][i] for e, i in samples])) / model.z_scale
    ltruth = to(np.asarray([e['load'][i] for e, i in samples])[:, None]) / model.load_scale
    pe = (z[:, :10] - ztruth[:, :10]).square().mean()
    voltage = (z[:, 10:] - ztruth[:, 10:]).square().mean()
    load_loss = (load - ltruth).square().mean()
    electrical = (pe + voltage + load_loss) / 3
    complexity = model.complexity(states) if complexity_weight else states.new_zeros(())
    total = 10 * risk + local + .1 * latent + .1 * (force + electrical) + complexity_weight * complexity
    return total, {"risk": risk, "long_mean": long_mean, "long_tail": (risk - .8 * long_mean) / .2,
                   "local": local, "latent_consistency": latent, "force": force,
                   "electrical": electrical, "pe": pe, "voltage": voltage, "load": load_loss,
                   "complexity": complexity}


def _parameter_lr(model: nn.Module, name: str) -> float:
    """The 5aeaa8d preregistered parameter-type learning-rate blocks."""
    if name.startswith(("gru.", "head.", "decoder.")):
        return 1e-4
    if ("dissipation_rho" in name or name.endswith(("coupling", "coupling_raw", "damping", "jpi_p", "rpp", "storage_base", "damping_raw", "rate_raw", "gain_raw"))):
        return 1e-2
    return 1e-3


def optimizer_for(model: nn.Module) -> torch.optim.Adam:
    blocks: dict[float,list[nn.Parameter]]={1e-4:[],1e-3:[],1e-2:[]}
    for name,p in model.named_parameters():
        if p.requires_grad: blocks[_parameter_lr(model,name)].append(p)
    return torch.optim.Adam([{"params":items,"lr":lr} for lr,items in blocks.items() if items])


def add_growth_parameters(optimizer: torch.optim.Adam, model: nn.Module,
                          parameters: Iterable[nn.Parameter]) -> None:
    """Append new groups without resetting Adam moments for existing tensors."""
    names={id(p):name for name,p in model.named_parameters()}
    blocks: dict[float,list[nn.Parameter]]={}
    for p in parameters:
        if p.requires_grad: blocks.setdefault(_parameter_lr(model,names[id(p)]),[]).append(p)
    for lr,items in blocks.items(): optimizer.add_param_group({"params":items,"lr":lr})


def _allowed(model: nn.Module) -> list[nn.Parameter]:
    for name,p in model.named_parameters():
        # direction vectors for non-port H/J/R remain frozen by construction.
        if "exchange_a" in name or "exchange_b" in name or "dissipation_a" in name: p.requires_grad_(False)
    return [p for p in model.parameters() if p.requires_grad]


def _assert_structure(model: _ObserverDecoder) -> float:
    zero=torch.zeros((2,33),dtype=torch.float64); f=model.field(zero); value=float(f.abs().max().detach())
    if not np.isfinite(value) or value>1e-10: raise AssertionError(f"f(0)={value}")
    if isinstance(model,GROM):
        if any("xi" in name for name,_ in model.named_parameters()): raise AssertionError("G must not retain legacy Xi")
        if not torch.allclose(f[:,:10],zero[:,10:20],atol=1e-12,rtol=0): raise AssertionError("kinematics")
    return value


def _freeze_pruned(model: GROM, groups: list[tuple[str,int]], optimizer: torch.optim.Optimizer) -> None:
    with torch.no_grad():
        for kind,index in groups:
            ps=model._group_parameters(kind,index)
            for p in ps:
                p.zero_(); p.requires_grad_(False)
                for state in optimizer.state.get(p,{}).values():
                    if torch.is_tensor(state): state.zero_()
            if kind=="R": model.structured.dissipation_mask[index]=0.


def _selection_scale(model: GROM, x: torch.Tensor) -> torch.Tensor:
    return torch.cat((x.new_full((10,),METRIC[0]), model.mass*METRIC[1], model.p_scale, x.new_ones(3)))


def _h_field(model: GROM, x: torch.Tensor, spec: PrimitiveSpec, J: torch.Tensor, R: torch.Tensor) -> torch.Tensor:
    n=model.structured.n; eta=x[:,:n]@model.structured.D.T
    _,ge,gs=primitive_value_and_gradient(spec,eta,x[:,2*n:])
    gradient=torch.cat((ge@model.structured.D,torch.zeros_like(x[:,n:2*n]),gs),-1)
    return gradient@(J-R).T


def _j_field(gradient: torch.Tensor, a: int, b: int) -> torch.Tensor:
    aa=torch.zeros(gradient.shape[-1],dtype=gradient.dtype,device=gradient.device); bb=aa.clone(); aa[10+a]=1.; bb[10+b]=1.
    return gradient@(torch.outer(aa,bb)-torch.outer(bb,aa)).T


def _r_field(gradient: torch.Tensor, coordinate: int) -> torch.Tensor:
    aa=torch.zeros(gradient.shape[-1],dtype=gradient.dtype,device=gradient.device); aa[10+coordinate]=1.
    return -(gradient@aa)[...,None]*aa


def _selection_gradient(model: GROM, samples, histories: torch.Tensor):
    """The registered Euler residual and its derivative w.r.t. internal f."""
    x=model.structured.interface_to_state(torch.cat((histories[:,-1],model.encode(histories)),-1))
    shifted=tensor(np.stack([event['x'][endpoint-9:endpoint+2] for event,endpoint in samples]))
    nxt=model.structured.interface_to_state(torch.cat((shifted[:,-1],model.encode(shifted)),-1))
    f=model.structured.field(x); scale=_selection_scale(model,x)
    residual=(x+DT*f-nxt)/scale; loss=residual.square().mean()
    gradient=torch.autograd.grad(loss,f)[0].detach(); J,R=model.structured.matrices()
    return x.detach(),model.structured.energy_and_gradient(x)[1].detach(),gradient,J.detach(),R.detach(),scale


def _best_candidate(model: GROM, x, energy_gradient, sensitivity, J, R, scale, kind: str):
    """Stream legal candidates, retaining one deterministic best score only."""
    best=None
    def consider(key, field, payload):
        nonlocal best
        rms=_rms(field/scale)
        if float(rms) <= 1e-12: return
        derivative=float((sensitivity*(field/rms)).sum())
        if kind == 'R' and derivative >= 0: return
        score=abs(derivative) if kind != 'R' else -derivative
        item=(score, tuple(key), derivative, payload, float(rms))
        if best is None or score>best[0] or (score==best[0] and item[1]<best[1]): best=item
    # Existing groups are represented by their exact primitive/direction keys.
    existing={}
    for spec in model.structured.storage_specs: existing[("H",spec.kind,tuple(spec.indices))]=True
    for spec in model.structured.exchange_specs: existing[("J",tuple(spec.a),tuple(spec.b))]=True
    for spec in model.structured.dissipation_specs: existing[("R",tuple(spec.a))]=True
    # Base diagonal damping already spans each pi coordinate; base Rpp already
    # spans each p coordinate.  They are legal learned base coefficients, not
    # new R directions, so lazy growth may only add genuinely absent directions.
    if kind == 'R':
        for index, active in enumerate(model.structured.damping_mask.detach().cpu().numpy()):
            if active != 0: existing[("R",_basis(23,index))]=True
        for index, active in enumerate(model.structured.rpp_mask.detach().cpu().numpy()):
            if active != 0: existing[("R",_basis(23,10+index))]=True
    if kind == 'H-product':
        for i in range(13):
            for j in range(i+1,13):
                if i<10 and j<10: continue
                spec=PrimitiveSpec('product',(i,j),1e-3,(float(model.structured.scales[i]),float(model.structured.scales[j])))
                if ('H',spec.kind,tuple(spec.indices)) not in existing: consider((i,j),_h_field(model,x,spec,J,R),spec)
    elif kind in ('H-sin','H-cos'):
        primitive='state_sin' if kind=='H-sin' else 'state_cos'
        for q in range(10,13):
            for edge in range(26):
                spec=PrimitiveSpec(primitive,(q,edge),1e-3,(float(model.structured.scales[q]),),float(model.structured.phi[edge]))
                if ('H',spec.kind,tuple(spec.indices)) not in existing: consider((q,edge),_h_field(model,x,spec,J,R),spec)
    elif kind == 'J':
        for a in range(23):
            for b in range(a+1,23):
                if a<20 and b<20: continue
                aa,bb=_basis(23,a),_basis(23,b)
                if ('J',aa,bb) not in existing: consider((a,b),_j_field(energy_gradient,a,b),(a,b))
    elif kind == 'R':
        for a in range(23):
            aa=_basis(23,a)
            if ('R',aa) not in existing: consider((a,),_r_field(energy_gradient,a),a)
    else: raise ValueError(kind)
    return best


def grow_auto(model: GROM, train_events, rng: np.random.Generator, optimizer: torch.optim.Adam, windows=None) -> dict[str,Any]:
    """One registered eight-slot TRAIN-only growth opportunity."""
    samples,histories=sample_batch(train_events,rng,BATCH) if windows is None else windows
    x,e,g,J,R,scale=_selection_gradient(model,samples,histories)
    report={"event_ids":[event['id'] for event,_ in samples],"endpoints":[int(i) for _,i in samples],"proposals":[]}
    for kind in ('H-product','H-product','H-sin','H-cos','J','J','R','R'):
        item=_best_candidate(model,x,e,g,J,R,scale,kind)
        if item is None or item[0] <= 1e-12:
            report['proposals'].append({"kind":kind,"missing":True}); continue
        _,key,derivative,payload,rms=item
        if kind.startswith('H'):
            payload=PrimitiveSpec(payload.kind,payload.indices,-np.sign(derivative)*1e-3,payload.scale,payload.phi,payload.name); params=model.add_storage(payload,1.)
        elif kind=='J': params=model.add_exchange(payload[0],payload[1],-np.sign(derivative)*1e-3)
        else: params=model.add_dissipation(payload,1e-3)
        add_growth_parameters(optimizer,model,params)
        report['proposals'].append({"kind":kind,"key":list(key),"derivative":derivative,"normalized_rms":rms,"missing":False})
    return report


def prune_g(model: GROM, train_events, rng: np.random.Generator, optimizer: torch.optim.Adam, windows=None) -> dict[str,Any]:
    """Freeze (rather than delete) whole extra groups and whole port columns."""
    samples,histories=sample_batch(train_events,rng,BATCH) if windows is None else windows
    z=torch.cat((histories[:,-1],model.encode(histories)),-1); x=model.structured.interface_to_state(z)
    total=_rms(_internal_normalized(model.field(z),model)); threshold=1e-3*total
    J,R=model.structured.matrices(); _,energy_gradient=model.structured.energy_and_gradient(x); n=model.structured.n
    drop=[]; rows=[]
    for kind,index,name in model.extra_groups:
        if kind=='H': field=_h_field(model,x,model.structured.storage_specs[index],J,R)
        if kind=='H': field=field*model.structured.storage_gamma[index]
        elif kind=='J':
            spec=model.structured.exchange_specs[index]; field=_j_field(energy_gradient,int(np.argmax(spec.a)),int(np.argmax(spec.b))) * model.structured.exchange_kappa[index]
        else:
            a=model.structured.dissipation_a[index]; aa=torch.zeros(model.structured.dim,dtype=x.dtype); aa[n:]=a/torch.linalg.vector_norm(a)
            field=-(energy_gradient@aa)[...,None]*aa*torch.nn.functional.softplus(model.structured.dissipation_rho[index])*model.structured.dissipation_mask[index]
        value=_rms(field/_selection_scale(model,x)); rows.append({"kind":kind,"index":index,"name":name,"normalized_rms":float(value)})
        if value < threshold: drop.append((kind,index))
    port_drop=[]
    pv=model.structured.port_values(x); B=model.structured.port_matrix()
    for col in range(pv.shape[-1]):
        value=_rms(_internal_normalized(model.structured.state_to_interface(pv[:,col,None]*B[:,col]),model)); rows.append({"kind":"port","index":col,"normalized_rms":float(value)})
        if value < threshold: port_drop.append(col)
    _freeze_pruned(model,drop,optimizer)
    for col in port_drop:
        p=model.structured.port_coefficient[col]
        direction=model.structured.port_direction[col]
        with torch.no_grad():
            p.zero_()
            for parameter in (p,direction):
                # Keep a valid unit-direction coordinate for numeric reload;
                # a zero, frozen coefficient vector removes the entire port.
                parameter.requires_grad_(False)
                for value in optimizer.state.get(parameter,{}).values():
                    if torch.is_tensor(value): value.zero_()
    return {"event_ids":[event['id'] for event,_ in samples],"endpoints":[int(i) for _,i in samples],"total_normalized_rms":float(total),"threshold":float(threshold),"groups_before":rows,"pruned_groups":[list(x) for x in drop],"pruned_ports":port_drop}


def load_model(model_dir: Path, base: MechanicalROM | None = None,
               source_model: JointROM | None = None) -> _ObserverDecoder:
    """Model-only reload protocol used by independent evaluation.

    The numeric export reconstructs the final appended G lists *before*
    loading ``model.pt``.  This is essential for G-auto because its final
    ParameterList length is not implied by the initial arm name.
    """
    meta=json.loads((model_dir/'checkpoint.json').read_text())
    if base is None: base=MechanicalROM.load(BASE)
    if source_model is None:
        network=dict(np.load(NETWORK,allow_pickle=False)); network['observed_buses']=np.unique(np.r_[network['load_buses'],network['generator_buses']]); network['load_positions']=np.searchsorted(network['observed_buses'],network['load_buses'])
        source_model=_construct(Path(meta.get('source',SOURCE)),base,network,'challenger')
    arm=meta['arm']
    if arm == 'C-free':
        model=CFreeROM(source_model)
    else:
        model=GROM(base,source_model,arm)
        model.structured=StructuredROM.from_export(model_dir/'structured_numeric_export.json').double()
    model.load_state_dict(torch.load(model_dir/'model.pt',map_location='cpu',weights_only=True),strict=True)
    for name,p in model.named_parameters(): p.requires_grad_(bool(meta.get('requires_grad',{}).get(name,p.requires_grad)))
    if isinstance(model,GROM): model.extra_groups=[tuple(v) for v in meta.get('extra_groups',model.extra_groups)]
    model.eval(); return model


def _source_hashes(source: Path) -> dict[str,str]:
    files={"source_model":source/'model.pt',"source_normalization":source/'normalization.npz',
           "base":BASE,"network":NETWORK,"code":Path(__file__),
           "structured_core":ROOT/'experiments'/'structured_rom.py',
           "grammar":ROOT/'experiments'/'structure_grammar.py',
           "0058_model":ROOT/'private_data'/'task2_joint_0058'/'model.pt',
           "0058_normalization":ROOT/'private_data'/'task2_joint_0058'/'normalization.npz',
           "pilot_index":ROOT/'private_data'/'ieee39_postfault'/'dataset_index.csv',
           "expansion_index":ROOT/'private_data'/'ieee39_expansion'/'dataset_index.csv',
           "pilot_splits":ROOT/'private_data'/'ieee39_postfault'/'splits.json',
           "expansion_splits":ROOT/'private_data'/'ieee39_expansion'/'splits.json'}
    return {name:digest(path) for name,path in files.items()}


def _checkpoint(output: Path, name: str, model: _ObserverDecoder, optimizer: torch.optim.Optimizer,
                fit_rng: np.random.Generator, selection_rng: np.random.Generator, source: Path,
                arm: str, *, step: int) -> Path:
    """Write a self-contained, strict-loadable checkpoint without overwriting another phase."""
    directory=output/name
    if directory.exists(): raise FileExistsError(directory)
    directory.mkdir()
    torch.save(model.state_dict(),directory/'model.pt'); torch.save(optimizer.state_dict(),directory/'optimizer.pt')
    torch.save({"fit_numpy":fit_rng.bit_generator.state,"selection_numpy":selection_rng.bit_generator.state,
                "torch":torch.get_rng_state()},directory/'rng_state.pt')
    if isinstance(model,GROM):
        counts=model.structured.export(directory/'structured_numeric_export.json')['counts']
        counts['nonzero_groups']={
            'H':sum(int(p.detach()!=0) for p in model.structured.storage_gamma),
            'J':sum(int(p.detach()!=0) for p in model.structured.exchange_kappa),
            'R':int(model.structured.dissipation_mask.sum()),
            'ports':sum(int(torch.count_nonzero(p.detach())>0) for p in model.structured.port_coefficient)}
    else:
        raw=sum(p.numel() for n,p in model.named_parameters() if p.requires_grad and not n.startswith(('gru.','head.','decoder.')))
        masked=int(model.core.support.numel()-model.core.support.sum()+model.core.alpha_support.numel()-model.core.alpha_support.sum())
        spatial_nnz=(model.core.spatial().detach()*model.core.amplitude.detach()!=0).sum(0)
        base_terms=10+10+10+20+int(torch.count_nonzero(model.active_incidence))
        closure_terms=int(model.core.support.sum()+spatial_nnz.sum()+(model.core.alpha_support.sum(0)*spatial_nnz).sum())
        counts={'raw_dynamics_scalar_count':raw,'masked_inactive_scalar_coordinates':masked,
                'normalized_direction_radial_redundancies':3,
                'effective_parameter_coordinate_upper_bound':raw-masked-3,
                'active_xi':int(model.core.support.sum()),'active_alpha':int(model.core.alpha_support.sum()),
                'expanded_rhs_uncollected_scalar_contributions':{'base':base_terms,'closure':closure_terms,'total':base_terms+closure_terms}}
    counts['shared_observer_decoder_trainable_scalars']=sum(p.numel() for n,p in model.named_parameters()
        if p.requires_grad and n.startswith(('gru.','head.','decoder.')))
    shutil.copyfile(source/'normalization.npz',directory/'normalization.npz')
    parameter_names={id(p):name for name,p in model.named_parameters()}
    meta={"arm":arm,"step":step,"source":str(source),"extra_groups":getattr(model,'extra_groups',[]),
          "requires_grad":{name:bool(p.requires_grad) for name,p in model.named_parameters()},
          "optimizer_parameter_names":[[parameter_names[id(p)] for p in group['params']] for group in optimizer.param_groups],
          "counts":counts}
    (directory/'checkpoint.json').write_text(json.dumps(meta,indent=2),encoding='utf-8')
    # Verify this is independently reconstructible before reporting it.
    rng_before_reload=torch.get_rng_state()
    reloaded=load_model(directory)
    torch.set_rng_state(rng_before_reload)
    probe=torch.linspace(-.02,.02,66,dtype=torch.float64).reshape(2,33)
    if not torch.allclose(model.field(probe),reloaded.field(probe),atol=1e-12,rtol=0):
        raise AssertionError(f'checkpoint reload mismatch: {name}')
    return directory


def _train_monitor(model: _ObserverDecoder, events: list[dict[str,Any]]) -> dict[str,Any]:
    """Equal-event full-TRAIN 1/2/5s risk; no batch-tail surrogate is used."""
    if len(events) != 346: raise AssertionError('monitor requires all 346 TRAIN events')
    was_training=model.training; model.eval(); rows=[]; summaries={}
    with torch.no_grad():
        if isinstance(model,GROM): model.begin_graph_cache()
        try:
          for horizon in (1.,2.,5.):
            values=[]
            for start in range(0,len(events),BATCH):
                block=events[start:start+BATCH]; histories=tensor(np.stack([e['x'][:11] for e in block]))
                errors=_event_rollout_errors(model,[(e,10) for e in block],histories,horizon)
                for event,value in zip(block,errors.detach().cpu().numpy(),strict=True):
                    value=float(value); values.append(value); rows.append({"event_id":event['id'],"horizon_s":horizon,"error":value})
            values=np.asarray(values); tail=np.sort(values)[-35:]
            summaries[str(int(horizon))]={"mean":float(values.mean()),"tail35":float(tail.mean()),
                                           "risk":float(.8*values.mean()+.2*tail.mean()),"n_events":len(values)}
        finally:
          if isinstance(model,GROM): model.end_graph_cache()
    if was_training: model.train()
    return {"per_event":rows,"horizons":summaries,"average_risk":float(np.mean([v['risk'] for v in summaries.values()]))}


def _initial_diagnostic(model: _ObserverDecoder, source_model: JointROM, base: MechanicalROM,
                        train: list[dict[str,Any]], rng: np.random.Generator) -> dict[str,Any]:
    """Fixed TRAIN-only, non-gating field and 1s rollout comparison to S/M3."""
    samples,histories=sample_batch(train,rng,BATCH)
    with torch.no_grad():
        z=torch.cat((histories[:,-1],model.encode(histories)),-1); candidate_field=model.field(z)[:,:30]
        source_state=torch.cat((histories[:,-1],source_model.encode(histories)),-1); source_field=source_model.field(source_state)[:,:30]
        raw=histories[:,-1].detach().cpu().numpy().copy(); raw[:,20:30]+=base.pm_star
        m3_field=torch.as_tensor(base.field(raw),dtype=torch.float64)
        candidate_roll=model.rollout(histories,100,DT)[...,:30]
        source_roll=source_model.rollout(histories,100,DT)[...,:30]
        m3_roll=base.rollout(raw,None,np.arange(101)*DT,max_step=DT); m3_roll[...,20:30]-=base.pm_star
    return {"event_ids":[e['id'] for e,_ in samples],"endpoints":[int(i) for _,i in samples],
            "field_candidate_minus_S_rms":float(_rms(candidate_field-source_field)),
            "field_candidate_minus_M3_rms":float(_rms(candidate_field-m3_field)),
            "rollout1s_candidate_minus_S_rms":float(_rms(candidate_roll-source_roll)),
            "rollout1s_candidate_minus_M3_rms":float(_rms(candidate_roll-torch.as_tensor(m3_roll)))}


def _invariants(model: _ObserverDecoder) -> dict[str,Any]:
    zero=_assert_structure(model)
    probe=torch.linspace(-.03,.03,66,dtype=torch.float64).reshape(2,33); field=model.field(probe)
    answer={"f0":zero,"probe_finite":bool(torch.isfinite(field).all()),"probe_nonzero":float(field.abs().max()),"kinematic_max_abs":float((field[:,:10]-probe[:,10:20]).abs().max())}
    if not answer['probe_finite'] or answer['probe_nonzero'] == 0. or answer['kinematic_max_abs'] > 1e-12: raise AssertionError(f'field probe: {answer}')
    return answer


def _failure(output: Path, model: _ObserverDecoder, *, arm: str, step: int, error: Exception,
             traces: list[dict[str,Any]], hashes: dict[str,str]) -> None:
    torch.save(model.state_dict(),output/'failure_model.pt')
    (output/'failure_evidence.json').write_text(json.dumps({"arm":arm,"step":step,"exception_type":type(error).__name__,
        "reason":str(error),"traces":traces,"hashes_before":hashes,"hashes_after":_source_hashes(Path(hashes['_source_path'])),
        "test_read":False},indent=2),encoding='utf-8')


def run(arm: str, output: Path, *, source: Path = SOURCE, smoke: bool = False) -> dict[str,Any]:
    """Formal execution entry point.  ``--smoke`` performs no optimizer update."""
    if output.exists(): raise FileExistsError(output)
    output.mkdir(parents=True); started=time.perf_counter(); git_commit=__import__('subprocess').check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(); torch.set_num_threads(1); torch.manual_seed(SEED)
    fit_rng=np.random.default_rng(SEED); selection_rng=np.random.default_rng(SELECTION_SEED); traces=[]; step=0
    base=MechanicalROM.load(BASE); network=dict(np.load(NETWORK,allow_pickle=False)); network['observed_buses']=np.unique(np.r_[network['load_buses'],network['generator_buses']]); network['load_positions']=np.searchsorted(network['observed_buses'],network['load_buses'])
    source_model=_construct(source,base,network,'challenger'); model=CFreeROM(source_model) if arm=='C-free' else GROM(base,source_model,arm)
    hashes={**_source_hashes(source),"_source_path":str(source)}; optimizer=optimizer_for(model)
    try:
        initial_checks=_invariants(model)
        # The only event/H5 access is here, after all source provenance checks.
        events,excluded,identities=read_development(base,network,full_train=True); train=events['train']
        if len(train)!=346: raise AssertionError('unexpected explicit TRAIN count')
        if smoke:
            samples,histories=sample_batch(train,fit_rng,BATCH); before={n:v.detach().clone() for n,v in model.state_dict().items()}
            if isinstance(model,GROM): model.begin_graph_cache()
            try: loss,terms=objective(model,samples,histories,horizon_for_step(1))
            finally:
                if isinstance(model,GROM): model.end_graph_cache()
            if not torch.isfinite(loss): raise FloatingPointError('nonfinite smoke loss')
            loss.backward(); grad_rows=[]
            for name,p in model.named_parameters():
                if p.requires_grad:
                    if p.grad is None or not torch.isfinite(p.grad).all(): raise FloatingPointError(f'smoke gradient: {name}')
                    grad_rows.append({"name":name,"norm":float(p.grad.norm())})
            if any(not torch.equal(before[n],v) for n,v in model.state_dict().items()): raise AssertionError('smoke mutated model state')
            if _source_hashes(source) != {k:v for k,v in hashes.items() if k != '_source_path'}: raise AssertionError('smoke source changed')
            traces.append({"step":1,"smoke_no_optimizer_update":True,"loss":float(loss.detach()),"terms":{k:float(v.detach()) for k,v in terms.items()},"gradients":grad_rows})
            result={"arm":arm,"smoke":True,"test_read":False,"hashes":hashes,"hashes_after":_source_hashes(source),"git_commit":git_commit,"state_unchanged":True,"initial_checks":initial_checks,"traces":traces,"excluded":excluded,"identities":identities}
            (output/'evidence.json').write_text(json.dumps(result,indent=2,allow_nan=False)); return result
        _checkpoint(output,'step0000',model,optimizer,fit_rng,selection_rng,source,arm,step=0)
        print(json.dumps({"event":"train_monitor_start","step":0}),flush=True)
        monitors={"0":_train_monitor(model,train)}
        (output/'monitor_step0000.json').write_text(json.dumps(monitors['0'],indent=2),encoding='utf-8'); print(json.dumps({"event":"train_monitor_finish","step":0}),flush=True)
        if not np.isfinite(monitors['0']['average_risk']): raise FloatingPointError('nonfinite TRAIN monitor at step0')
        initial_diagnostic=_initial_diagnostic(model,source_model,base,train,np.random.default_rng(SELECTION_SEED))
        (output/'initial_diagnostic.json').write_text(json.dumps(initial_diagnostic,indent=2),encoding='utf-8')
        selection_windows=None
        for step in range(1,STEPS+1):
            optimizer.zero_grad(set_to_none=True); samples,histories=sample_batch(train,fit_rng,BATCH)
            if isinstance(model,GROM): model.begin_graph_cache()
            try: loss,terms=objective(model,samples,histories,horizon_for_step(step),complexity_weight=0. if step>1000 else 1e-4)
            finally:
                if isinstance(model,GROM): model.end_graph_cache()
            if not torch.isfinite(loss): raise FloatingPointError('nonfinite loss')
            loss.backward(); gradients=[]
            for name,p in model.named_parameters():
                if p.requires_grad:
                    if p.grad is None: raise FloatingPointError(f'missing required gradient: {name}')
                    if not torch.isfinite(p.grad).all(): raise FloatingPointError(f'nonfinite gradient: {name}')
                    gradients.append(p.grad.reshape(-1))
            raw_norm=float(torch.linalg.vector_norm(torch.cat(gradients))); clipped=float(torch.nn.utils.clip_grad_norm_(_allowed(model),100.)); optimizer.step()
            if any(not torch.isfinite(p).all() for p in _allowed(model)): raise FloatingPointError('nonfinite parameter')
            checks=_invariants(model)
            traces.append({"step":step,"horizon_s":horizon_for_step(step),"loss":float(loss.detach()),"gradient_norm_before_cap":raw_norm,"gradient_norm_clip_return":clipped,"terms":{k:float(v.detach()) for k,v in terms.items()},"checks":checks,"wall_seconds":time.perf_counter()-started})
            if step%25==0: print(json.dumps({"step":step,"loss":traces[-1]['loss'],"seconds":traces[-1]['wall_seconds']}),flush=True)
            if step==400:
                selection_windows=sample_batch(train,selection_rng,BATCH)
                if isinstance(model,GROM) and model.arm=='G-auto':
                    traces.append({"step":step,"growth":grow_auto(model,train,selection_rng,optimizer,selection_windows)})
                _checkpoint(output,'step0400_after_growth',model,optimizer,fit_rng,selection_rng,source,arm,step=step)
            if step==800:
                print(json.dumps({"event":"train_monitor_start","step":800}),flush=True)
                monitors['800_before_growth']=_train_monitor(model,train)
                (output/'monitor_step0800_before_growth.json').write_text(json.dumps(monitors['800_before_growth'],indent=2),encoding='utf-8'); print(json.dumps({"event":"train_monitor_finish","step":800}),flush=True)
                if not np.isfinite(monitors['800_before_growth']['average_risk']): raise FloatingPointError('nonfinite TRAIN monitor at step800')
                _checkpoint(output,'step0800_before_growth',model,optimizer,fit_rng,selection_rng,source,arm,step=step)
                selection_windows=sample_batch(train,selection_rng,BATCH)
                if isinstance(model,GROM) and model.arm=='G-auto': traces.append({"step":step,"growth":grow_auto(model,train,selection_rng,optimizer,selection_windows)})
                _checkpoint(output,'step0800_after_growth',model,optimizer,fit_rng,selection_rng,source,arm,step=step)
            if step==1000:
                if selection_windows is None: raise AssertionError('missing second selection windows')
                _checkpoint(output,'step1000_before_prune',model,optimizer,fit_rng,selection_rng,source,arm,step=step)
                pruning=model.prune(selection_windows[1],optimizer) if isinstance(model,CFreeROM) else prune_g(model,train,selection_rng,optimizer,selection_windows)
                traces.append({"step":step,"pruning":pruning}); _checkpoint(output,'step1000_after_prune',model,optimizer,fit_rng,selection_rng,source,arm,step=step)
        print(json.dumps({"event":"train_monitor_start","step":1200}),flush=True)
        monitors['1200']=_train_monitor(model,train)
        (output/'monitor_step1200.json').write_text(json.dumps(monitors['1200'],indent=2),encoding='utf-8'); print(json.dumps({"event":"train_monitor_finish","step":1200}),flush=True)
        if not np.isfinite(monitors['1200']['average_risk']): raise FloatingPointError('nonfinite TRAIN monitor at step1200')
        final_dir=_checkpoint(output,'step1200',model,optimizer,fit_rng,selection_rng,source,arm,step=STEPS)
        after={**_source_hashes(source),"_source_path":str(source)}
        if after != hashes: raise AssertionError('source artifact changed during fit')
        evidence={"arm":arm,"seed":SEED,"steps":STEPS,"smoke":False,"test_read":False,"hashes":hashes,"hashes_after":after,"git_commit":git_commit,"initial_checks":initial_checks,"initial_diagnostic":initial_diagnostic,"monitors":monitors,"traces":traces,"excluded":excluded,"identities":identities,"final_checkpoint":str(final_dir),"seconds":time.perf_counter()-started}
        (output/'evidence.json').write_text(json.dumps(evidence,indent=2,allow_nan=False)); return evidence
    except Exception as error:
        _failure(output,model,arm=arm,step=step,error=error,traces=traces,hashes=hashes)
        raise


def main() -> None:
    parser=argparse.ArgumentParser(); parser.add_argument('--arm',choices=('C-free','G-fixed','G-auto'),required=True); parser.add_argument('--output',type=Path,required=True); parser.add_argument('--source',type=Path,default=SOURCE); parser.add_argument('--smoke',action='store_true')
    args=parser.parse_args(); run(args.arm,args.output,source=args.source,smoke=args.smoke)


if __name__ == '__main__': main()
