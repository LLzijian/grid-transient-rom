"""Fixed-action memory model for the registered 2026-10-25 comparison.

The interface is [delta, omega, p, r]; energy and matrices use pi=M*omega.
Only the strict M3 backbone is copied.  No source q template or added port is
retained.  B/C differ exclusively by C's parameter-free causal initial-state
path.  This module reads no event data and performs no fitting.
"""
from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
from torch import nn

from structured_rom import StructuredROM

FORMAT = "r_memory_model_20261025"
_PHYSICAL = ("coupling", "damping", "jpi_p", "rpp", "storage_base")


def _cpu(value: Any) -> torch.Tensor:
    return torch.as_tensor(value, dtype=torch.float64).detach().cpu().clone()


def _list(value: Any):
    return _cpu(value).tolist()


def _source_config(source: Any) -> dict[str, Any]:
    """Extract fixed features and the effective strict M3 parameters only."""
    if isinstance(source, Mapping):
        return copy.deepcopy(dict(source))
    core = getattr(source, "structured", source)
    if not isinstance(core, StructuredROM):
        raise TypeError("source must contain a StructuredROM M3 backbone")
    D, mass = _cpu(core.D), _cpu(core.mass)
    cfg = {"mass": mass.tolist(), "D": D.tolist(),
           "delta_star": _list(core.delta_star), "gain": _list(getattr(source, "gain", core.gain)),
           "rate": _list(getattr(source, "rate", torch.ones(10))),
           "base_coupling": _list(getattr(source, "base_coupling", core._pos("coupling"))),
           "input_scale_delta": _list(getattr(source, "input_scale_delta", torch.ones(10))),
           "input_scale_omega": _list(getattr(source, "input_scale_omega", torch.ones(10))),
           "p_scale": _list(getattr(source, "p_scale", torch.ones(10)))}
    for name in _PHYSICAL:
        cfg[name] = _list(core._pos(name)[:10] if name == "storage_base" else core._pos(name))
    # Copy the raw coordinates as well: softplus->inverse-softplus need not
    # round-trip bitwise.  These tensors have no old memory components.
    cfg["backbone_raw"] = {}
    for name in _PHYSICAL:
        for suffix in ("", "_mask"):
            value = getattr(core, name + suffix)
            if name == "storage_base":
                value = value[:10]
            cfg["backbone_raw"][name + suffix] = _list(value)
    if hasattr(source, "neighbor_A"):
        cfg["neighbor_A"] = _list(source.neighbor_A)
    if hasattr(source, "active_edge"):
        cfg["active_edge"] = source.active_edge.detach().cpu().tolist()
    return cfg


class RMemoryROM(nn.Module):
    """Forty-state strict-M3 plus one shared exponential memory per machine.

    kappa=kappa_reference*exp(kappa_raw), lambda=exp(lambda_raw)/s.
    Thus this implementation uses the strictly positive lambda subfamily of
    the registered nonnegative family.  There are two independent memory
    coefficients, shared by all machines.  rstar is a frozen physical scale.
    """
    def __init__(self, source: Any, mode: str = "B", seed: int = 25101,
                 source_identity: Mapping[str, Any] | None = None):
        super().__init__()
        if mode not in ("B", "C"):
            raise ValueError("mode must be B or C")
        self.mode, self.arm, self.seed = mode, "r-memory-" + mode, int(seed)
        self.source_identity = copy.deepcopy(dict(source_identity or {}))
        cfg = _source_config(source)
        mass, D = _cpu(cfg["mass"]), _cpu(cfg["D"])
        if mass.shape != (10,) or D.ndim != 2 or D.shape[1] != 10 or not len(D):
            raise ValueError("ten-machine nonempty fixed M3 graph required")
        if not torch.isfinite(mass).all() or torch.any(mass <= 0):
            raise ValueError("finite positive M3 masses required")
        self.structured = StructuredROM(
            mass=mass, D=D, coupling=cfg["coupling"], delta_star=cfg["delta_star"],
            damping=cfg["damping"], gain=cfg["gain"], jpi_p=cfg["jpi_p"],
            rpp=cfg["rpp"], storage_base=cfg["storage_base"], n_q=0,
            storage=(), exchange=(), dissipation=(), ports=(), trainable=True).double()
        if "backbone_raw" in cfg:
            with torch.no_grad():
                for name, value in cfg["backbone_raw"].items():
                    target = getattr(self.structured, name)
                    target.copy_(torch.as_tensor(value, dtype=target.dtype))
        for name, value in (("mass", mass), ("base_D", D),
                            ("delta_star", cfg["delta_star"]),
                            ("base_coupling", cfg["base_coupling"]),
                            ("rate", cfg["rate"]), ("gain", cfg["gain"]),
                            ("input_scale_delta", cfg["input_scale_delta"]),
                            ("input_scale_omega", cfg["input_scale_omega"]),
                            ("p_scale", cfg["p_scale"])):
            self.register_buffer(name, _cpu(value))
        for name in ("base_coupling", "rate", "gain", "input_scale_delta", "input_scale_omega", "p_scale"):
            value = getattr(self, name)
            expected = len(D) if name == "base_coupling" else 10
            if value.shape != (expected,) or not torch.isfinite(value).all() or torch.any(value <= 0):
                raise ValueError(f"{name} must be a finite positive vector of length {expected}")
        self.register_buffer("input_scale", torch.cat((self.input_scale_delta, self.input_scale_omega)))
        self.register_buffer("phi", D @ self.delta_star)
        if "neighbor_A" in cfg:
            adjacency = _cpu(cfg["neighbor_A"])
        else:
            adjacency = D.abs().T @ torch.diag(self.base_coupling) @ D.abs()
            adjacency.fill_diagonal_(0.)
            adjacency = adjacency / adjacency.sum(-1, keepdim=True).clamp_min(1e-15)
        if adjacency.shape != (10, 10) or not torch.isfinite(adjacency).all():
            raise ValueError("fixed neighbor_A must be a finite 10x10 matrix")
        self.register_buffer("neighbor_A", adjacency)
        self.register_buffer("active_edge", torch.as_tensor(cfg.get("active_edge", [True] * len(D)), dtype=torch.bool))
        self.register_buffer("W", torch.eye(10, dtype=torch.float64))
        mstar = mass.log().mean().exp()
        self.register_buffer("kappa_reference", mstar.clone())  # tau*=1 second
        self.register_buffer("lambda_reference", torch.tensor(1., dtype=torch.float64))
        self.register_buffer("rstar", mstar * self.input_scale_omega)
        # Times are seconds.  The second polynomial coefficient is the
        # derivative at the right endpoint, using only these eleven points.
        t = np.linspace(-.1, 0., 11)
        design = np.column_stack((np.ones(11), t, t * t))
        self.register_buffer("slope_weights", torch.tensor(np.linalg.pinv(design)[1], dtype=torch.float64))
        self.kappa_raw = nn.Parameter(torch.tensor(0., dtype=torch.float64))
        self.lambda_raw = nn.Parameter(torch.tensor(0., dtype=torch.float64))
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(self.seed)
            self.local_gru = nn.GRU(7, 8, batch_first=True, dtype=torch.float64)
            self.local_head = nn.Linear(8, 1, bias=False, dtype=torch.float64)
            with torch.no_grad():
                self.local_head.weight.zero_()
        self._runtime_cache = None
        self._graph_cache = None
        self._cache_mode = None

    @property
    def n(self):
        return 10

    @property
    def rank(self):
        return 10

    @property
    def dim(self):
        return 40

    @property
    def interface_dim(self):
        return 40

    @property
    def kappa(self):
        return self.kappa_reference * self.kappa_raw.exp()

    @property
    def lambda_(self):
        return self.lambda_reference * self.lambda_raw.exp()

    @classmethod
    def from_source(cls, source, mode="B", seed=25101, source_identity=None):
        return cls(source, mode=mode, seed=seed, source_identity=source_identity)

    def _tensor(self, value):
        return torch.as_tensor(value, dtype=self.mass.dtype, device=self.mass.device)

    def _clear_runtime(self):
        self._runtime_cache = self._graph_cache = self._cache_mode = None

    def _apply(self, fn, recurse=True):
        self._clear_runtime()
        return super()._apply(fn, recurse=recurse)

    def load_state_dict(self, *args, **kwargs):
        self._clear_runtime()
        return super().load_state_dict(*args, **kwargs)

    def _constants(self):
        core = self.structured
        return (core._pos("coupling"), core._pos("damping"), core._pos("jpi_p"),
                core._pos("rpp"), core._pos("storage_base"), self.kappa,
                self.lambda_, self.phi.sin())

    def _current_constants(self):
        mode = (torch.is_grad_enabled(), torch.is_inference_mode_enabled())
        if self._runtime_cache is not None and self._cache_mode != mode:
            self._clear_runtime()
        return self._runtime_cache if self._runtime_cache is not None else self._constants()

    def begin_graph_cache(self):
        if self._runtime_cache is not None:
            raise RuntimeError("nested r-memory training graph cache")
        self._cache_mode = (torch.is_grad_enabled(), torch.is_inference_mode_enabled())
        self._runtime_cache = self._constants()

    def end_graph_cache(self):
        self._clear_runtime()

    def base_momentum_field(self, physical):
        """Strict M3 pi derivative, never an old 40-state slice at q=0."""
        z = self._tensor(physical)
        if z.shape[-1] != 30:
            raise ValueError("physical state width must be 30")
        d, w, p = z.split(10, dim=-1)
        k, damping, G, _, Sp, _, _, sin0 = self._current_constants()
        gd = ((torch.sin(self.phi + d @ self.base_D.T) - sin0) * k) @ self.base_D
        return -gd - damping * w + G * Sp * p

    def _features(self, history):
        d, w = history[..., :10], history[..., 10:20]
        coi = d - (d * self.mass).sum(-1, keepdim=True) / self.mass.sum()
        dn, wn = coi / self.input_scale_delta, w / self.input_scale_omega
        nd = (coi @ self.neighbor_A.T - coi) / self.input_scale_delta
        nw = (w @ self.neighbor_A.T) / self.input_scale_omega
        m, rate, gain = (v / v.mean().clamp_min(1e-15) for v in (self.mass, self.rate, self.gain))
        return torch.stack((dn, wn, nd, nw, m.expand_as(dn), rate.expand_as(dn), gain.expand_as(dn)), -1)

    def endpoint_slope(self, history):
        h = self._tensor(history)
        if h.shape[-2:] != (11, 30):
            raise ValueError("exactly eleven causal Nx30 samples required")
        return torch.einsum("t,...tn->...n", self.slope_weights, h[..., 10:20])

    def encode(self, history):
        h = self._tensor(history)
        if h.shape[-2:] != (11, 30):
            raise ValueError("encoder requires exactly eleven 30-column causal samples")
        scalar = h.ndim == 2
        if scalar:
            h = h.unsqueeze(0)
        if h.ndim != 3:
            raise ValueError("history must have shape [batch,11,30] or [11,30]")
        b = len(h)
        seq = self._features(h).permute(0, 2, 1, 3).reshape(b * 10, 11, 7)
        origin = self._features(torch.zeros_like(h)).permute(0, 2, 1, 3).reshape(b * 10, 11, 7)
        raw = self.local_head(self.local_gru(seq)[0][:, -1]).reshape(b, 10)
        raw0 = self.local_head(self.local_gru(origin)[0][:, -1]).reshape(b, 10)
        r = self.rstar * (raw - raw0)
        if self.mode == "C":
            # Only p0 (the final physical row) enters this direct path.
            r = r + self.base_momentum_field(h[:, -1]) - self.mass * self.endpoint_slope(h)
        return r[0] if scalar else r

    def interface_to_state(self, z):
        z = self._tensor(z)
        if z.shape[-1] != 40:
            raise ValueError("interface state width must be 40")
        return torch.cat((z[..., :10], z[..., 10:20] * self.mass, z[..., 20:]), -1)

    def state_to_interface(self, x):
        x = self._tensor(x)
        if x.shape[-1] != 40:
            raise ValueError("internal state width must be 40")
        return torch.cat((x[..., :10], x[..., 10:20] / self.mass, x[..., 20:]), -1)

    def field(self, state, u=None):
        if u is not None:
            raise ValueError("the strict r-memory candidate has no external ports")
        z = self._tensor(state)
        if z.shape[-1] != 40:
            raise ValueError("interface state width must be 40")
        d, w, p, r = z.split(10, dim=-1)
        k, damping, G, rpp, Sp, kappa, lam, sin0 = self._current_constants()
        gd = ((torch.sin(self.phi + d @ self.base_D.T) - sin0) * k) @ self.base_D
        pi_dot = -gd - damping * w + G * Sp * p - r
        return torch.cat((w, pi_dot / self.mass, -G * w - rpp * Sp * p, kappa * w - lam * r), -1)

    def internal_field(self, internal, u=None):
        return self.interface_to_state(self.field(self.state_to_interface(internal), u=u))

    def energy_and_gradient(self, internal):
        x = self._tensor(internal)
        if x.shape[-1] != 40:
            raise ValueError("internal state width must be 40")
        base_H, base_gradient = self.structured.energy_and_gradient(x[..., :30])
        r = x[..., 30:]
        return base_H + .5 * (r.square() / self.kappa).sum(-1), torch.cat((base_gradient, r / self.kappa), -1)

    def energy(self, internal):
        return self.energy_and_gradient(internal)[0]

    def gradient(self, internal, create_graph=True):
        return self.energy_and_gradient(internal)[1]

    def matrices(self):
        j0, r0 = self.structured.matrices()
        z = self.mass.new_zeros((40, 40))
        J, R = z.clone(), z.clone()
        J[:30, :30], R[:30, :30] = j0, r0
        eye = torch.eye(10, dtype=z.dtype, device=z.device)
        J[10:20, 30:40], J[30:40, 10:20] = -self.kappa * eye, self.kappa * eye
        R[30:40, 30:40] = self.lambda_ * self.kappa * eye
        return J, R

    def port_matrix(self):
        return self.mass.new_zeros((40, 0))

    def port_values(self, internal):
        x = self._tensor(internal)
        return x.new_zeros(x.shape[:-1] + (0,))

    def rk4(self, state, dt):
        state = self._tensor(state)
        if not np.isfinite(dt) or dt <= 0:
            raise ValueError("dt must be finite and positive")
        a = self.field(state)
        b = self.field(state + dt * a / 2)
        c = self.field(state + dt * b / 2)
        d = self.field(state + dt * c)
        return state + dt * (a + 2 * b + 2 * c + d) / 6

    def rollout(self, history, steps, dt):
        h = self._tensor(history)
        if isinstance(steps, bool) or int(steps) != steps or steps < 0:
            raise ValueError("nonnegative integer rollout steps required")
        state = torch.cat((h[..., -1, :], self.encode(h)), -1)
        out = [state]
        for _ in range(int(steps)):
            state = self.rk4(state, dt)
            out.append(state)
        return torch.stack(out, dim=-2)

    def complexity(self, state=None):
        # No within-family structure search or coefficient-size regularizer.
        return self.mass.new_tensor(2.)

    def complexity_counts(self):
        raw = sum(p.numel() for p in self.parameters() if p.requires_grad)
        encoder = sum(p.numel() for name, p in self.named_parameters() if p.requires_grad and ("gru" in name or "head" in name))
        masked = sum(int((getattr(self.structured, name + "_mask") == 0).sum()) for name in _PHYSICAL)
        return {"machines": 10, "states": 40, "rank": 10, "active_edges": len(self.base_D),
                "shared_memory_scalar_coordinates": 2, "raw_trainable_scalar_count": raw,
                "encoder_trainable_scalar_count": encoder,
                "effective_active_parameter_coordinate_upper_bound": raw - masked,
                "expanded_rhs_scalar_contributions": {"total": 2 * len(self.base_D) + 80,
                    "convention": "centered edge sine is one atom per incident node; delta 10, momentum 2m+30, p 20, r 20; before algebraic collection"},
                "scope": "counts are representation conventions, not an identifiable minimum"}

    def config(self):
        cfg = _source_config(self)
        cfg["neighbor_A"] = _list(self.neighbor_A)
        return cfg

    def export(self, path):
        if self._runtime_cache is not None:
            raise RuntimeError("export after ending the active optimizer graph")
        J, R = self.matrices()
        payload = {"format": FORMAT, "model_kind": FORMAT, "mode": self.mode,
                   "seed": self.seed, "source_identity": copy.deepcopy(self.source_identity),
                   "config": self.config(),
                   "state_dict": {k: v.detach().cpu().tolist() for k, v in self.state_dict().items()},
                   "J": J.detach().cpu().tolist(), "R": R.detach().cpu().tolist(),
                   "B": self.port_matrix().detach().cpu().tolist(),
                   "counts": self.complexity_counts(), "lambda_parameterization": "strictly_positive_exponential"}
        Path(path).write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        return payload

    @classmethod
    def from_export(cls, path):
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if payload.get("format") != FORMAT or payload.get("model_kind") != FORMAT:
            raise ValueError("wrong r-memory model format")
        model = cls(payload["config"], mode=payload["mode"], seed=payload["seed"],
                    source_identity=payload.get("source_identity"))
        template = model.state_dict()
        supplied = payload.get("state_dict", {})
        if set(supplied) != set(template):
            raise ValueError("exported state names differ")
        state = {k: torch.as_tensor(supplied[k], dtype=v.dtype) for k, v in template.items()}
        model.load_state_dict(state, strict=True)
        return model


def from_export(path):
    return RMemoryROM.from_export(path)
