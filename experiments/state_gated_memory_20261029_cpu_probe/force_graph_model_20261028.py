"""Fixed-M3 static-force / propagating-force-memory comparison.

Interface: [delta, omega, p, r]. Internal power coordinates use pi=M*omega.
Only kappa/lambda and (for S or M) 45 nonnegative spatial coordinates train.
The 100 ms encoder uses delta/omega history and terminal p0, without p history.
This module contains no event reader, optimizer, or scientific stopping logic.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

FORMAT = "force_graph_model_20261028"
MODES = ("L", "S", "M")
PAIRS = tuple((i, j) for i in range(10) for j in range(i + 1, 10))
RIDGE = 1.e-4


def _cpu(x):
    return torch.as_tensor(x, dtype=torch.float64).detach().cpu().clone()


def strict_m3_config(source):
    """Extract effective M3 values, discarding every old memory/port term.

    A symbolic export's config can predate its optimized raw coordinates, so
    effective structured.* values come from state_dict, never that stale config.
    A plain physical mapping is also accepted for data-free fixtures/exports.
    """
    if "state_dict" not in source:
        return copy.deepcopy(dict(source))
    state = source["state_dict"]
    cfg = source["config"]
    result = {"mass": state["mass"], "D": state["base_D"],
              "delta_star": state["delta_star"],
              "input_scale_delta": state["input_scale_delta"],
              "input_scale_omega": state["input_scale_omega"],
              "p_scale": state["p_scale"],
              "gain": state.get("gain", cfg["gain"]),
              "rate": state.get("rate", cfg["rate"])}
    for name in ("coupling", "damping", "jpi_p", "rpp", "storage_base"):
        value = F.softplus(_cpu(state["structured." + name])) * _cpu(state["structured." + name + "_mask"])
        result[name] = (value[:10] if name == "storage_base" else value).tolist()
    result["deleted_source_components"] = {
        "state_keys": [k for k in state if ("gru" in k or "head" in k or "template" in k
                      or "port" in k or "q_quadratic" in k)],
        "symbolic_terms": copy.deepcopy(source.get("terms", [])),
        "note": "strict M3 only; source q, cross terms, added ports and GRU are absent"}
    return copy.deepcopy(result)


class ForceGraphROM(nn.Module):
    n, rank, dim, interface_dim = 10, 10, 40, 40

    def __init__(self, source, mode="L", *, source_identity=None):
        super().__init__()
        if mode not in MODES:
            raise ValueError("registered modes are L/S/M")
        self.mode = mode
        self.source_identity = copy.deepcopy(source_identity or {})
        cfg = strict_m3_config(source)
        self._source_config = copy.deepcopy(cfg)
        for name in ("mass", "delta_star", "damping", "jpi_p", "rpp", "storage_base",
                     "input_scale_delta", "input_scale_omega", "p_scale", "gain", "rate"):
            value = _cpu(cfg[name])
            if value.shape != (10,) or not torch.isfinite(value).all():
                raise ValueError("finite ten-machine vector required: " + name)
            positive = name in ("mass", "jpi_p", "storage_base", "input_scale_delta", "input_scale_omega", "p_scale", "gain")
            invalid_sign = (value <= 0).any() if positive else (name != "delta_star" and (value < 0).any())
            if invalid_sign:
                raise ValueError("invalid physical sign: " + name)
            self.register_buffer(name, value)
        D, coupling = _cpu(cfg["D"]), _cpu(cfg["coupling"])
        if (D.ndim != 2 or D.shape[1] != 10 or coupling.shape != (len(D),)
                or not torch.isfinite(D).all() or not torch.isfinite(coupling).all()
                or (coupling < 0).any()):
            raise ValueError("fixed M3 incidence and nonnegative coupling required")
        B = torch.zeros(10, 45, dtype=torch.float64)
        for e, (i, j) in enumerate(PAIRS):
            B[i, e], B[j, e] = 1., -1.
        k45 = torch.zeros(45, dtype=torch.float64)
        for row, k in zip(D, coupling):
            indices = torch.nonzero(row, as_tuple=False).flatten().tolist()
            if len(indices) != 2 or sorted(row[indices].tolist()) != [-1., 1.]:
                raise ValueError("M3 D rows must be signed pair incidence")
            k45[PAIRS.index(tuple(sorted(indices)))] += k
        for name, value in (("base_D", D), ("base_coupling", coupling), ("B", B),
                            ("reference_coupling45", k45), ("phi", B.T @ self.delta_star),
                            ("input_scale", torch.cat((self.input_scale_delta, self.input_scale_omega))),
                            ("W", torch.eye(10, dtype=torch.float64))):
            self.register_buffer(name, value)
        mstar = self.mass.log().mean().exp()
        self.register_buffer("kappa_reference", mstar)
        self.register_buffer("lambda_reference", torch.tensor(1., dtype=torch.float64))
        self.register_buffer("rstar", mstar * self.input_scale_omega)
        self.kappa_raw = nn.Parameter(torch.tensor(0., dtype=torch.float64))
        self.lambda_raw = nn.Parameter(torch.tensor(0., dtype=torch.float64))
        if mode == "S":
            self.k_bar = nn.Parameter(k45 / mstar)
        if mode == "M":
            self.a_bar = nn.Parameter(torch.zeros(45, dtype=torch.float64))
        # Exact fixed mechanical reverse propagation for piecewise-linear omega.
        P = torch.zeros(30, 30, dtype=torch.float64)
        P[:10, :10] = -torch.diag(self.rpp * self.storage_base)
        P[:10, 10:20] = -torch.diag(self.jpi_p)
        P[10:20, 20:] = torch.eye(10, dtype=torch.float64)
        self.register_buffer("p_backward", torch.matrix_exp(-.01 * P)[:10])
        self._cache = None
        self._cache_stamp = None

    @classmethod
    def from_source_export(cls, path, mode="L", *, source_identity=None):
        path = Path(path)
        raw = path.read_bytes()
        identity = {**(source_identity or {}), "source_path": str(path.resolve()),
                    "source_sha256": hashlib.sha256(raw).hexdigest()}
        return cls(json.loads(raw), mode, source_identity=identity)

    @property
    def kappa(self):
        return self.kappa_reference * self.kappa_raw.exp()

    @property
    def lambda_(self):
        return self.lambda_reference * self.lambda_raw.exp()

    def _tensor(self, value):
        return torch.as_tensor(value, dtype=self.mass.dtype, device=self.mass.device)

    def clear_cache(self):
        self._cache = self._cache_stamp = None

    def _stamp(self):
        return (tuple(p._version for p in self.parameters()), torch.is_grad_enabled(),
                torch.is_inference_mode_enabled(), self.training, self.mass.device, self.mass.dtype)

    def _apply(self, fn, recurse=True):
        self.clear_cache()
        return super()._apply(fn, recurse=recurse)

    def train(self, mode=True):
        self.clear_cache()
        return super().train(mode)

    def load_state_dict(self, *args, **kwargs):
        self.clear_cache()
        return super().load_state_dict(*args, **kwargs)

    def _new_context(self):
        kappa, lam = self.kappa, self.lambda_
        # Use feasible parameters directly. ReLU/softplus here would lock the
        # initially zero edges or fail to represent an exactly absent edge.
        coupling = self.kappa_reference * self.k_bar if self.mode == "S" else self.reference_coupling45
        Q = lam * self.W
        if self.mode == "M":
            Q = Q + self.lambda_reference * (self.B * self.a_bar) @ self.B.T
        return {"kappa": kappa, "lambda": lam, "coupling": coupling, "Q": Q}

    def _context(self):
        if self._cache is not None:
            if self._cache_stamp != self._stamp():
                self.clear_cache()
            else:
                return self._cache
        return self._new_context()

    def begin_graph_cache(self):
        if self._cache is not None:
            raise RuntimeError("end the previous model graph cache first")
        self._cache_stamp = self._stamp()
        self._cache = self._new_context()

    def end_graph_cache(self):
        self.clear_cache()

    @torch.no_grad()
    def project_nonnegative_(self):
        """Post-update Euclidean projection; caller's Adam moments are untouched."""
        if self.mode == "S":
            self.k_bar.clamp_min_(0.)
        if self.mode == "M":
            self.a_bar.clamp_min_(0.)
        self.clear_cache()
        return self

    def _potential_gradient(self, delta, context):
        eta = delta @ self.B
        return ((torch.sin(self.phi + eta) - self.phi.sin()) * context["coupling"]) @ self.B.T

    def base_momentum_field(self, physical, context=None):
        d, w, p = self._tensor(physical).split(10, dim=-1)
        context = self._context() if context is None else context
        return -self._potential_gradient(d, context) - self.damping * w + self.jpi_p * self.storage_base * p

    def field(self, state, u=None):
        if u is not None:
            raise ValueError("no external ports")
        z = self._tensor(state)
        if z.shape[-1] != 40:
            raise ValueError("interface state width must be 40")
        d, w, p, r = z.split(10, dim=-1)
        context = self._context()
        pdot = -self.jpi_p * w - self.rpp * self.storage_base * p
        rdot = context["kappa"] * w - r @ context["Q"].T
        return torch.cat((w, (self.base_momentum_field(z[..., :30], context) - r) / self.mass, pdot, rdot), -1)

    def interface_to_state(self, z):
        z = self._tensor(z)
        return torch.cat((z[..., :10], z[..., 10:20] * self.mass, z[..., 20:]), -1)

    def state_to_interface(self, x):
        x = self._tensor(x)
        return torch.cat((x[..., :10], x[..., 10:20] / self.mass, x[..., 20:]), -1)

    def internal_field(self, internal, u=None):
        return self.interface_to_state(self.field(self.state_to_interface(internal), u=u))

    def energy_and_gradient(self, internal):
        x = self._tensor(internal)
        d, pi, p, r = x.split(10, dim=-1)
        context = self._context()
        eta = d @ self.B
        potential = (context["coupling"] * (self.phi.cos() - (self.phi + eta).cos() - self.phi.sin() * eta)).sum(-1)
        H = ((pi.square() / self.mass + self.storage_base * p.square()
              + r.square() / context["kappa"]).sum(-1) / 2 + potential)
        grad = torch.cat((self._potential_gradient(d, context), pi / self.mass,
                          self.storage_base * p, r / context["kappa"]), -1)
        return H, grad

    def energy(self, internal):
        return self.energy_and_gradient(internal)[0]

    def gradient(self, internal, create_graph=True):
        return self.energy_and_gradient(internal)[1]

    def matrices(self):
        context = self._context()
        J = self.mass.new_zeros(40, 40)
        R = J.clone()
        J[:10, 10:20], J[10:20, :10] = self.W, -self.W
        G = torch.diag(self.jpi_p)
        J[10:20, 20:30], J[20:30, 10:20] = G, -G
        J[10:20, 30:], J[30:, 10:20] = -context["kappa"] * self.W, context["kappa"] * self.W
        R[10:20, 10:20] = torch.diag(self.damping)
        R[20:30, 20:30] = torch.diag(self.rpp)
        R[30:, 30:] = context["kappa"] * context["Q"]
        return J, R

    def _history_context(self, context):
        if "history" not in context:
            A = self.mass.new_zeros(30, 30)
            A[:10, :10] = -context["Q"]
            A[:10, 10:20] = context["kappa"] * self.W
            A[10:20, 20:] = self.W
            propagator = torch.matrix_exp(.01 * A)[:10]
            E = [self.W]
            for _ in range(10):
                E.append(E[-1] @ propagator[:, :10])
            E = torch.stack(E)
            C = (.02 / 6) * (E[0:10:2] + 4 * E[1:10:2] + E[2:11:2])
            design = (C * self.rstar[None, None, :] / (.02 * self.rstar[None, :, None])).reshape(50, 10)
            normal = design.T @ design + RIDGE * self.W
            context["history"] = (propagator, E, C, design, normal)
        return context["history"]

    def history_system(self, history):
        h = self._tensor(history)
        if h.ndim not in (2, 3) or h.shape[-2:] != (11, 30):
            raise ValueError("exactly 11 causal 30-column rows required")
        if not torch.isfinite(h[..., :20]).all() or not torch.isfinite(h[..., -1, 20:30]).all():
            raise ValueError("finite delta/omega history and terminal p0 required")
        context = self._context()
        propagator, E, C, design, normal = self._history_context(context)
        omega = h[..., 10:20]
        slope = (omega[..., 1:, :] - omega[..., :-1, :]) / .01
        p = [h[..., -1, 20:30]]
        for i in range(9, -1, -1):
            augmented = torch.cat((p[-1], omega[..., i + 1, :], slope[..., i, :]), -1)
            p.append(augmented @ self.p_backward.T)
        p = torch.stack(list(reversed(p)), dim=-2)
        v = [torch.zeros_like(p[..., 0, :])]
        for i in range(10):
            augmented = torch.cat((v[-1], omega[..., i, :], slope[..., i, :]), -1)
            v.append(augmented @ propagator.T)
        v = torch.stack(v, dim=-2)
        fbase = self.base_momentum_field(torch.cat((h[..., :20], p), -1), context)
        fminusv = fbase - v
        b = (.02 / 6) * (fminusv[..., 0:10:2, :] + 4 * fminusv[..., 1:10:2, :] + fminusv[..., 2:11:2, :])
        b = b - self.mass * (omega[..., 2:11:2, :] - omega[..., 0:10:2, :])
        bscaled = (b / (.02 * self.rstar)).flatten(-2)
        rhs = bscaled @ design
        s = torch.linalg.solve(normal, rhs.unsqueeze(-1)).squeeze(-1)
        u = self.rstar * s
        r0 = u @ E[-1].T + v[..., -1, :]
        return {"r0": r0, "u": u, "s": s, "p_history": p, "v_history": v,
                "E": E, "C": C, "A": design, "normal": normal,
                "b": b, "bscaled": bscaled, "fbase": fbase}

    def encode(self, history):
        return self.history_system(history)["r0"]

    def initialize(self, history):
        h = self._tensor(history)
        return torch.cat((h[..., -1, :30], self.encode(h)), -1)

    def initialization_diagnostics(self, history):
        system = self.history_system(history)
        singular = torch.linalg.svdvals(system["A"])
        residual = system["s"] @ system["A"].T - system["bscaled"]
        return {"design_singular_values": singular.detach().cpu().tolist(),
                "design_condition_number": float((singular[0] / singular[-1]).detach()),
                "regularized_normal_condition_number": float(torch.linalg.cond(system["normal"]).detach()),
                "ridge_singular_direction_retention": (singular.square() / (singular.square() + RIDGE)).detach().cpu().tolist(),
                "ridge": RIDGE, "scaled_residual_l2": residual.norm(dim=-1).detach().cpu().tolist(),
                "ridge_penalty": (RIDGE * system["s"].square().sum(-1)).detach().cpu().tolist(),
                "claim_scope": "conditioning of ten-state initial-value elimination; not identifiability of 47 dynamical parameters"}

    def _linear_generator(self, context):
        # Internal coordinates: [delta, pi, p, r]. Sine potential is excluded.
        A = self.mass.new_zeros(40, 40)
        invM = torch.diag(1 / self.mass)
        A[:10, 10:20] = invM
        A[10:20, 10:20] = -torch.diag(self.damping / self.mass)
        A[10:20, 20:30] = torch.diag(self.jpi_p * self.storage_base)
        A[10:20, 30:] = -self.W
        A[20:30, 10:20] = -torch.diag(self.jpi_p / self.mass)
        A[20:30, 20:30] = -torch.diag(self.rpp * self.storage_base)
        A[30:, 10:20] = context["kappa"] * invM
        A[30:, 30:] = -context["Q"]
        return A

    @staticmethod
    def _dt(dt):
        if isinstance(dt, bool) or not isinstance(dt, (int, float)) or not math.isfinite(dt) or dt <= 0:
            raise ValueError("finite positive scalar time step required")

    def fast_step(self, state, dt=.01):
        """Strang exact-linear half / centered-sine momentum kick / half."""
        self._dt(dt)
        context = self._context()
        key = ("linear_half", float(dt))
        if key not in context:
            context[key] = torch.matrix_exp(self._linear_generator(context) * (dt / 2))
        flow = context[key]
        x = self.interface_to_state(state) @ flow.T
        kick = -dt * self._potential_gradient(x[..., :10], context)
        x = torch.cat((x[..., :10], x[..., 10:20] + kick, x[..., 20:]), -1)
        return self.state_to_interface(x @ flow.T)

    def rk4(self, state, dt):
        """The actual classical RK4 reference, separate from fast_step."""
        self._dt(dt)
        z = self._tensor(state)
        a = self.field(z)
        b = self.field(z + dt * a / 2)
        c = self.field(z + dt * b / 2)
        d = self.field(z + dt * c)
        return z + (dt / 6) * (a + 2 * b + 2 * c + d)

    def rollout(self, history, steps, dt, *, method="rk4"):
        if type(steps) is not int or steps < 0 or method not in ("rk4", "fast"):
            raise ValueError("finite registered rollout steps and method required")
        self._dt(dt)
        owns_cache = self._cache is None
        if owns_cache:
            self.begin_graph_cache()
        try:
            state = self.initialize(history)
            out = [state]
            step = self.rk4 if method == "rk4" else self.fast_step
            for _ in range(steps):
                state = step(state, dt)
                out.append(state)
            return torch.stack(out, dim=-2)
        finally:
            if owns_cache:
                self.end_graph_cache()

    def complexity(self, state=None):
        return self.mass.new_tensor(float(2 + (45 if self.mode != "L" else 0)))

    def complexity_counts(self):
        return {"machines": 10, "states": 40, "candidate_pairs": 45,
                "raw_trainable_scalar_count": sum(p.numel() for p in self.parameters()),
                "encoder_trainable_scalar_count": 0, "initial_state_solve_dimension": 10,
                "spatial_scalar_coordinates": 0 if self.mode == "L" else 45,
                "scope": "representation counts; no parameter identifiability claim"}

    def export(self, path):
        if self._cache is not None:
            raise RuntimeError("end the active graph cache before export")
        payload = {"format": FORMAT, "mode": self.mode, "config": self._source_config,
                   "source_identity": copy.deepcopy(self.source_identity),
                   "state_dict": {k: v.detach().cpu().tolist() for k, v in self.state_dict().items()},
                   "counts": self.complexity_counts(), "ridge": RIDGE,
                   "integration": {"rk4": "classical reference", "fast_step": "Strang linear/potential splitting"}}
        Path(path).write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        return payload

    @classmethod
    def from_export(cls, path):
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if payload.get("format") != FORMAT or payload.get("ridge") != RIDGE:
            raise ValueError("wrong force-graph format or initializer regularization")
        model = cls(payload["config"], payload["mode"], source_identity=payload["source_identity"])
        template = model.state_dict()
        if set(template) != set(payload["state_dict"]):
            raise ValueError("exported state keys differ")
        supplied = {k: torch.as_tensor(payload["state_dict"][k], dtype=v.dtype) for k, v in template.items()}
        if any(v.shape != template[k].shape or not torch.isfinite(v).all() for k, v in supplied.items()):
            raise ValueError("invalid exported state shapes or values")
        model.load_state_dict(supplied)
        if ((model.k_bar < 0).any() if model.mode == "S" else
                (model.a_bar < 0).any() if model.mode == "M" else False):
            raise ValueError("exported spatial coordinates must be feasible")
        return model
