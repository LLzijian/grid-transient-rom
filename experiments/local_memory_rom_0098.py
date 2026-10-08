"""Loop 0098 local/global memory structured ROM.

This module is deliberately self contained at the campaign boundary.  It wraps
the small :class:`StructuredROM` implementation used by the earlier structured
experiments, while keeping the memory coordinates and their shared encoder
local to this experiment.  No data reader or training loop is imported here.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from structured_rom import StructuredROM


_DTYPE = torch.float64
_TEMPLATE_NAMES = (
    "H_pq", "H_edge_sin", "H_edge_cos", "J_pi_q",
    "J_p_q", "R_q", "R_pi_q", "local_port",
)


def _tensor(x: Any) -> torch.Tensor:
    return torch.as_tensor(x, dtype=_DTYPE)


def _inv_softplus(x: torch.Tensor) -> torch.Tensor:
    x = x.clamp_min(1e-14)
    return torch.where(x > 20.0, x, torch.log(torch.expm1(x)))


def _effective_parameter(obj: Any, name: str, positive: bool = False) -> torch.Tensor:
    """Read an effective value from either StructuredROM or MechanicalROM."""
    value = getattr(obj, name, None)
    if value is not None:
        value = value() if callable(value) else value
        value = _tensor(value).detach().reshape(-1)
        if positive and hasattr(obj, "_pos") and hasattr(obj, name + "_mask"):
            try:
                value = obj._pos(name).detach().reshape(-1)
            except Exception:
                pass
        return value
    raw = getattr(obj, name + "_raw", None)
    if raw is not None:
        raw = _tensor(raw).detach().reshape(-1)
        return F.softplus(raw) if positive else raw
    raise AttributeError(f"base has no {name} or {name}_raw")


def _incidence(base: Any, n: int) -> torch.Tensor:
    D = getattr(base, "D", None)
    if D is not None:
        D = _tensor(D)
        if D.ndim != 2 or D.shape[1] != n:
            raise ValueError("base.D must have shape (edges, machines)")
        return D
    incidence = getattr(base, "incidence", None)
    if incidence is None:
        raise ValueError("base must expose D or incidence")
    incidence = _tensor(incidence)
    if incidence.ndim != 2:
        raise ValueError("base.incidence must be a matrix")
    if incidence.shape[1] == n:
        return incidence
    if incidence.shape[0] == n:
        return incidence.T
    raise ValueError("base incidence has no machine-sized dimension")


def _base_values(base: Any, name: str, n: int, default: Any) -> torch.Tensor:
    if hasattr(base, "_pos") and hasattr(base, name + "_mask"):
        try:
            value = base._pos(name).detach()
        except Exception:
            value = None
    else:
        value = None
    if value is None:
        value = getattr(base, name, None)
    if value is None:
        value = getattr(base, name + "_raw", None)
        if value is not None:
            value = F.softplus(_tensor(value))
    if value is None:
        value = default
    out = _tensor(value).detach().reshape(-1)
    if out.numel() == 1:
        out = out.repeat(n)
    if out.numel() != n:
        raise ValueError(f"{name} must have {n} entries")
    return out


def _parse_scale(value: Any, n: int, *, which: int | None = None) -> torch.Tensor:
    if value is None:
        return torch.ones(n, dtype=_DTYPE)
    if isinstance(value, dict):
        value = value["delta" if which == 0 else "omega"]
    arr = _tensor(value)
    if arr.ndim == 0:
        return arr.repeat(n)
    if arr.numel() == n:
        return arr.reshape(n)
    if arr.numel() == 2 * n and which is not None:
        return arr.reshape(2, n)[which]
    if arr.numel() == 2 and which is not None:
        return arr.reshape(2)[which].repeat(n)
    raise ValueError("input_scale must be scalar, length n, length 2n, or a delta/omega mapping")


class LocalMemoryROM(nn.Module):
    """Structured ROM with shared local memory and G3/L10/HL11 arms.

    The public interface state is ``[delta, omega, p, q]``.  The internal
    StructuredROM state is ``[delta, pi, p, q]`` and is converted using the
    copied M3 inertias.  All learned templates are shared; machine-specific
    instances arise only through the fixed physical graph and parameter map.
    """

    template_names = _TEMPLATE_NAMES

    def __init__(self, base: Any, input_scale: Any = None, p_scale: Any = None,
                 arm: str = "L10-S", seed: int = 98):
        super().__init__()
        if arm not in {"G3-S", "L10-S", "HL11-S"}:
            raise ValueError("arm must be G3-S, L10-S, or HL11-S")
        self.arm = arm
        self.seed = int(seed)
        mass = _tensor(getattr(base, "mass")).detach().reshape(-1)
        n = int(mass.numel())
        if n != 10:
            raise ValueError("0098 M3 arms require ten machines")
        D_all = _incidence(base, n)
        coupling_all = _base_values(base, "coupling", D_all.shape[0], torch.ones(D_all.shape[0]))
        active = coupling_all > 0
        if not bool(active.any()):
            raise ValueError("base must contain at least one active coupling edge")
        D = D_all[active]
        coupling = coupling_all[active]
        delta_star = _base_values(base, "delta_star", n, torch.zeros(n))
        damping = torch.clamp(_base_values(base, "damping", n, torch.ones(n)), min=1e-4)
        gain = _base_values(base, "gain", n, torch.ones(n)).clamp_min(1e-8)
        jpi_p = _base_values(base, "jpi_p", n, gain)
        rate = _base_values(base, "rate", n, torch.ones(n))
        rpp = _base_values(base, "rpp", n, rate * gain)
        pbase = getattr(base, "storage_base", None)
        if pbase is not None and hasattr(base, "_pos"):
            try:
                pbase = base._pos("storage_base").detach()[:n]
            except Exception:
                pbase = _tensor(pbase).detach().reshape(-1)[:n]
        elif pbase is not None:
            pbase = _tensor(pbase).detach().reshape(-1)[:n]
        else:
            pbase = 1.0 / gain
        if pbase.numel() != n:
            pbase = (1.0 / gain)
        p_scale_t = _base_values(base, "p_scale", n, torch.ones(n)) if p_scale is None else _tensor(p_scale).reshape(-1)
        if p_scale_t.numel() == 1:
            p_scale_t = p_scale_t.repeat(n)
        if p_scale_t.numel() != n or torch.any(p_scale_t <= 0):
            raise ValueError("p_scale must be a positive length-10 vector")
        self.register_buffer("mass", mass)
        self.register_buffer("p_scale", p_scale_t.detach())
        self.register_buffer("input_scale_delta", _parse_scale(input_scale, n, which=0))
        self.register_buffer("input_scale_omega", _parse_scale(input_scale, n, which=1))
        self.register_buffer("input_scale", torch.cat((self.input_scale_delta, self.input_scale_omega)))
        self.register_buffer("active_edge", active)
        self.register_buffer("base_D", D)
        self.register_buffer("base_coupling", coupling)
        self.register_buffer("delta_star", delta_star)
        self.register_buffer("rate", rate)
        self.register_buffer("gain", gain)

        rank = 3 if arm == "G3-S" else (10 if arm == "L10-S" else 11)
        self.rank = rank
        W = self._make_W(arm, mass, rate)
        self.register_buffer("W", W)

        # Fixed graph used by both the encoder and the local port.  The
        # normalized adjacency is a physical summary, not a learned ID.
        raw_adj = torch.abs(D).T @ torch.diag(coupling) @ torch.abs(D)
        raw_adj.fill_diagonal_(0.0)
        A = raw_adj / raw_adj.sum(-1, keepdim=True).clamp_min(1e-15)
        self.register_buffer("neighbor_A", A)
        # Fixed, parameter-independent endpoint summaries for the explicit
        # port.  Sin uses the oriented incidence; cos uses endpoint magnitudes.
        weighted_D = coupling[:, None] * D
        endpoint_weight = (coupling[:, None] * torch.abs(D)).sum(0).clamp_min(1e-15)
        self.register_buffer("port_s_weight", weighted_D / endpoint_weight)
        self.register_buffer("port_c_weight", torch.abs(weighted_D) / endpoint_weight)
        self.register_buffer("phi", D @ delta_star)

        # Reuse the StructuredROM physical terms, replacing any old q storage
        # with the requested dynamic q dimensions and one shared q quadratic.
        self.structured = StructuredROM(
            mass=mass, D=D, coupling=coupling, delta_star=delta_star,
            damping=damping, gain=gain, n_q=rank,
            jpi_p=jpi_p, rpp=rpp, storage_base=torch.cat((pbase, torch.zeros(rank))),
            storage=(), trainable=True,
        ).double()
        self._q_quadratic_raw = nn.Parameter(_inv_softplus(torch.tensor(1.0, dtype=_DTYPE)))

        # Fresh, shared encoders.  Their fixed seed is only for reproducible
        # initialization and has no role in the physics or feature map.
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(self.seed)
            if arm == "G3-S":
                self.local_gru = None
                self.local_head = None
                self.global_gru = nn.GRU(12, 8, batch_first=True, dtype=_DTYPE)
                self.global_head = nn.Linear(8, 3, bias=False, dtype=_DTYPE)
            elif arm == "HL11-S":
                self.local_gru = nn.GRU(7, 8, batch_first=True, dtype=_DTYPE)
                self.local_head = nn.Linear(8, 1, bias=False, dtype=_DTYPE)
                self.global_gru = nn.GRU(4, 8, batch_first=True, dtype=_DTYPE)
                self.global_head = nn.Linear(8, 1, bias=False, dtype=_DTYPE)
            else:
                self.local_gru = nn.GRU(7, 8, batch_first=True, dtype=_DTYPE)
                self.local_head = nn.Linear(8, 1, bias=False, dtype=_DTYPE)
                self.global_gru = None
                self.global_head = None

        # Signed H/port terms and positive J/R terms share one compact vector;
        # inactive entries are masked so activation is a single explicit edit.
        initial = torch.tensor([1e-3, 1e-3, 1e-3, 1e-3, 1e-3, 1e-2, 1e-2, 1e-3], dtype=_DTYPE)
        raw = initial.clone()
        for i in (5, 6):
            raw[i] = _inv_softplus(raw[i])
        self._template_scalars = nn.ParameterList([nn.Parameter(raw[i]) for i in range(7)])
        self.port_raw = nn.Parameter(torch.full((6,), 1e-3, dtype=_DTYPE))
        self.register_buffer("template_mask", torch.tensor([0., 0., 0., 1., 0., 1., 0., 0.], dtype=_DTYPE))
        self._graph_cache: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None

    @staticmethod
    def _make_W(arm: str, mass: torch.Tensor, rate: torch.Tensor) -> torch.Tensor:
        n = mass.numel()
        if arm == "L10-S":
            return torch.eye(n, dtype=_DTYPE)
        if arm == "HL11-S":
            return torch.cat((torch.eye(n, dtype=_DTYPE), torch.ones(n, 1, dtype=_DTYPE) / np.sqrt(n)), dim=1)
        cols = [torch.ones(n, dtype=_DTYPE), mass / mass.mean().clamp_min(1e-15), rate / rate.mean().clamp_min(1e-15)]
        return torch.stack([c / torch.linalg.vector_norm(c).clamp_min(1e-15) for c in cols], dim=1)

    @property
    def n(self) -> int:
        return 10

    @property
    def dim(self) -> int:
        return 30 + self.rank

    @property
    def interface_dim(self) -> int:
        return self.dim

    @property
    def q_quadratic(self) -> torch.Tensor:
        return F.softplus(self._q_quadratic_raw)

    @property
    def template_coefficients(self) -> torch.Tensor:
        out = torch.stack(tuple(p for p in self._template_scalars))
        for i in (5, 6):
            out = out.clone()
            out[i] = F.softplus(self._template_scalars[i])
        # The seventh slot is the activation amplitude for the six shared port
        # coefficients; the individual coefficients live in ``port_raw``.
        out = torch.cat((out, torch.ones_like(self.template_mask[7:8])))
        return out * self.template_mask

    def template_parameters(self, index: int | None = None):
        if index is not None:
            if not 0 <= int(index) < 8:
                raise IndexError("template index must be in [0, 7]")
            return [self.port_raw] if int(index) == 7 else [self._template_scalars[int(index)]]
        return {name: self.template_parameters(i) for i, name in enumerate(self.template_names)}

    def set_active(self, index: int, active: bool = True) -> None:
        if not 0 <= int(index) < len(self.template_names):
            raise IndexError("template index must be in [0, 7]")
        with torch.no_grad():
            self.template_mask[int(index)] = 1.0 if active else 0.0

    def config(self) -> dict[str, Any]:
        return {
            "mass": self.mass.detach().tolist(), "D": self.base_D.detach().tolist(),
            "coupling": self.base_coupling.detach().tolist(), "delta_star": self.delta_star.detach().tolist(),
            "damping": self.structured._pos("damping").detach().tolist(), "gain": self.gain.detach().tolist(),
            "jpi_p": self.structured._pos("jpi_p").detach().tolist(), "rpp": self.structured._pos("rpp").detach().tolist(),
            "input_scale": self.input_scale.detach().tolist(), "p_scale": self.p_scale.detach().tolist(),
            "rate": self.rate.detach().tolist(), "active_edge": self.active_edge.detach().tolist(),
            "arm": self.arm, "seed": self.seed,
        }

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "LocalMemoryROM":
        base = StructuredROM(mass=config["mass"], D=config["D"], coupling=config["coupling"],
                             delta_star=config.get("delta_star"), damping=config["damping"],
                             gain=config["gain"], jpi_p=config.get("jpi_p"), rpp=config.get("rpp"), n_q=0,
                             storage=(), trainable=True)
        base.rate = _tensor(config.get('rate', np.ones(10)))
        model = cls(base, input_scale=config.get("input_scale"), p_scale=config.get("p_scale"),
                    arm=config["arm"], seed=int(config.get("seed", 98)))
        if 'active_edge' in config:
            model.active_edge = torch.as_tensor(config['active_edge'], dtype=torch.bool)
        return model

    def export(self, path: str | Path) -> dict[str, Any]:
        state = {k: v.detach().cpu().tolist() for k, v in self.state_dict().items()}
        coeff = self.template_coefficients.detach().tolist()
        J, R = self.matrices()
        payload: dict[str, Any] = {
            "format": "local_memory_rom_0098", "config": self.config(), "W": self.W.detach().tolist(),
            "J": J.detach().tolist(), "R": R.detach().tolist(), "B": self.port_matrix().detach().tolist(),
            "template_names": list(self.template_names), "template_mask": self.template_mask.detach().tolist(),
            "template_coefficients": coeff, "q_quadratic": float(self.q_quadratic.detach()),
            "state_dict": state,
            "counts": self.complexity_counts(),
        }
        Path(path).write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return payload

    @classmethod
    def from_export(cls, path: str | Path) -> "LocalMemoryROM":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        model = cls.from_config(payload["config"])
        state = {k: _tensor(v) for k, v in payload.get("state_dict", {}).items()}
        model.load_state_dict(state, strict=True)
        return model

    def complexity_counts(self):
        """Count expanded scalar RHS contributions, not matrix dimensions.

        First collect constant spatial maps within each named energy primitive;
        then count each scalar gradient atom propagated by the assembled J-R.
        Centered trig functions are one atom. Distinct named energy primitives
        are not collected against each other. This is a reproducible expression
        convention, not a minimum symbolic complexity or identifiability claim.
        """
        D=self.base_D.detach().numpy(); W=self.W.detach().numpy()
        c=self.template_coefficients.detach().numpy()
        atoms=np.zeros(self.dim, dtype=int)
        atoms[:10]=(D!=0).sum(0)
        atoms[10:30]=1; atoms[30:]=1
        if c[0]!=0:
            atoms[20:30]+=(W!=0).sum(1)
            atoms[30:]+=(W!=0).sum(0)
        for index, mapping in ((1,D@W),(2,np.abs(D)@W)):
            if c[index]!=0:
                for edge in range(len(D)):
                    nz=np.count_nonzero(mapping[edge])
                    atoms[:10]+=(D[edge]!=0)*nz
                    atoms[30:]+=(mapping[edge]!=0)
        J,R=self.matrices()
        matrix=(J-R).detach().numpy()
        propagated=int(((matrix!=0).sum(0)*atoms).sum())
        port=0
        if bool(self.template_mask[7]):
            coeff=self.port_raw.detach().numpy()
            if coeff[0]!=0: port+=10
            if coeff[1]!=0: port+=10
            if coeff[2]!=0: port+=int(np.count_nonzero(W))
            if coeff[3]!=0: port+=int(torch.count_nonzero(self.port_s_weight))
            if coeff[4]!=0: port+=int(torch.count_nonzero(self.port_c_weight))
            if coeff[5]!=0:
                port+=int(np.count_nonzero((self.neighbor_A.detach().numpy()-np.eye(10))@W))
        raw=sum(p.numel() for p in self.parameters() if p.requires_grad)
        encoder=sum(p.numel() for name,p in self.named_parameters()
                    if p.requires_grad and ('gru' in name or 'head' in name))
        masked_base=self.rank if self.structured.storage_base.requires_grad else 0
        inactive=sum(sum(p.numel() for p in self.template_parameters(i) if p.requires_grad)
                     for i in range(8) if not bool(self.template_mask[i]))
        return {'machines':self.n,'rank':self.rank,'active_edges':len(D),
                'template_groups_available':8,'template_groups_active':int(self.template_mask.sum()),
                'shared_memory_scalar_coordinates':14,
                'template_instances':dict(zip(self.template_names,(10,len(D),len(D),
                    int(np.count_nonzero(W)),int(np.count_nonzero(W)),self.rank,10,10))),
                'raw_trainable_scalar_count':raw,'encoder_trainable_scalar_count':encoder,
                'masked_base_storage_coordinates':masked_base,'inactive_template_coordinates':inactive,
                'effective_active_parameter_coordinate_upper_bound':raw-masked_base-inactive,
                'expanded_rhs_scalar_contributions':{'gradient_atoms':atoms.tolist(),
                    'assembled_J_minus_R_times_gradient_atoms':propagated,
                    'explicit_port_basis_contributions':port,'total':propagated+port,
                    'convention':self.complexity_counts.__doc__}}

    def _features(self, history: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if history.shape[-2:] != (11, 30):
            raise ValueError("encoder requires [batch, 11, 30] histories")
        d = history[..., :10]
        w = history[..., 10:20]
        coi = d - (d * self.mass).sum(-1, keepdim=True) / self.mass.sum()
        dnorm = coi / self.input_scale_delta
        wnorm = w / self.input_scale_omega
        # A[i,j] is the fixed normalized weight from machine i to j.
        nei_d = (coi @ self.neighbor_A.T) - coi
        nei_w = w @ self.neighbor_A.T
        nei_d = nei_d / self.input_scale_delta
        nei_w = nei_w / self.input_scale_omega
        m = self.mass / self.mass.mean().clamp_min(1e-15)
        r = self.rate / self.rate.mean().clamp_min(1e-15)
        g = self.gain / self.gain.mean().clamp_min(1e-15)
        local = torch.stack((dnorm, wnorm, nei_d, nei_w,
                             m.expand_as(dnorm), r.expand_as(dnorm), g.expand_as(dnorm)), dim=-1)
        return local, torch.stack((dnorm, wnorm, nei_d, nei_w), dim=-1)

    def encode(self, history: torch.Tensor) -> torch.Tensor:
        history = history if torch.is_tensor(history) else _tensor(history)
        history = history.to(dtype=self.mass.dtype, device=self.mass.device)
        local, first4 = self._features(history)
        b, t, n, _ = local.shape
        zero = torch.zeros_like(history)
        _, first40 = self._features(zero)
        if self.arm == "G3-S":
            global_input = torch.einsum("btna,nr->btra", first4, self.W).reshape(b, t, 12)
            global0 = torch.einsum("btna,nr->btra", first40, self.W).reshape(b, t, 12)
            raw = self.global_head(self.global_gru(global_input)[0][:, -1])
            raw0 = self.global_head(self.global_gru(global0)[0][:, -1])
            return 0.1 * (raw - raw0)
        local_seq = local.permute(0, 2, 1, 3).reshape(b * n, t, 7)
        local0, _ = self._features(zero)
        raw_local = self.local_head(self.local_gru(local_seq)[0][:, -1]).reshape(b, n)
        raw_local0 = self.local_head(self.local_gru(local0.permute(0, 2, 1, 3).reshape(b * n, t, 7))[0][:, -1]).reshape(b, n)
        q_local = 0.1 * (raw_local - raw_local0)
        if self.arm == "L10-S":
            return q_local
        global_input = first4.mean(dim=2)
        global0 = first40.mean(dim=2)
        raw = self.global_head(self.global_gru(global_input)[0][:, -1])
        raw0 = self.global_head(self.global_gru(global0)[0][:, -1])
        return torch.cat((q_local, 0.1 * (raw - raw0)), dim=-1)

    def interface_to_state(self, z: torch.Tensor) -> torch.Tensor:
        z = z if torch.is_tensor(z) else _tensor(z)
        if z.shape[-1] != self.dim:
            raise ValueError(f"expected interface state width {self.dim}")
        return torch.cat((z[..., :10], z[..., 10:20] * self.mass, z[..., 20:]), dim=-1)

    def state_to_interface(self, x: torch.Tensor) -> torch.Tensor:
        x = x if torch.is_tensor(x) else _tensor(x)
        return torch.cat((x[..., :10], x[..., 10:20] / self.mass, x[..., 20:]), dim=-1)

    def _split(self, x: torch.Tensor):
        return x[..., :10], x[..., 10:20], x[..., 20:30], x[..., 30:]

    def _template_energy(self, x: torch.Tensor) -> torch.Tensor:
        d, pi, p, q = self._split(x)
        eta = d @ self.base_D.T
        v = q @ self.W.T
        c = self.template_coefficients
        H = x.new_zeros(x.shape[:-1])
        H = H + c[0] * ((p / self.p_scale) * v).sum(-1)
        norm_k = self.base_coupling / self.base_coupling.mean().clamp_min(1e-15)
        trig_s = torch.sin(self.phi + eta) - torch.sin(self.phi)
        trig_c = torch.cos(self.phi + eta) - torch.cos(self.phi)
        edge_difference = v @ self.base_D.T
        H = H + c[1] * (norm_k * edge_difference * trig_s).sum(-1)
        endpoint_sum = v @ torch.abs(self.base_D).T
        H = H + c[2] * (norm_k * endpoint_sum * trig_c).sum(-1)
        H = H + 0.5 * self.q_quadratic * q.square().sum(-1)
        return H

    def energy_and_gradient(self, internal: torch.Tensor):
        x = internal if torch.is_tensor(internal) else _tensor(internal)
        x = x.to(dtype=self.mass.dtype, device=self.mass.device)
        if x.shape[-1] != self.dim:
            raise ValueError(f"expected internal state width {self.dim}")
        d, pi, p, q = self._split(x)
        eta = d @ self.base_D.T
        v = q @ self.W.T
        c = self.template_coefficients
        norm_k = self.base_coupling / self.base_coupling.mean().clamp_min(1e-15)
        trig_s = torch.sin(self.phi + eta) - torch.sin(self.phi)
        trig_c = torch.cos(self.phi + eta) - torch.cos(self.phi)
        edge_difference = v @ self.base_D.T
        endpoint_sum = v @ torch.abs(self.base_D).T
        extra_H = (c[0] * ((p / self.p_scale) * v).sum(-1)
                   + c[1] * (norm_k * edge_difference * trig_s).sum(-1)
                   + c[2] * (norm_k * endpoint_sum * trig_c).sum(-1)
                   + 0.5 * self.q_quadratic * q.square().sum(-1))
        base_H, base_e = self.structured.energy_and_gradient(x)
        gd = c[1] * ((norm_k * edge_difference * torch.cos(self.phi + eta)) @ self.base_D)
        gd = gd + c[2] * ((-norm_k * endpoint_sum * torch.sin(self.phi + eta)) @ self.base_D)
        gv = c[0] * (p / self.p_scale)
        gv = gv + c[1] * (norm_k * trig_s) @ self.base_D
        gv = gv + c[2] * (norm_k * trig_c) @ torch.abs(self.base_D)
        gq = gv @ self.W + self.q_quadratic * q
        gp = c[0] * v / self.p_scale
        extra_e = torch.cat((gd, torch.zeros_like(pi), gp, gq), dim=-1)
        return base_H + extra_H, base_e + extra_e

    def energy(self, internal: torch.Tensor) -> torch.Tensor:
        return self.energy_and_gradient(internal)[0]

    def gradient(self, internal: torch.Tensor, create_graph: bool = True) -> torch.Tensor:
        return self.energy_and_gradient(internal)[1]

    def matrices(self):
        J, R = self.structured.matrices()
        c = self.template_coefficients
        n, r = self.n, self.rank
        # pi-q and p-q paired exchange blocks.
        J = J.clone()
        J[n:2*n, 2*n+n:2*n+n+r] += c[3] * self.W
        J[2*n+n:2*n+n+r, n:2*n] -= c[3] * self.W.T
        scaled = self.W / self.p_scale[:, None]
        J[2*n:2*n+n, 2*n+n:2*n+n+r] += c[4] * scaled
        J[2*n+n:2*n+n+r, 2*n:2*n+n] -= c[4] * scaled.T
        R = R.clone()
        R[2*n+n:2*n+n+r, 2*n+n:2*n+n+r] += c[5] * torch.eye(r, dtype=R.dtype, device=R.device)
        for i in range(n):
            a = torch.cat((self.W[i].new_zeros(n), self.W[i].new_zeros(n), self.W[i].new_zeros(10), self.W[i]))
            a[n+i] = 1.0
            a = a / torch.linalg.vector_norm(a).clamp_min(1e-15)
            R = R + c[6] * torch.outer(a, a)
        return J, R

    def _port_values_internal(self, x: torch.Tensor) -> torch.Tensor:
        d, pi, p, q = self._split(x)
        eta = d @ self.base_D.T
        v = q @ self.W.T
        sin_d = torch.sin(self.phi + eta) - torch.sin(self.phi)
        cos_d = torch.cos(self.phi + eta) - torch.cos(self.phi)
        # edge summaries are accumulated to their endpoint machines.
        edge_s = sin_d @ self.port_s_weight
        edge_c = cos_d @ self.port_c_weight
        nei_v = v @ self.neighbor_A.T - v
        omega = pi / self.mass
        bases = torch.stack((omega / self.input_scale_omega, p / self.p_scale, v,
                             edge_s, edge_c, nei_v), dim=-1)
        return (self.template_mask[7] * bases * self.port_raw).sum(-1)

    def port_values(self, internal: torch.Tensor) -> torch.Tensor:
        x = internal if torch.is_tensor(internal) else _tensor(internal)
        if x.shape[-1] != self.dim:
            x = self.interface_to_state(x)
        return self._port_values_internal(x)

    def port_matrix(self) -> torch.Tensor:
        B = torch.zeros((self.dim, self.n), dtype=self.mass.dtype, device=self.mass.device)
        B[10:20, :] = torch.eye(self.n, dtype=B.dtype, device=B.device)
        return B

    def field(self, z: torch.Tensor, u: torch.Tensor | None = None) -> torch.Tensor:
        z = z if torch.is_tensor(z) else _tensor(z)
        z = z.to(dtype=self.mass.dtype, device=self.mass.device)
        x = self.interface_to_state(z)
        _, e = self.energy_and_gradient(x)
        if self._graph_cache is None:
            J, R = self.matrices()
            B = self.port_matrix()
        else:
            J, R, B = self._graph_cache
        pv = self.port_values(x) if u is None else (u if torch.is_tensor(u) else _tensor(u))
        if pv.shape[-1] != self.n:
            raise ValueError(f"external port input must have width {self.n}")
        f = e @ (J - R).T + pv @ B.T
        return self.state_to_interface(f)

    def internal_field(self, x: torch.Tensor, u: torch.Tensor | None = None) -> torch.Tensor:
        return self.interface_to_state(self.field(self.state_to_interface(x), u=u))

    def begin_graph_cache(self) -> None:
        J, R = self.matrices()
        self._graph_cache = (J, R, self.port_matrix())

    def end_graph_cache(self) -> None:
        self._graph_cache = None

    def rk4(self, state: torch.Tensor, dt: float) -> torch.Tensor:
        a = self.field(state)
        b = self.field(state + dt * a / 2)
        c = self.field(state + dt * b / 2)
        d = self.field(state + dt * c)
        return state + dt * (a + 2*b + 2*c + d) / 6

    def rollout(self, history: torch.Tensor, steps: int, dt: float) -> torch.Tensor:
        state = torch.cat((history[..., -1, :], self.encode(history)), dim=-1)
        out = [state]
        for _ in range(int(steps)):
            state = self.rk4(state, dt)
            out.append(state)
        return torch.stack(out, dim=1)

    def complexity(self, state: torch.Tensor | None = None) -> torch.Tensor:
        c = self.template_coefficients
        return c[:7].abs().sum() + self.template_mask[7]*self.port_raw.abs().sum()


def from_export(path: str | Path) -> LocalMemoryROM:
    return LocalMemoryROM.from_export(path)
