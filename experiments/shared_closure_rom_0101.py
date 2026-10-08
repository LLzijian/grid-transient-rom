"""Shared L10-S23/L10-F23 closure adapters for preregistered stage 0101.

S23 keeps the frozen LocalMemoryROM vector field and changes only the reported
complexity to the collected 23-coordinate closure.  F23 keeps the same L10
encoder, M3 positive skeleton, fixed graph and interface, while replacing the
H/J/R closure by the protocol's direct 19+4 RHS basis.  No data reader or
training loop is imported here.
"""
from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import torch
from torch import nn

from local_memory_rom_0098 import LocalMemoryROM
from structured_rom import StructuredROM


N = 10
CLOSURE_NAMES = (
    "pi_neg_Es", "pi_Ec", "pi_j_q_over_ps", "pi_p_over_ps", "pi_Ts", "pi_Tc", "pi_q",
    "p_neg_r_q_over_ps", "p_p_over_ps2", "p_Ts_over_ps", "p_Tc_over_ps", "p_q_over_ps",
    "q_neg_omega", "q_neg_A_p_over_ps", "q_neg_q_over_ps2", "q_neg_p_over_ps", "q_neg_Ts", "q_neg_Tc", "q_neg_q",
    "port_omega", "port_sin", "port_cos", "port_neighbor_q",
)
PRIMITIVE_NAMES = CLOSURE_NAMES
EXPECTED_S_MASK = (1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 1.0)
RHS_PRIMITIVE_DEFINITIONS = {
    "fixed_graph": {
        "eta": "D @ delta",
        "Ts": "D.T @ (kn * (sin(phi + eta) - sin(phi)))",
        "Tc": "abs(D).T @ (kn * (cos(phi + eta) - cos(phi)))",
        "Es": "D.T @ (kn * (D @ q) * cos(phi + eta))",
        "Ec": "D.T @ (kn * (abs(D) @ q) * sin(phi + eta))",
        "kn": "base_coupling / mean(base_coupling)",
        "edge_port_s": "(sin(phi + eta) - sin(phi)) @ port_s_weight",
        "edge_port_c": "(cos(phi + eta) - cos(phi)) @ port_c_weight",
        "neighbor_q": "neighbor_A @ q - q (row-vector convention: q @ neighbor_A.T - q)",
    },
    "pi_rhs": ["-Es", "Ec", "jpi_p*q/p_scale", "p/p_scale", "Ts", "Tc", "q"],
    "p_rhs": ["-rpp*q/p_scale", "p/p_scale**2", "Ts/p_scale", "Tc/p_scale", "q/p_scale"],
    "q_rhs": ["-omega", "-storage_base*p/p_scale", "-q/p_scale**2", "-p/p_scale", "-Ts", "-Tc", "-q"],
    "port_rhs": ["omega/input_scale_omega", "edge_port_s", "edge_port_c", "neighbor_q"],
    "m3_skeleton": {
        "energy_gradient": "e_M3 = grad_x H_M3(delta, pi, p)",
        "internal_rhs": "f_M3 = e_M3 @ (J_M3 - R_M3).T",
        "interface_rhs": "[delta_dot, omega_dot, p_dot] = [f_delta, f_pi/mass, f_p]",
    },
}
UNIT_METADATA = {
    "state": "[delta, omega, p, q]",
    "internal_momentum": "pi=M*omega",
    "delta": "rad",
    "omega": "rad_s",
    "time": "seconds",
    "q": "latent",
    "p": "inherited M3 units; no new physical-unit claim",
}


def _effective(source: Any, name: str) -> torch.Tensor:
    value = getattr(source.structured, "_pos")(name)
    return value.detach().clone()


def _copy_buffer(value: torch.Tensor) -> torch.Tensor:
    return value.detach().clone()


def mapped_coefficients(model: LocalMemoryROM) -> torch.Tensor:
    """Map the seven shared S coordinates and six raw port coordinates to C23.

    The order is exactly ``pi7 + p5 + q7 + port4`` in ``CLOSURE_NAMES``.
    Port p/q are merged into the pi p/q coefficients as registered by 0101.
    """
    c = model.template_coefficients
    port = model.port_raw
    a, b, cc = c[0], c[1], c[2]
    q_quad, u, v, r = model.q_quadratic, c[3], c[4], c[5]
    rho = c[6]
    p_port, q_port = port[1], port[2]
    # R_pi_q is a rank-one PSD term on (pi_i, q_i).  Its pi-q cross
    # contribution is represented in C23's direct RHS coordinates, while
    # its pi diagonal contribution belongs to F23's M3 damping skeleton.
    half_rho = rho * 0.5
    return torch.stack((
        b, cc, a, u * a + p_port - half_rho * a,
        u * b - half_rho * b, u * cc - half_rho * cc,
        u * q_quad + q_port - half_rho * q_quad,
        a, v * a, v * b, v * cc, v * q_quad,
        u + half_rho, v, v * a, r * a + half_rho * a,
        r * b + half_rho * b, r * cc + half_rho * cc,
        r * q_quad + half_rho * q_quad,
        port[0], port[3], port[4], port[5],
    ))


class S23(LocalMemoryROM):
    """Frozen-field S arm with collected 23-coordinate complexity."""

    model_kind = "S23"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # The protocol closes R_pi_q.  Keep its state slot for raw-export
        # compatibility, but do not expose an inactive trainable scalar.
        self._template_scalars[6].requires_grad_(False)

    def complexity(self, state: torch.Tensor | None = None) -> torch.Tensor:
        del state
        return mapped_coefficients(self).abs().sum()

    def config_0101(self) -> dict[str, Any]:
        return {"model_kind": self.model_kind, "config": self.config(), "closure_names": list(CLOSURE_NAMES)}

    def export(self, path: str | Path) -> dict[str, Any]:
        payload = super().export(path)
        payload.update({
            "format": "shared_closure_rom_0101",
            "model_kind": self.model_kind,
            "closure_names": list(CLOSURE_NAMES),
            "rhs_primitives": list(PRIMITIVE_NAMES),
            "rhs_primitive_definitions": RHS_PRIMITIVE_DEFINITIONS,
            "rhs_coefficients": mapped_coefficients(self).detach().cpu().tolist(),
            "unit_metadata": {**UNIT_METADATA, "time_seconds": True},
        })
        Path(path).write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return payload


class F23(nn.Module):
    """Direct 19+4 RHS closure on the common L10 M3/encoder skeleton."""

    model_kind = "F23"

    def __init__(self, source: LocalMemoryROM):
        super().__init__()
        if source.arm != "L10-S":
            raise ValueError("0101 F23 requires the L10-S source")
        self.arm = source.arm
        self.seed = int(source.seed)
        self.rank = 10
        self.n = N
        self.register_buffer("mass", _copy_buffer(source.mass))
        self.register_buffer("p_scale", _copy_buffer(source.p_scale))
        self.register_buffer("input_scale_delta", _copy_buffer(source.input_scale_delta))
        self.register_buffer("input_scale_omega", _copy_buffer(source.input_scale_omega))
        self.register_buffer("input_scale", _copy_buffer(source.input_scale))
        self.register_buffer("active_edge", _copy_buffer(source.active_edge))
        self.register_buffer("base_D", _copy_buffer(source.base_D))
        self.register_buffer("base_coupling", _copy_buffer(source.base_coupling))
        self.register_buffer("delta_star", _copy_buffer(source.delta_star))
        self.register_buffer("rate", _copy_buffer(source.rate))
        self.register_buffer("gain", _copy_buffer(source.gain))
        self.register_buffer("W", _copy_buffer(source.W))
        self.register_buffer("neighbor_A", _copy_buffer(source.neighbor_A))
        self.register_buffer("port_s_weight", _copy_buffer(source.port_s_weight))
        self.register_buffer("port_c_weight", _copy_buffer(source.port_c_weight))
        self.register_buffer("phi", _copy_buffer(source.phi))

        self.structured = StructuredROM(
            # The fixed graph weight is retained in ``base_coupling`` for the
            # closure bases.  The common M3 skeleton must use the source's
            # learned/effective coupling so that the pair starts with the
            # identical physical RHS.
            mass=source.mass, D=source.base_D, coupling=_effective(source, "coupling"),
            delta_star=source.delta_star,
            damping=_effective(source, "damping") + source.template_coefficients[6] * 0.5,
            gain=source.gain,
            jpi_p=_effective(source, "jpi_p"), rpp=_effective(source, "rpp"),
            n_q=0, storage=(), storage_base=_effective(source, "storage_base")[:N],
            trainable=True,
        ).double()
        self.local_gru = copy.deepcopy(source.local_gru)
        self.local_head = copy.deepcopy(source.local_head)
        self.global_gru = None
        self.global_head = None
        self.closure_coefficients = nn.Parameter(mapped_coefficients(source).detach().clone())
        self._graph_cache: tuple[torch.Tensor, torch.Tensor] | None = None

    @property
    def dim(self) -> int:
        return 40

    @property
    def interface_dim(self) -> int:
        return self.dim

    @property
    def closure_names(self) -> tuple[str, ...]:
        return CLOSURE_NAMES

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "F23":
        """Construct the F arm skeleton from an exported configuration.

        Continuous tensors (including the 23 direct coefficients) are loaded
        separately by ``load_model``/``load_state_dict`` so this constructor is
        deterministic and does not consume the caller's RNG stream.
        """
        source = S23.from_config({**config, "arm": config.get("arm", "L10-S")})
        return cls(source)

    def _split(self, x: torch.Tensor):
        return x[..., :N], x[..., N:2 * N], x[..., 2 * N:3 * N], x[..., 3 * N:4 * N]

    def interface_to_state(self, z: torch.Tensor) -> torch.Tensor:
        d, w, p, q = self._split(z)
        return torch.cat((d, w * self.mass, p, q), dim=-1)

    def state_to_interface(self, x: torch.Tensor) -> torch.Tensor:
        d, pi, p, q = self._split(x)
        return torch.cat((d, pi / self.mass, p, q), dim=-1)

    def _features(self, history: torch.Tensor):
        # Reuse the frozen source implementation verbatim.  In particular,
        # its neighborhood angle/speed columns remain per-machine vectors and
        # are not reduced to a node mean.
        return LocalMemoryROM._features(self, history)

    def encode(self, history: torch.Tensor) -> torch.Tensor:
        return LocalMemoryROM.encode(self, history)

    def _base_field(self, base_x: torch.Tensor) -> torch.Tensor:
        if self._graph_cache is None:
            _, e = self.structured.energy_and_gradient(base_x)
            j, r = self.structured.matrices()
        else:
            j, r = self._graph_cache
            _, e = self.structured.energy_and_gradient(base_x)
        return e @ (j - r).T

    def _basis(self, x: torch.Tensor):
        d, pi, p, q = self._split(x)
        eta = d @ self.base_D.T
        centered_sin = torch.sin(self.phi + eta) - torch.sin(self.phi)
        centered_cos = torch.cos(self.phi + eta) - torch.cos(self.phi)
        kn = self.base_coupling / self.base_coupling.mean().clamp_min(1e-15)
        ts = (kn * centered_sin) @ self.base_D
        tc = (kn * centered_cos) @ torch.abs(self.base_D)
        es = (kn * (q @ self.base_D.T) * torch.cos(self.phi + eta)) @ self.base_D
        ec = (kn * (q @ torch.abs(self.base_D).T) * torch.sin(self.phi + eta)) @ self.base_D
        edge_s = centered_sin @ self.port_s_weight
        edge_c = centered_cos @ self.port_c_weight
        neighbor_q = q @ self.neighbor_A.T - q
        return d, pi, p, q, ts, tc, es, ec, edge_s, edge_c, neighbor_q

    def port_values(self, internal: torch.Tensor) -> torch.Tensor:
        x = internal if internal.shape[-1] == self.dim else self.interface_to_state(internal)
        _, pi, p, q, _, _, _, _, edge_s, edge_c, neighbor_q = self._basis(x)
        c = self.closure_coefficients
        return (c[19] * (pi / self.mass) / self.input_scale_omega
                + c[20] * edge_s + c[21] * edge_c + c[22] * neighbor_q)

    def port_matrix(self) -> torch.Tensor:
        b = torch.zeros((self.dim, N), dtype=self.mass.dtype, device=self.mass.device)
        b[N:2 * N] = torch.eye(N, dtype=b.dtype, device=b.device)
        return b

    def field(self, z: torch.Tensor, u: torch.Tensor | None = None) -> torch.Tensor:
        z = z if torch.is_tensor(z) else torch.as_tensor(z, dtype=self.mass.dtype)
        z = z.to(dtype=self.mass.dtype, device=self.mass.device)
        x = self.interface_to_state(z)
        d, pi, p, q, ts, tc, es, ec, edge_s, edge_c, neighbor_q = self._basis(x)
        c = self.closure_coefficients
        ps = self.p_scale
        j = self.structured._pos("jpi_p")
        rpp = self.structured._pos("rpp")
        storage = self.structured._pos("storage_base")[:N]
        omega = pi / self.mass
        pi_extra = (
            c[0] * (-es) + c[1] * ec + c[2] * (j * q / ps) + c[3] * (p / ps)
            + c[4] * ts + c[5] * tc + c[6] * q
        )
        p_extra = (
            c[7] * (-rpp * q / ps) + c[8] * (p / ps.square())
            + c[9] * (ts / ps) + c[10] * (tc / ps) + c[11] * (q / ps)
        )
        q_extra = (
            c[12] * (-omega) + c[13] * (-storage * p / ps)
            + c[14] * (-q / ps.square()) + c[15] * (-p / ps)
            + c[16] * (-ts) + c[17] * (-tc) + c[18] * (-q)
        )
        base = self._base_field(torch.cat((d, pi, p), dim=-1))
        if u is None:
            pi_extra = pi_extra + self.port_values(x)
        else:
            supplied = u if torch.is_tensor(u) else torch.as_tensor(u, dtype=x.dtype, device=x.device)
            pi_extra = pi_extra + supplied
        internal_rhs = torch.cat((base[..., :N], base[..., N:2 * N] + pi_extra,
                                  base[..., 2 * N:3 * N] + p_extra, q_extra), dim=-1)
        return self.state_to_interface(internal_rhs)

    def internal_field(self, x: torch.Tensor, u: torch.Tensor | None = None) -> torch.Tensor:
        return self.interface_to_state(self.field(self.state_to_interface(x), u=u))

    def begin_graph_cache(self) -> None:
        self._graph_cache = self.structured.matrices()

    def end_graph_cache(self) -> None:
        self._graph_cache = None

    def rk4(self, state: torch.Tensor, dt: float) -> torch.Tensor:
        a = self.field(state)
        b = self.field(state + dt * a / 2)
        c = self.field(state + dt * b / 2)
        d = self.field(state + dt * c)
        return state + dt * (a + 2 * b + 2 * c + d) / 6

    def rollout(self, history: torch.Tensor, steps: int, dt: float) -> torch.Tensor:
        state = torch.cat((history[..., -1, :], self.encode(history)), dim=-1)
        out = [state]
        for _ in range(int(steps)):
            state = self.rk4(state, dt)
            out.append(state)
        return torch.stack(out, dim=1)

    def complexity(self, state: torch.Tensor | None = None) -> torch.Tensor:
        del state
        return self.closure_coefficients.abs().sum()

    def config(self) -> dict[str, Any]:
        return {
            "model_kind": self.model_kind, "arm": self.arm, "seed": self.seed,
            "mass": self.mass.detach().tolist(), "D": self.base_D.detach().tolist(),
            "coupling": self.base_coupling.detach().tolist(), "delta_star": self.delta_star.detach().tolist(),
            "damping": self.structured._pos("damping").detach().tolist(), "gain": self.gain.detach().tolist(),
            "jpi_p": self.structured._pos("jpi_p").detach().tolist(), "rpp": self.structured._pos("rpp").detach().tolist(),
            "storage_base": self.structured._pos("storage_base")[:N].detach().tolist(),
            "input_scale": self.input_scale.detach().tolist(), "p_scale": self.p_scale.detach().tolist(),
            "rate": self.rate.detach().tolist(),
            # Preserve the source's full-graph activity identity.  The F
            # graph below contains only active rows, but checkpoints retain
            # this original mask as a buffer and strict reload must recreate
            # its exact shape.
            "active_edge": self.active_edge.detach().tolist(),
            "closure_names": list(CLOSURE_NAMES),
        }

    def export(self, path: str | Path) -> dict[str, Any]:
        payload = {
            "format": "shared_closure_rom_0101", "model_kind": self.model_kind,
            "config": self.config(), "closure_names": list(CLOSURE_NAMES),
            "rhs_primitives": list(PRIMITIVE_NAMES),
            "rhs_primitive_definitions": RHS_PRIMITIVE_DEFINITIONS,
            "rhs_coefficients": self.closure_coefficients.detach().cpu().tolist(),
            "fixed_graph": {"D": self.base_D.detach().cpu().tolist(), "abs_D": self.base_D.abs().detach().cpu().tolist(),
                            "phi": self.phi.detach().cpu().tolist(), "neighbor_A": self.neighbor_A.detach().cpu().tolist()},
            "unit_metadata": {**UNIT_METADATA, "time_seconds": True},
            "state_dict": {name: value.detach().cpu().tolist() for name, value in self.state_dict().items()},
        }
        Path(path).write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return payload


def _s23_from_source(source: LocalMemoryROM) -> S23:
    if isinstance(source, S23):
        return copy.deepcopy(source)
    model = S23.from_config(source.config())
    model.load_state_dict(copy.deepcopy(source.state_dict()), strict=True)
    with torch.no_grad():
        for index, active in enumerate(EXPECTED_S_MASK):
            model.template_mask[index] = active
    return model


def make_pair(source: LocalMemoryROM) -> tuple[S23, F23]:
    """Create identical-field S23/F23 models without consuming caller RNG."""
    s_model = _s23_from_source(source)
    f_model = F23(s_model)
    with torch.no_grad():
        f_model.closure_coefficients.copy_(mapped_coefficients(s_model).detach())
    return s_model, f_model


def save_model(model: nn.Module, path: str | Path) -> dict[str, Any]:
    payload = {"format": "shared_closure_rom_0101", "model_kind": model.model_kind,
               "config": model.config(), "state_dict": model.state_dict(),
               "closure_names": list(CLOSURE_NAMES)}
    torch.save(payload, path)
    return payload


def _load_payload(path: Path) -> dict[str, Any]:
    if path.is_dir():
        # Runner/evaluator endpoints have used both names while retaining the
        # same JSON schema.  Resolve only known export/checkpoint filenames;
        # never guess from a latest checkpoint.
        candidates = (
            path / "numeric_export.json",
            path / "structured_numeric_export.json",
            path / "model.json",
            path / "model.pt",
        )
        path = next((candidate for candidate in candidates if candidate.exists()), candidates[0])
    if path.suffix.lower() == ".json":
        return json.loads(path.read_text(encoding="utf-8"))
    return torch.load(path, map_location="cpu", weights_only=False)


def load_model(path: str | Path):
    """Load an 0101 export or checkpoint saved by :func:`save_model`."""
    source_path = Path(path)
    payload = _load_payload(source_path)
    kind = payload.get("model_kind")
    if kind == "S23":
        config = payload.get("config", {}).get("config", payload.get("config"))
        model = S23.from_config(config)
        model.load_state_dict({key: torch.as_tensor(value, dtype=torch.float64) for key, value in payload["state_dict"].items()}, strict=True)
        return model
    if kind == "F23":
        config = payload["config"]
        model = _f23_from_config(config)
        model.load_state_dict({key: torch.as_tensor(value, dtype=torch.float64) for key, value in payload["state_dict"].items()}, strict=True)
        return model
    if payload.get("format") == "local_memory_rom_0098":
        model = S23.from_config(payload["config"])
        model.load_state_dict({key: torch.as_tensor(value, dtype=torch.float64) for key, value in payload.get("state_dict", {}).items()}, strict=True)
        return model
    raise ValueError(f"unsupported 0101 model export: {source_path}")


def _f23_from_config(config: dict[str, Any]) -> F23:
    return F23.from_config(config)


__all__ = [
    "CLOSURE_NAMES", "PRIMITIVE_NAMES", "S23", "F23", "mapped_coefficients",
    "make_pair", "save_model", "load_model",
]
