"""Single signed, power-preserving p/r exchange; data-free model only.

Reuses the frozen ForceGraphROM physical skeleton. No optimizer, event access,
source selection, or scientific admission policy is defined here.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
from pathlib import Path

import torch
from torch import nn

from force_graph_model_20261028 import ForceGraphROM, RIDGE

FORMAT = "force_exchange_model_20261028"


class ForceExchangeROM(ForceGraphROM):
    def __init__(self, source, mode="L", *, source_identity=None, train_gamma=True):
        super().__init__(source, mode, source_identity=source_identity)
        self.register_buffer("gamma_reference", torch.tensor(1., dtype=torch.float64))
        self.raw_gamma = nn.Parameter(torch.tensor(0., dtype=torch.float64), requires_grad=train_gamma)

    @property
    def gamma(self):
        return self.gamma_reference * self.raw_gamma

    def _new_context(self):
        c = super()._new_context()
        c["gamma"] = self.gamma
        c["exchange_pr"] = c["gamma"] / (c["kappa"] * self.storage_base).sqrt()
        c["exchange_rp"] = -c["gamma"] * (c["kappa"] * self.storage_base).sqrt()
        return c

    def _stamp(self):
        # Buffers can be changed by explicit physical-fixture edits as well as
        # parameters by updates. Neither may leave a stale propagation graph.
        return super()._stamp() + (tuple(b._version for b in self.buffers()),
                                  tuple(p.requires_grad for p in self.parameters()))

    def field(self, state, u=None):
        base = super().field(state, u=u)
        z = self._tensor(state)
        c = self._context()
        return torch.cat((base[..., :20], base[..., 20:30] + c["exchange_pr"] * z[..., 30:],
                          base[..., 30:] + c["exchange_rp"] * z[..., 20:30]), -1)

    def matrices(self):
        J, R = super().matrices()
        c = self._context()
        link = torch.diag(c["gamma"] * (c["kappa"] / self.storage_base).sqrt())
        J[20:30, 30:], J[30:, 20:30] = link, -link.T
        return J, R

    def exchange_power(self, state):
        z = self._tensor(state)
        p, r = z[..., 20:30], z[..., 30:]
        c = self._context()
        pp = (self.storage_base * p * c["exchange_pr"] * r).sum(-1)
        pr = (r / c["kappa"] * c["exchange_rp"] * p).sum(-1)
        return {"p_power": pp, "r_power": pr, "net_power": pp + pr}

    def _linear_generator(self, context):
        A = super()._linear_generator(context)
        A[20:30, 30:] = torch.diag(context["exchange_pr"])
        A[30:, 20:30] = torch.diag(context["exchange_rp"])
        return A

    @staticmethod
    def _simpson(v, axis):
        return (.02 / 6) * (v.index_select(axis, torch.arange(0, 10, 2, device=v.device))
                            + 4 * v.index_select(axis, torch.arange(1, 10, 2, device=v.device))
                            + v.index_select(axis, torch.arange(2, 11, 2, device=v.device)))

    def _history_context(self, context):
        if "exchange_history" not in context:
            T = self.mass.new_zeros(20, 20)
            T[:10, :10] = -torch.diag(self.rpp * self.storage_base)
            T[:10, 10:] = torch.diag(context["exchange_pr"])
            T[10:, :10] = torch.diag(context["exchange_rp"])
            T[10:, 10:] = -context["Q"]
            B = torch.cat((-torch.diag(self.jpi_p), context["kappa"] * self.W))
            augmented = self.mass.new_zeros(40, 40)
            # Unit reverse interval s: inputs are omega_now and
            # Delta_omega=omega_prev-omega_now, avoiding 1/dt amplification.
            augmented[:20, :20], augmented[:20, 20:30] = -.01 * T, -.01 * B
            augmented[20:30, 30:] = self.W
            flow = torch.matrix_exp(augmented)[:20]
            Z = [torch.cat((torch.zeros_like(self.W), self.W))]
            for _ in range(10):
                Z.append(flow[:, :20] @ Z[-1])
            Z = torch.stack(list(reversed(Z)))
            C = self._simpson(Z[:, 10:] - (self.jpi_p * self.storage_base)[None, :, None] * Z[:, :10], 0)
            A = (C * self.rstar[None, None, :] / (.02 * self.rstar[None, :, None])).reshape(50, 10)
            F = Z[0, 10:] * self.rstar[None, :] / self.rstar[:, None]
            design = torch.cat((A, math.sqrt(RIDGE) * F))
            # Reduced QR avoids normal-equation conditioning and repeated-root
            # eigenderivatives; all factors remain in the autograd graph.
            qr_q, qr_r = torch.linalg.qr(design, mode="reduced")
            context["exchange_history"] = dict(T=T, B=B, flow=flow, Z=Z, C=C, A=A, F=F,
                design=design, qr_q=qr_q, qr_r=qr_r, normal=design.T @ design)
        return context["exchange_history"]

    def history_system(self, history):
        h = self._tensor(history)
        if h.ndim not in (2, 3) or h.shape[-2:] != (11, 30):
            raise ValueError("exactly 11 causal 30-column rows required")
        if not torch.isfinite(h[..., :20]).all() or not torch.isfinite(h[..., -1, 20:30]).all():
            raise ValueError("finite delta/omega history and terminal p0 required")
        context = self._context()
        hc = self._history_context(context)
        omega = h[..., 10:20]
        delta_omega = omega[..., :-1, :] - omega[..., 1:, :]
        v = [torch.cat((h[..., -1, 20:30], torch.zeros_like(h[..., -1, 20:30])), -1)]
        for i in range(9, -1, -1):
            a = torch.cat((v[-1], omega[..., i + 1, :], delta_omega[..., i, :]), -1)
            v.append(a @ hc["flow"].T)
        vbar = torch.stack(list(reversed(v)), -2)
        pbar, rbar = vbar[..., :10], vbar[..., 10:]
        fbase = self.base_momentum_field(torch.cat((h[..., :20], pbar), -1), context)
        b = self._simpson(fbase - rbar, -2) - self.mass * (omega[..., 2:11:2, :] - omega[..., 0:10:2, :])
        bscaled = (b / (.02 * self.rstar)).flatten(-2)
        q = rbar[..., 0, :] / self.rstar
        target = torch.cat((bscaled, -math.sqrt(RIDGE) * q), -1)
        rhs = target @ hc["qr_q"]
        s = torch.linalg.solve_triangular(hc["qr_r"], rhs.unsqueeze(-1), upper=True).squeeze(-1)
        r0 = self.rstar * s
        reconstructed = vbar + torch.einsum("tij,...j->...ti", hc["Z"], r0)
        return {**hc, "s": s, "r0": r0, "u": reconstructed[..., 0, 10:],
                "vbar": vbar, "q": q, "p_history": reconstructed[..., :10],
                "r_history": reconstructed[..., 10:], "b": b, "bscaled": bscaled,
                "target": target, "fbase": fbase}

    def initialization_diagnostics(self, history):
        s = self.history_system(history)
        residual = s["s"] @ s["A"].T - s["bscaled"]
        penalty = s["s"] @ s["F"].T + s["q"]
        return {"design_singular_values": torch.linalg.svdvals(s["A"]).detach().cpu().tolist(),
                "augmented_design_condition_number": float(torch.linalg.cond(s["design"]).detach()),
                "reverse_flow_norm": float(s["flow"][:, :20].norm().detach()),
                "ridge": RIDGE, "scaled_residual_l2": residual.norm(dim=-1).detach().cpu().tolist(),
                "ridge_penalty": (RIDGE * penalty.square().sum(-1)).detach().cpu().tolist(),
                "claim_scope": "ten unknown terminal forces; no parameter identifiability claim"}

    def complexity(self, state=None):
        return super().complexity(state) + int(self.raw_gamma.requires_grad)

    def complexity_counts(self):
        result = super().complexity_counts()
        result.update(raw_trainable_scalar_count=sum(p.numel() for p in self.parameters() if p.requires_grad),
                      exchange_scalar_coordinates=int(self.raw_gamma.requires_grad), history_linear_dimension=20)
        return result

    @classmethod
    def from_force_graph_export(cls, path, *, train_gamma=True):
        path = Path(path)
        original = ForceGraphROM.from_export(path)
        identity = {"force_graph_export_path": str(path.resolve()),
                    "force_graph_export_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                    "parent_source_identity": copy.deepcopy(original.source_identity)}
        model = cls(original._source_config, original.mode, source_identity=identity, train_gamma=train_gamma)
        model.load_state_dict({**model.state_dict(), **original.state_dict()})
        return model

    def export(self, path):
        if self._cache is not None:
            raise RuntimeError("end the active graph cache before export")
        payload = dict(format=FORMAT, mode=self.mode, config=self._source_config,
            source_identity=copy.deepcopy(self.source_identity), ridge=RIDGE,
            state_dict={k: v.detach().cpu().tolist() for k, v in self.state_dict().items()},
            requires_grad={k: p.requires_grad for k, p in self.named_parameters()},
            units={"gamma": "1/s", "gamma_reference": "1/s", "p": "inherited physical p", "r": "physical force"},
            counts=self.complexity_counts(), integration="classical RK4; fast interface is not scientifically qualified")
        Path(path).write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        return payload

    @classmethod
    def from_export(cls, path):
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if payload.get("format") != FORMAT or payload.get("ridge") != RIDGE:
            raise ValueError("wrong exchange format or initializer regularization")
        model = cls(payload["config"], payload["mode"], source_identity=payload["source_identity"],
                    train_gamma=payload["requires_grad"]["raw_gamma"])
        template = model.state_dict()
        if set(template) != set(payload["state_dict"]):
            raise ValueError("exported state keys differ")
        supplied = {k: torch.as_tensor(payload["state_dict"][k], dtype=v.dtype) for k, v in template.items()}
        if any(v.shape != template[k].shape or not torch.isfinite(v).all() for k, v in supplied.items()):
            raise ValueError("invalid exported state")
        model.load_state_dict(supplied)
        if model.gamma_reference.item() != 1.:
            raise ValueError("gamma reference must remain 1/s")
        if ((model.k_bar < 0).any() if model.mode == "S" else
                (model.a_bar < 0).any() if model.mode == "M" else False):
            raise ValueError("infeasible source graph")
        if set(payload["requires_grad"]) != set(dict(model.named_parameters())):
            raise ValueError("parameter trainability keys differ")
        for name, parameter in model.named_parameters():
            parameter.requires_grad_(payload["requires_grad"][name])
        return model
