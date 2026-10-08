"""E7 comparison baselines, ported verbatim from the old kron-reduce project.

Three reference learners that are NOT energy/Lyapunov structured.  They learn
time-derivatives from training-context trajectories and roll out autoregressively
from the clearing state, sharing the M0 deployment contract (initial delta0 /
omega0 + exogenous Pm schedule).  None carries an energy structure, so all three
declare ``certificate_claim: False`` -- this is the qualitative axis on which the
modular model's local energy structure is compared. The modular model's current
candidate-energy search is also heuristic and makes no certificate claim.

The only change from the kron-reduce sources is import-rewriting (kiki shares
the base / integration / m0 module layout, so the relative imports are already
identical); the learners themselves are reproduced unmodified so E7 is an
honest cross-project baseline.
  SINDYcModel            : global sparse ridge over a controlled polynomial
                           library in (delta_rel, omega, |.|, Pm), fitted to
                           finite-difference time-derivatives; rk4 rollout.
  PINNModel              : M0 force plus a residual fully-connected NN (physics
                           residual), derivative + equilibrium-penalty loss.
  AutoregressiveLSTMModel: LSTMCell predicting (delta, omega) derivatives,
                           teacher-forced during fit, autoregressive rollout.

Ported 2026-09-04 for kiki E7.  Uses ONLY the causal deployment input (clearing
state + exogenous Pm schedule); observed Pm appears only inside the training
context as teacher signal.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from .base import DynamicsModel, FitResult
from .integration import rk4_rollout
from .m0 import M0Model

__all__ = ["AutoregressiveLSTMModel", "PINNModel", "SINDYcModel"]


class _PINNCore(torch.nn.Module):
    def __init__(self, n_nodes: int, hidden: int):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(3 * n_nodes, hidden),
            torch.nn.Tanh(),
            torch.nn.Linear(hidden, hidden),
            torch.nn.Tanh(),
            torch.nn.Linear(hidden, n_nodes),
        )

    def forward(self, delta_relative, omega, pm):
        inputs = torch.cat((delta_relative, omega, pm), dim=-1)
        zeros = torch.zeros_like(inputs)
        return self.net(inputs) - self.net(zeros)


class _AutoregressiveCore(torch.nn.Module):
    def __init__(self, n_nodes: int, hidden: int):
        super().__init__()
        self.n_nodes = n_nodes
        self.hidden = hidden
        self.init_h = torch.nn.Linear(2 * n_nodes, hidden)
        self.init_c = torch.nn.Linear(2 * n_nodes, hidden)
        self.cell = torch.nn.LSTMCell(3 * n_nodes, hidden)
        self.derivative = torch.nn.Linear(hidden, 2 * n_nodes)

    def initial_hidden(self, state):
        return torch.tanh(self.init_h(state)), torch.tanh(self.init_c(state))

    def step(self, state, pm, hidden):
        h, c = self.cell(torch.cat((state, pm), dim=-1), hidden)
        return self.derivative(h), (h, c)


class SINDYcModel(DynamicsModel):
    method_name = "SINDYc"

    def __init__(self, coefficients=None, n_nodes: int | None = None):
        self.coefficients = None if coefficients is None else np.asarray(coefficients, dtype=np.float64)
        self.n_nodes = n_nodes

    @staticmethod
    def _library(delta, omega, pm, equilibrium):
        relative = delta - equilibrium
        return np.concatenate((
            np.ones(1), relative, omega, relative * np.abs(relative),
            omega * np.abs(omega), pm,
        ))

    def fit(self, train, val, run_context):
        features, derivatives = [], []
        for sample in train[: run_context.max_cases]:
            target = sample["training_context"]["targets"]
            static = sample["model_inputs"]["static"]
            pm = sample["training_context"]["Pm_observed"]
            limit = min(len(target["t"]), max(20, run_context.horizon_steps * 5))
            ddelta = np.gradient(target["delta"][:limit], target["t"][:limit], axis=0)
            domega = np.gradient(target["omega"][:limit], target["t"][:limit], axis=0)
            for k in range(limit):
                features.append(self._library(target["delta"][k], target["omega"][k], pm[k], static["equilibrium_delta"]))
                derivatives.append(np.concatenate((ddelta[k], domega[k])))
        theta = np.asarray(features)
        y = np.asarray(derivatives)
        ridge = 1e-5
        self.coefficients = np.linalg.solve(theta.T @ theta + ridge * np.eye(theta.shape[1]), theta.T @ y)
        self.coefficients[np.abs(self.coefficients) < 1e-6] = 0.0
        self.n_nodes = y.shape[1] // 2
        losses = []
        # kron-reduce divergence (documented in E7 report): the unconstrained
        # global polynomial can diverge non-finitely when rolled out on
        # near-critical val events, so the validation rollout that feeds
        # FitResult.best_validation_loss is guarded -- a diverged case records a
        # large loss instead of aborting the fit.
        for sample in val[: run_context.max_cases]:
            target = sample["training_context"]["targets"]
            steps = min(len(target["t"]), run_context.horizon_steps)
            try:
                dp, wp = self.rollout(sample["model_inputs"]["static"], target["t"][:steps], {"Pm": sample["model_inputs"]["exogenous"]["Pm"][:steps]})
                losses.append(np.mean((dp - target["delta"][:steps]) ** 2) + np.mean((wp - target["omega"][:steps]) ** 2))
            except FloatingPointError:
                losses.append(1e3)
        return FitResult(self.method_name, "completed", tuple(s["event_id"] for s in train[:run_context.max_cases]), tuple(s["event_id"] for s in val[:run_context.max_cases]), float(np.mean(losses)), {"library": "controlled_centered_quadratic", "fit_val_rollout_guard": True})

    def vector_field(self, delta, omega, pm, static):
        if self.coefficients is None or self.n_nodes is None:
            raise RuntimeError("SINDYc is not fitted")
        out = self._library(delta, omega, pm, static["equilibrium_delta"]) @ self.coefficients
        return out[: self.n_nodes], out[self.n_nodes:]

    def rollout(self, initial_state, time_grid, exogenous):
        return rk4_rollout(self.vector_field, np.asarray(initial_state["delta0"]).reshape(-1), np.asarray(initial_state["omega0"]).reshape(-1), time_grid, exogenous["Pm"], initial_state)

    def save(self, path):
        path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, coefficients=self.coefficients, n_nodes=self.n_nodes)
        path.with_suffix(".json").write_text(json.dumps({"method": self.method_name}, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path):
        with np.load(path) as data:
            return cls(data["coefficients"], int(data["n_nodes"]))

    def physics_audit(self):
        return {"status": "pass" if self.coefficients is not None else "not_fitted", "structure": "unconstrained_sparse_controlled", "certificate_claim": False}


class PINNModel(DynamicsModel):
    """Residual NN on top of a self-fitted M0 swing base.

    The M0 base is fit weakly from the same training context (mass-scaled
    force).  The residual maps (delta_rel, omega, Pm) -> force residual, with
    the origin pinned to zero by construction so equilibrium is preserved.  The
    learned residual does NOT inherit the energy/Lyapunov form of the modular
    model, hence certificate_claim=False.
    """

    method_name = "PINN"

    def __init__(self, base=None, n_nodes: int | None = None, hidden: int = 24):
        self.base = base or M0Model()
        self.n_nodes = n_nodes
        self.hidden = hidden
        self.core = _PINNCore(n_nodes, hidden).double() if n_nodes is not None else None

    def _dataset(self, samples, context):
        xs, ps, ys = [], [], []
        for sample in samples[: context.max_cases]:
            target = sample["training_context"]["targets"]
            static = sample["model_inputs"]["static"]
            pm = sample["training_context"]["Pm_observed"]
            limit = min(len(target["t"]), max(20, context.horizon_steps * 4))
            domega = np.gradient(target["omega"][:limit], target["t"][:limit], axis=0)
            for k in range(limit):
                residual = static["mass"] * domega[k] - self.base.force(target["delta"][k], target["omega"][k], pm[k], static)
                xs.append(np.concatenate((target["delta"][k] - static["equilibrium_delta"], target["omega"][k])))
                ps.append(pm[k]); ys.append(residual)
        return tuple(torch.as_tensor(np.asarray(v), dtype=torch.float64) for v in (xs, ps, ys))

    def fit(self, train, val, run_context):
        if self.base.damping is None:
            self.base.fit(train, val, run_context)
        self.n_nodes = train[0]["model_inputs"]["static"]["mass"].size
        torch.manual_seed(run_context.seed)
        self.core = _PINNCore(self.n_nodes, self.hidden).double()
        optimizer = torch.optim.Adam(self.core.parameters(), lr=run_context.learning_rate)
        train_x, train_p, train_y = self._dataset(train, run_context)
        val_x, val_p, val_y = self._dataset(val, run_context)
        best_loss, best_state = float("inf"), None
        for _ in range(max(1, run_context.epochs)):
            optimizer.zero_grad()
            prediction = self.core(train_x[:, :self.n_nodes], train_x[:, self.n_nodes:], train_p)
            equilibrium_penalty = torch.mean(self.core(torch.zeros_like(train_x[:1, :self.n_nodes]), torch.zeros_like(train_x[:1, self.n_nodes:]), torch.zeros_like(train_p[:1])) ** 2)
            loss = torch.mean((prediction - train_y) ** 2) + equilibrium_penalty
            loss.backward(); optimizer.step()
            with torch.no_grad():
                vp = self.core(val_x[:, :self.n_nodes], val_x[:, self.n_nodes:], val_p)
                validation = float(torch.mean((vp - val_y) ** 2))
            if validation < best_loss:
                best_loss = validation
                best_state = {key: value.detach().clone() for key, value in self.core.state_dict().items()}
        assert best_state is not None
        self.core.load_state_dict(best_state)
        return FitResult(self.method_name, "completed", tuple(s["event_id"] for s in train[:run_context.max_cases]), tuple(s["event_id"] for s in val[:run_context.max_cases]), best_loss, {"shared_across_cases": True, "checkpoint_restored": True, "loss": "derivative_plus_equilibrium_physics"})

    def vector_field(self, delta, omega, pm, static):
        if self.core is None:
            raise RuntimeError("PINN is not fitted")
        with torch.no_grad():
            residual = self.core(
                torch.as_tensor(delta - static["equilibrium_delta"], dtype=torch.float64).unsqueeze(0),
                torch.as_tensor(omega, dtype=torch.float64).unsqueeze(0),
                torch.as_tensor(pm, dtype=torch.float64).unsqueeze(0),
            ).squeeze(0).numpy()
        force = self.base.force(delta, omega, pm, static) + residual
        return omega, force / np.asarray(static["mass"])

    def rollout(self, initial_state, time_grid, exogenous):
        return rk4_rollout(self.vector_field, np.asarray(initial_state["delta0"]).reshape(-1), np.asarray(initial_state["omega0"]).reshape(-1), time_grid, exogenous["Pm"], initial_state)

    def save(self, path):
        if self.core is None:
            raise RuntimeError("PINN is not fitted")
        path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"method": self.method_name, "n_nodes": self.n_nodes, "hidden": self.hidden, "state_dict": self.core.state_dict(), "base_damping": self.base.damping, "base_coupling": self.base.coupling}, path)

    @classmethod
    def load(cls, path):
        payload = torch.load(path, map_location="cpu", weights_only=False)
        model = cls(M0Model(payload["base_damping"], payload["base_coupling"]), int(payload["n_nodes"]), int(payload["hidden"]))
        model.core.load_state_dict(payload["state_dict"])
        return model

    def physics_audit(self):
        return {"status": "pass" if self.core is not None else "not_fitted", "shared_model": True, "equilibrium_zero_by_construction": True, "certificate_claim": False}


class AutoregressiveLSTMModel(DynamicsModel):
    method_name = "LSTM"

    def __init__(self, n_nodes: int | None = None, hidden: int = 32):
        self.n_nodes = n_nodes
        self.hidden = hidden
        self.core = _AutoregressiveCore(n_nodes, hidden).double() if n_nodes is not None else None

    def _loss(self, samples, context, train_mode):
        losses = []
        for sample in samples[: context.max_cases]:
            target = sample["training_context"]["targets"]
            pm = torch.as_tensor(sample["training_context"]["Pm_observed"], dtype=torch.float64)
            delta = torch.as_tensor(target["delta"], dtype=torch.float64)
            omega = torch.as_tensor(target["omega"], dtype=torch.float64)
            t = torch.as_tensor(target["t"], dtype=torch.float64)
            steps = min(len(t), context.horizon_steps)
            state0 = torch.cat((delta[0], omega[0])).unsqueeze(0)
            hidden = self.core.initial_hidden(state0)
            for k in range(steps - 1):
                # Teacher forcing is confined to optimizer/validation sequences.
                current = torch.cat((delta[k], omega[k])).unsqueeze(0)
                derivative, hidden = self.core.step(current, pm[k].unsqueeze(0), hidden)
                true_derivative = (torch.cat((delta[k + 1], omega[k + 1])) - current.squeeze(0)) / (t[k + 1] - t[k])
                losses.append(torch.mean((derivative.squeeze(0) - true_derivative) ** 2))
        return torch.mean(torch.stack(losses))

    def fit(self, train, val, run_context):
        self.n_nodes = train[0]["model_inputs"]["static"]["mass"].size
        torch.manual_seed(run_context.seed)
        self.core = _AutoregressiveCore(self.n_nodes, self.hidden).double()
        optimizer = torch.optim.Adam(self.core.parameters(), lr=run_context.learning_rate)
        best_loss, best_state = float("inf"), None
        for _ in range(max(1, run_context.epochs)):
            optimizer.zero_grad(); loss = self._loss(train, run_context, True); loss.backward(); optimizer.step()
            with torch.no_grad():
                validation = float(self._loss(val, run_context, False))
            if validation < best_loss:
                best_loss = validation
                best_state = {key: value.detach().clone() for key, value in self.core.state_dict().items()}
        assert best_state is not None
        self.core.load_state_dict(best_state)
        return FitResult(self.method_name, "completed", tuple(s["event_id"] for s in train[:run_context.max_cases]), tuple(s["event_id"] for s in val[:run_context.max_cases]), best_loss, {"test_teacher_forcing": False, "hidden_initialization": "initial_state", "checkpoint_restored": True})

    def rollout(self, initial_state, time_grid, exogenous):
        if self.core is None:
            raise RuntimeError("LSTM is not fitted")
        t = np.asarray(time_grid, dtype=np.float64); pm = np.asarray(exogenous["Pm"], dtype=np.float64)
        delta = np.empty((len(t), self.n_nodes)); omega = np.empty_like(delta)
        delta[0] = np.asarray(initial_state["delta0"]).reshape(-1); omega[0] = np.asarray(initial_state["omega0"]).reshape(-1)
        with torch.no_grad():
            state = torch.as_tensor(np.concatenate((delta[0], omega[0])), dtype=torch.float64).unsqueeze(0)
            hidden = self.core.initial_hidden(state)
            for k in range(len(t) - 1):
                derivative, hidden = self.core.step(state, torch.as_tensor(pm[k], dtype=torch.float64).unsqueeze(0), hidden)
                dt = float(t[k + 1] - t[k])
                state = state + dt * derivative
                values = state.squeeze(0).numpy()
                delta[k + 1], omega[k + 1] = values[:self.n_nodes], values[self.n_nodes:]
                if not np.isfinite(values).all():
                    raise FloatingPointError(f"nonfinite LSTM rollout at step {k + 1}")
        return delta, omega

    def save(self, path):
        if self.core is None:
            raise RuntimeError("LSTM is not fitted")
        path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"method": self.method_name, "n_nodes": self.n_nodes, "hidden": self.hidden, "state_dict": self.core.state_dict()}, path)

    @classmethod
    def load(cls, path):
        payload = torch.load(path, map_location="cpu", weights_only=False)
        model = cls(int(payload["n_nodes"]), int(payload["hidden"])); model.core.load_state_dict(payload["state_dict"]); return model

    def physics_audit(self):
        return {"status": "pass" if self.core is not None else "not_fitted", "autoregressive_inference": True, "test_teacher_forcing": False, "certificate_claim": False}
