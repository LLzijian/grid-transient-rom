"""TRAIN-only multiple-shooting helpers for the preregistered 0102 M arm.

The module deliberately owns no data split, sampler, optimizer, or evaluation
logic.  A caller supplies the TRAIN event records and the finite boundary
indices that are actually present in those records.  The first segment of an
event is initialized once by the shared GRU.  Every later segment starts from
the observed delta/omega/p and a TRAIN-only auxiliary q parameter, then uses
only explicit 40-state ``model.rk4`` steps.
"""
from __future__ import annotations

import copy
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import torch
from torch import nn
from train_structured_rom import METRIC, _event_rollout_errors, _point_error


PHYSICAL_DIM = 30
STATE_DIM = 40
Q_DIM = 10
DEFAULT_DT = 0.01
DEFAULT_CONTINUITY_WEIGHT = 1.0
DEFAULT_TRAJECTORY_WEIGHT = 1.0


def _risk(values: torch.Tensor) -> torch.Tensor:
    """The registered 0098 event weighting used for the long term."""
    if values.ndim != 1 or values.numel() == 0:
        raise ValueError("risk values must be a nonempty event vector")
    top = values.topk(max(1, int(torch.ceil(torch.tensor(.1 * values.numel())).item()))).values.mean()
    return .8 * values.mean() + .2 * top


def _as_event_tensor(event: Mapping[str, Any]) -> torch.Tensor:
    if "x" not in event:
        raise KeyError("TRAIN event must contain x")
    x = event["x"] if torch.is_tensor(event["x"]) else torch.as_tensor(event["x"], dtype=torch.float64)
    if x.ndim != 2 or x.shape[1] != PHYSICAL_DIM:
        raise ValueError("event x must have shape [time, 30]")
    if x.shape[0] < 11:
        raise ValueError("event must contain the 11-point initialization history")
    if not torch.isfinite(x).all():
        raise FloatingPointError("nonfinite TRAIN event state")
    return x.to(dtype=torch.float64)


def train_state_scales(
    events: Sequence[Mapping[str, Any]],
    q_reference: torch.Tensor | None = None,
    *,
    model: Any | None = None,
    physical_scales: torch.Tensor | None = None,
    eps: float = 1e-12,
    q_eps: float = 1e-3,
) -> torch.Tensor:
    """Build a fixed diagonal scale from TRAIN observations only.

    The first 30 entries use deterministic uncentered RMS values of the
    supplied TRAIN records.  q has no observed truth, so its scale is one
    unless a TRAIN-only source-rollout reference is explicitly supplied; in
    that case its ten RMS values are frozen from that reference at construction.
    The returned tensor is detached and must be retained by the caller.
    """
    if not events:
        raise ValueError("at least one TRAIN event is required")
    if physical_scales is None and model is not None and hasattr(model, "p_scale"):
        p_scale = torch.as_tensor(model.p_scale, dtype=torch.float64).reshape(-1)
        if p_scale.numel() != 10:
            raise ValueError("model p_scale must contain ten physical scales")
        physical = torch.cat((torch.full((10,), float(METRIC[0]), dtype=torch.float64),
                              torch.full((10,), float(METRIC[1]), dtype=torch.float64),
                              p_scale.detach())).clamp_min(float(eps))
    elif physical_scales is not None:
        physical = torch.as_tensor(physical_scales, dtype=torch.float64).reshape(-1)
        if physical.shape != (PHYSICAL_DIM,):
            raise ValueError("physical_scales must be a 30-vector")
        physical = physical.clamp_min(float(eps))
    else:
        observed = torch.cat([_as_event_tensor(event) for event in events], dim=0)
        physical = torch.sqrt(torch.mean(observed.square(), dim=0)).clamp_min(float(eps))
    if q_reference is None:
        q_scale = torch.ones(Q_DIM, dtype=physical.dtype)
    else:
        q = q_reference if torch.is_tensor(q_reference) else torch.as_tensor(q_reference, dtype=physical.dtype)
        if q.shape[-1] != Q_DIM:
            raise ValueError("q_reference must end in ten latent coordinates")
        q_scale = torch.sqrt(torch.mean(q.detach().to(dtype=physical.dtype).square(), dim=tuple(range(q.ndim - 1))))
        q_scale = q_scale.clamp_min(float(q_eps))
    return torch.cat((physical, q_scale)).detach()


def _event_key(event: Mapping[str, Any], index: int) -> Any:
    return event.get("id", index)


def _resolve_boundaries(
    events: Sequence[Mapping[str, Any]],
    boundaries: Mapping[Any, Sequence[int]] | Sequence[Sequence[int]],
) -> list[list[int]]:
    """Resolve boundary indices while preserving the caller's finite TRAIN data."""
    if isinstance(boundaries, Mapping):
        values = []
        for index, event in enumerate(events):
            key = _event_key(event, index)
            if key not in boundaries:
                raise KeyError(f"missing boundaries for TRAIN event {key!r}")
            values.append(list(boundaries[key]))
    else:
        if len(boundaries) != len(events):
            raise ValueError("one boundary list is required for every TRAIN event")
        values = [list(value) for value in boundaries]
    resolved = []
    for event, points in zip(events, values):
        length = _as_event_tensor(event).shape[0]
        if len(points) < 2 or any(int(point) != point for point in points):
            raise ValueError("each event needs at least two integer boundaries")
        points = [int(point) for point in points]
        if points[0] != 10 or any(right <= left for left, right in zip(points, points[1:])):
            raise ValueError("the first boundary must be the 100 ms index 10 and boundaries must increase")
        if points[-1] >= length:
            raise ValueError("boundary exceeds the observed TRAIN event")
        resolved.append(points)
    return resolved


def _source_q_at_boundary(
    source_model: Any,
    event: Mapping[str, Any],
    boundary: int,
    *,
    start: int = 10,
    dt: float = DEFAULT_DT,
) -> torch.Tensor:
    """Deterministically obtain q at an observed boundary from source autonomy."""
    x = _as_event_tensor(event)
    if boundary < start or boundary >= x.shape[0]:
        raise ValueError("source q boundary outside observed TRAIN event")
    history = x[start - 10:start + 1].unsqueeze(0)
    with torch.no_grad():
        state = torch.cat((history[:, -1], source_model.encode(history)), dim=-1)
        for _ in range(boundary - start):
            state = source_model.rk4(state, float(dt))
        q = state[..., PHYSICAL_DIM:]
    return q[0].detach().clone()


def nonlong_0098_objective(model: Any, samples, history: torch.Tensor, horizon: float, force_scale: float):
    """Compute unchanged 0098 terms other than its long autonomous risk.

    This is the efficient M path: the caller supplies the same selected
    ``samples`` and causal ``history`` as the ordinary objective, while the
    multiple-shooting risk replaces only the long term.  No long autonomous
    rollout is performed here.
    """
    del horizon
    local = _event_rollout_errors(model, samples, history, .2).mean()
    q = model.encode(history)
    state = torch.cat((history[:, -1], q), -1)
    adjacent = torch.stack([_as_event_tensor(event)[index - 9:index + 2] for event, index in samples]).to(history)
    latent = ((model.rk4(state, DEFAULT_DT)[:, PHYSICAL_DIM:] - model.encode(adjacent)) / .1).square().mean()
    predicted_torque = state[:, 20:30] - model.mass * model.field(state)[:, 10:20]
    truth_torque = torch.stack([torch.as_tensor(event["torque"][index], dtype=history.dtype, device=history.device)
                                for event, index in samples])
    torque = ((predicted_torque - truth_torque) / force_scale).square().mean()
    complexity = model.complexity(state)
    total = local + .1 * latent + .1 * torque + 1e-4 * complexity
    return total, {"local": local, "latent_consistency": latent,
                   "torque": torque, "complexity": complexity}


@dataclass(frozen=True)
class SegmentRecord:
    event_index: int
    start: int
    end: int
    auxiliary_index: int | None


class MultipleShootingProblem(nn.Module):
    """Finite TRAIN shooting windows and their trainable interior q starts."""

    def __init__(
        self,
        model: Any,
        events: Sequence[Mapping[str, Any]],
        boundaries: Mapping[Any, Sequence[int]] | Sequence[Sequence[int]],
        *,
        source_model: Any | None = None,
        scales: torch.Tensor | None = None,
        dt: float = DEFAULT_DT,
        q_initial: Mapping[Any, Sequence[float]] | Sequence[Sequence[float]] | None = None,
    ) -> None:
        super().__init__()
        if not events:
            raise ValueError("multiple shooting requires TRAIN events")
        self.dt = float(dt)
        if self.dt <= 0:
            raise ValueError("dt must be positive")
        self.events = tuple(events)
        self.boundaries = tuple(tuple(points) for points in _resolve_boundaries(events, boundaries))
        source = model if source_model is None else source_model
        initial_values: list[torch.Tensor] = []
        self.segment_records: list[SegmentRecord] = []
        aux_index = 0
        for event_index, points in enumerate(self.boundaries):
            for start, end in zip(points[:-1], points[1:]):
                record = SegmentRecord(event_index, start, end, None if start == 10 else aux_index)
                self.segment_records.append(record)
                if start != 10:
                    if q_initial is None:
                        value = _source_q_at_boundary(source, events[event_index], start, dt=self.dt)
                    else:
                        if isinstance(q_initial, Mapping):
                            event_key = _event_key(events[event_index], event_index)
                            boundary_key = (event_key, start)
                            if boundary_key in q_initial:
                                supplied = q_initial[boundary_key]
                            elif event_key in q_initial:
                                supplied = q_initial[event_key]
                            else:
                                raise KeyError(f"missing q_initial for {boundary_key!r}")
                        else:
                            supplied = q_initial[event_index][aux_index]
                        value = torch.as_tensor(supplied, dtype=torch.float64)
                    if value.shape != (Q_DIM,) or not torch.isfinite(value).all():
                        raise ValueError("every auxiliary q start must be a finite ten-vector")
                    initial_values.append(value.detach().clone())
                    aux_index += 1
        if initial_values:
            self.q_aux = nn.ParameterList([nn.Parameter(value) for value in initial_values])
        else:
            self.q_aux = nn.ParameterList()
        if scales is None:
            q_reference = torch.stack(initial_values) if initial_values else None
            scales = train_state_scales(events, q_reference=q_reference, model=model)
        scales = scales if torch.is_tensor(scales) else torch.as_tensor(scales, dtype=torch.float64)
        if scales.shape != (STATE_DIM,) or torch.any(~torch.isfinite(scales)) or torch.any(scales <= 0):
            raise ValueError("scales must be a finite positive 40-vector")
        self.register_buffer("scales", scales.detach().clone().to(dtype=torch.float64))

    @property
    def auxiliary_parameters(self) -> tuple[nn.Parameter, ...]:
        return tuple(self.q_aux)

    @property
    def auxiliary_count(self) -> int:
        return len(self.q_aux)

    def auxiliary_state_dict(self) -> dict[str, torch.Tensor]:
        return {f"q_aux.{index}": value.detach().clone() for index, value in enumerate(self.q_aux)}

    def load_auxiliary_state_dict(self, state: Mapping[str, torch.Tensor]) -> None:
        expected = set(self.auxiliary_state_dict())
        if set(state) != expected:
            raise ValueError("auxiliary q state does not match this TRAIN shooting layout")
        with torch.no_grad():
            for index, value in enumerate(self.q_aux):
                tensor = state[f"q_aux.{index}"]
                if tuple(tensor.shape) != tuple(value.shape):
                    raise ValueError("auxiliary q shape mismatch")
                value.copy_(tensor)

    def _initial_state(self, model: Any, record: SegmentRecord) -> torch.Tensor:
        event = self.events[record.event_index]
        x = _as_event_tensor(event)
        physical = x[record.start]
        if record.auxiliary_index is None:
            history = x[record.start - 10:record.start + 1].unsqueeze(0)
            q = model.encode(history)[0]
        else:
            q = self.q_aux[record.auxiliary_index]
        return torch.cat((physical, q), dim=-1)

    def segment_rollout(self, model: Any, record: SegmentRecord) -> torch.Tensor:
        """Return the finite segment states, including its fixed/start state."""
        state = self._initial_state(model, record)
        states = [state]
        for _ in range(record.start, record.end):
            state = model.rk4(state, self.dt)
            states.append(state)
        return torch.stack(states, dim=0)

    def _selected_occurrences(self, samples: Sequence[tuple[Mapping[str, Any], int]] | None):
        if samples is None:
            return [(index, None) for index in range(len(self.events))]
        occurrences = []
        for event, endpoint in samples:
            found = None
            for index, candidate in enumerate(self.events):
                if event is candidate or (event.get("id", object()) == candidate.get("id", object())):
                    found = index
                    break
            if found is None:
                raise ValueError("sample event is outside the TRAIN shooting problem")
            occurrences.append((found, int(endpoint)))
        if not occurrences:
            raise ValueError("multiple-shooting samples cannot be empty")
        return occurrences

    def loss(
        self,
        model: Any,
        samples: Sequence[tuple[Mapping[str, Any], int]] | None = None,
        horizon: float | None = None,
        *,
        trajectory_weight: float = DEFAULT_TRAJECTORY_WEIGHT,
        continuity_weight: float = DEFAULT_CONTINUITY_WEIGHT,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Compute selected finite TRAIN segments and full-state continuity.

        ``samples`` preserves the runner's event multiplicity.  The endpoint
        is used to reject windows without a real causal history; segment
        boundaries remain the registered event boundaries and are capped at
        the requested real horizon.
        """
        occurrences = self._selected_occurrences(samples)
        if horizon is None:
            horizon_limit = None
        else:
            if float(horizon) <= 0:
                raise ValueError("horizon must be positive")
            horizon_limit = 10 + round(float(horizon) / self.dt)
        trajectory_terms = []
        continuity_terms = []
        continuity_residuals = []
        continuity_event_indices = []
        event_terms: list[list[torch.Tensor]] = []
        for event_index, endpoint in occurrences:
            if endpoint is not None and endpoint < 10:
                raise ValueError("sample endpoint must have ten points of causal history")
            if endpoint is not None and endpoint >= _as_event_tensor(self.events[event_index]).shape[0] - 1:
                raise ValueError("sample endpoint must have at least one real future point")
            event_limit = _as_event_tensor(self.events[event_index]).shape[0] - 1
            cutoff = event_limit if horizon_limit is None else min(event_limit, horizon_limit)
            event_records = []
            for record in self.segment_records:
                if record.event_index != event_index or record.start >= cutoff:
                    continue
                end = min(record.end, cutoff)
                if end > record.start:
                    event_records.append(SegmentRecord(record.event_index, record.start, end,
                                                       record.auxiliary_index))
            if not event_records:
                raise ValueError("selected TRAIN sample has no real segment at this horizon")
            event_values = []
            event_point_sum = None
            event_point_count = 0
            previous_end = None
            for record in event_records:
                states = self.segment_rollout(model, record)
                truth = _as_event_tensor(self.events[record.event_index])[record.start:record.end + 1]
                # The initial point is fixed/observed; score each real future
                # point once, so adjacent segments do not double-count a
                # shared boundary.
                if states.shape[0] <= 1:
                    continue
                point_values = torch.stack([
                    _point_error(prediction, target, model)
                    for prediction, target in zip(states[1:], truth[1:])
                ])
                event_error = point_values.mean()
                event_values.append(event_error)
                event_point_sum = point_values.sum() if event_point_sum is None else event_point_sum + point_values.sum()
                event_point_count += int(point_values.numel())
                if record.auxiliary_index is not None:
                    if previous_end is None:
                        raise RuntimeError("interior segment has no preceding segment")
                    residual = (previous_end - states[0]) / self.scales
                    continuity_residuals.append(residual)
                    continuity_terms.append(residual.square().sum())
                    continuity_event_indices.append(event_index)
                previous_end = states[-1]
            if not event_values:
                raise ValueError("selected TRAIN sample has no real future points")
            # Average all real future points in the occurrence; segment
            # boundaries therefore do not make short final segments equal in
            # weight to longer segments.
            event_mean = event_point_sum / event_point_count
            event_terms.append([event_mean])
            trajectory_terms.append(event_mean)
        if not trajectory_terms:
            raise ValueError("no finite shooting segments")
        trajectory = torch.stack(trajectory_terms).mean()
        continuity = (torch.stack(continuity_terms).mean()
                      if continuity_terms else trajectory.new_zeros(()))
        total = float(trajectory_weight) * trajectory + float(continuity_weight) * continuity
        event_values = torch.stack([torch.stack(values).mean() for values in event_terms])
        residual_tensor = (torch.stack(continuity_residuals)
                           if continuity_residuals else trajectory.new_zeros((0, STATE_DIM)))
        event_index_tensor = torch.as_tensor(continuity_event_indices, dtype=torch.long,
                                             device=trajectory.device)
        return total, {"shooting_trajectory": trajectory, "shooting_continuity": continuity,
                       "shooting_event_values": event_values,
                       "shooting_continuity_residuals": residual_tensor,
                       "shooting_continuity_event_indices": event_index_tensor,
                       "shooting_risk": _risk(event_values), "shooting_total": total}


def make_objective(
    problem: MultipleShootingProblem,
    autonomous_objective: Callable[..., tuple[torch.Tensor, Mapping[str, torch.Tensor]]] | None = None,
    *,
    nonlong_objective: Callable[..., tuple[torch.Tensor, Mapping[str, torch.Tensor]]] | None = None,
    trajectory_weight: float = DEFAULT_TRAJECTORY_WEIGHT,
    continuity_weight: float = DEFAULT_CONTINUITY_WEIGHT,
    risk_weight: float = 10.0,
) -> Callable[..., tuple[torch.Tensor, dict[str, torch.Tensor]]]:
    """Return a runner-compatible objective with M auxiliary terms.

    ``autonomous_objective`` should be the unchanged 0098 objective over the
    same TRAIN samples.  Omitting it is useful for isolated helper QA only; it
    does not claim a deployable training objective.
    """
    def objective(model, samples, history, horizon, force_scale):
        shooting_total, shooting_terms = problem.loss(
            model, samples=samples, horizon=horizon,
            trajectory_weight=trajectory_weight, continuity_weight=continuity_weight)
        if autonomous_objective is None and nonlong_objective is None:
            return shooting_total, dict(shooting_terms)
        if nonlong_objective is not None:
            base_total, base_terms = nonlong_objective(model, samples, history, horizon, force_scale)
            replacement = float(risk_weight) * shooting_terms["shooting_risk"]
            continuity_penalty = float(continuity_weight) * shooting_terms["shooting_continuity"]
            total = base_total + replacement + continuity_penalty
            terms = dict(base_terms)
            terms.update(shooting_terms)
            terms["risk"] = shooting_terms["shooting_risk"]
            terms["shooting_replaced_risk"] = replacement
            terms["shooting_continuity_penalty"] = continuity_penalty
            return total, terms
        assert autonomous_objective is not None
        base_total, base_terms = autonomous_objective(model, samples, history, horizon, force_scale)
        if "risk" not in base_terms:
            raise KeyError("autonomous objective must expose the registered risk term")
        # The M strategy replaces only the long autonomous term.  Local,
        # latent, torque, and complexity terms remain exactly those returned
        # by the unchanged 0098 objective.  Continuity is the separate M
        # auxiliary penalty; the segment fit supplies the replacement risk.
        replacement = float(risk_weight) * shooting_terms["shooting_risk"]
        continuity_penalty = float(continuity_weight) * shooting_terms["shooting_continuity"]
        total = base_total - float(risk_weight) * base_terms["risk"] + replacement + continuity_penalty
        terms = dict(base_terms)
        terms.update(shooting_terms)
        terms["risk"] = shooting_terms["shooting_risk"]
        terms["shooting_replaced_risk"] = replacement
        terms["shooting_continuity_penalty"] = continuity_penalty
        return total, terms
    return objective


__all__ = [
    "PHYSICAL_DIM", "STATE_DIM", "Q_DIM", "DEFAULT_DT",
    "train_state_scales", "SegmentRecord", "MultipleShootingProblem",
    "make_objective", "nonlong_0098_objective",
]
