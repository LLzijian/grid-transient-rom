"""Differentiable, event-balanced fitting primitives for motion-only OSGKD."""
from __future__ import annotations

from collections import defaultdict
from contextlib import contextmanager

import numpy as np
import torch
from torch.utils.checkpoint import checkpoint


class IntegrationWork:
    def __init__(self):
        self.phase = "forward"
        self.rhs = {"forward": 0, "recompute": 0}
        self.encoder_events = 0

    @contextmanager
    def during(self, phase):
        previous = self.phase
        self.phase = phase
        try:
            yield
        finally:
            self.phase = previous

    def snapshot(self):
        return {"rhs_forward": self.rhs["forward"],
                "rhs_recompute": self.rhs["recompute"],
                "encoder_events": self.encoder_events}


def integrate(model, state, steps, *, internal_step=.005, recompute=False, work=None):
    """Stop each grouped trajectory at its actual horizon; return 0.01 s outputs."""
    if type(steps) is not int or steps < 0 or internal_step not in (.005, .0025):
        raise ValueError("integer output steps and registered internal step required")
    work = IntegrationWork() if work is None else work
    batch = int(state.numel() // (5 * model.n))
    substeps = round(.01 / internal_step)
    outputs_per_segment = 20 // substeps

    def field(x):
        work.rhs[work.phase] += batch
        return model.rhs(None, x)

    def step(x):
        h = internal_step
        k1 = field(x)
        k2 = field(x + h / 2 * k1)
        k3 = field(x + h / 2 * k2)
        k4 = field(x + h * k3)
        return x + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)

    paths = [state.unsqueeze(-2)]
    for start in range(0, steps, outputs_per_segment):
        count = min(outputs_per_segment, steps - start)

        def segment(x, count=count):
            output = []
            for _ in range(count):
                for _ in range(substeps):
                    x = step(x)
                output.append(x)
            return torch.stack(output, dim=-2)

        if recompute and torch.is_grad_enabled():
            part = checkpoint(segment, state, use_reentrant=False,
                              context_fn=lambda: (work.during("forward"), work.during("recompute")))
        else:
            part = segment(state)
        paths.append(part)
        state = part[..., -1, :]
    return torch.cat(paths, dim=-2)


def tensor_events(events, device, *, max_steps=200):
    """Keep metadata outside the only two tensors accepted by initialize()."""
    converted = []
    for event in events:
        times = np.asarray(event["t"], dtype=float)
        if (times.ndim != 1 or not len(times) or not np.isfinite(times).all() or abs(times[0]) > 1e-10
                or not np.allclose(np.diff(times), .01, rtol=0, atol=1e-9)):
            raise ValueError("targets must begin at the decision point on the 0.01 s grid")
        limit = min(len(times), max_steps + 1)
        item = {"id": event["id"], "history": {}, "targets": {}, "masks": {}}
        for key in ("theta", "omega"):
            value = np.asarray(event["history"][key], dtype=float)
            if value.ndim != 2 or value.shape[0] != 11 or not np.isfinite(value).all():
                raise ValueError("eleven finite motion observations required")
            item["history"][key] = torch.as_tensor(value, dtype=torch.float64, device=device)
        n = item["history"]["theta"].shape[1]
        for key in ("theta", "omega", "tau_m", "tau_e"):
            value = np.asarray(event["targets"][key], dtype=float)[:limit]
            mask = np.asarray(event["masks"][key], dtype=bool)[:limit]
            if value.shape != (limit, n) or mask.shape != value.shape:
                raise ValueError("node-aligned target and measurement mask required")
            if np.any(mask & ~np.isfinite(value)):
                raise ValueError("an observed target is nonfinite")
            if key in ("theta", "omega") and not mask.all():
                raise ValueError("motion targets must be complete within the actual record")
            # Missing measurements never participate in a loss or its gradient.
            item["targets"][key] = torch.as_tensor(np.where(mask, value, 0.),
                                                    dtype=torch.float64, device=device)
            item["masks"][key] = torch.as_tensor(mask, dtype=torch.bool, device=device)
        converted.append(item)
    return converted


def event_objective(model, events, steps, mechanical_scale, *, recompute=False, work=None,
                    path_finite=None):
    """Return one [motion, ports, consistency] row per event, without padded futures."""
    work = IntegrationWork() if work is None else work
    if not events or steps < 1:
        raise ValueError("a nonempty batch and positive future horizon required")
    history_theta = torch.stack([e["history"]["theta"] for e in events])
    history_omega = torch.stack([e["history"]["omega"] for e in events])
    work.encoder_events += len(events)
    initial = model.initialize(history_theta, history_omega)
    groups = defaultdict(list)
    for index, event in enumerate(events):
        count = min(steps, len(event["targets"]["theta"]) - 1)
        if count < 1:
            raise ValueError("no available future; retain as unavailable in the evaluation denominator")
        groups[count].append(index)
    losses = [None] * len(events)
    finite = [None] * len(events)
    mechanical_scale = torch.as_tensor(mechanical_scale, dtype=torch.float64, device=initial.device)
    if (mechanical_scale.shape != (model.n,) or not torch.isfinite(mechanical_scale).all()
            or not torch.all(mechanical_scale > 0)):
        raise ValueError("positive node-aligned mechanical output scale required")
    for count, indices in groups.items():
        path = integrate(model, initial[indices], count, recompute=recompute, work=work)
        ports = model.ports(path[:, 1:])
        for local, index in enumerate(indices):
            event = events[index]
            if path_finite is not None:
                finite[index] = torch.isfinite(path[local]).all()
            target = event["targets"]
            motion = .5 * ((path[local, 1:, :model.n] - target["theta"][1:count+1]).square().mean()
                           + ((path[local, 1:, model.n:2*model.n] - target["omega"][1:count+1])
                              / model.omega_scale).square().mean())
            port_terms = []
            for key, prediction, scale in (("tau_m", ports.tau_m[local], mechanical_scale),
                                            ("tau_e", ports.tau_e[local], model.Se)):
                mask = event["masks"][key][1:count+1]
                if mask.any():
                    error = (prediction - target[key][1:count+1]) / scale
                    port_terms.append(error.square()[mask].mean())
            port_loss = torch.stack(port_terms).mean() if port_terms else motion.new_zeros(())
            consistency = motion.new_zeros(())
            if count >= 10 and len(target["theta"]) >= 11:
                work.encoder_events += 1
                encoded = model.initialize(target["theta"][:11], target["omega"][:11])
                difference = (path[local, 10, 2*model.n:] - encoded[2*model.n:]) / model.init_bounds
                consistency = difference.square().mean()
            losses[index] = torch.stack((motion, port_loss, consistency))
    if path_finite is not None:
        path_finite.extend(finite)
    return torch.stack(losses)


def joint_loss(components):
    return (components[:, 0] + .1 * components[:, 1] + .01 * components[:, 2]).mean()
