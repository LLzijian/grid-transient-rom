"""Read-only val scoring with complete original denominators."""
from time import perf_counter, process_time

import numpy as np
import torch

from .osgkd import score_event, summarize
from kiki.models.osgkd_fitting import IntegrationWork, integrate


def missing_row(event):
    return dict(event_id=event["id"], group=event["group"], label=event["label"],
                missing_reason=event["missing_reason"], actual_future_points=None,
                scopes={scope: dict(truth_available=False, prediction_available=False,
                                    requested_points=None, full_horizon_truth=False, points=0,
                                    numerical=dict(passed=False, finite=False, rms=None, maximum=None),
                                    metrics=None) for scope in ("1", "2", "5", "8", "actual")},
                classification_8s=None, classification_actual=None)


@torch.no_grad()
def evaluate_split(model, events, constants, *, split):
    """Evaluate supplied val packets without optimizer updates or writing trajectories."""
    if not events or len({e["id"] for e in events}) != len(events):
        raise ValueError("nonempty unique packets required")
    start, cpu = perf_counter(), process_time()
    work, rows, groups = IntegrationWork(), {}, {}
    was_training = model.training
    model.eval()
    try:
        for event in events:
            if event["history"] is None:
                rows[event["id"]] = missing_row(event)
                continue
            steps = max(800, min(1500, len(event["t"])-1)) if event["label"] == "unstable" else 800
            groups.setdefault(steps, []).append(event)
        for steps, cohort in groups.items():
            for begin in range(0, len(cohort), 16):
                batch = cohort[begin:begin+16]
                theta, omega = (torch.as_tensor(np.stack([e["history"][key] for e in batch]),
                                                dtype=torch.float64, device=model.mass.device)
                                for key in ("theta", "omega"))
                work.encoder_events += len(batch)
                initial = model.initialize(theta, omega)
                if initial.shape != (len(batch), 50) or not torch.equal(initial[:, :10].contiguous().view(torch.int64), theta[:, -1].contiguous().view(torch.int64)) or not torch.equal(initial[:, 10:20].contiguous().view(torch.int64), omega[:, -1].contiguous().view(torch.int64)):
                    raise ValueError("motion-only initial state must preserve observed endpoints")
                main = integrate(model, initial, steps, internal_step=.005, work=work)
                fine = integrate(model, initial, steps, internal_step=.0025, work=work)
                ports = model.ports(fine)
                main, fine = main.cpu().numpy(), fine.cpu().numpy()
                mechanical, electrical = ports.tau_m.cpu().numpy(), ports.tau_e.cpu().numpy()
                for index, event in enumerate(batch):
                    rows[event["id"]] = score_event(event, main[index], fine[index],
                                                    dict(tau_m=mechanical[index], tau_e=electrical[index]), constants)
    finally:
        model.train(was_training)
    rows = [rows[e["id"]] for e in events]
    mapped = summarize(rows, {**constants, "fit_ids": [], "check_ids": []})
    # Fixed horizons require the whole recorded horizon even when truncated metrics exist.
    for name, metric in mapped["splits"]["train_all"].items():
        scope = "actual" if name == "unstable_actual" else name.split("_")[-1][:-1]
        selected = [r for r in rows if name.startswith("all_") or
                    r["label"] == ("unstable" if name == "unstable_actual" else "stable")]
        if name.startswith("stable_") and any(not r["scopes"][scope]["full_horizon_truth"] or
               r["scopes"][scope]["metrics"] is None for r in selected):
            metric["official"] = None
    missing = [e["id"] for e in events if e["history"] is None]
    summary = dict(metric=mapped["metric"], splits={split: mapped["splits"]["train_all"]},
                   classification={split: mapped["classification"]["train_all"]},
                   coverage=dict(denominator=len(events), input_eligible=len(events)-len(missing),
                                 missing_history_ids=missing,
                                 stable=sum(e["label"] == "stable" for e in events),
                                 unstable=sum(e["label"] == "unstable" for e in events)),
                   work=work.snapshot(), cost=dict(wall_s=perf_counter()-start, cpu_s=process_time()-cpu))
    return dict(rows=rows, summary=summary)


def evaluate_val(model, events, constants):
    """Preserve the original val API and denominator behavior."""
    return evaluate_split(model, events, constants, split="val")


def evaluate_train(model, events, constants):
    """Score complete supplied train packets under the same numerical contract."""
    return evaluate_split(model, events, constants, split="train")
