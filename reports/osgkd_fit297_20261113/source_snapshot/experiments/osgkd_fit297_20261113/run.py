"""Release-bound execution of the registered OSGKD fit297 comparison."""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import hashlib
import json
import math
import os
from pathlib import Path
import re
import sys
from time import perf_counter, process_time

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
if __name__ == "__main__":
    sys.path[:0] = [str(ROOT / "src"), str(ROOT)]

from kiki.data.osgkd import group_order, load_fit297
from kiki.models.osgkd import OSGKD, OSGKDConfig
from kiki.models.osgkd_fitting import IntegrationWork, event_objective, integrate, joint_loss, tensor_events


RUN = "osgkd_fit297_20261113"
PROTOCOL = "campaigns/osgkd_ieee39/first_fit297_protocol.md"
SEED = 20261113
VARIANTS = ("C0", "C1")
MODEL_CONSTANTS = ("mass", "damping", "F", "Sp", "Se", "omega_scale", "theta_ref")
CHECKPOINTS = (0, 200, 400, 600, 800, 1000, 1250, 1500)
SELECTABLE = frozenset(CHECKPOINTS) - {200}
COMPONENTS = ("motion", "ports", "consistency")


@contextmanager
def measured(costs, category):
    start_wall, start_cpu = perf_counter(), process_time()
    try:
        yield
    finally:
        entry = costs.setdefault(category, {"wall_s": 0.0, "cpu_s": 0.0})
        entry["wall_s"] += perf_counter() - start_wall
        entry["cpu_s"] += process_time() - start_cpu


def new_model(variant, constants, device):
    torch.manual_seed(SEED)
    config = OSGKDConfig(
        variant, hidden_size=32, p_init_bound=4.0, a_init_bound=1.0, e_init_bound=4.0,
        c_init=float(np.median(constants["Se"])) / 10, beta_fraction_init=.1,
    )
    if constants.get("candidate_graph") == "undirected_complete_K10":
        if len(constants["theta_ref"]) != 10:
            raise ValueError("audited K10 graph requires ten machines")
        edges = np.stack(np.triu_indices(10, k=1))
        if "edge_index" in constants and not np.array_equal(edges, constants["edge_index"]):
            raise ValueError("audited edge ordering differs from complete K10 declaration")
    else:
        edges = constants["edge_index"]
    model = OSGKD(config, edge_index=edges, **{name: constants[name] for name in MODEL_CONSTANTS})
    with torch.no_grad():
        model.mechanical_coeff[0].fill_(-1)
        model.amplitude_coeff[0].fill_(-1)
        model.memory_coeff[0].fill_(-1)
        model.initial_head.weight.normal_(0, .01)
        model.initial_head.bias.zero_()
    return model.to(device)


def parameter_groups(model):
    encoder = list(model.encoder.parameters()) + list(model.initial_head.parameters())
    edges = [model.c] if model.config.variant == "C0" else [model.c, model.beta]
    excluded = {id(p) for p in encoder + edges}
    dynamics = [p for p in model.parameters() if id(p) not in excluded]
    return encoder, dynamics, edges


def new_optimizer(model):
    encoder, dynamics, edges = parameter_groups(model)
    return torch.optim.Adam(
        [{"params": encoder, "lr": .001}, {"params": dynamics + edges, "lr": .0003}],
        betas=(.9, .999), eps=1e-8, weight_decay=0,
    )


class EventBatches:
    def __init__(self, count):
        if count < 1:
            raise ValueError("training has no eligible fit events")
        self.rng = np.random.default_rng(SEED)
        self.count = count
        self.order = []
        self.position = 0

    def next(self):
        if self.position == len(self.order):
            self.order = self.rng.permutation(self.count).tolist()
            self.position = 0
        batch = self.order[self.position:self.position + 8]
        self.position += len(batch)
        return batch

    def state(self):
        return {"rng": self.rng.bit_generator.state, "order": self.order, "position": self.position}


def event_sets(events, constants, device):
    by_id = {event["id"]: event for event in events}
    rank = {group: i for i, group in enumerate(group_order(events))}
    fit_ids, check_ids = (constants[f"{side}_ids"] for side in ("fit", "check"))
    if (len(set(fit_ids + check_ids)) != len(fit_ids) + len(check_ids)
            or set(fit_ids + check_ids) != set(by_id) or len(by_id) != len(events)
            or any(rank[by_id[event_id]["group"]] % 5 == 0 for event_id in fit_ids)
            or any(rank[by_id[event_id]["group"]] % 5 != 0 for event_id in check_ids)):
        raise ValueError("invalid frozen fit/check identity")
    sides, excluded, probes = {}, {}, {}
    for side in ("fit", "check"):
        members = [by_id[event_id] for event_id in constants[f"{side}_ids"]]
        eligible = [event for event in members if len(event["t"]) > 1]
        excluded[side] = [event["id"] for event in members if len(event["t"]) <= 1]
        sides[side] = tensor_events(eligible, device)
        ordered = sorted(eligible, key=lambda event: (rank[event["group"]], str(event["id"])))
        probes[side] = tensor_events(ordered[:12], device)
        if f"{side}_probe_ids" in constants and (
            [event["id"] for event in probes[side]] != constants[f"{side}_probe_ids"]
        ):
            raise ValueError(f"{side} numerical probe identity changed")
        if not sides[side]:
            raise ValueError(f"no eligible events for {side} objective or probe")
    return sides, excluded, probes


@torch.no_grad()
def numeric_probe(model, events, work):
    def initial_from(motion_events):
        theta = torch.stack([event["history"]["theta"] for event in motion_events])
        omega = torch.stack([event["history"]["omega"] for event in motion_events])
        work.encoder_events += len(motion_events)
        return model.initialize(theta, omega), torch.cat((theta[:, -1], omega[:, -1]), dim=-1)

    initial, endpoint = initial_from(events)
    altered = []
    for event in events:
        changed_p0 = event["targets"]["tau_m"].clone()
        changed_p0[0] += 1
        altered.append({**event, "label": "changed-for-input-probe",
                        "targets": {**event["targets"], "tau_m": changed_p0}})
    alternate, _ = initial_from(altered)
    unchanged = (initial == alternate).all(dim=-1).cpu().tolist()
    main = integrate(model, initial, 200, work=work)
    fine = integrate(model, initial, 200, internal_step=.0025, work=work)
    scale = torch.cat((torch.ones_like(model.omega_scale), model.omega_scale, model.init_bounds))
    diff = (main - fine) / scale
    values = torch.stack((
        diff.square().mean(dim=(-2, -1)).sqrt(), diff.abs().amax(dim=(-2, -1)),
        (initial[:, :2 * model.n] - endpoint).abs().amax(dim=-1),
    ), dim=-1).cpu().tolist()
    rows = []
    for event, (rms, maximum, endpoint_error), input_masked in zip(events, values, unchanged, strict=True):
        finite = all(map(math.isfinite, (rms, maximum, endpoint_error)))
        rows.append({
            "id": event["id"], "finite": finite, "rms": rms, "max": maximum,
            "endpoint_error": endpoint_error, "withheld_input_masked": input_masked,
            "passed": finite and rms <= 1e-4 and maximum <= 1e-3 and endpoint_error <= 1e-12 and input_masked,
        })
    return {"passed": bool(rows) and all(row["passed"] for row in rows), "events": rows}


def _fd_edge_feasible(model, direction):
    for eps in (1e-5, 5e-6):
        for sign in (-1, 1):
            c = model.c.detach() + sign * eps * direction[0]
            beta = model.beta.detach()
            if model.config.variant == "C1":
                beta = beta + sign * eps * direction[1]
            if not bool(torch.all((c >= 0) & (beta >= 0) & (beta <= c))):
                return False
    return True


def gradient_probe(model, events, mechanical_scale, work, *, fd_work=None):
    fd_work = work if fd_work is None else fd_work
    groups = parameter_groups(model)
    generator = torch.Generator(device="cpu").manual_seed(SEED)
    directions = []
    for group in groups:
        vector = torch.randn(sum(p.numel() for p in group), dtype=torch.float64, generator=generator)
        vector = (vector / vector.norm()).to(model.mass.device)
        directions.append([v.reshape_as(p) for v, p in zip(
            vector.split([p.numel() for p in group]), group, strict=True
        )])
    edge_feasible = _fd_edge_feasible(model, directions[-1])
    if not edge_feasible:
        return {"passed": False, "reason": "infeasible_fixed_edge_direction", "checks": []}
    names = ("encoder", "dynamics", "edges")
    parameters = [p for group in groups for p in group]
    rows = []
    for event in events[:2]:
        loss = joint_loss(event_objective(model, [event], 20, mechanical_scale, work=work))
        gradients = torch.autograd.grad(loss, parameters, allow_unused=True)
        position = 0
        for name, group, direction in zip(names, groups, directions, strict=True):
            ad = sum(float((grad * v).sum()) for grad, v in zip(
                gradients[position:position + len(group)], direction, strict=True
            ) if grad is not None)
            position += len(group)
            with torch.no_grad():
                original = [p.detach().clone() for p in group]
                for eps in (1e-5, 5e-6):
                    values = []
                    try:
                        for sign in (-1, 1):
                            for p, x, v in zip(group, original, direction, strict=True):
                                p.copy_(x + sign * eps * v)
                            values.append(float(joint_loss(event_objective(
                                model, [event], 20, mechanical_scale, work=fd_work
                            ))))
                    finally:
                        for p, x in zip(group, original, strict=True):
                            p.copy_(x)
                    fd = (values[1] - values[0]) / (2 * eps)
                    relative = abs(fd - ad) / max(1.0, abs(fd), abs(ad))
                    rows.append({
                        "id": event["id"], "direction": name, "eps": eps, "ad": ad, "fd": fd,
                        "relative_error": relative, "feasible": True,
                        "passed": math.isfinite(relative) and relative <= 1e-3,
                    })
    return {"passed": bool(rows) and all(row["passed"] for row in rows), "checks": rows}


def qualify(constants, fit_probes, device):
    results = {}
    for variant in VARIANTS:
        model = new_model(variant, constants, device)
        numeric_work, ad_work, fd_work = (IntegrationWork() for _ in range(3))
        costs = {}
        with measured(costs, "numerical_probe"):
            numeric = numeric_probe(model, fit_probes, numeric_work)
        with measured(costs, "gradient_probe"):
            gradient = gradient_probe(model, fit_probes, constants["S_m"], ad_work, fd_work=fd_work)
        results[variant] = {
            "numeric": numeric, "gradient": gradient, "passed": numeric["passed"] and gradient["passed"],
            "work": {"numeric": _work_receipt(numeric_work),
                     "gradient_ad": _work_receipt(ad_work, with_backward=True),
                     "gradient_fd": _work_receipt(fd_work)}, "function_costs": costs,
        }
    return {"passed": all(row["passed"] for row in results.values()), "variants": results}


def horizon(attempt):
    return 20 if attempt <= 100 else 50 if attempt <= 200 else 200


def adaptation_reason(checkpoints, attempts):
    latest = {row["attempt"]: row for row in checkpoints}
    if len(attempts) != 1000 or not all(step in latest for step in (600, 1000)):
        return None
    prior, last = (latest[step] for step in (600, 1000))
    a, b = prior["fit"]["J"], last["fit"]["J"]
    if not (prior["numeric"]["passed"] and last["numeric"]["passed"]
            and math.isfinite(a) and math.isfinite(b)):
        return None
    recent = attempts[-200:]
    if not all(row["accepted"] for row in recent):
        return None
    if a > 0 and (a - b) / a >= .01:
        return "fit_J_improved"
    if sum(row["clipped"] for row in recent) > 50:
        return "reduce_lr_for_clipping"
    return None


def select_candidate(checkpoints):
    eligible = [
        row for row in checkpoints
        if row["attempt"] in SELECTABLE and row["numeric"]["passed"] and math.isfinite(row["check"]["J"])
    ]
    return min(eligible, key=lambda row: (row["check"]["J"], row["attempt"]), default=None)


@torch.no_grad()
def check_objective(model, events, mechanical_scale, work):
    valid = []
    failed = []
    for start in range(0, len(events), 8):
        batch = events[start:start + 8]
        path_finite = []
        components = event_objective(model, batch, 200, mechanical_scale, work=work,
                                     path_finite=path_finite)
        finite = (components.isfinite().all(dim=-1) & torch.stack(path_finite)).cpu().tolist()
        for event, row, ok in zip(batch, components.cpu().tolist(), finite, strict=True):
            if ok:
                valid.append(row)
            else:
                failed.append(event["id"])
    means = np.mean(valid, axis=0).tolist() if valid else [None] * 3
    score = means[0] + .1 * means[1] + .01 * means[2] if not failed and valid else math.inf
    return {"J": score, "eligible": len(events), "completed": len(valid), "failed_ids": failed,
            "finite_subset_components": dict(zip(COMPONENTS, means, strict=True))}


def training_attempt(model, optimizer, events, attempt, mechanical_scale, work):
    optimizer.zero_grad(set_to_none=True)
    path_finite = []
    components = event_objective(model, events, horizon(attempt), mechanical_scale,
                                 recompute=False, work=work, path_finite=path_finite)
    loss = joint_loss(components)
    row = {"attempt": attempt, "horizon_s": horizon(attempt) / 100,
           "event_ids": [e["id"] for e in events], "J": float(loss.detach()),
           "components": dict(zip(COMPONENTS, components.detach().mean(dim=0).tolist(), strict=True)),
           "accepted": False, "clipped": False, "gradient_norm": None, "projection": None}
    if not bool(torch.stack(path_finite).all()):
        return {**row, "rejection": "nonfinite_trajectory"}
    if not math.isfinite(row["J"]):
        return {**row, "rejection": "nonfinite_loss"}
    loss.backward()
    norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0))
    row["gradient_norm"] = norm
    if not math.isfinite(norm):
        return {**row, "rejection": "nonfinite_gradient"}

    optimizer.step()
    with torch.no_grad():
        c_before, beta_before = model.c.clone(), model.beta.clone()
        model.project_parameters_()
        change = torch.cat((model.c - c_before, model.beta - beta_before))
        changes = [
            (change != 0).sum(), change.abs().amax(), (model.c == 0).sum(),
            (model.beta == 0).sum(), (model.beta == model.c).sum(),
        ]
        count, magnitude, c_zero, beta_zero, beta_cap = torch.stack(changes).cpu().tolist()
    return {**row, "accepted": True, "clipped": norm > 5.0,
            "projection": {"changed_coordinates": int(count), "max_change": magnitude,
                           "c_zero": int(c_zero), "beta_zero": int(beta_zero),
                           "beta_at_c": int(beta_cap)}}


def _work_receipt(work, with_backward=False):
    counts = work.snapshot()
    return {**counts, "rhs_backward_vjp": None if with_backward else 0,
            "rhs_backward_vjp_upper_bound": counts["rhs_forward"] if with_backward else 0}


def _json_safe(value):
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _json(value):
    return json.dumps(_json_safe(value), ensure_ascii=False, allow_nan=False, indent=2) + "\n"


def _identity(path):
    with path.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
        size = stream.tell()
    return {"path": path.resolve().relative_to(ROOT).as_posix(), "sha256": digest, "bytes": size}


def _write_json(path, value):
    temp = path.with_suffix(path.suffix + ".tmp")
    with temp.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(_json(value))
        stream.flush()
        os.fsync(stream.fileno())
    temp.replace(path)
    return _identity(path)


def _checkpoint(path, model, optimizer, batches, attempt, provenance, device):
    temp = path.with_suffix(".tmp")
    payload = {
        "model": model.checkpoint(), "optimizer": optimizer.state_dict(),
        "attempt": attempt, "provenance": provenance, "numpy_batch_rng": batches.state(),
        "torch_rng": torch.get_rng_state(),
        "cuda_rng": torch.cuda.get_rng_state(device) if device.type == "cuda" else None,
    }
    with temp.open("xb") as stream:
        torch.save(payload, stream)
        stream.flush()
        os.fsync(stream.fileno())
    temp.replace(path)
    return _identity(path)


def train_variant(variant, constants, sides, probes, device, directory, provenance):
    model = new_model(variant, constants, device)
    optimizer = new_optimizer(model)
    batches = EventBatches(len(sides["fit"]))
    train_work, check_work, probe_work = (IntegrationWork() for _ in range(3))
    checkpoints, attempts = [], []
    costs = {}
    rejected = consecutive = 0
    adaptation = None
    stop = "budget_complete"
    mechanical_scale = constants["S_m"]

    def save_check(attempt):
        with measured(costs, "checkpoint_serialization"):
            artifact = _checkpoint(directory / f"attempt_{attempt:04d}.pt", model, optimizer, batches,
                                   attempt, provenance, device)
        with measured(costs, "full_objectives"):
            fit = check_objective(model, sides["fit"], mechanical_scale, check_work)
            check = check_objective(model, sides["check"], mechanical_scale, check_work)
        with measured(costs, "check_numerical_probe"):
            probe = numeric_probe(model, probes["check"], probe_work)
        row = {"attempt": attempt, "artifact": artifact, "fit": fit, "check": check, "numeric": probe}
        with measured(costs, "checkpoint_serialization"):
            _write_json(directory / f"attempt_{attempt:04d}.json", row)
        checkpoints.append(row)
        print(f"{variant} attempt {attempt}: fit J={fit['J']:.7g}, check J={check['J']:.7g}, "
              f"numerical={probe['passed']}", flush=True)

    save_check(0)
    limit = 1000
    with (directory / "attempts.jsonl").open("x", encoding="utf-8", newline="\n") as log:
        for attempt in range(1, 1501):
            batch = [sides["fit"][i] for i in batches.next()]
            with measured(costs, "train_attempts"):
                row = training_attempt(model, optimizer, batch, attempt, mechanical_scale, train_work)
            attempts.append(row)
            with measured(costs, "attempt_serialization"):
                log.write(json.dumps(_json_safe(row), allow_nan=False) + "\n")
                log.flush()
            if row["accepted"]:
                consecutive = 0
            else:
                rejected += 1
                consecutive += 1
            if attempt in CHECKPOINTS:
                save_check(attempt)
            if consecutive >= 10 or rejected >= 30:
                stop = "nonfinite_rejection_limit"
                if attempt not in CHECKPOINTS:
                    with measured(costs, "checkpoint_serialization"):
                        _checkpoint(directory / "terminated.pt", model, optimizer, batches, attempt,
                                    provenance, device)
                break
            if attempt == 1000:
                adaptation = adaptation_reason(checkpoints, attempts)
                if adaptation is not None:
                    limit = 1500
                    if adaptation == "reduce_lr_for_clipping":
                        for group in optimizer.param_groups:
                            group["lr"] /= 3
            if attempt == limit:
                break
    return {
        "variant": variant, "parameter_count": sum(p.numel() for p in model.parameters()),
        "attempt_count": len(attempts), "accepted": len(attempts) - rejected, "rejected": rejected,
        "clipped": sum(row["clipped"] for row in attempts), "stop": stop,
        "adaptation": adaptation, "checkpoints": checkpoints, "selected": select_candidate(checkpoints),
        "function_costs": costs,
        "work": {"train": _work_receipt(train_work, with_backward=True),
                 "objectives": _work_receipt(check_work), "probe": _work_receipt(probe_work)},
    }


def train(variant, constants, sides, probes, device, directory, provenance):
    target = directory / variant
    target.mkdir()
    started = (perf_counter(), process_time())
    trial = train_variant(variant, constants, sides, probes, device, target, provenance)
    trial["cost"] = _cost(started, device)
    _write_json(target / "trial.json", trial)
    return {"trials": {variant: trial}}


def choose_pair(trials):
    static, dynamic = (trials[name]["selected"] for name in VARIANTS)
    use_static = static is not None and (dynamic is None or static["check"]["J"] < dynamic["check"]["J"])
    return {"C0": static, "C1_dynamic": dynamic,
            "C1": static if use_static else dynamic, "C1_used_static_fallback": use_static}


def _cost(started, device):
    return {
        "wall_s": perf_counter() - started[0], "cpu_s": process_time() - started[1],
        "cuda_peak_allocated_bytes": torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None,
        "cuda_peak_reserved_bytes": torch.cuda.max_memory_reserved(device) if device.type == "cuda" else None,
    }


REQUIRED_SOURCES = {
    PROTOCOL,
    "campaigns/osgkd_ieee39/execution_amendment.md",
    "campaigns/osgkd_ieee39/port_audit_amendment.md",
    "designs/osgkd_architecture.md",
    "designs/osgkd_observation_adoption_20261113.md",
    f"experiments/{RUN}/run.py",
    "src/kiki/data/osgkd.py",
    "src/kiki/models/osgkd.py",
    "src/kiki/models/osgkd_fitting.py",
    "src/kiki/evaluation/osgkd.py",
    "experiments/evaluate_l1_latent.py",
}


def _repo_path(relative):
    if not isinstance(relative, str) or Path(relative).is_absolute():
        raise ValueError("release references must be repository-relative paths")
    path = (ROOT / relative).resolve()
    if path.relative_to(ROOT).as_posix() != relative:
        raise ValueError("release reference must use its canonical repository path")
    return path


def _verified_ref(ref):
    path = _repo_path(ref["path"])
    if _identity(path) != ref:
        raise ValueError(f"released file identity differs: {ref['path']}")
    return path


def _receipt(ref, phase):
    receipt = json.loads(_verified_ref(ref).read_bytes())
    if receipt["run_id"] != RUN or receipt["phase"] != phase or receipt["status"] != "completed":
        raise ValueError(f"{phase} receipt is not a completed job in this Run")
    return receipt


def _require_same_inputs(previous, release):
    for key in ("audit_ref", "source_hashes", "torch_version", "device", "threads"):
        if previous["release"][key] != release[key]:
            raise ValueError(f"phase receipts disagree on {key}")


def validate_release(release, phase, job_id, variant):
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,95}", job_id):
        raise ValueError("job_id must be a short filesystem-safe identifier")
    if (
        release.get("released") is not True or release.get("run_id") != RUN
        or release.get("phase") != phase or release.get("job_id") != job_id
        or release.get("variant") != variant or (phase == "train") != (variant in VARIANTS)
    ):
        raise ValueError("job is outside the root release")
    if release.get("torch_version") != str(torch.__version__):
        raise ValueError("Torch version differs from the release")
    threads = release.get("threads")
    if type(threads) is not int or threads != 1 or release.get("device") != "cpu":
        raise ValueError("execution amendment requires one CPU thread")
    sources = release.get("source_hashes", {})
    if REQUIRED_SOURCES - sources.keys():
        raise ValueError("release omits required protocol or source hashes")
    for name, digest in sources.items():
        if _identity(_repo_path(name))["sha256"] != digest:
            raise ValueError(f"released source changed: {name}")
    audit = json.loads(_verified_ref(release["audit_ref"]).read_bytes())
    if audit["status"] != "adopted_for_prediction":
        raise ValueError("discrete observation contract not adopted for prediction")
    if phase != "qualify":
        qualified = _receipt(release["qualification_ref"], "qualify")
        _require_same_inputs(qualified, release)
        if qualified["result"]["passed"] is not True:
            raise ValueError("initial numerical or gradient qualification did not pass")
    trials = {}
    if phase == "evaluate":
        for name in VARIANTS:
            trained = _receipt(release["train_refs"][name], "train")
            _require_same_inputs(trained, release)
            if (
                trained["release"]["variant"] != name
                or trained["release"]["qualification_ref"] != release["qualification_ref"]
            ):
                raise ValueError("training receipt identity differs")
            trials[name] = trained["result"]["trials"][name]
        selection = choose_pair(trials)
        for name in ("C0", "C1_dynamic"):
            artifact = selection[name]["artifact"] if selection[name] is not None else None
            if release["selected_model_refs"][name] != artifact:
                raise ValueError(f"released {name} checkpoint differs from the registered selection")
            if artifact is not None:
                _verified_ref(artifact)
    return audit, trials


def evaluate(trials, events, constants, device, directory):
    from kiki.evaluation.osgkd import evaluate_fit297

    selection = choose_pair(trials)
    scores = {}
    for name in ("C0", "C1_dynamic"):
        candidate = selection[name]
        if candidate is None:
            scores[name] = None
            continue
        path = _verified_ref(candidate["artifact"])
        saved = torch.load(path, map_location="cpu", weights_only=True)
        model = OSGKD.from_checkpoint(saved["model"]).to(device).eval()
        if model.config.variant != name.split("_")[0] or saved["attempt"] != candidate["attempt"]:
            raise ValueError("selected checkpoint content differs from its receipt")
        target = directory / name
        target.mkdir()
        with torch.no_grad():
            scores[name] = evaluate_fit297(model, events, constants, output_directory=target)
    selected_name = "C0" if selection["C1_used_static_fallback"] else "C1_dynamic"
    return {"selection": selection, "scores": scores, "C1_score_source": selected_name}


def main(argv=None):
    started = (perf_counter(), process_time())
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("qualify", "train", "evaluate"))
    parser.add_argument("--job-id", required=True)
    parser.add_argument("--variant", choices=VARIANTS)
    args = parser.parse_args(argv)

    release_path = ROOT / "results" / RUN / f"{args.phase}_release.json"
    raw_release = release_path.read_bytes()
    release = json.loads(raw_release)
    audit, trials = validate_release(release, args.phase, args.job_id, args.variant)
    device = torch.device(release["device"])
    torch.set_num_threads(release["threads"])
    if device.type == "cuda":
        if not torch.cuda.is_available() or (device.index is not None and device.index >= torch.cuda.device_count()):
            raise ValueError("released CUDA device is unavailable")
        torch.cuda.reset_peak_memory_stats(device)

    local = ROOT / "artifacts" / RUN / args.job_id
    output = ROOT / "results" / RUN / args.job_id
    if local.exists() or output.exists():
        raise FileExistsError("job id already used; existing jobs are never overwritten")
    local.mkdir(parents=True)
    output.mkdir(parents=True)
    (local / "release.json").write_bytes(raw_release)
    release_ref = _identity(local / "release.json")
    provenance = {
        "release_ref": release_ref, "audit_ref": release["audit_ref"],
        "data_bundle": audit["bundle"], "source_hashes": release["source_hashes"],
    }
    receipt = {
        "run_id": RUN, "phase": args.phase, "job_id": args.job_id, "release": release,
        "provenance": provenance, "status": "running", "result": None,
        "environment": {
            "command": [sys.executable, *sys.argv],
            "python": sys.version, "numpy": np.__version__, "torch": str(torch.__version__),
            "device": str(device), "threads": torch.get_num_threads(),
            "interop_threads": torch.get_num_interop_threads(),
            "cuda_device": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
            "float": "float64",
        },
    }
    _write_json(local / "started.json", receipt)
    costs = {}
    try:
        with measured(costs, "load_frozen_data"):
            events, constants = load_fit297(release["audit_ref"])
        if args.phase == "evaluate":
            result = evaluate(trials, events, constants, device, local)
        else:
            sides, excluded, probes = event_sets(events, constants, device)
            if args.phase == "qualify":
                result = qualify(constants, probes["fit"], device)
            else:
                result = train(args.variant, constants, sides, probes, device, local, provenance)
            result["objective_excluded_ids"] = excluded
        receipt.update(status="completed", result=result)
    except BaseException as error:
        receipt.update(status="interrupted" if isinstance(error, KeyboardInterrupt) else "failed",
                       error=f"{type(error).__name__}: {error}")
        raise
    finally:
        receipt["cost"] = _cost(started, device)
        receipt["function_costs"] = costs
        _write_json(output / "summary.json", receipt)
    print(output / "summary.json", flush=True)
    return 1 if args.phase == "qualify" and not receipt["result"]["passed"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
