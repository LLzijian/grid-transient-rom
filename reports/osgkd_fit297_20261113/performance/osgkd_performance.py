"""Synthetic OSGKD timing and floating-point equivalence check."""
import hashlib
import json
import platform
import sys
import time
from contextlib import nullcontext
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from kiki.models import osgkd_fitting as fitting
from kiki.models.osgkd import OSGKD, OSGKDConfig

torch.set_num_threads(1)
torch.set_num_interop_threads(1)
N = 10
BATCH = 8
SCALE = np.full(N, .1)
DTYPE = torch.float64


def model_and_batch(variant, device):
    rng = np.random.default_rng(20261113)
    torch.manual_seed(20261113)
    model = OSGKD(
        OSGKDConfig(variant, p_init_bound=4, e_init_bound=4),
        mass=np.ones(N), damping=np.zeros(N), F=np.eye(N), Sp=SCALE,
        Se=SCALE, omega_scale=np.ones(N), theta_ref=np.linspace(-.15, .15, N),
        edge_index=np.array([(i, j) for i in range(N) for j in range(i + 1, N)]).T,
    ).to(device)
    with torch.no_grad():
        for name, value in model.named_parameters():
            if name in ("c", "beta"):
                value.fill_(.2 if name == "c" else .08)
            elif "coeff" in name:
                value.normal_(std=.02)
    events = []
    for i in range(BATCH):
        targets = {key: rng.normal(0., .08, (201, N)) for key in ("theta", "omega", "tau_m", "tau_e")}
        masks = {key: np.ones_like(value, dtype=bool) for key, value in targets.items()}
        masks["tau_e"][::3, :3] = False
        history = {key: rng.normal(0., .08, (11, N)) for key in ("theta", "omega")}
        events.append({"id": str(i), "history": history, "t": np.arange(201)*.01,
                       "targets": targets, "masks": masks})
    return model, fitting.tensor_events(events, device)


class TraceRhs(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, state):
        return self.model.rhs(None, state)


def setup(model, device, method):
    if method != "trace":
        return nullcontext()
    example = torch.zeros(BATCH, 5 * N, device=device, dtype=DTYPE)
    traced = torch.jit.trace(TraceRhs(model), (example,), check_trace=False)
    assert {id(parameter) for parameter in traced.parameters()} == {id(parameter) for parameter in model.parameters()}
    return patch.object(model, "rhs", lambda t, x: traced(x))


def evaluate(model, batch, device, steps, recompute):
    def sync():
        if device == "cuda":
            torch.cuda.synchronize()

    model.zero_grad(set_to_none=True)
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    work = fitting.IntegrationWork()
    sync()
    start = time.perf_counter()
    loss = fitting.joint_loss(fitting.event_objective(model, batch, steps, SCALE, recompute=recompute, work=work))
    sync()
    forward = time.perf_counter() - start
    loss.backward()
    sync()
    total = time.perf_counter() - start
    return {"forward_s": forward, "backward_s": total - forward, "total_s": total,
            "loss": loss.item(), "work": work.snapshot(),
            "cuda_peak_allocated_bytes": torch.cuda.max_memory_allocated() if device == "cuda" else None,
            "cuda_peak_reserved_bytes": torch.cuda.max_memory_reserved() if device == "cuda" else None}


def signatures(model, batch, device, method):
    with setup(model, device, method):
        result = evaluate(model, batch, device, 12, recompute=method == "checkpoint")
        initial = model.initialize(torch.stack([e["history"]["theta"] for e in batch]),
                                   torch.stack([e["history"]["omega"] for e in batch]))
        path = fitting.integrate(model, initial, 12, recompute=method == "checkpoint")
        return {"path": path.detach().clone(), "loss": torch.tensor(result["loss"], dtype=DTYPE),
                **{name: param.grad.detach().clone() for name, param in model.named_parameters()}}


def run():
    device, phase, *methods = sys.argv[1:]
    receipt = Path(f".tmp/osgkd_performance_{device}_{phase}.json")
    sources = [Path(__file__), Path("src/kiki/models/osgkd.py"), Path(fitting.__file__)]
    results = {
        "device": torch.cuda.get_device_name() if device == "cuda" else platform.processor(),
        "torch": str(torch.__version__), "threads": torch.get_num_threads(), "synthetic_only": True,
        "batch": BATCH, "internal_step": .005, "dtype": "float64", "optimizer_updates": 0,
        "command": [sys.executable, "-B", *sys.argv], "pythonpath": str(Path("src").resolve()),
        "source_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
        "comparisons": {}, "timing": {},
    }
    steps = 20 if phase == "screen" else 200
    results["future_seconds"] = steps * .01
    for variant in ("C0", "C1") if phase == "screen" else ("C1",):
        model, batch = model_and_batch(variant, device)
        reference = signatures(model, batch, device, "checkpoint")
        for method in methods:
            if method == "checkpoint":
                continue
            candidate = signatures(model, batch, device, method)
            errors = {}
            for name, expected in reference.items():
                actual = candidate[name]
                torch.testing.assert_close(actual, expected, atol=2e-12, rtol=2e-11)
                errors[name] = (actual - expected).abs().max().item()
            results["comparisons"][f"{variant}:{method}"] = errors
            print(variant, method, "largest absolute error", max(errors.values()), flush=True)
        receipt.write_text(json.dumps(results, indent=2))
    for method in methods:
        model, batch = model_and_batch("C1", device)
        with setup(model, device, method):
            evaluate(model, batch, device, 4, recompute=method == "checkpoint")
            row = evaluate(model, batch, device, steps, recompute=method == "checkpoint")
        results["timing"][method] = row
        print(method, json.dumps(row), flush=True)
        receipt.write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    run()
