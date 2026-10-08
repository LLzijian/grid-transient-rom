"""Refit one fold/stage without touching the consumed test population.

Usage::

    python experiments/run_refit_v2.py F1 backbone [bootstrap_draws]
    python experiments/run_refit_v2.py F1 full [bootstrap_draws]
    python experiments/run_refit_v2.py F1 prune [bootstrap_draws]

The output is immutable once created.  Only ``split.train_ids`` and
``split.val_ids`` are adapted or loaded; this runner has no test-read path.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
import tempfile
import time
import uuid
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import scipy

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from kiki.data import CanonicalStore, ModelDataRequest, adapt_event, load_split
from kiki.evaluation.metrics import aggregate_trajectory_metrics, trajectory_metrics
from kiki.models.base import RunContext, sample_rollout_inputs
from kiki.models.prunable_energy_model import PrunableEnergyModel

H100_POINTS = 101
H100_NOMINAL_SECONDS = 1.0
MODULE_ADOPTION_MIN_IMPROVEMENT = 0.02
STAGES = ("backbone", "full", "prune")
FOLDS = ("F1", "F2", "F3", "F4", "F5")


def sha256_file(path: str | Path) -> str:
    """Return the SHA-256 digest of a file without loading it all at once."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_ids(ids: Sequence[str]) -> str:
    payload = json.dumps(list(ids), ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def json_safe(value: Any) -> Any:
    """Convert scientific Python values to strict-JSON-compatible values."""
    if isinstance(value, np.ndarray):
        return [json_safe(item) for item in value.tolist()]
    if isinstance(value, np.generic):
        return json_safe(value.item())
    if isinstance(value, Path):
        return str(value)
    if dataclasses.is_dataclass(value):
        return json_safe(dataclasses.asdict(value))
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, float):
        return value if np.isfinite(value) else None
    if isinstance(value, (str, int, bool)) or value is None:
        return value
    return str(value)


def atomic_write_json(path: str | Path, value: Any) -> None:
    """Write strict JSON through a same-directory temporary file and replace."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(
        json_safe(value), indent=2, ensure_ascii=False, allow_nan=False
    )
    fd, temporary = tempfile.mkstemp(
        prefix=f".{target.name}.", suffix=".tmp", dir=target.parent
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(payload)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        Path(temporary).replace(target)
    finally:
        temporary_path = Path(temporary)
        if temporary_path.exists():
            temporary_path.unlink()


def normalize_json_file(path: str | Path) -> None:
    """Rewrite a model sidecar through the strict JSON sanitizer."""
    source = Path(path).read_text(encoding="utf-8")
    parsed = json.loads(source, parse_constant=lambda token: float(token))
    atomic_write_json(path, parsed)


def _git_value(*args: str) -> str:
    result = subprocess.run(
        ["git", *args], cwd=ROOT, check=True, capture_output=True, text=True
    )
    return result.stdout.strip()


def git_provenance() -> dict[str, Any]:
    status = _git_value("status", "--porcelain", "--untracked-files=all")
    if status:
        raise RuntimeError("refit v2 requires a clean git worktree before startup")
    return {"head": _git_value("rev-parse", "HEAD"), "worktree": "clean"}


def require_frozen_git(expected_head: str) -> None:
    """Refuse publication if tracked code/config changed during the run."""
    if _git_value("rev-parse", "HEAD") != expected_head:
        raise RuntimeError("git HEAD changed while the refit stage was running")
    unstaged = _git_value("diff", "--name-only")
    staged = _git_value("diff", "--cached", "--name-only")
    if unstaged or staged:
        raise RuntimeError("tracked files changed while the refit stage was running")


def _package_version(name: str, fallback: str = "unknown") -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return fallback


def _event_provenance(store: CanonicalStore, ids: Sequence[str]) -> list[dict[str, Any]]:
    records = []
    for event_id in ids:
        metadata = store.metadata(event_id)
        records.append(
            {
                "event_id": event_id,
                "source_dataset": metadata.source_dataset,
                "fault_line": metadata.fault_line,
                "outcome": metadata.outcome_label,
                "event_signature_sha256": metadata.event_signature_sha256,
            }
        )
    return records


def build_provenance(
    config: dict[str, Any],
    store: CanonicalStore,
    meta: Any,
    split: Any,
    context: RunContext,
    fold: str,
    git: dict[str, Any],
    stage: str,
    dependency_hashes: dict[str, dict[str, str]],
) -> dict[str, Any]:
    dataset_dir = Path(config["dataset_dir"])
    split_path = dataset_dir / "splits" / f"{split.split_name}.json"
    train_ids = list(split.train_ids)
    val_ids = list(split.val_ids)
    split_provenance: dict[str, Any] = {
        "view": split.view,
        "split_name": split.split_name,
        "n_train": len(train_ids),
        "n_val": len(val_ids),
        "train_ids_sha256": sha256_ids(train_ids),
        "val_ids_sha256": sha256_ids(val_ids),
    }
    if stage == "backbone":
        split_provenance.update(
            {
                "train_ids": train_ids,
                "val_ids": val_ids,
                "train_event_signatures": _event_provenance(store, train_ids),
                "val_event_signatures": _event_provenance(store, val_ids),
            }
        )
    else:
        manifest_key = "backbone_summary" if stage == "full" else "full_summary"
        split_provenance["population_manifest"] = dependency_hashes[manifest_key]
    return {
        "fold": fold,
        "dataset": {
            "dataset_name": config["dataset_name"],
            "dataset_revision": meta.dataset_revision,
            "build_id": meta.build_id,
            "schema_version": meta.schema_version,
            "files": {
                "dataset_index_csv": sha256_file(dataset_dir / "dataset_index.csv"),
                "schema_json": sha256_file(dataset_dir / "schema.json"),
                "split_config_json": sha256_file(split_path),
            },
            "h5_strategy": (
                "The 390 MB H5 payload is not hashed; build_id, dataset index, "
                "schema, and per-event signatures are recorded instead."
            ),
        },
        "split": split_provenance,
        "git": git,
        "runtime": {
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "h5py": h5py.__version__,
            "kiki": _package_version("kiki", "0.1.0"),
        },
        "run_context": dataclasses.asdict(context),
    }


def h100_eligible(time_grid: Sequence[float] | np.ndarray) -> bool:
    return len(time_grid) >= H100_POINTS


def h100_event_report(model: Any, sample: dict[str, Any]) -> dict[str, Any]:
    """Report exactly H100 or record an explicit exclusion/failure."""
    event_id = str(sample["event_id"])
    metadata = sample["training_context"].get("metadata", {})
    initial, time, exogenous = sample_rollout_inputs(sample)
    base = {
        "event_id": event_id,
        "source_dataset": metadata.get("source_dataset"),
        "fault_line": metadata.get("fault_line"),
        "outcome": sample["training_context"].get("label"),
    }
    if not h100_eligible(time):
        return {
            **base,
            "status": "excluded_short",
            "available_samples": len(time),
            "required_samples": H100_POINTS,
        }
    try:
        pred_delta, pred_omega = model.rollout(
            initial,
            time[:H100_POINTS],
            {
                "Pm": exogenous["Pm"][:H100_POINTS],
                "interpolation": exogenous["interpolation"],
            },
        )
        targets = sample["training_context"]["targets"]
        metrics = trajectory_metrics(
            np.asarray(targets["delta"][:H100_POINTS]),
            np.asarray(targets["omega"][:H100_POINTS]),
            pred_delta,
            pred_omega,
        )
        return {
            **base,
            "status": "success",
            "n_samples": H100_POINTS,
            "endpoint_seconds": float(time[H100_POINTS - 1]),
            "trajectory_metrics": metrics,
        }
    except Exception as error:  # noqa: BLE001 - retain per-event failure evidence
        return {
            **base,
            "status": "failure",
            "n_samples": H100_POINTS,
            "endpoint_seconds": float(time[H100_POINTS - 1]),
            "error": {"type": type(error).__name__, "message": str(error)},
        }


def h100_report(model: Any, samples: Sequence[dict[str, Any]]) -> dict[str, Any]:
    declared_events = [h100_event_report(model, sample) for sample in samples]
    eligible = [
        row for row in declared_events if row["status"] != "excluded_short"
    ]
    successes = [row for row in eligible if row["status"] == "success"]
    failures = [row for row in eligible if row["status"] == "failure"]
    metrics = [row["trajectory_metrics"] for row in successes]
    aggregate = aggregate_trajectory_metrics(metrics)
    n_eligible = len(eligible)
    aggregate.update(
        {
            "n_declared": len(declared_events),
            "n_eligible": n_eligible,
            "n_success": len(successes),
            "n_failures": len(failures),
            "failure_rate": len(failures) / n_eligible if n_eligible else None,
        }
    )
    return {
        "horizon": "H100",
        "definition": "exactly_100_integration_intervals_101_endpoint_inclusive_samples",
        "required_samples": H100_POINTS,
        "nominal_duration_seconds_approximate": H100_NOMINAL_SECONDS,
        "aggregate": aggregate,
        "events": eligible,
        "excluded_short_events": [
            row for row in declared_events if row["status"] == "excluded_short"
        ],
    }


def _run_context(config: dict[str, Any], train: Sequence[Any], val: Sequence[Any]) -> RunContext:
    return RunContext(
        seed=int(config["seed"]),
        epochs=100,
        horizon_steps=100,
        max_cases=max(len(train), len(val)),
        ridge_values=(1e-8, 1e-6, 1e-4, 1e-2),
        integral_windows=(1, 5, 20),
        rollout_refine=True,
        refinement_min_improvement=0.02,
        max_internal_step=0.01,
        weak_max_windows_per_case_horizon=120,
    )


def _adapt(config: dict[str, Any], store: CanonicalStore, meta: Any, ids: Sequence[str]) -> list[dict[str, Any]]:
    request = ModelDataRequest.initial_state_rollout(config["pm_mode"])
    return [
        adapt_event(
            store.load_event(event_id),
            inertia_prior=meta.M_prior,
            request=request,
            representation=config["representation"],
        )
        for event_id in ids
    ]


def _model_report(model: PrunableEnergyModel, train: list[dict[str, Any]], val: list[dict[str, Any]], context: RunContext) -> dict[str, Any]:
    return {
        "active_blocks": list(model.active_blocks),
        "selected_ridge": model.selected_ridge,
        "physics_audit": model.physics_audit(train[0]["model_inputs"]["static"]),
        "validation_summary": {
            "role": "selection_diagnostic_not_fixed_H100",
            "mixed_available_length_selection_summary": model.validation_summary(val, context),
        },
        "val_metrics_H100": h100_report(model, val),
    }


def _verify_reload(
    model_path: Path,
    model: PrunableEnergyModel,
    train: list[dict[str, Any]],
    val: list[dict[str, Any]],
    context: RunContext,
    stage: str,
) -> dict[str, Any]:
    reloaded = PrunableEnergyModel.load(model_path)
    if tuple(reloaded.active_blocks) != tuple(model.active_blocks):
        raise RuntimeError("reload changed active blocks")
    if reloaded.selected_ridge != model.selected_ridge:
        raise RuntimeError("reload changed selected ridge")
    if reloaded.validation_scales != model.validation_scales:
        raise RuntimeError("reload changed validation scales")
    gate = reloaded.diagnostics.get("module_adoption_gate")
    if stage == "prune" and not isinstance(gate, dict):
        raise RuntimeError("prune reload lacks module_adoption_gate")
    if stage == "prune" and json_safe(gate) != json_safe(
        model.diagnostics.get("module_adoption_gate")
    ):
        raise RuntimeError("reload changed module_adoption_gate")
    report = _model_report(reloaded, train, val, context)
    return {"model": reloaded, "report": report, "module_adoption_gate": gate}


def _stage_dependencies(fold: str, stage: str) -> dict[str, Path]:
    fold_root = ROOT / "results" / "e10_refit_v2" / fold
    if stage == "full":
        return {
            "backbone_proposal": fold_root / "backbone" / "proposal.json",
            "backbone_summary": fold_root / "backbone" / "summary.json",
        }
    if stage == "prune":
        full = fold_root / "full" / "model_full.npz"
        return {
            "full_model_npz": full,
            "full_model_json": full.with_suffix(".json"),
            "full_summary": fold_root / "full" / "summary.json",
        }
    return {}


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("fold", choices=FOLDS)
    parser.add_argument("stage", choices=STAGES)
    parser.add_argument("bootstrap_draws", nargs="?", type=int, default=200)
    args = parser.parse_args(argv)
    if args.bootstrap_draws <= 0:
        parser.error("bootstrap_draws must be positive")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    started = datetime.now(UTC)
    clock_start = time.perf_counter()
    git = git_provenance()
    run_dir = ROOT / "results" / "e10_refit_v2" / args.fold / args.stage
    if run_dir.exists():
        raise RuntimeError(f"refusing to overwrite existing output directory: {run_dir}")
    dependencies = _stage_dependencies(args.fold, args.stage)
    missing = [str(path) for path in dependencies.values() if not path.is_file()]
    if missing:
        raise RuntimeError(f"missing current-run stage dependencies: {missing}")
    dependency_hashes = {
        name: {"path": str(path.relative_to(ROOT)), "sha256": sha256_file(path)}
        for name, path in dependencies.items()
    }
    config = json.loads((ROOT / "configs" / "dataset.json").read_text(encoding="utf-8"))
    store = CanonicalStore.open(config["dataset_dir"])
    meta = store.dataset_metadata
    split = load_split(
        config["dataset_dir"],
        view=config["view"],
        split_name=config["split_name"],
        fold=args.fold,
    )
    train = _adapt(config, store, meta, split.train_ids)
    val = _adapt(config, store, meta, split.val_ids)
    context = _run_context(config, train, val)

    report: dict[str, Any] = {
        "stage": f"E10-refit-v2-{args.fold}-{args.stage}",
        "started_at": started.isoformat(),
        "fold": args.fold,
        "stage_name": args.stage,
        "bootstrap_draws": args.bootstrap_draws,
        "representation": config["representation"],
        "pm_mode": config["pm_mode"],
        "n_train": len(train),
        "n_val": len(val),
        "provenance": build_provenance(
            config,
            store,
            meta,
            split,
            context,
            args.fold,
            git,
            args.stage,
            dependency_hashes,
        ),
    }
    report["provenance"]["stage_dependencies"] = dependency_hashes

    if args.stage == "backbone":
        model = PrunableEnergyModel(active_blocks=(), max_internal_step=0.01)
        fit = model.fit(train, val, context)
        proposal = PrunableEnergyModel.propose_candidates(
            train,
            model.node_damping,
            model.coupling,
            max_steps=1001,
            correlation_threshold=0.20,
            seed=int(config["seed"]),
        )
        report["fit"] = dataclasses.asdict(fit)
        report["proposal"] = proposal
        model_filename = "model_base.npz"
    elif args.stage == "full":
        proposal_path = dependencies["backbone_proposal"]
        proposal = json.loads(proposal_path.read_text(encoding="utf-8"))
        model = PrunableEnergyModel(
            active_blocks=proposal["active_blocks"],
            memory_taus=proposal["memory_taus"],
            resonances=proposal["resonances"],
            node_tau=0.2,
            potential_scale=0.25,
            max_internal_step=0.01,
            enforce_identifiability=True,
        )
        fit = model.fit(train, val, context)
        report["fit"] = dataclasses.asdict(fit)
        report["proposal_source"] = str(proposal_path.relative_to(ROOT))
        report["block_parameter_norms"] = fit.details.get("block_parameter_norms", {})
        model_filename = "model_full.npz"
    else:
        full_path = dependencies["full_model_npz"]
        model = PrunableEnergyModel.load(full_path)
        expected_scales = model._training_scales(train)
        full_metadata = json.loads(
            full_path.with_suffix(".json").read_text(encoding="utf-8")
        )
        if int(full_metadata.get("format_version", 1)) < 4:
            model.validation_scales = expected_scales
            report["validation_scale_provenance"] = {
                "status": "reconstructed",
                "reason": "accepted F1 full checkpoint predates model format v4",
                "source": "exact frozen training population",
                "values": expected_scales,
            }
        else:
            if model.validation_scales != expected_scales:
                raise RuntimeError(
                    "saved validation scales differ from the frozen training population"
                )
            report["validation_scale_provenance"] = {
                "status": "verified_saved_format_v4",
                "source": "full checkpoint and exact frozen training population",
                "values": expected_scales,
            }
        model = model.prune(
            train,
            val,
            context,
            tolerance=0.02,
            closed_loop_refine_cases=0,
            closed_loop_refine_evaluations=0,
            bootstrap_draws=args.bootstrap_draws,
            seed=int(config["seed"]),
            minimum_module_improvement=MODULE_ADOPTION_MIN_IMPROVEMENT,
        )
        report["module_adoption_minimum_improvement"] = MODULE_ADOPTION_MIN_IMPROVEMENT
        report["pruning_trace"] = model.pruning_trace
        report["module_adoption_gate"] = model.diagnostics.get("module_adoption_gate")
        model_filename = "model_pruned.npz"

    run_dir.parent.mkdir(parents=True, exist_ok=True)
    # ``TemporaryDirectory`` applies a private Windows ACL that survives a
    # directory rename and prevents the normal Git process from traversing the
    # published result.  A unique ordinary child directory inherits the result
    # tree ACL while retaining the same publish-after-verification semantics.
    staging_dir = run_dir.parent / f".{args.stage}.inflight-{uuid.uuid4().hex}"
    staging_dir.mkdir(exist_ok=False)
    if args.stage == "backbone":
        atomic_write_json(staging_dir / "proposal.json", report["proposal"])
    model_path = staging_dir / model_filename
    model.save(model_path)
    normalize_json_file(model_path.with_suffix(".json"))
    reload_result = _verify_reload(model_path, model, train, val, context, args.stage)
    report.update(reload_result["report"])
    model_hashes = {
        "npz": sha256_file(model_path),
        "json": sha256_file(model_path.with_suffix(".json")),
    }
    report["model_files_sha256"] = model_hashes
    report["provenance"]["model_files_sha256"] = model_hashes
    report["reload_verified"] = True
    report["finished_at"] = datetime.now(UTC).isoformat()
    report["wall_time_seconds"] = time.perf_counter() - clock_start
    require_frozen_git(git["head"])
    atomic_write_json(staging_dir / "summary.json", report)
    staging_dir.replace(run_dir)
    console_summary = {
        "stage": report["stage"],
        "wall_time_seconds": report["wall_time_seconds"],
        "fit_status": report.get("fit", {}).get("status", "completed_prune"),
        "active_blocks": report["active_blocks"],
        "selected_ridge": report["selected_ridge"],
        "physics_status": report["physics_audit"]["status"],
        "h100_aggregate": report["val_metrics_H100"]["aggregate"],
        "module_adoption_gate": report.get("module_adoption_gate"),
        "model_files_sha256": report["model_files_sha256"],
        "reload_verified": report["reload_verified"],
    }
    print(
        json.dumps(
            json_safe(console_summary), indent=2, ensure_ascii=False, allow_nan=False
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
