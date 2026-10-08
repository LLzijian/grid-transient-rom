"""Compact, read-only factual extractor for the completed/live 0099 artifacts.

This module reads JSON evidence, traces and numeric exports already written by the
0099 runner/monitor.  It never loads a model, integrates a trajectory, reads TEST,
or evaluates a checkpoint.  Missing endpoint evaluation artifacts remain explicit
``None`` values with a reason rather than being inferred from training files.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA = ROOT / "private_data" / "task2_joint_adequacy_0099"
ARMS = ("G3-S", "L10-S")
HORIZONS = ("1", "2", "5")
EVAL_FIELDS = (
    "validation_complete",
    "validation_severe_prefix",
    "validation_common_rmse_rad_s",
    "validation_numerical_pass",
    "l1_counts",
    "validation_numerical_scope",
    "modal_matches",
    "modal_qa",
    "local_alpha_per_s",
    "power_qa",
    "train_complete",
    "evaluation_seconds",
)


def _read(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _finite(value: Any) -> bool:
    return isinstance(value, (int, float)) and math.isfinite(float(value))


def _json_number(value: Any) -> Any:
    """Keep JSON numbers JSON-safe without silently converting missing values."""
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _trace_summary(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"path": str(path), "available": False, "missing": "trace.jsonl"}

    rows: list[dict[str, Any]] = []
    parse_errors: list[int] = []
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
            if isinstance(value, dict):
                rows.append(value)
            else:
                parse_errors.append(line_no)
        except json.JSONDecodeError:
            parse_errors.append(line_no)

    losses = [row.get("loss") for row in rows]
    gradients = [row.get("gradient_norm") for row in rows]
    loss_finite = [value for value in losses if _finite(value)]
    gradient_finite = [value for value in gradients if _finite(value)]
    clip_keys = sorted({key for row in rows for key in row if "clip" in key.lower()})
    clip_values = [row[key] for row in rows for key in clip_keys if key in row]
    clip_count = sum(bool(value) for value in clip_values)
    preclip_gt_100 = sum(float(value) > 100.0 for value in gradient_finite)
    preclip_eq_100 = sum(float(value) == 100.0 for value in gradient_finite)

    result: dict[str, Any] = {
        "path": str(path),
        "available": True,
        "rows": len(rows),
        "parse_error_lines": parse_errors,
        "first_step": rows[0].get("step") if rows else None,
        "last_step": rows[-1].get("step") if rows else None,
        "first_elapsed_seconds": rows[0].get("elapsed_seconds") if rows else None,
        "last_elapsed_seconds": rows[-1].get("elapsed_seconds") if rows else None,
        "finite_loss_rows": len(loss_finite),
        "nonfinite_loss_rows": len(rows) - len(loss_finite),
        "finite_gradient_norm_rows": len(gradient_finite),
        "nonfinite_gradient_norm_rows": len(rows) - len(gradient_finite),
        "loss_finite_all": bool(rows) and len(loss_finite) == len(rows),
        "gradient_norm_finite_all": bool(rows) and len(gradient_finite) == len(rows),
        "gradient_norm_min": min(gradient_finite) if gradient_finite else None,
        "gradient_norm_max": max(gradient_finite) if gradient_finite else None,
        "gradient_norm_mean": (sum(gradient_finite) / len(gradient_finite)) if gradient_finite else None,
        "gradient_norm_recorded": bool(gradient_finite),
        "preclip_norm_source": "trace.gradient_norm; runner records the value returned by torch.nn.utils.clip_grad_norm_(params, 100.)",
        "preclip_norm_threshold": 100.0,
        "preclip_norm_gt_100_count": preclip_gt_100,
        "preclip_norm_gt_100_fraction": (preclip_gt_100 / len(gradient_finite)) if gradient_finite else None,
        "preclip_norm_equal_100_count": preclip_eq_100,
        "internal_clip_event_count": None,
        "internal_clip_event_precision_note": "Strictly greater than 100 is reported from the recorded pre-clip norm; exact internal clip events at the floating threshold are not claimed.",
        "clip_field_names": clip_keys,
        "clip_frequency": (clip_count / len(rows)) if clip_keys and rows else None,
        "clip_count": clip_count if clip_keys else None,
        "clip_frequency_available": bool(clip_keys),
    }
    return result


def _term_means(batch_terms: Any) -> dict[str, Any] | None:
    """Average recorded fixed-batch terms; no new numerical calculation is run."""
    if not isinstance(batch_terms, dict):
        return None
    result: dict[str, Any] = {}
    for grid in ("coarse", "fine", "finer"):
        payload = batch_terms.get(grid)
        if not isinstance(payload, dict):
            continue
        names = payload.get("term_names")
        values = payload.get("terms")
        if not isinstance(names, list) or not isinstance(values, list):
            continue
        by_horizon: dict[str, Any] = {}
        for horizon_index, horizon in enumerate(HORIZONS):
            columns = [[] for _ in names]
            for batch in values:
                if not isinstance(batch, list) or horizon_index >= len(batch):
                    continue
                record = batch[horizon_index]
                if not isinstance(record, list):
                    continue
                for index in range(min(len(names), len(record))):
                    if _finite(record[index]):
                        columns[index].append(float(record[index]))
            by_horizon[horizon] = {
                name: {
                    "mean": (sum(column) / len(column)) if column else None,
                    "finite_batches": len(column),
                }
                for name, column in zip(names, columns)
            }
        result[grid] = {"term_names": names, "horizons": by_horizon}
    return result or None


def _top5(rows: Any, qa: Any) -> dict[str, list[dict[str, Any]]]:
    if not isinstance(rows, list):
        return {horizon: [] for horizon in HORIZONS}
    numerical = {}
    if isinstance(qa, dict):
        for row in qa.get("autonomous", []):
            if isinstance(row, dict):
                numerical[(row.get("event_id"), str(row.get("horizon")))] = row.get("pass")
    result: dict[str, list[dict[str, Any]]] = {}
    for horizon in HORIZONS:
        selected = [
            row for row in rows
            if isinstance(row, dict)
            and str(row.get("horizon")) == horizon
            and _finite(row.get("point_error"))
        ]
        selected.sort(key=lambda row: float(row["point_error"]), reverse=True)
        compact = []
        for row in selected[:5]:
            compact.append({
                "event_id": row.get("event_id"),
                "group": row.get("group"),
                "label": row.get("label"),
                "horizon": row.get("horizon"),
                "steps": row.get("steps"),
                "complete_horizon": row.get("complete_horizon"),
                "point_error": row.get("point_error"),
                "all_substeps_finite": row.get("all_substeps_finite"),
                "numerically_valid": numerical.get((row.get("event_id"), horizon)),
            })
        result[horizon] = compact
    return result


def _numeric_counts(evidence: dict[str, Any]) -> dict[str, Any]:
    progress = evidence.get("progress") if isinstance(evidence.get("progress"), dict) else {}
    final_qa = evidence.get("final_grid_qa") if isinstance(evidence.get("final_grid_qa"), dict) else {}
    counts: dict[str, Any] = {}
    autonomous = final_qa.get("autonomous") if isinstance(final_qa.get("autonomous"), list) else []
    local = final_qa.get("local") if isinstance(final_qa.get("local"), list) else []
    for name, rows in (("autonomous", autonomous), ("local", local)):
        counts[name] = {
            "rows": len(rows),
            "pass": sum(bool(row.get("pass")) for row in rows if isinstance(row, dict)),
            "finite": sum(bool(row.get("all_substeps_finite")) for row in rows if isinstance(row, dict)),
        }
    counts["event_rows_fine"] = len(evidence.get("event_rows_fine", [])) if isinstance(evidence.get("event_rows_fine"), list) else 0
    counts["progress"] = progress
    counts["initial_grid_pass"] = (evidence.get("initial_grid_qa") or {}).get("pass") if isinstance(evidence.get("initial_grid_qa"), dict) else None
    counts["final_grid_pass"] = final_qa.get("pass")
    return counts


def _monitor_summary(path: Path, *, step: int | None, label: str) -> dict[str, Any]:
    if not path.exists():
        return {"path": str(path), "available": False, "step": step, "step_label": label, "missing": "evidence.json"}
    evidence = _read(path)
    if not isinstance(evidence, dict):
        return {"path": str(path), "available": False, "step": step, "step_label": label, "invalid": "top-level JSON is not an object"}
    raw_s = evidence.get("raw_S_fine")
    valid_s = evidence.get("S_fine")
    horizons = {}
    for horizon in HORIZONS:
        horizons[horizon] = {
            "raw": raw_s.get(horizon) if isinstance(raw_s, dict) else None,
            "valid": valid_s.get(horizon) if isinstance(valid_s, dict) else None,
        }
    return {
        "path": str(path),
        "available": True,
        "step": step,
        "step_label": label,
        "status": evidence.get("status"),
        "complete": evidence.get("complete"),
        "refined": evidence.get("refined"),
        "elapsed_seconds": evidence.get("seconds"),
        "J": {
            "raw": evidence.get("raw_J_fine"),
            "valid": evidence.get("J_fine"),
            "raw_coarse": evidence.get("raw_J_coarse"),
            "valid_coarse": evidence.get("J_coarse"),
            "raw_finer": evidence.get("raw_J_finer"),
            "numerical_pass": evidence.get("numerical_pass"),
        },
        "numerical_sensitivity_d": evidence.get("numerical_sensitivity_d"),
        "S": horizons,
        "S_grid_differences": evidence.get("S_grid_differences"),
        "numeric_counts": _numeric_counts(evidence),
        "term_means": _term_means(evidence.get("batch_terms")),
        "top5_point_error_by_horizon": _top5(evidence.get("event_rows_fine"), evidence.get("final_grid_qa")),
        "final_grid_qa_pass": (evidence.get("final_grid_qa") or {}).get("pass") if isinstance(evidence.get("final_grid_qa"), dict) else None,
        "manifest_sha256": evidence.get("manifest_sha256"),
        "model_state_sha256_before": evidence.get("model_state_sha256_before"),
        "model_state_sha256_after": evidence.get("model_state_sha256_after"),
        "sources_unchanged": evidence.get("sources_unchanged"),
        "rng_unchanged": evidence.get("rng_unchanged"),
        "test_read": evidence.get("test_read"),
        "validation_read": evidence.get("validation_read"),
    }


def _checkpoint_rows(arm_dir: Path) -> list[dict[str, Any]]:
    rows = []
    for path in sorted(list(arm_dir.glob("step*/checkpoint.json")) + list(arm_dir.glob("last_step*/checkpoint.json"))):
        try:
            value = _read(path)
        except (OSError, json.JSONDecodeError):
            continue
        if isinstance(value, dict):
            rows.append({"path": str(path.parent), "metadata": value})
    rows.sort(key=lambda row: (row["metadata"].get("step", -1), row["path"]))
    return rows


def _export_summary(checkpoint_path: Path | None) -> dict[str, Any]:
    if checkpoint_path is None:
        return {"available": False, "missing": "checkpoint reference absent; export was not selected by inference"}
    path = checkpoint_path / "structured_numeric_export.json"
    if not path.exists():
        return {"available": False, "path": str(path), "missing": "structured_numeric_export.json"}
    value = _read(path)
    if not isinstance(value, dict):
        return {"available": False, "path": str(path), "invalid": "top-level JSON is not an object"}
    return {
        "available": True,
        "path": str(path),
        "format": value.get("format"),
        "template_names": value.get("template_names"),
        "template_mask": value.get("template_mask"),
        "template_coefficients": value.get("template_coefficients"),
        "q_quadratic": value.get("q_quadratic"),
        "counts": value.get("counts"),
    }


def _find_eval_files(wrapper_dir: Path) -> dict[str, Path | None]:
    """Locate the exact validation/train filenames emitted by evaluate_joint_adequacy."""
    candidates = {
        "validation": (wrapper_dir / "validation" / "trajectories" / "evidence.json",),
        "modal": (wrapper_dir / "validation" / "modal" / "evidence.json",),
        "power": (wrapper_dir / "validation" / "evidence.json",),
        "train_evaluation": (wrapper_dir / "train" / "evidence.json",),
    }
    return {role: next((path for path in paths if path.exists()), None) for role, paths in candidates.items()}


def _evaluation_fields(paths: dict[str, Path | None]) -> dict[str, Any]:
    loaded = {}
    for role, path in paths.items():
        if path is not None:
            try:
                loaded[role] = _read(path)
            except (OSError, json.JSONDecodeError):
                loaded[role] = None

    result: dict[str, Any] = {
        "paths": {role: (str(path) if path is not None else None) for role, path in paths.items()},
        "fields": {field: None for field in EVAL_FIELDS},
        "missing_fields": [],
        "available": bool(loaded),
    }
    validation = loaded.get("validation") if isinstance(loaded.get("validation"), dict) else None
    modal = loaded.get("modal") if isinstance(loaded.get("modal"), dict) else None
    power = loaded.get("power") if isinstance(loaded.get("power"), dict) else None
    train = loaded.get("train_evaluation") if isinstance(loaded.get("train_evaluation"), dict) else None
    if validation is not None:
        # Same interval-by-interval rule as the frozen0098 capability assembly.
        numerical = {(r['event_id'], r['horizon']): r
                     for key in ('numerical_scored_initial', 'numerical_scored_finer')
                     for r in validation.get(key, [])}
        for row in numerical.values():
            row['pass'] = bool(row.get('available') and row.get('finite_to_endpoint')
                               and row.get('los_pair_pass')
                               and row['coi_normalized_rmse'] <= .001
                               and row['common_normalized_rmse'] <= .001)
        scope = {h: {'eligible': sum(r.get('available', False) for (e, rh), r in numerical.items() if rh == h),
                     'passing': sum(r['pass'] for (e, rh), r in numerical.items() if rh == h),
                     'unresolved_event_ids': [e for (e, rh), r in numerical.items()
                                              if rh == h and r.get('available') and not r['pass']]}
                 for h in ('1', '2', '5', '8', 'prefix')}
        common = {}
        for h in ('1', '2', '5', '8'):
            rows = [r for r in validation.get('common_frequency', [])
                    if str(r['horizon']) == h and r.get('n', 0) > 0]
            common[h] = (math.sqrt(sum(r['sse'] for r in rows) / sum(r['n'] for r in rows))
                         if rows and all(r.get('sse') is not None for r in rows) else None)
        result["fields"].update({
            "validation_complete": validation.get("summary"),
            "validation_severe_prefix": validation.get("unstable_available_prefix"),
            "validation_common_rmse_rad_s": common,
            "validation_numerical_pass": validation.get("model_numerical_gate_pass"),
            "l1_counts": validation.get("l1_numerically_gated_counts"),
            "validation_numerical_scope": scope,
        })
    if modal is not None:
        result["fields"].update({"modal_matches": modal.get("matched_count"), "modal_qa": modal.get("qa"), "local_alpha_per_s": modal.get("spectral_alpha")})
    if power is not None:
        result["fields"]["power_qa"] = power.get("energy_power_qa")
    if train is not None:
        result["fields"]["train_complete"] = train.get("summary")
    # Nested durations overlap. Only the outer wrapper owns the charged duration.
    result["missing_fields"] = [field for field, value in result["fields"].items() if value is None]
    if not loaded:
        result["missing_reason"] = "wrapper contains no validation/modal/power/train evaluation evidence"
    return result


def _same_ref(left: Any, right: Any) -> bool:
    if not isinstance(left, str) or not isinstance(right, str):
        return False
    a = left.replace("\\", "/").rstrip("/").lower()
    b = right.replace("\\", "/").rstrip("/").lower()
    return a == b or a.endswith("/" + b) or b.endswith("/" + a)


def _evaluation_summary(data_root: Path, arm_dir: Path, arm: str, formal: dict[str, Any] | None) -> dict[str, Any]:
    """Read root eval_* wrappers and retain selected/last checkpoint identity separately."""
    wrapper_dirs = []
    seen = set()
    for path in sorted(tuple(data_root.glob("eval_*")) + tuple(arm_dir.glob("eval_*"))):
        if path.is_dir() and str(path) not in seen:
            seen.add(str(path))
            wrapper_dirs.append(path)
    selected_ref = formal.get("selected_checkpoint") if isinstance(formal, dict) else None
    last_ref = formal.get("last_checkpoint") if isinstance(formal, dict) else None
    evaluations: dict[str, Any] = {}
    ignored: list[str] = []
    for wrapper_dir in wrapper_dirs:
        wrapper_evidence_path = wrapper_dir / "evidence.json"
        wrapper_evidence = _read(wrapper_evidence_path) if wrapper_evidence_path.exists() else {}
        checkpoint_ref = wrapper_evidence.get("checkpoint") if isinstance(wrapper_evidence, dict) else None
        name_matches = arm.lower() in wrapper_dir.name.lower()
        ref_matches = isinstance(checkpoint_ref, str) and arm.lower() in checkpoint_ref.lower()
        if not name_matches and not ref_matches:
            ignored.append(str(wrapper_dir))
            continue
        if _same_ref(checkpoint_ref, selected_ref) or "selected" in wrapper_dir.name.lower():
            role = "selected"
        elif _same_ref(checkpoint_ref, last_ref) or "last" in wrapper_dir.name.lower():
            role = "last"
        else:
            role = "unknown"
        if role in evaluations:
            role = f"{role}_{len([key for key in evaluations if key.startswith(role)]) + 1}"
        paths = _find_eval_files(wrapper_dir)
        fields = _evaluation_fields(paths)
        fields["fields"]["evaluation_seconds"] = wrapper_evidence.get("seconds")
        fields["missing_fields"] = [k for k, v in fields["fields"].items() if v is None]
        evaluations[role] = {
            "wrapper_path": str(wrapper_dir),
            "wrapper_evidence_path": str(wrapper_evidence_path) if wrapper_evidence_path.exists() else None,
            "checkpoint": checkpoint_ref,
            "status": wrapper_evidence.get("status") if isinstance(wrapper_evidence, dict) else None,
            "seconds": wrapper_evidence.get("seconds") if isinstance(wrapper_evidence, dict) else None,
            "within_cap": wrapper_evidence.get("within_cap") if isinstance(wrapper_evidence, dict) else None,
            "sources_unchanged": wrapper_evidence.get("sources_unchanged") if isinstance(wrapper_evidence, dict) else None,
            "test_read": wrapper_evidence.get("test_read") if isinstance(wrapper_evidence, dict) else None,
            "validation_read": wrapper_evidence.get("validation_read") if isinstance(wrapper_evidence, dict) else None,
            "jobs": wrapper_evidence.get("jobs") if isinstance(wrapper_evidence, dict) else None,
            "evaluation": fields,
        }
    return {
        "schema": "0098 assemble_capability arms schema",
        "available": bool(evaluations),
        "evaluations": evaluations,
        "ignored_eval_dirs": ignored,
        "missing_reason": None if evaluations else "0099 contains no matching eval_* wrapper evidence",
    }


def _frozen_0098_schema() -> dict[str, Any]:
    path = ROOT / "private_data" / "task2_local_memory_0098" / "capability.json"
    if not path.exists():
        return {"available": False, "path": str(path), "missing": "capability.json"}
    value = _read(path)
    return {
        "available": True,
        "path": str(path),
        "top_level_keys": sorted(value) if isinstance(value, dict) else None,
        "arm_names": sorted(value.get("arms", {})) if isinstance(value, dict) and isinstance(value.get("arms"), dict) else [],
        "arm_field_keys": {
            arm: sorted(row) for arm, row in (value.get("arms", {}) if isinstance(value, dict) else {}).items() if isinstance(row, dict)
        },
        "evaluation_schema_source": str(ROOT / "private_data" / "task2_local_memory_0098" / "assemble_capability.py"),
    }


def _arm_summary(data_root: Path, arm: str) -> dict[str, Any]:
    arm_dir = data_root / f"formal_{arm}"
    trace_path = arm_dir / "trace.jsonl"
    checkpoints = _checkpoint_rows(arm_dir)
    formal_path = arm_dir / "evidence.json"
    formal = _read(formal_path) if formal_path.exists() else None
    trace = _trace_summary(trace_path)
    ancestor_trace = ROOT / "private_data" / "task2_local_memory_0098" / f"formal_{arm}" / "trace.jsonl"
    counts = {h: Counter() for h in HORIZONS}
    phase_counts = {h: Counter() for h in HORIZONS}
    for source, phase in ((ancestor_trace, False), (trace_path, True)):
        if source.exists():
            for line in source.read_text(encoding="utf-8").splitlines():
                try:
                    row = json.loads(line)
                    h = str(int(row["horizon"]))
                    counts[h].update(row["sample_ids"])
                    if phase:
                        phase_counts[h].update(row["sample_ids"])
                except (ValueError, KeyError):
                    continue  # active partial final line is covered by trace parse errors
    monitor_manifest = data_root / f"qa_initial_{arm}" / "manifest.json"
    pool = _read(monitor_manifest).get("event_ids", []) if monitor_manifest.exists() else []
    trace["sampling_coverage_by_horizon"] = {
        h: {"ancestor_plus_0099_unique_events": len(counts[h]),
            "0099_unique_events": len(phase_counts[h]),
            "ancestor_plus_0099_unsampled_ids": [e for e in pool if not counts[h][e]],
            "0099_unsampled_ids": [e for e in pool if not phase_counts[h][e]],
            "ancestor_plus_0099_draws_per_event": dict(counts[h]),
            "0099_draws_per_event": dict(phase_counts[h])}
        for h in HORIZONS}
    monitor_records = {}
    for path in sorted(arm_dir.glob("monitor_step*/evidence.json")):
        match = re.fullmatch(r"monitor_step(\d+)", path.parent.name)
        if match is None:
            continue
        step = int(match.group(1))
        monitor_records[path.parent.name] = _monitor_summary(path, step=step, label=path.parent.name)
    initial_monitor = _monitor_summary(data_root / f"qa_initial_{arm}" / "evidence.json", step=0, label="qa_initial")
    restore_path = data_root / f"qa_restore_{arm}" / "evidence.json"
    restore = None
    if restore_path.exists():
        value = _read(restore_path)
        restore = {
            "path": str(restore_path),
            "status": value.get("status") if isinstance(value, dict) else None,
            "elapsed_seconds": value.get("seconds") if isinstance(value, dict) else None,
            "monitor": _monitor_summary_from_nested(value.get("monitor"), restore_path.parent, label="qa_restore") if isinstance(value, dict) else None,
        }

    trace_last_step = trace.get("last_step")
    selected_ref = formal.get("selected_checkpoint") if isinstance(formal, dict) else None
    last_ref = formal.get("last_checkpoint") if isinstance(formal, dict) else None

    def resolve_checkpoint(ref: Any) -> Path | None:
        if not isinstance(ref, str) or not ref:
            return None
        candidates = [Path(ref)]
        if not Path(ref).is_absolute():
            candidates.extend((ROOT / ref, data_root / ref, arm_dir / ref))
        for candidate in candidates:
            if (candidate / "checkpoint.json").exists():
                return candidate.resolve()
        return Path(ref)

    selected_path = resolve_checkpoint(selected_ref)
    last_path = resolve_checkpoint(last_ref)

    def metadata_for(path: Path | None) -> dict[str, Any] | None:
        if path is None:
            return None
        for row in checkpoints:
            if _same_ref(row["path"], str(path)):
                return row["metadata"]
        return None

    selected_metadata = metadata_for(selected_path)
    last_metadata = metadata_for(last_path)
    checkpoint_last_step = last_metadata.get("step") if isinstance(last_metadata, dict) else None
    continuation = None
    if _finite(trace_last_step) and _finite(checkpoint_last_step):
        continuation = {
            "trace_last_step": trace_last_step,
            "latest_checkpoint_step": checkpoint_last_step,
            "trace_beyond_latest_checkpoint": int(trace_last_step) > int(checkpoint_last_step),
            "interpretation": "checkpoint absence after the trace is retained as run-state information; it is not classified as failure",
        }

    return {
        "arm": arm,
        "paths": {
            "formal_dir": str(arm_dir),
            "trace": str(trace_path),
            "formal_evidence": str(formal_path) if formal_path.exists() else None,
            "monitor_steps": {key: value["path"] for key, value in monitor_records.items()},
        },
        "trace": trace,
        "monitors": {"qa_initial": initial_monitor, "qa_restore": restore, **monitor_records},
        "checkpoints": [{"path": row["path"], **row["metadata"]} for row in checkpoints],
        "formal_run_evidence": {
            "available": isinstance(formal, dict),
            "path": str(formal_path) if formal_path.exists() else None,
            "status": formal.get("status") if isinstance(formal, dict) else None,
            "stop_reason": formal.get("stop_reason") if isinstance(formal, dict) else None,
            "seconds": formal.get("seconds") if isinstance(formal, dict) else None,
            "within_cap": formal.get("within_cap") if isinstance(formal, dict) else None,
            "sources_unchanged": formal.get("sources_unchanged") if isinstance(formal, dict) else None,
        },
        "selected_checkpoint": selected_ref,
        "last_checkpoint": last_ref,
        "last_checkpoint_step": checkpoint_last_step,
        "available_checkpoint_steps": [row["metadata"].get("step") for row in checkpoints],
        "run_continuation": continuation,
        "status": formal.get("status") if isinstance(formal, dict) else None,
        "status_reason": None if isinstance(formal, dict) and formal.get("status") is not None else "formal status field is absent",
        "checkpoint_exports": {
            "selected": _export_summary(selected_path),
            "last": _export_summary(last_path),
        },
        "checkpoint_metadata": {"selected": selected_metadata, "last": last_metadata},
        "final_evaluation": _evaluation_summary(data_root, arm_dir, arm, formal),
    }


def _monitor_summary_from_nested(value: Any, parent: Path, *, label: str) -> dict[str, Any] | None:
    if not isinstance(value, dict):
        return None
    # Restore evidence embeds the same monitor schema rather than a separate file.
    temp = parent / "evidence.json"
    result = _monitor_summary_from_dict(value, temp, label=label)
    return result


def _monitor_summary_from_dict(evidence: dict[str, Any], path: Path, *, label: str) -> dict[str, Any]:
    raw_s = evidence.get("raw_S_fine")
    valid_s = evidence.get("S_fine")
    horizons = {h: {"raw": raw_s.get(h) if isinstance(raw_s, dict) else None,
                    "valid": valid_s.get(h) if isinstance(valid_s, dict) else None} for h in HORIZONS}
    return {
        "path": str(path) + " (embedded)", "available": True, "step": None, "step_label": label,
        "status": evidence.get("status"), "complete": evidence.get("complete"), "refined": evidence.get("refined"),
        "elapsed_seconds": evidence.get("seconds"),
        "J": {"raw": evidence.get("raw_J_fine"), "valid": evidence.get("J_fine"),
               "raw_coarse": evidence.get("raw_J_coarse"), "valid_coarse": evidence.get("J_coarse"),
               "raw_finer": evidence.get("raw_J_finer"), "numerical_pass": evidence.get("numerical_pass")},
        "numerical_sensitivity_d": evidence.get("numerical_sensitivity_d"), "S": horizons,
        "S_grid_differences": evidence.get("S_grid_differences"), "numeric_counts": _numeric_counts(evidence),
        "term_means": _term_means(evidence.get("batch_terms")),
        "top5_point_error_by_horizon": _top5(evidence.get("event_rows_fine"), evidence.get("final_grid_qa")),
        "final_grid_qa_pass": (evidence.get("final_grid_qa") or {}).get("pass") if isinstance(evidence.get("final_grid_qa"), dict) else None,
        "manifest_sha256": evidence.get("manifest_sha256"), "model_state_sha256_before": evidence.get("model_state_sha256_before"),
        "model_state_sha256_after": evidence.get("model_state_sha256_after"), "sources_unchanged": evidence.get("sources_unchanged"),
        "rng_unchanged": evidence.get("rng_unchanged"), "test_read": evidence.get("test_read"), "validation_read": evidence.get("validation_read"),
    }


def build_report(data_root: Path = DEFAULT_DATA) -> dict[str, Any]:
    report = {
        "loop": "0099",
        "scope": "factual artifact extraction only; no training, integration, evaluation, TEST access, or checkpoint selection",
        "test_read": False,
        "validation_read": False,
        "data_root": str(data_root),
        "arms": {arm: _arm_summary(data_root, arm) for arm in ARMS},
        "frozen_0098_schema": _frozen_0098_schema(),
    }
    paired = {}
    candidate_path = data_root / 'eval_L10-S_last/validation/trajectories/evidence.json'
    if candidate_path.exists():
        candidate = _read(candidate_path)
        for role in ('selected', 'last'):
            reference_path = data_root / f'eval_G3-S_{role}/validation/trajectories/evidence.json'
            if not reference_path.exists():
                continue
            reference = _read(reference_path)
            rows = []
            scopes = []
            for evidence in (reference, candidate):
                numeric = {(r['event_id'], r['horizon']): r
                           for key in ('numerical_scored_initial', 'numerical_scored_finer')
                           for r in evidence.get(key, [])}
                scopes.append({k: bool(r.get('available') and r.get('finite_to_endpoint')
                                       and r.get('los_pair_pass')
                                       and r['coi_normalized_rmse'] <= .001
                                       and r['common_normalized_rmse'] <= .001)
                               for k, r in numeric.items()})
            for h in ('1', '2', '5', '8', 'prefix'):
                a = {r['event_id']: r for r in reference['records'] if r['horizon'] == h and r.get('available')}
                b = {r['event_id']: r for r in candidate['records'] if r['horizon'] == h and r.get('available')}
                if a.keys() != b.keys():
                    raise ValueError('paired development event identity changed')
                for group in sorted({r['group'] for r in a.values()}):
                    ids = [e for e in a if a[e]['group'] == group]
                    finite = all(a[e].get('finite') and b[e].get('finite') for e in ids)
                    rows.append({'horizon': h, 'group': group, 'event_count': len(ids),
                                 'numerically_admissible_group': all(scopes[0].get((e, h)) and scopes[1].get((e, h)) for e in ids),
                                 'mean_event_RMSE_difference_L10_minus_G3':
                                     sum(b[e]['rmse'] - a[e]['rmse'] for e in ids)/len(ids) if finite else None})
            paired[f'L10last_vs_G3{role}'] = rows
    report['paired_validation_groups'] = paired
    report['paired_scope'] = 'Descriptive paired groups; last-vs-last matches600 new updates, selectedG3 has400. No independent generalization or isolated locality claim.'
    input_paths = [
        path for path in data_root.rglob("*.json")
        if path.name != "capability_0099.json"
    ] + list(data_root.glob("formal_*/trace.jsonl"))
    report["input_artifacts"] = sorted({str(path): _sha256(path) for path in input_paths}.items())
    return report


def write_report(data_root: Path = DEFAULT_DATA, output: Path | None = None, *, overwrite: bool = False) -> dict[str, Any]:
    output = output or data_root / "capability_0099.json"
    if output.exists() and not overwrite:
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    report = build_report(data_root)
    report["report_code"] = {"path": str(Path(__file__)), "sha256": _sha256(Path(__file__))}
    output.write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--overwrite", action="store_true", help="overwrite only the derived capability JSON")
    args = parser.parse_args()
    result = write_report(args.data_root, args.output, overwrite=args.overwrite)
    print(json.dumps({"output": str(args.output or args.data_root / "capability_0099.json"),
                      "arms": list(result["arms"]),
                      "trace_last_steps": {arm: result["arms"][arm]["trace"].get("last_step") for arm in ARMS}},
                     sort_keys=True))


if __name__ == "__main__":
    main()
