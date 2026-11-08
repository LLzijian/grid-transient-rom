"""E4: energy-structure and candidate-margin audit on the frozen F1 model.

Loads the pruned E2 model, runs the numeric certificate suite, computes the
integration-free candidate margin for every validation event. The candidate
search is not a verified global lower bound, so formal stability is refused;
the heuristic screen is reported separately with group-level uncertainty.

Stability labels here come from metadata outcome_label for validation of the
margin rule; the rule itself reads only delta0/omega0/Pm_clear (the exact
clearing state) -- no future trajectory. A trajectory-threshold classifier on
a uniform fixed H100 causal rollout is reported only as a method comparison.
"""

from __future__ import annotations

import json
import sys
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from kiki.data import CanonicalStore, ModelDataRequest, adapt_event, load_split
from kiki.evaluation.energy_structure import evaluate_energy_structure
from kiki.evaluation.stability import grouped_missed_instability_audit
from kiki.models.base import sample_rollout_inputs
from kiki.models.prunable_energy_model import PrunableEnergyModel

ANGLE_LIMIT = np.pi / 2
OMEGA_S = 376.99111843077515


def main() -> int:
    config = json.loads((ROOT / "configs" / "dataset.json").read_text(encoding="utf-8"))
    store = CanonicalStore.open(config["dataset_dir"])
    meta = store.dataset_metadata
    split = load_split(config["dataset_dir"], view=config["view"],
                       split_name=config["split_name"], fold="F1")
    request = ModelDataRequest.initial_state_rollout("zoh")
    val = []
    for eid in split.val_ids:
        sample = adapt_event(store.load_event(eid), inertia_prior=meta.M_prior,
                             request=request, representation=config["representation"])
        sample["outcome_label"] = sample["training_context"]["label"]
        sample["fault_line"] = sample["training_context"]["metadata"]["fault_line"]
        val.append(sample)

    model = PrunableEnergyModel.load(ROOT / "results" / "e02_full" / "model_full.npz")

    report = {
        "stage": "E4",
        "timestamp": datetime.now(UTC).isoformat(),
        "fold": "F1",
        "model": "model_full.npz (pruned F1 modular)",
        "n_val": len(val),
        "angle_limit": ANGLE_LIMIT,
    }

    structure_audit = evaluate_energy_structure(model, val)
    report["energy_structure_audit"] = structure_audit.to_dict()

    # Integration-free candidate margins on every validation event.
    margins = []
    n_certified = 0
    for sample in val:
        static = sample["model_inputs"]["static"]
        try:
            m = model.stability_margin(static, angle_limit=ANGLE_LIMIT)
        except (ValueError, RuntimeError) as error:
            margins.append({
                "event_id": sample["event_id"],
                "true_stable": sample["outcome_label"] == "stable",
                "source": sample["training_context"]["metadata"]["source_dataset"],
                "fault_line": sample["fault_line"],
                "certified": False,
                "predicted_stable": False,
                "candidate_search_completed": False,
                "screening_predicted_stable": False,
                "error": str(error),
            })
            continue
        m = dict(m)
        m["event_id"] = sample["event_id"]
        m["true_stable"] = sample["outcome_label"] == "stable"
        m["source"] = sample["training_context"]["metadata"]["source_dataset"]
        m["fault_line"] = sample["fault_line"]
        margins.append(m)
        if m.get("certified"):
            n_certified += 1
    report["n_margin_certified"] = n_certified

    # No verified global escape-energy lower bound exists, so formal decisions
    # are refused. Preserve the local-search result as a heuristic screen only.
    for m in margins:
        m["verdict"] = "refused"
        m["screening_verdict"] = (
            "stable"
            if m.get("candidate_search_completed") and m.get("screening_predicted_stable")
            else "unstable"
            if m.get("candidate_search_completed")
            else "refused"
        )
    stable = [m for m in margins if m["true_stable"]]
    unstable = [m for m in margins if not m["true_stable"]]
    stable_safe = sum(1 for m in stable if m["verdict"] == "stable")
    stable_rejected = len(stable) - stable_safe
    unstable_unsafe = sum(1 for m in unstable if m["verdict"] == "stable")
    unstable_refused = sum(1 for m in unstable if m["verdict"] == "refused")
    unstable_called = sum(1 for m in unstable if m["verdict"] == "unstable")
    report["confusion"] = {
        "n_val": len(margins),
        "n_stable": len(stable),
        "n_unstable": len(unstable),
        "stable_safe": stable_safe,
        "stable_rejected_or_refused": stable_rejected,
        "unstable_unsafe_called_stable": unstable_unsafe,
        "unstable_refused": unstable_refused,
        "unstable_called_unstable": unstable_called,
    }
    report["classification"] = {
        "missed_instability_rate": (
            unstable_unsafe / len(unstable) if unstable else float("nan")
        ),
        "stable_safe_coverage": stable_safe / len(stable) if stable else float("nan"),
        "formal_rule": "no verified global lower bound => refused",
    }

    report["screening_classification"] = {
        "missed_instability_rate": sum(
            1 for m in unstable if m["screening_verdict"] == "stable"
        ) / len(unstable),
        "stable_screen_coverage": sum(
            1 for m in stable if m["screening_verdict"] == "stable"
        ) / len(stable),
        "status": "retrospective_heuristic_not_certificate",
        "uncertainty": grouped_missed_instability_audit(
            margins,
            predicted_stable_key="screening_predicted_stable",
            eligible_key="candidate_search_completed",
        ),
    }

    # Fault-line cluster bootstrap for heuristic stable-screen coverage only.
    lines = sorted({m["fault_line"] for m in margins})
    by_line = {line: [m for m in margins if m["fault_line"] == line] for line in lines}
    rng = np.random.default_rng(324)
    mir_draws, cov_draws = [], []
    for _ in range(200):
        sample_lines = rng.choice(lines, size=len(lines), replace=True)
        pick = [m for line in sample_lines for m in by_line[line]]
        st = [m for m in pick if m["true_stable"]]
        us = [m for m in pick if not m["true_stable"]]
        if st:
            cov_draws.append(
                sum(1 for m in st if m["screening_verdict"] == "stable") / len(st)
            )
        if us:
            mir_draws.append(
                sum(1 for m in us if m["screening_verdict"] == "stable") / len(us)
            )
    report["bootstrap"] = {
        "screening_missed_instability_bootstrap_95_upper": (
            float(np.percentile(mir_draws, 95)) if mir_draws else float("nan")
        ),
        "screening_missed_instability_median": (
            float(np.median(mir_draws)) if mir_draws else float("nan")
        ),
        "stable_screen_coverage_bootstrap_95_lower": (
            float(np.percentile(cov_draws, 5)) if cov_draws else float("nan")
        ),
        "stable_screen_coverage_bootstrap_median": (
            float(np.median(cov_draws)) if cov_draws else float("nan")
        ),
        "n_fault_lines": len(lines),
        "clusters": "fault_line",
    }

    # Comparison: trajectory-threshold classifier on a uniform H100 causal rollout
    thr_rows = []
    thr_failures = []
    for sample in val:
        initial, t, exo = sample_rollout_inputs(sample)
        h = min(101, len(t))
        try:
            pd, pw = model.rollout(initial, t[:h],
                                   {"Pm": exo["Pm"][:h], "interpolation": exo["interpolation"]})
        except Exception as error:  # noqa: BLE001 - preserve every event-level failure
            thr_failures.append(
                {"event_id": sample["event_id"], "error": f"{type(error).__name__}: {error}"}
            )
            continue
        span = float(np.max(np.ptp(pd, axis=1)))
        speed = float(np.max(np.abs(pw)))
        thr_rows.append({"event_id": sample["event_id"],
                         "true_stable": sample["outcome_label"] == "stable",
                         "pred_stable": (span < np.pi) and (speed < 0.05 * OMEGA_S)})
    n_us = sum(1 for r in thr_rows if not r["true_stable"])
    fsr_thr = sum(1 for r in thr_rows if r["pred_stable"] and not r["true_stable"]) / max(1, n_us)
    report["trajectory_threshold_comparison"] = {
        "fsr": fsr_thr, "n_unstable": n_us,
        "n_rollout_failures": len(thr_failures),
        "rollout_failures": thr_failures,
        "note": "threshold on range>=pi or |w|>=0.05pu over uniform H=100 causal rollout",
    }

    run_dir = ROOT / "results" / "e04"
    run_dir.mkdir(parents=True, exist_ok=True)
    out = run_dir / "summary.json"
    if out.exists():
        raise SystemExit(f"refusing to overwrite {out}")
    out.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    print(json.dumps(report, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
