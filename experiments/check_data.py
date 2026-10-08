"""E0: data-path validation for the kiki rebuild.

Checks, against configs/dataset.json:
  1. Store opens, event count and stable/unstable counts match.
  2. All 5 folds load; train/val/test are event-disjoint and line-disjoint.
  3. Every F1 event adapts under the causal contract (generator_10, ZOH):
     COI centering, COI power balance, delta[0]==delta0, Pm ZOH == Pm_clear.
  4. delta/omega targets are finite; t starts at 0.

Writes results/e00_data_check/summary.json. Never touches test trajectories
beyond metadata (fault_line for the disjointness audit comes from the index).
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from kiki.data import CanonicalStore, ModelDataRequest, adapt_event, load_split


def main() -> int:
    config = json.loads((ROOT / "configs" / "dataset.json").read_text(encoding="utf-8"))
    store = CanonicalStore.open(config["dataset_dir"])
    report: dict[str, object] = {
        "stage": "E0",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "dataset_dir": config["dataset_dir"],
        "checks": {},
    }

    # 1. event counts
    ids = store.event_ids(config["view"])
    labels = [store.metadata(eid).outcome_label for eid in ids]
    n_stable = sum(1 for x in labels if x == "stable")
    n_unstable = sum(1 for x in labels if x == "unstable")
    exp = config["expected_events"]
    report["checks"]["event_counts"] = {
        "total": len(ids),
        "stable": n_stable,
        "unstable": n_unstable,
        "pass": (len(ids), n_stable, n_unstable)
        == (exp["total"], exp["stable"], exp["unstable"]),
    }

    # 2. folds: event- and line-disjointness
    folds = {}
    all_ok = True
    for fold in ("F1", "F2", "F3", "F4", "F5"):
        split = load_split(
            config["dataset_dir"],
            view=config["view"],
            split_name=config["split_name"],
            fold=fold,
        )
        line = lambda eid: store.metadata(eid).fault_line  # noqa: E731
        train_lines = {line(e) for e in split.train_ids} | {line(e) for e in split.val_ids}
        test_lines = {line(e) for e in split.test_ids}
        line_disjoint = not (train_lines & test_lines)
        folds[fold] = {
            "n_train": len(split.train_ids),
            "n_val": len(split.val_ids),
            "n_test": len(split.test_ids),
            "line_disjoint": line_disjoint,
        }
        all_ok &= line_disjoint
    report["checks"]["folds"] = folds
    report["checks"]["folds_line_disjoint_all"] = bool(all_ok)

    # 3. adapt all F1 train+val events under the causal contract
    meta = store.dataset_metadata
    split = load_split(
        config["dataset_dir"],
        view=config["view"],
        split_name=config["split_name"],
        fold=config["development_fold"],
    )
    request = ModelDataRequest.initial_state_rollout(config["pm_mode"])
    n_adapted = 0
    max_coi_state = 0.0
    max_coi_power = 0.0
    n_finite_fail = 0
    for eid in split.train_ids + split.val_ids:
        event = store.load_event(eid)
        sample = adapt_event(
            event,
            inertia_prior=meta.M_prior,
            request=request,
            representation=config["representation"],
        )
        static = sample["model_inputs"]["static"]
        exo = sample["model_inputs"]["exogenous"]
        targets = sample["training_context"]["targets"]
        mass = static["mass"]
        max_coi_state = max(
            max_coi_state,
            abs(float(np.sum(static["delta0"].reshape(-1) * mass))),
            abs(float(np.sum(static["omega0"].reshape(-1) * mass))),
        )
        max_coi_power = max(
            max_coi_power, abs(float(np.sum(exo["Pm"][0])))
        )
        if not (
            np.isfinite(targets["delta"]).all() and np.isfinite(targets["omega"]).all()
        ):
            n_finite_fail += 1
        # ZOH: every Pm row equals the clearing value
        assert np.allclose(exo["Pm"], static["Pm_clear"][0], atol=0.0), eid
        # initial state matches first target row
        assert np.allclose(static["delta0"][0], targets["delta"][0], atol=0.0), eid
        assert exo["t"][0] == 0.0, eid
        n_adapted += 1
    report["checks"]["adaptation"] = {
        "n_adapted": n_adapted,
        "expected": len(split.train_ids) + len(split.val_ids),
        "max_abs_coi_state_residual": max_coi_state,
        "max_abs_coi_power_residual": max_coi_power,
        "n_nonfinite_targets": n_finite_fail,
        "pass": n_adapted == len(split.train_ids) + len(split.val_ids)
        and max_coi_state < 1e-10
        and max_coi_power < 1e-10
        and n_finite_fail == 0,
    }

    report["status"] = (
        "PASS"
        if all(
            c.get("pass", True)
            for c in report["checks"].values()
            if isinstance(c, dict)
        )
        and report["checks"]["folds_line_disjoint_all"]
        else "FAIL"
    )

    run_dir = ROOT / "results" / "e00_data_check"
    run_dir.mkdir(parents=True, exist_ok=True)
    out = run_dir / "summary.json"
    if out.exists():
        raise SystemExit(f"refusing to overwrite {out}")
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
