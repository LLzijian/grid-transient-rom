"""Generate the three paper figures from frozen models and immutable results.

Figures (written to paper/figures/, read-only of results/):
  fig1_pipeline.pdf              modular identification pipeline (schematic).
  fig2_difficulty_trajectories.pdf  representative F1 validation trajectories by
                                 difficulty band (truth vs modular model, all 10
                                 COI machines, H=1 s ZOH rollout). Uses the
                                 DEVELOPMENT validation set only (test untouched).
  fig3_margin_scatter.pdf        heuristic candidate margin over 315 legacy test
                                 events (results/e06/{F1..F5}/test_margin.json):
                                 local candidate energy vs cleared-state storage,
                                 screen vs truth. Aggregate results only; no
                                 test trajectory is re-read and no model changes.

Run with the reused project environment:
  E:/03-nn_Transfer_Func/kron-reduce/.venv/Scripts/python.exe experiments/paper_figures.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

plt.switch_backend("Agg")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from kiki.data import CanonicalStore, ModelDataRequest, adapt_event, load_split
from kiki.models.base import sample_rollout_inputs
from kiki.models.prunable_energy_model import PrunableEnergyModel

H_STEPS = 101  # 1.00 s at dt = 0.01 (t[0]..t[100])
OUT = ROOT / "paper" / "figures"

# Matplotlib's PDF backend adds the current time unless the fields are
# explicitly removed.  Keep the remaining metadata fixed as well so repeated
# figure builds can be compared byte-for-byte.  PNG metadata is fixed for the
# same reason, even though the default PNG writer does not add a timestamp.
_PDF_METADATA = {
    "Creator": "kiki.paper_figures",
    "Producer": "kiki.paper_figures",
    "CreationDate": None,
    "ModDate": None,
}
_PNG_METADATA = {
    "Software": "kiki.paper_figures",
}


def _save_figure(fig, stem: str) -> None:
    """Write one figure in deterministic PDF and PNG forms."""
    fig.savefig(OUT / f"{stem}.pdf", metadata=_PDF_METADATA)
    fig.savefig(OUT / f"{stem}.png", dpi=160, metadata=_PNG_METADATA)


def _band(peak: float) -> str:
    if peak <= 0.5:
        return "mild"
    if peak <= 1.5:
        return "moderate"
    return "severe"


def _adapt_ids(config, store, meta, ids):
    request = ModelDataRequest.initial_state_rollout(config["pm_mode"])
    return [
        adapt_event(
            store.load_event(eid),
            inertia_prior=meta.M_prior,
            request=request,
            representation=config["representation"],
        )
        for eid in ids
    ]


def fig1_pipeline() -> None:
    """Modular identification pipeline overview (schematic, not data)."""
    fig, ax = plt.subplots(figsize=(7.0, 4.1))
    ax.axis("off")
    boxes = {
        "backbone": (0.03, 0.78, "M0 swing backbone\n(never removed)\nD,K >= 0"),
        "proposal": (0.03, 0.44, "Residual impulse\ncorrelation + fault-line\nbootstrap -> candidates"),
        "fit": (0.03, 0.10, "Joint weak-form fit\n(no numerical derivatives)"),
        "prune": (0.62, 0.78, "Block-deletion audit\n+ total candidate vs fresh M0\n2% adoption gate"),
        "energy": (0.62, 0.44, "Local storage audit\n(center, gradient, Hessian,\nsupply identity)"),
        "deploy": (0.62, 0.10, "Candidate-energy screen\n(heuristic; no formal\nstability verdict)"),
    }
    for x, y, text in boxes.values():
        ax.add_patch(
            plt.Rectangle(
                (x, y), 0.30, 0.16,
                fill=True, facecolor="white", edgecolor="black", linewidth=1.2,
            )
        )
        ax.text(
            x + 0.15, y + 0.08, text,
            ha="center", va="center", fontsize=8,
        )
    arrows = [
        ((0.33, 0.86), (0.62, 0.86)),   # backbone -> prune (modules ride backbone)
        ((0.18, 0.78), (0.18, 0.60)),   # backbone down to proposal
        ((0.18, 0.44), (0.18, 0.26)),   # proposal down to fit
        ((0.33, 0.18), (0.62, 0.18)),   # fit -> prune (refit on deletion)
        ((0.92, 0.18), (0.92, 0.60)),   # up to energy
        ((0.77, 0.60), (0.77, 0.26)),   # energy -> deploy
    ]
    for (x0, y0), (x1, y1) in arrows:
        ax.annotate(
            "", xy=(x1, y1), xytext=(x0, y0),
            arrowprops={"arrowstyle": "-|>", "color": "black", "lw": 1.2},
        )
    ax.text(0.18, 0.965, "Frozen-data pipeline (train/val only)",
            ha="center", fontsize=9, style="italic")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    fig.tight_layout()
    _save_figure(fig, "fig1_pipeline")
    plt.close(fig)
    print("[fig1] wrote fig1_pipeline.pdf")


def fig2_trajectories(config, store, meta, model) -> None:
    split = load_split(
        config["dataset_dir"],
        view=config["view"],
        split_name=config["split_name"],
        fold="F1",
    )
    val = _adapt_ids(config, store, meta, split.val_ids)
    records = []  # per stable event
    unstable = []
    failures = 0
    for sample in val:
        initial, t, exo = sample_rollout_inputs(sample)
        h = min(H_STEPS, len(t))
        try:
            _pd, pw = model.rollout(
                initial, t[:h], {"Pm": exo["Pm"][:h], "interpolation": exo["interpolation"]}
            )
        except Exception:  # noqa: BLE001 - figure reports aggregate rollout failures
            failures += 1
            continue
        tr = sample["training_context"]["targets"]
        truth_omega = np.asarray(tr["omega"][:h], dtype=float)
        peak = float(np.max(np.abs(truth_omega)))
        label = str(sample["training_context"]["label"])
        rmse = float(np.sqrt(np.mean((pw - truth_omega) ** 2)))
        rec = {
            "event_id": sample["event_id"],
            "band": _band(peak),
            "peak": peak,
            "label": label,
            "rmse": rmse,
            "t": np.asarray(t[:h]),
            "truth": truth_omega,
            "model": pw,
        }
        if label == "stable":
            records.append(rec)
        else:
            unstable.append(rec)
    if failures:
        print(f"[fig2] {failures} rollout failures skipped")

    def medoid(items):
        med = float(np.median([r["rmse"] for r in items]))
        return min(items, key=lambda r: abs(r["rmse"] - med))

    rows = [
        ("mild stable", medoid([r for r in records if r["band"] == "mild"])),
        ("moderate stable", medoid([r for r in records if r["band"] == "moderate"])),
        ("severe stable", medoid([r for r in records if r["band"] == "severe"])),
    ]
    if unstable:
        rows.append(("severe unstable (margin domain)", medoid(unstable)))

    n_rows = len(rows)
    fig, axes = plt.subplots(n_rows, 1, figsize=(6.6, 1.9 * n_rows), sharex=True)
    if n_rows == 1:
        axes = [axes]
    cmap = plt.get_cmap("tab10")
    for ax, (band_title, r) in zip(axes, rows):
        n_mach = r["truth"].shape[1]
        for k in range(n_mach):
            ax.plot(r["t"], r["truth"][:, k], color=cmap(k % 10), lw=0.9,
                    solid_joinstyle="round")
            ax.plot(r["t"], r["model"][:, k], color=cmap(k % 10), ls="--", lw=0.9)
        ax.set_ylabel("speed (rad/s)")
        ax.set_title(
            f"{band_title}  |  {r['event_id']}  |  "
            f"peak $|\\omega|$={r['peak']:.2f}  |  speed RMSE={r['rmse']:.3f}",
            fontsize=9,
        )
    axes[-1].set_xlabel("time after clearing (s)")
    from matplotlib.lines import Line2D
    legend = [
        Line2D([0], [0], color="0.35", lw=1.2, label="truth"),
        Line2D([0], [0], color="0.35", ls="--", lw=1.2, label="modular model"),
    ]
    fig.legend(handles=legend, loc="upper right", fontsize=8, frameon=False, ncol=2)
    fig.tight_layout(rect=(0, 0, 1, 0.985))
    _save_figure(fig, "fig2_difficulty_trajectories")
    plt.close(fig)
    print(f"[fig2] wrote fig2_difficulty_trajectories.pdf ({len(records)} stable, "
          f"{len(unstable)} unstable val events scanned)")


def fig3_margin_scatter(config) -> None:
    rows = []
    for fold in ("F1", "F2", "F3", "F4", "F5"):
        path = ROOT / "results" / "e06" / fold / "test_margin.json"
        data = json.loads(path.read_text(encoding="utf-8"))
        rows.extend(data)
    n_total = len(rows)

    fig, ax = plt.subplots(figsize=(6.6, 5.2))
    searched = [r for r in rows if r["certified"]]
    failed_search = [r for r in rows if not r["certified"]]

    def cv(rec):
        return rec["predicted_stable"]

    safe = [r for r in searched if r["true_stable"] and cv(r)]
    conserv = [r for r in searched if r["true_stable"] and not cv(r)]
    un_correct = [r for r in searched if not r["true_stable"] and not cv(r)]
    missed = [r for r in searched if not r["true_stable"] and cv(r)]
    ref_stable = [r for r in failed_search if r["true_stable"]]
    ref_unstable = [r for r in failed_search if not r["true_stable"]]

    def plot(recs, color, marker, label):
        if not recs:
            return
        x = np.asarray([r["storage_at_clearing"] for r in recs], dtype=float)
        y = np.asarray(
            [r.get("critical_energy_candidate", r.get("critical_energy")) for r in recs],
            dtype=float,
        )
        ax.scatter(x, y, s=16, color=color, marker=marker, label=label,
                   linewidths=0.4, edgecolors="none")

    plot(safe, "#1b9e77", "o", f"stable, positive screen (n={len(safe)})")
    plot(conserv, "#e6ab02", "D", f"stable, non-positive screen (n={len(conserv)})")
    plot(un_correct, "#d95f02", "X", f"unstable, non-positive screen (n={len(un_correct)})")
    plot(missed, "#000000", "*", f"unstable, positive screen (n={len(missed)})")
    plot(ref_stable, "#66a61e", "^", f"search failed, stable (n={len(ref_stable)})")
    plot(ref_unstable, "#e7298a", "v", f"search failed, unstable (n={len(ref_unstable)})")

    lim = [0.0, max(ax.get_xlim()[1], ax.get_ylim()[1], 10.0)]
    ax.plot(lim, lim, "k--", lw=0.9)
    ax.annotate("margin = 0", (lim[1] * 0.72, lim[1] * 0.72), fontsize=8, rotation=40,
                ha="center")
    ax.annotate("candidate margin > 0 (heuristic screen)", (lim[1] * 0.06, lim[1] * 0.94),
                fontsize=8, color="#1b9e77")
    ax.set_xlim(*lim)
    ax.set_ylim(0.0, lim[1])
    ax.set_xlabel("cleared-state storage $H(\\delta_0,\\omega_0,h_0)$")
    ax.set_ylabel(r"candidate energy $H_{c,\mathrm{local}}$")
    ax.legend(loc="lower right", fontsize=7.5, framealpha=0.9)
    ax.set_title(
        f"Heuristic candidate margin, all {n_total} legacy test events (5 folds)",
        fontsize=9,
    )
    fig.tight_layout()
    _save_figure(fig, "fig3_margin_scatter")
    plt.close(fig)
    print(f"[fig3] wrote fig3_margin_scatter.pdf ({n_total} events; "
          f"positive-screen misses={len(missed)}, searches={len(searched)})")


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    config = json.loads((ROOT / "configs" / "dataset.json").read_text(encoding="utf-8"))
    store = CanonicalStore.open(config["dataset_dir"])
    meta = store.dataset_metadata

    model = PrunableEnergyModel.load(ROOT / "results" / "e02_prune" / "model_pruned.npz")
    print(f"[main] loaded F1 pruned model: "
          f"blocks={model.active_blocks} taus={model.memory_taus} "
          f"resonances={model.resonances}")

    fig1_pipeline()
    fig2_trajectories(config, store, meta, model)
    fig3_margin_scatter(config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
