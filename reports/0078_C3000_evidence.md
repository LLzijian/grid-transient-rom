# Loop0078 C-free3000 — retained trajectory control

Intermediate result only; G-fixed3000 is still running. Allocation and
settings are those frozen in `reports/task2_structure_campaign_0078.md`
and b0172b8, based on TRAIN risks, not validation. No G-auto primary rescue,
seed confirmation, final model promotion or TEST access.

C completes the single1800-update extension (5683.04s) with exact model/
Adam/RNG continuation, unchanged support and frozen state, zero complexity
penalty and no further selection. Independent continuation/trace/monitor/
source checks all pass. Xi34/alpha18/effective dynamics count141 remain.
TRAIN3000 risk averaged over1/2/5s is .3207898362;5s risk .3220110638.

## Fixed3000 validation

| Cohort | Events | Mean | Pooled | CVaR10 |
|---|---:|---:|---:|---:|
| Complete1s | 96 | .253897258 | .341181022 | .740749303 |
| Complete2s | 95 | .249002255 | .301891462 | .616628150 |
| Complete5s | 95 | .198073952 | .240044721 | .490982770 |
| Unstable available prefix | 12 | .706272247 | 1.238937861 | 1.741488599 |

Every registered S-relative trajectory retention gate passes. The closest
gate is unstable-prefix CVaR: ratio1.044586273 versus allowed1.05; pooled
prefix ratio1.007102736. Common5 RMS=.216180851 rad/s (95 events,47500
samples). L1 detects10/12 eligible unstable events, with0 false LOS and2
false window-settled,0 unresolved. Raw117/eligible107/latency10 and raw22
unstable denominators remain explicit. No finite filtering drops events.

All488 available scored numerical intervals (535 total horizon rows) and
required LOS intervals pass; no .0025 diagnostic is triggered. Reporting
forecasts remain .005. The companion cached physical/numerical replay uses
existing evaluator helpers and is NOT independent software implementation.
Separately implemented direct NumPy metric arithmetic reproduces all above
aggregates and common5 within5.56e-17. It asserts the exact cohorts and
fails on any nonfinite/short eligible forecast. It reads validation caches,
not H5 or TEST. Historical per-step event IDs were not saved; exact resume
and deterministic RNG consistency are not a historical event-ID log.

## Decision and boundary

Retain as an ordinary sparse trajectory-control candidate. This is not a
structured H/J/R/port model and cannot substantiate G-auto discovery.
Do not apply the G-auto >=5% superiority gate to C against itself.
The separate physical-modal majority hypothesis fails1/13 versus>=7;
see `reports/failed/0078_control_physical_modes.md`. Local linear stability
and improved trajectories do not establish modal fidelity or a nonlinear
certificate. Accepted M3/0058/S remain unchanged. Finish G-fixed3000 and
its fixed audit before closing the campaign or choosing its next hypothesis.

Evidence: `private_data/task2_structure_0078_extend_C-free` (training and
`independent_integrity.json`); corresponding `_eval` directory (forecasts,
per-event results, `independent_evaluation.json` shared-helper replay, and
`independent_metrics_direct.py/.json` separate cached arithmetic).
