# Accepted loop 0008: F1 refit-v2 prune accepted under the corrected total gate

> 范围更正标记（2026-09-22）：本接受结论保持不变；早期关于 scale bug 不影响 deletion path 的过强推断已由 [0009](0009_five_fold_refit_acceptance.md) 更正。

Date: 2026-09-06

Starting commit: `ea00c71`

Decision: **accepted F1 prune rerun** (corrects the rejected 0007 comparison; no
new confirmatory claim)

## Registered claim

The F1 `prune` stage rerun under model format v4 support produces a
gate-compliant final checkpoint: the corrected total candidate-versus-fresh-M0
adoption gate accepts the three energy modules with an honest observed
improvement of 66.3%, the validation scales used in that comparison are
reconstructed from the exact frozen F1 training population (the accepted F1
`full` checkpoint predates format v4), and the pruned model reloads with
unchanged active blocks, ridge, and scales.

The loop fails if the accepted pruned artifact lacks a finite `module_adoption_gate`
dict, if its sidecar does not persist positive finite `validation_scales`, or if
reload changes active blocks / ridge / scales.

## Evidence

- F1 `prune` rerun: `results/e10_refit_v2/F1/prune/summary.json`, recorded git
  head `ea00c712`, wall time 2144.5 s, `reload_verified: true`.
- `validation_scale_provenance`: `status=reconstructed`,
  `source="exact frozen training population"`,
  `reason="accepted F1 full checkpoint predates model format v4"`;
  values `{omega: 0.3273, edge_angle: 0.4939}`.
- `module_adoption_gate`: `accepted=true`,
  `observed_improvement=0.6631`, backbone standardized loss `5.6549` vs
  candidate `1.9052`, both failure rates `0.0`,
  `criterion="candidate improves validation loss by threshold without higher
  failure rate"`. The invalid 96.23% figure from the rejected 0007 run
  (candidate 0.2130 under default scales) is **not** carried forward.
- Active blocks after prune: `NODE_HIGH_ORDER`, `EDGE_MEMORY`,
  `RESONANT_MEMORY` (all three adopted). Fixed-H100 val report unchanged from
  the accepted `full` stage: pooled RMSE 0.317, pooled R2 0.766, pooled omega
  R2 0.631, 34 eligible / 0 failures.
- **Hash evidence that the 0007 bug was confined to the gate denominator**: the
  pruned NPZ is byte-identical to the rejected 0007 artifact
  (`2b8592a4bc20febd...c455dcc`), while the JSON sidecar now differs
  (`f0f7f4d9...`, format_version 4 with persisted `validation_scales` and the
  corrected gate). The scale bug therefore never entered the fit / whole-block
  deletion path; it corrupted only the reported standardized-loss comparison.

## Gate record

| Gate | Result |
|---|---|
| G1 syntax | pass: `compileall -q src experiments tests` |
| G2 unit | pass: `pytest -q` |
| G6 JSON | pass: `verify_artifacts.py --json-only` |
| diff integrity | pass: `git diff --check` |

## Scientific boundary

Acceptance covers the F1 refit-v2 prune checkpoint only, as corrected
development/validation evidence under loops 0001/0006. It is not a confirmatory
out-of-sample result. F2–F5 refits and the corrected validation diagnostics
remain pending; the confirmatory release stays blocked on a genuinely new
dataset build.
