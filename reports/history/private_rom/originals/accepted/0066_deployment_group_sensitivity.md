# Loop0066 — conditional deployment-loss group insensitivity

## Question and hypothesis

Does omitting TRAIN group31 from a matched coefficient-only continuation
materially change that group's genuine-prefix autonomous predictions?
Accept only the preregistered conditional insensitivity hypothesis, not
coefficient uniqueness or honest unseen-group generalization.

Preregistration: `reports/task2_deployment_group_sensitivity.md`, commit
761cd9a, starting2336a8a. Training implementation ef34c09; reviewed evaluator
e33922f. Authority: `docs/TASK2_after_loop_59.md`.

## Change and execution

Source S remains `private_data/task2_long_0060_5s`. A uses all346 eligible
TRAIN events; B excludes group31's13 events from optimization only, leaving333.
Both start from S, with fresh Adam moments, seed60, lr.0003, batch32,
700 steps and5s horizon. Exact existing loss/sampler retained; no extension,
early stopping, checkpoint selection, clipping, anchoring or new loss.
Only36 active Xi coefficients change. All other tensors, inactive Xi,
normalization, encoder, r3, topology, library and M3 remain bitwise frozen.

Both exit0. A takes3098.32s; B2695.21s. Each records22400 sampled event IDs;
B samples none of the13 excluded events. Source hashes remain unchanged.

Model SHA256:

- S: `d34d9f5c36fb296e1d49e59f5f7b18a5da33b56bf371a16169f82bc25664995c`
- A: `c39125efcb0dd4d084f7aa319b760f346738dd4e0e55fdd203188ff7d9481c55`
- B: `37222dd15148342adf876ff6c69243279db9fb0c88a21b24303a1ec6c443634c`

Commands use `E:/03-nn_Transfer_Func/kron-reduce/.venv/Scripts/python.exe`,
OMP/OPENBLAS threads1:

```text
experiments/train_long_latent.py --start-model private_data/task2_long_0060_5s --horizon 5 --steps 700 --output private_data/task2_group31_all_0066
experiments/train_long_latent.py --start-model private_data/task2_long_0060_5s --horizon 5 --steps 700 --exclude-train-group 31 --output private_data/task2_group31_omit_0066
experiments/audit_deployment_group_sensitivity.py --output private_data/task2_group_sensitivity_0066
experiments/evaluate_long_latent.py --model challenger --model-dir private_data/task2_group31_all_0066 --output private_data/task2_group31_all_eval_0066
experiments/evaluate_long_latent.py --model challenger --model-dir private_data/task2_group31_omit_0066 --output private_data/task2_group31_omit_eval_0066
```

## Evidence

### Optimization adequacy

Deterministic midpoint panels evaluate the same full loss, all events in each
panel together. Every term, ID and endpoint is in `panels.json`; no batch-tail
approximation. All panels finite. Registered A-all and B-complement checks
both pass <= source*(1+1e-6). This is not a stationarity/global-optimum proof.

| Panel | S total | A total | B total |
|---|---:|---:|---:|
| All346 | 3.067480245 | 2.667455879 | 2.667796499 |
| Complement31,333 | 3.093228197 | 2.682235183 | 2.681603425 |
| Group31,13 | 1.963117557 | 1.911622342 | 1.932073841 |

### Primary group31 prediction

All13 events (10 stable,3 unstable), genuine prefix capped at5s, first100ms
encoded once, no subsequent truth input. Initial forecast point excluded
from error. No events or short prefixes removed.

| Model | Mean | Pooled | CVaR10 |
|---|---:|---:|---:|
| S | .339229770 | .369697023 | .607925738 |
| A | .335034593 | .364104041 | .598098409 |
| B | .336730101 | .366468007 | .601383893 |
| B/A | 1.005060696 | 1.006492556 | 1.005493216 |

All three ratios pass <=1.10. Pooled B-A trajectory disagreement .007220322;
dividing by A pooled error gives **.019830381**, passing <=.10. Denominator
floor1e-8 is inactive. Numerical and optimization prerequisites both pass.

Group31 complete1/2/5/8 cohorts each contain10 events; capped8 retains all13.
Complete-horizon, capped5/8, stable/unstable summaries and198 score rows are
preserved in paired `evidence.json`. In particular, unstable-prefix pooled
errors S/A/B are .570558359/.565817066/.567308348; these short genuine
prefixes are the same under the5s and8s caps, not completed8s truth.

Common synchronous-frequency errors are separate from COI metrics. Pooled
group31 RMSE (rad/s) worsens despite the small COI improvements:

| Genuine prefix cap | S | A | B |
|---|---:|---:|---:|
| 5s | .677942972 | .693344887 | .716184718 |
| 8s | .540755077 | .551924262 | .570940374 |

All78 common-frequency rows, including individual SSE/counts, are preserved.

### Development benchmark and numerical limits

Same107 validation events; complete counts96/95/95/95 for1/2/5/8s. Values
below are mean / pooled / CVaR10, using the unchanged .005 reference grid.

| Horizon | S | A | B |
|---|---|---|---|
| 1s | .298393/.373210/.767441 | .298403/.378676/.773011 | .298513/.378727/.773601 |
| 2s | .328373/.381051/.734655 | .328066/.380192/.728295 | .328156/.380355/.729253 |
| 5s | .288326/.333571/.628208 | .286298/.330798/.620761 | .286318/.330802/.621196 |
| 8s | .240233/.277769/.522296 | .238235/.275257/.516337 | .238347/.275372/.516724 |
| Unstable genuine prefix,12 | .724623/1.230200/1.667156 | .728947/1.265218/1.734205 | .728535/1.264082/1.733700 |

Thus5–8s complete trajectories improve slightly but unstable-prefix tails
worsen. Per-group comparisons are saved for all8 validation groups; e.g.
group26 at1s CVaR rises from1.173401 to1.249257(A)/1.247905(B). No validation
feedback altered either fit, thresholds or budget. A/B do not dominate S.

Group31 .0025/.005 maximum primary-prefix step discrepancies S/A/B are
3.39509e-7/3.37907e-7/3.40445e-7; all33 states finite on both grids.
No .00125 diagnostic is triggered. Unobserved fixed8s continuations remain
separate: discrepancies .023875924/.015231398/.022026750 exceed1e-3.
Finite output alone does not establish convergence or stability.

Validation .005/.01 genuine-prefix/complete maximum discrepancies A/B are
8.63531e-6/8.67433e-6; both pass, so no .0025 refinement is triggered.
All-fixed8 discrepancies .018455822/.002482937 remain unresolved on
unobserved tails. This does not imply numerical failure of observed-prefix
prediction and does not rescue prior closed-loop all8 convergence failures.

### Coefficients and coupled poles

Both fits change36/36 active coefficients, with0 sign flips. Maximum absolute
changes versus S: .1247882(A), .1138757(B); largest absolute coefficient-scale
ratios to nonzero S:1.19936/1.19736. Full signed Xi/support arrays are saved.
Small prediction sensitivity does not prove parameter uniqueness.

| Model | Aq poles | Dominant gauge-quotient J32 poles |
|---|---|---|
| S | -.709166; -.950451 +/- .273285i | -.156568875 +/-9.944463548i |
| A | -.741443; -1.000433 +/- .245313i | -.153519995 +/-9.957516018i |
| B | -.743891; -.993492 +/- .246648i | -.152193362 +/-9.957028934i |

All32 coupled poles remain strictly negative in real part. J32 removes only
the rotational angle gauge, preserving common frequency. Live field/RK4
equivalence max errors <=3.553e-15/1.111e-16. No new basin, energy-function,
global boundedness or full-order safety-margin claim follows.

### Verification and retained artifacts

Training preflight:124 tests and both one-step training smokes passed before
the700-step fits. Evaluator preflight, completed while training ran:131 full
tests, strengthened13-test regression, source-identity evaluator smoke and
independent identity QA passed before the actual paired evaluation.
Afterward, independent saved-array metric QA checks
13 events/198 score rows/78 common-frequency rows with0 mismatches;
independent structural/provenance QA rebuilds J32, matches source J33/J32
exactly to0065, and checks both training pools/hashes with0 mismatches.
A/B validation rows, aggregates and step arithmetic independently reproduce
with maximum difference0. No raw TEST payload was read.

Retained data under `private_data/`: `task2_group31_all_0066`,
`task2_group31_omit_0066`, `task2_group_sensitivity_0066`,
`task2_group31_all_eval_0066`, `task2_group31_omit_eval_0066`.
These contain checkpoints, sampled IDs, traces, normalization, panel terms,
truth-aligned forecasts, source parameters, Xi, J32 and pole arrays.
Independent scripts: `.loop-work/check_long0066_group.py`,
`.loop-work/check_long0066_structure.py`, `.loop-work/check_long_saved.py`.
QA results: `.loop-work/long0066_group_independent_checks.json`,
`long0066_structure_independent_checks.json`, `long0066_A_validation_checks.json`,
`long0066_B_validation_checks.json` (all in `.loop-work/`).

## Classification and scientific interpretation

**Accepted — conditional fixed-budget deployment-prediction insensitivity
diagnostic.** Root acceptance follows unchanged preregistered criteria and
independent QA, not the evaluator's label alone.

The large0062 encoder-reference inconsistency for group31 does not manifest
as comparable physical-rollout sensitivity in these two matched700-step
continuations. This weakens the inference from that proxy diagnostic to
deployment failure; it does not erase0062, identify true latent states, prove
global optimization or establish consistency across all fault groups.
The encoder, topology and starting coefficients already saw group31, and
group31 was chosen from prior TRAIN evidence. This is conditional, not a
fresh group-held-out generalization result. No covariance gate is used.

No model promotion or destructive rollback is appropriate: preserve A/B as
diagnostic candidates, leave S and the0058 short-horizon champion unchanged.

## Next implication and pause

Remaining issues include unstable validation tails/common-frequency error,
unobserved long-time numerical convergence, broader coefficient robustness,
nonlinear storage/boundary and final model selection. Fresh TEST remains
closed; these issues must not be resolved with TEST access.

Per the latest user instruction, **pause after closing0066**. No0067
preregistration, new fit, integration extension or subsequent experiment
is authorized by this closure. The project-level objective is not complete.
