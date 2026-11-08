# Loop0094 — correction worsens the conditioned speed balance

Work type: diagnostic. Result: supported. Model promotion: none.
Preregistration `reports/task2_conditional_balance_0094.md` (7826458),
implementation55e6222. Loop0093 remains rejected; no candidate was changed.

## Decisive evidence

On the unchanged346TRAIN events (311stable/35unstable), the exact0093k/d
correction worsens the shared-q, truephysical-path speed-balance defect on
both registered RK4 grids. Fine/coarse full/base average1/2/5risk ratios are
1.548026040/1.548026040; severe35mean5s-prefix ratios4.467770566/4.467770574.
Both exceed the preregistered1.25 joint threshold. Descriptive severe worst4
ratios2.840464231/2.840464238; not an additional acceptance gate.

| Fine-grid arm | Average risk | Severe35 mean | Severe worst4 |
|---|---:|---:|---:|
| Baseline | 18.320644714 | 1.771554179 | 11.734780674 |
| k only | 29.676490043 | 6.006392751 | 28.218372387 |
| d only | 17.846951228 | 4.296609649 | 19.382078278 |
| Full | 28.360835087 | 7.914897619 | 33.332224763 |

The six fixed-severe-cohort quadratic terms at5s are
[1.771554179,4.919751223,3.147137516,-.684912651,-.622082046,-.616550602].
The negative cross terms partially cancel positive correction norms; they
remain reported, not clipped. No additive attribution of changing tail sets
or autonomous errors is claimed. All1038 event/horizon rows per arm/grid and
all fixed-cohort/horizon terms are retained in the metric arrays.

Numerical QA: maximum paired-grid weighted RMS3.028334254e-8 <=.001;
quadratic identity maxrelative6.355540561e-16. All baseline denominators pass
the >1e-12 and >100x corresponding numerical-risk/mean conditions.
Three paired probes confirm omega-only field changes; no AD, fit, eigensolve
or autonomous rollout. All source/candidate hashes unchanged.
Independent checker passes692 event/grid Simpson checks of known increments,
all scores/terms and188 summary scalars: maxarrayrelative6.696889674e-16,
maxscalarabsolute2.842170943e-14. It does not independently integrate q/I0.

## Interpretation and next question

Autonomous physical departure is not necessary for the observed worsening
of this conditioned diagnostic. Both isolated corrections worsen severe
events, so a damping-only rescue is not justified by these data. Sharedq is
still model-generated, not latent truth; compensation and conditional-state
incompatibility remain possible. Piecewise-linear physical replay is not an
unsampled FOM solution. This does not quantify causes of0093 autonomouserror,
reject latent order or the model family, or establish optimization adequacy.

Next smallest question: can other positive amplitudes in the SAME36-column
backbone retain majority harmonic capture without worsening the fixed-path
TRAIN balance? A separately preregistered convex compatibility diagnostic,
not another neural fit or capacity increase, can test that distinction.
Independent route review requests exact tail epigraphs, postchecked constraints,
baseline feasibility and an optimality certificate before any impossibility
interpretation. Root will register those limits before another real calculation.

## Cost, commands and preservation

One main20.5125492s + one independent3.7252237s + metadata20s =44.2377729s
within1100s; stage closed. No real retry. One coherent preexecution repair;
15main/metric synthetics pass2.61s,8independent synthetics pass.117s.
Main command: `python experiments/audit_conditional_balance_0094.py --output
private_data/task2_conditional_balance_0094`.
Checker command: `python private_data/task2_conditional_balance_0094_independent/check_balance.py
--input-dir private_data/task2_conditional_balance_0094 --output-json
private_data/task2_conditional_balance_0094_independent/independent.json`.

SHA256 anchors:
- evidence61373d0088cdd7b0271918cbb55a0291d08db753a8344ff6a323bace16134225
- arraysd33cbb83b656e1275ac3af01aa779444db2f33e6a13bbeb285eed2dcf8dae2ec
- metricse50396f14a1d01f937ddb31f3bba8fe0ac21fc32789a73c39d04cd0278496dbb
- independent37bbb42ad09f3b23768ee747a45b256a9d9b88a9ebf75f805d1eafffe2908060
- checker6547d981a3a92ece29585fe853315b71126f1b3de3d8c0cbe7be693b6d2460be

All artifacts retained in `private_data/task2_conditional_balance_0094` and
`_independent`. Accepted M3/0058/S and all deployed/default files unchanged;
no rollback needed. Fresh TEST unopened; consumed Pilot TEST excluded.
