# Loop0088 — safe but insufficient fixed-budget improvement

Work type: diagnostic intervention/model screen. Result: **rejected**.
Model promotion: **none**. Original thresholds unchanged.

Preregistration04d5aee, implementationab891c7. Root session20779 exited0:
`python experiments/train_guarded_rollout_0088.py --train`.
All300 updates/projection solves completed, one final candidate and one
346-event RK4 pair. Training2824.9646s, forecast41.5970s, new main2870.8563s.
Including0087's failed no-update preflight35.8327s, SAME-branch cost2906.6890s.
0087 remains resource-inconclusive; no reset, replication or post-hoc pass.
All caps satisfied; this allocation is CLOSED, no further resource-only grant.

| Autonomous TRAIN quantity | Baseline | Candidate | Ratio | Gate |
|---|---:|---:|---:|---:|
| Average1/2/5s risk | .893819553 | .870870042 | .974324224 | <=.90 fail |
| Unstable5s-prefix mean | .665464698 | .632003908 | .949718159 | <=1.05 pass |
| Unstable5s-prefix worst4 | 2.303873239 | 2.048963855 | .889356159 | <=1.05 pass |

Candidate1/2/5s risks .597922131/.985419403/1.029268593. All346 events
retained per horizon,35 unstable; genuine prefixes only. All registered
finite/substep/numerical checks pass. Unused VAL was materialized by the
original reader, not scored; no TEST or boundary-data merge.

Final AD/affine Jacobian relative discrepancy1.17e-17; fixed-P decay maximum
-.5100108115, spectral abscissa-.1334653780/s, eigen residual1.40e-16.
Origin/gauge/kinematic/export/power/frozen-key QA pass. These are model-local
properties, not modal fidelity, a nonlinear region or a full-order certificate.

Independent cached audits in `private_data/task2_guarded_rollout_0088_independent`:
`training_independent.json` reproduces all300 KKT/proposal/continuity checks,
final guard and cumulative caps, max relative discrepancy7.77e-16 (3.64s).
`forecast_independent.json` reproduces1038 event metrics per arm, summaries
and numerical rows, max absolute discrepancy7.11e-15 (2.31s). Both pass on
their first real run, no repair or scientific rerun. Fifteen combined synthetic
production tests had passed before launch. Independent arithmetic is not
independent-data confirmation.

Cached descriptive trace:105/300 solves have F-min<=1e-6; median/max
projection-to-proposal distance ratio8.10e-5/.811436. Raw gradient norms
2.173..229.541; final theta displacement norm.1027049151. These are coordinate-
dependent optimizer descriptions, not stationarity or convergence evidence.
The guard sometimes binds, but these traces cannot assign the missed gain to
the guard, Adam geometry, sampling or model expressivity.

Artifacts: `private_data/task2_guarded_rollout_0088/`; final model SHA
5bab5fba84fb6261821c8654ffa6e33b4e015d12f472114028d8fe0e119577ba.
Accepted/default M3/0058/S and G3000 inputs unchanged. Rejected standalone
candidate/evidence retained; no default implementation was replaced, so no
rollback is needed. No intermediate selection, validation screening or promotion.

Stage review: a2.6% aggregate improvement with better severe-event performance
does not meet the registered10% claim. It also does not establish optimization
adequacy or reject this model family. Close direct coefficient optimization;
do not add steps, change weights or select an earlier iterate. Before further
training/complexity, root proposes one bounded cached physical-output modal
diagnostic of G3000 and this final candidate using unchanged0071 targets/rules.
This targets the distinct L3 gap in after75, not a rescue of the failed TRAIN
screen. Independent route challenge and prospective registration precede it.
