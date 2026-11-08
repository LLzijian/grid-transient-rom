# Loop0082 — conditional coefficient regression did not transfer

Work type: diagnostic intervention / model screen. Result: **rejected** for
the corrected preregistered intervention. Model promotion: **none**.
Stage1 remains an implementation failure: `0082_attempt1_normalization.md`.

## Experiment and evidence

Preregistration `reports/task2_coefficient_block_0082.md`: initial40fc4c8,
scale clarification4e9b037, prospective correctionaa3d328; corrected source
7219d1f. One24-amplitude ridge fit,46514 windows/346 TRAIN events (311/35),
then one autonomous1/2/5s comparison. Only eight H and sixteen existing port
amplitudes change; observer, support, base and J/R/B frozen. Conditional speed
scale1.003367; autonomous uses exact inherited0080 scales. No VAL scoring/TEST;
inherited reader materializes unused VAL. No sample exclusion or fabricated tail.

Artifacts: `private_data/task2_coefficient_block_0082_attempt2/` contains
evidence.json, fit_arrays.npz, trajectories.npz and candidate export. Session34555
exited0; runtime132.025s (design94.736s, forecast18.808s).
Baseline model SHA2667480a75d7672ad49ed66c79731e754676bae52574229dbb5f98c581ac890b;
candidate SHA72fce608287d9e13dd23957e860ad5d20abb2e3130cd883e0f493e1021348db5.

| Registered quantity | Candidate / baseline | Gate | Outcome |
|---|---:|---:|---|
| Conditional total RMS | .908207356 | <=.90 | fail |
| Conditional physical RMS | .905487371 | <=1.05 | pass |
| Conditional unstable total RMS | .916223509 | <=1.05 | pass |
| Autonomous average risk | 4.590839868 | <=.90 | fail |
| Unstable5s available-prefix mean | 1.634105811 | <=1.05 | fail |
| Unstable5s available-prefix worst4 | 2.471272632 | <=1.05 | fail |

Conditional total/physical/latent RMS baseline2.461886875/2.435271191/.361027991,
candidate2.235903768/2.205107308/.369820795. Latent residual worsens. All24columns
active; design singular range.048157728 to3.134709057; ridge condition4061.936.
Unregularized objective6.060886983 ->4.999265660; penalized5.001495414;
stationarity2.14013e-15. Autonomous average risk.893819553 ->4.103382440.
Candidate1/2/5s risk1.116922820/3.820616390/7.372608110 versus
.602125281/1.018217616/1.061115762. All346 events at every available-prefix
horizon; unstable mean.665464698 ->1.087439730, worst4 2.303873239 ->5.693498883.

## QA and independent verification

Every-window quadrature maximum6.49619e-7/s <=.001/s. Autonomous numerical
maxima COI1.59818e-5/common3.23661e-6/Pm8.90501e-6, each<=.001. All genuine
RK stages finite. Full-field affinity, origin/gauge/kinematics, reload/export,
actual power identity, J antisymmetry/R PSD, allowed changes and source hashes
pass. Ten0082 synthetic tests pass; six unchanged0081 regressions previously pass.

Independent artifacts: `private_data/task2_coefficient_block_0082_attempt2_independent/`.
`fit_independent.json` reproduces residuals/mapping/quadrature but retains
strict internal1e-11 beta/delta discrepancies (max3.02322e-11). This is NOT
an all-checks pass. `roundoff_check.json` verifies the saved beta against
independently nested-summed normal equations: stationarity7.54308e-13 <= the
REGISTERED1e-9. Solving SAVED K/g by eig/QR differs1.45e-14/9.27e-15; re-summed
Gram differences1.33e-12 amplify to3.04e-11 in beta at condition4062.
Root disposition: reduction-order/conditioning explains the strict resummation
flag; the registered stationarity gate passes. No coefficient or threshold
changed. Supplemental audit cost16.40s.

`forecast_exact_independent_v2.json` reproduces all1038 event metrics per arm
and346 numerical rows per arm, maxdifference2.13e-14; eight synthetic/schema
checks pass, runtime2.44s. First schema-only attempt stopped on two new scale
metadata fields (2.16s), before metrics; failure report preserved. Root authorized
only permitting and exactly validating these fields. No metric or model change,
no experiment rerun. Fit-audit time was not precisely recorded; no invented
combined QA time. Original normalization-failure artifacts remain preserved.

## Interpretation, rollback and stage review

Reject this fixed conditional-surrogate intervention, not r3 or the generator
family. Less conditional physical residual does not provide autonomous transfer.
qenc is an observer construction, not truth; changing H affects all paired
coupled dynamics. Tractable-subproblem stationarity does not establish original
nonconvex rollout optimization adequacy. No threshold or tail weighting changed.

Both authorized executions spent: stage1 133.840s + stage2 132.025s =265.866s
main, plus QA/audits. No further fit, lambda, blend or extension authorized by0082.
Preserve standalone diagnostic source/candidates as evidence; no accepted/default
model was overwritten, hence no model rollback needed. M3/0058/S and C-control
status unchanged. Root next authorizes the separately bounded model-only pole
audit in `reports/task2_coefficient_poles_0083.md`: test introduced local
instability before another intervention, without adding complexity or claiming
local poles explain all nonlinear error.
