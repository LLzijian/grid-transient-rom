# Loop0080 — one-shot force-oracle initialization rejected

Question: does replacing only frozen G-fixed3000 q(100ms) by0079 qF reduce
TRAIN autonomous risk>=10%, with severe-prefix mean/tail<=1.05 baseline?
Preregistration fd50172; implementation2a665a9; start c15ea52. No fitting,
physical-initial-state change or truth injection after initialization.

## Evidence and decision

**Rejected.** Average1/2/5s risk rises .8938195532 ->2.8209719722,
qF/baseline ratio3.156086664 rather than<=.90. Severe35-event prefix mean
rises .6654646978->2.9020206823 (ratio4.360893511); worst4-event mean
2.3038732387->16.2240682644 (ratio7.042083736), both above1.05.

| Horizon | Baseline risk | qF initialization risk |
|---|---:|---:|
| 1s | .602125281 | 1.619969722 |
| 2s | 1.018217616 | 2.303190972 |
| 5s | 1.061115762 | 4.539755223 |

All346 TRAIN events (311 stable/35 unstable,24 groups,369 raw) retain
genuine available-prefix targets at every horizon. No complete-only
filter, artificial tail, initial-point scoring or hidden optimizer is used.
Reference forecasts are fixed .005s; comparison .01. All original substeps
on genuine intervals finite. Numerical COI/common/p maxima are baseline
1.54634e-5/8.59567e-7/7.64318e-6 and qF9.00124e-6/6.40653e-7/3.50691e-6,
each below .001. This is not an observed step-consistency failure.

All1038 coarse-baseline event/horizon errors reproduce the original0078
monitor with relative max6.89e-16 (independent absolute max3.56e-14,
horizon summary max8.89e-16). Seven targeted synthetic checks passed.
Separate cached NumPy arithmetic reproduces every reported risk/numerical
row exactly and confirms identical physical initial states, q values,
length masks and all cohort counts. Model tensors/source hashes/dataset
identities unchanged. Runtime416.87s. Inherited reader loads unused
validation; only TRAIN is analyzed. No TEST is read.

## Interpretation and next implication

The hidden-force projection fixes a particular instantaneous force error,
but is not a learned dynamical state. A one-shot replacement is insufficient
and materially harms subsequent autonomous prediction. This rejects only
this initial-state intervention, not all causal initialization improvements
or r3. It does not rescue0078 or promote a model. No deployed implementation
was changed, so accepted M3/0058/S remain; preserve the rejected diagnostic
trajectories and keep C3000's prior control status.

Next test the projected q trajectory's compatibility with the SAME frozen
latent dynamics over short physical-trajectory-conditioned windows, compared
with the causal encoder trajectory. This isolates latent closure consistency
from accumulated physical-state rollout error without fitting or more
latent dimensions. Explicitly classify truth-conditioned propagation as
diagnostic, never autonomous prediction or a deployable input path.

Artifacts: `private_data/task2_force_initialization_0080/{evidence.json,
trajectories.npz}`; independent checker and evidence:
`private_data/task2_force_initialization_0080_independent/{check.py,
independent.json}`.
