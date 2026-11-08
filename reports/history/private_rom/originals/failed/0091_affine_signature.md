# Loop0091 — sampled affine signature rejected

Work type: diagnostic. Result: rejected. Model promotion: none.
Protocol `reports/task2_affine_signature_0091.md`, prereg55f3e99;
implementation7f6df5e. No model/ODE/checkpoint changed.

## Evidence

One full13-frequency fit and13 leave-one-frequency-out fits, using only
validated0090 torque-response arrays. Main command:
`python experiments/audit_affine_signature_0091.py --output private_data/task2_affine_signature_0091`.
All14 constrained normal-equation, gauge, finite and objective checks pass.
Full capture .8106248699925952 passes.80; pooled LOO capture
.7849771131572072 fails.80. The joint preregistered hypothesis is rejected;
the full-fit pass does not rescue it. No thresholds/weights/frequencies changed.

Residuals concentrate toward the low-frequency end: at .06060–.08048Hz,
discrepancy norms74.52–75.05 versus held-out residual norms59.19–60.38.
At .6414–1.6866Hz, corresponding held-out residuals16.15–24.66.
All13 rows, full/LOO parameters, predictions and masks remain in the artifacts;
this descriptive concentration is not permission to drop those frequencies.

Independent checker uses a Helmert null-basis fit instead of the production
right-projector implementation. First real invocation passes14 fits, no errors;
maximum array relative discrepancy4.76973e-16 and scalar absolute2.91038e-11.
Root reran synthetic tests before launch: main4 pass(.10s), independent8
pass(.097s). One documented coherent preexecution implementation repair;
no real retry. Main.0673556s + independent.1126041s + metadata1.5360985s
= stage1.7160582s, within120s. Stage closed; no extra fit/degree/search grant.

Evidence directory `private_data/task2_affine_signature_0091`:
- arrays SHA e38c779c3d80f059c184c60e7c69aebeabfdd3fc4d74aa8ce55713738271ffcb;
- evidence SHA 4e96d916630ba1a3234ded0d4c33654645aa772e8c61577fd7ffbba65e2dfc40.
Independent `private_data/task2_affine_signature_0091_independent/independent.json`
SHA898f06fddf969a2bfe5938a4e7b90142b612cd2797103d62921f34eebf2ebce7.

## Scope and next implication

This rejects the registered sampled affine-signature criterion, not all
instantaneous corrections, all memory models, latent order3 or the structured
family. Most sampled discrepancy energy is nevertheless fitted descriptively.
Nearby-frequency LOO is development interpolation, not independent physical
confirmation. Dense190 real coefficients need not be physically realizable,
passive or stable; no deployment or physical-mechanism claim follows.

Before adding dynamics or another training branch, examine whether the existing
angle-potential/damping backbone can realize even a majority of this discrepancy.
This is a prospective legal-span question, not a renamed0091 retry; independent
route challenge is pending and no new fit is authorized by this conclusion.
Accepted M3/0058/S remain unchanged. No rollout, VAL or TEST read in0091;
fresh frozen TEST remains unopened, consumed PilotTEST excluded.
