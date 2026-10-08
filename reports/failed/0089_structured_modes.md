# Loop0089 — neither frozen structured system matches the physical modes

Work type: diagnostic. Result: **rejected** for each majority-modal claim.
Model promotion: **none**.0088's trajectory-screen rejection is unchanged.

Preregistration7c1e638, implementationc3de9af. Single command:
`python experiments/audit_structured_modes_0089.py --output private_data/task2_structured_modes_0089`.
Exited0; main1.4240s, exactly two32-state eigensolves. No training, model
loading, new derivative/FOM eigensolve, rollout, H5, VAL or TEST access.
Sixteen combined tests passed before this first/only real invocation. The
documented preexecution repair fixed schema/provenance and checker arithmetic;
it did not consume a scientific run or alter thresholds.

| Frozen system | Distinct matches | Required | Eigenvector condition | Min. relative pole separation |
|---|---:|---:|---:|---:|
| Original G3000 | 0/13 | >=7/13 | 59.8275 | .0377131 |
| Final0088 | 0/13 | >=7/13 | 41.4715 | .0372210 |

Each has12 eligible oscillatory ROM candidates and156 saved FOM/ROM pairs.
Neither triggers the preregistered numerical-sensitivity warning. Full spectra,
physical vectors, all pairs and unmatched lists are retained. Q removes only
common angle; full common frequency, exact inertia/units and complex phase
are preserved. New eigen-residual and cached-Q/source checks pass.

Both systems have the SAME gate-availability pattern. Eight targets
66,71,73,77,79,98,100,104 have no ROM physical-output MAC>=.8. The last three
also lack frequency- and damping-qualified partners. Targets62,64,68,84,91
have some individually frequency/damping/shape-qualified partners, but none
meets all three gates simultaneously. Individual gate availability does not
imply that the same partner passes; no many-to-one or nearest-frequency rescue.

The original699-state descriptor and its150 finite/549 infinite modes were
NOT recomputed. Their validated0078 residual/unit/equilibrium/reliability
checks were inherited with explicit current SHA links to original sources.
The13 physical output vectors were reconstructed from cached complex right
vectors. G3000 uses0083 baseline_A33, not its rejected0082 candidate matrix;
final0088 uses its actual exported-model AD Jacobian. Derivative/affine QA is
inherited from those anchored successful checks, not presented as fresh FD.

Independent checker `private_data/task2_structured_modes_0089_independent/independent.json`
passes on its first real invocation:312 total pair rows, exact matches/unmatched
sets and gate availability, physical outputs/Q/eigen residuals and current
source hashes. Maximum arithmetic difference1.11e-15; .05025s, no extra
eigensolve. Source/code/candidate files unchanged. Main arrays/evidence are
in `private_data/task2_structured_modes_0089/`.

Stage CLOSED; its two-object budget is spent. Local stability and modest
TRAIN improvement did not provide L3 fidelity. Frequency/damping adjustment
alone cannot fix missing shapes in these fixed eigensystems. This does NOT
identify which learned structure is deficient, prove inadequate latent order,
demonstrate guard causality or reject the model family. No regional certificate,
small-perturbation confirmation or promotion follows. All accepted/default
models and rejected evidence remain intact; no implementation rollback needed.

Root next implication: before more training or added structure, diagnose the
observable small-signal feedback rather than label q coordinates. A prospective
bounded comparison of instantaneous physical feedback and eliminated-state
memory against the same FOM is under independent review. It must preserve
physical units/common frequency, count all solves, and avoid interpreting a
frequency-domain decomposition as attribution of large-disturbance error.
