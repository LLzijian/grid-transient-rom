# Loop0095 — numerical feasible point, optimal-status gate unmet

Work type: diagnostic. Result: inconclusive (registered optimization-status
failure). Model promotion: none. Preregistration29d47ed, implementationf450a09.
No claim of constrained optimum or scientific rejection of the36-column span.

The ONE registered convex solve returned CVXPY `optimal_inaccurate`, native
Clarabel `AlmostSolved`, after51iterations/10.0358815solver seconds. The
preregistration requires `optimal`; this gate is not reinterpreted or relaxed.
No solver switch, changed tolerance, extra iterate selection or real retry.

## Evidence retained, not promoted

The returned point's directly computed harmonic capture is.571587392340133
(objective.428412607659867), versus unconstrained0092capture.686483616.
This arithmetic is valid for that point; it is not a certified constrained
optimum and does not make0095 supported. Native objectivegap8.04059e-11,
primalresidual4.57167e-8,dualresidual1.28748e-14 pass the separate numerical
thresholds, but do not override the status gate.

| Conditioned ratio to0094 baseline | Fine | Coarse |
|---|---:|---:|
| Average1/2/5 risk | .999999999574 | .999999999584 |
| Severe35 mean | .999999999982 | 1.000000001370 |
| Severe worst4 | .641080620515 | .641080621657 |

Both grids satisfy preregistered numerical nonworsening tolerance1+1e-7;
coarse severe mean exceeds1 by1.37e-9, explicitly retained. Minimum actual
k=.0314968250101,d=1.00007359346e-8: one damping coefficient lies essentially
at the imposed floor, not robust strict positivity. Fullprefix pairedRMS
2.085948968e-8 <=.001. No autonomous prediction/stability/modal claim follows.

Independent arithmetic passes: maxarrayrelative4.826264018e-15,
maxscalarabsolute2.842170943e-14. It reconstructs692 event/grid column paths,
2076scoreGrams,1038pairedGrams,1038cachedfactor products, all old0094 arm
anchors and final score/tail/constraint fields. Saved epigraph variables have
zero measured positive constraint violation. It checks saved solver fields
and finite dual arrays; it does NOT independently solve a dual or provide a
rigorous interval certificate. Main used1038 fine Gram eighs and one solve.

## Stage review and next question

Stage closed: main52.3107587s + independent8.7369362s + metadata20s
=81.0476949s <=1100s. One coherent preexecution repair, no actual retry.
10non-solver synthetic tests, one tiny known convex-problem solve and
8independent synthetics pass. No model exported/default changed; no rollback
required. Accepted M3/0058/S untouched; freshTEST unopened.

The frozen feasible numerical point is useful evidence even though the
optimal-status requirement failed. A separately preregistered0096 can ask
whether this EXACT point is dynamically usable: local coupled stability,
physical modal matching and autonomous TRAIN predictions. That is a new
deployment question, not a rescue/continuation of the0095 optimizer, and
requires no further coefficient adjustment. Independent route review supports
this distinction with strict frozen-byte and no-promotion limits. Root rejects
the review's overly broad suggestion that pointwise capture itself becomes
invalid: only optimality/0095acceptance remains unavailable.

Commands:
`python experiments/audit_backbone_compatibility_0095.py --output private_data/task2_backbone_compatibility_0095`
and `python private_data/task2_backbone_compatibility_0095_independent/check_compatibility.py
--input-dir private_data/task2_backbone_compatibility_0095 --output-json
private_data/task2_backbone_compatibility_0095_independent/independent.json`.

SHA256: arrays ef147edab2eb4c20aa962ede1e28a9fea80b0416b0544a3826e573e14074e75d;
evidence668d82bf9d7361308563d5c4f2fb478b4f4d933c8f45ec67522c0d1ff5354128;
independent9639b8f03710cd3f67cf82da5c059deacfdedc562d9555b71168edfc0cb3136a;
checker94a1c06fb5e2ebeae4f7bde1a44731d3d64279c88fbe2b7a9e501ff39d094a87.
Detailed artifacts remain in `private_data/task2_backbone_compatibility_0095`
and `_independent`; no source/candidate bytes changed.
