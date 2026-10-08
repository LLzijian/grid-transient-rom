# Loop0085 — local guard passed, useful-transfer gate failed

Work type: diagnostic intervention/model screen. Result: **rejected** for
the useful autonomous improvement claim. Model promotion: **none**.

Preregistration8ca6263 with prospective clarifications b219cc1/9ffda0f/80be3cd;
implementationc6914f5. One execution, root session10391 exited0. Command:
Python experiments/audit_lyapunov_guard_0085.py --output
private_data/task2_lyapunov_guard_0085. Main47.4529s, forecast40.0753s.
Five synthetic tests passed (root2.22s). No H5/VAL/TEST; only verified TRAIN caches.

The fixed baseline Lyapunov solve has relative residual1.21297e-12,
condition448.0496. Proposal mu87.63685 fixes alpha=.005591255341351045
(only.5591% of the rejected0082 step), before forecasts. Actual reloaded
candidate decay maximum=-.5100000000007106 and spectral abscissa=-.1385978131/s;
all guard, actual-Jacobian affinity, origin/gauge/export/power/frozen-state
checks pass. This is a sufficient numerical local guard, not a finite-amplitude
region or full-order stability claim.

| Autonomous quantity | Baseline | Guarded | Ratio | Gate |
|---|---:|---:|---:|---:|
| Average1/2/5s risk | .893819553 | .873926504 | .977743775 | <=.90 fail |
| Unstable5s-prefix mean | .665464698 | .666708916 | 1.001869698 | <=1.05 pass |
| Unstable5s-prefix worst4 | 2.303873239 | 2.319037750 | 1.006582181 | <=1.05 pass |

Guarded1/2/5s risks .600436681/1.018110857/1.003231975. All346 events at
each genuine horizon; severe short prefixes retained. Numerical maxima
COI3.10146e-4/common1.48478e-5/Pm1.49430e-4 all<=.001. Conditional total/
physical/unstable RMS ratios .999024947/.999002925/.999074390; prior reference
gates remain .90/1.05/1.05 (total fails). Conditional quadrature max6.49618e-7.
No reference gate is relaxed and0082 remains rejected.

Candidate SHA c64a10d7bbeef84af8dbccbe86ff08397566de7f98265ab1a910518fb5727c00.
P, actual/full/quotient matrices, coefficients, conditional arrays and full
forecasts: `private_data/task2_lyapunov_guard_0085/`.
Independent results: `private_data/task2_lyapunov_guard_0085_independent/`.
Guard/conditional arithmetic maxabsolute2.84e-14, relative2.73e-16 (1.08s).
Forecast arithmetic maxabsolute7.11e-15, twelve synthetic/schema checks (2.21s).
The first forecast checker stopped before metrics on0082-specific identity
fields absent in0085 (2.04s). That failure is preserved; root authorized only
an exact-hash-linked parent0082 identity lookup, no metric change, input edit
or experiment rerun. Production cache checks used0080 BASELINE numerical rows,
not0082 candidate rows; this distinction was fixed before real execution.

Stage closed, one candidate budget spent. The guarded step is locally stable
and slightly better in aggregate, but not the preregistered10% improvement.
Scalar contraction changes size and stability simultaneously; no causal error
fraction is inferred. No accepted/default model was changed, no rollback needed;
rejected artifact retained, M3/0058/S unchanged.

Root next tests ONE fixed-P-constrained convex coefficient solve, rather than
another scalar truncation. The question is whether optimizing direction within
this declared local guard makes the conditional surrogate useful autonomously.
Existing isolated CVXPY/Clarabel from0067 is available; no solver installation,
new framework, support, state or loss term is needed. Independent prior challenge
already bounds fixed-P conservatism and non-causal interpretations. Register
exact solve/optimality and rollout gates before execution; no automatic rollover.
