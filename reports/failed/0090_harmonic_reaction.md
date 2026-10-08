# Loop0090 — sampled harmonic reaction fidelity rejected

Work type: diagnostic. Result: **rejected**. Model promotion: **none**.
The clamped descriptor systems were numerically well posed on all13 specified
frequencies; this is not a failed implementation or an order/family rejection.

Preregistration908f5f0, implementationd89449d. Single main invocation:
`python experiments/audit_harmonic_reaction_0090.py --output private_data/task2_harmonic_reaction_0090`.
Exited0,2.2690484s,26 multi-RHS linear solves and26 conditioning SVDs.
Independent26 raw solves,0 SVDs,.885366s. Including metadata3.6526071s,
stage aggregate6.807021s. All finite stage budgets closed/spent; no retry,
shifted frequency, pseudoinverse, regularization or dropped input channel.
Nine main/helper synthetic tests and eight independent synthetic tests pass.

## Decisive evidence

- Aggregate normalized reaction error **1.044147166**, fails<=.20.
- Every fixed frequency fails<=.50: range **.823708588..1.956225669**.
- Descriptive normalized endogenous-torque error **.756457262**; frequency
  range .667280593..1.947953578. The shared inertial reaction does not erase
  the large feedback mismatch. This secondary metric did not replace a gate.
- Maximum equilibrated condition3348.81691; precision-floor-amplified
  rho5.04895e-10<=1e-6; maximum original backward error9.31351e-17.
  Full-equation, physical-unit/inertia, angle-kinematic and source QA pass.
- Independent raw-solve operators agree to3.489e-15 relative/1.140e-13
  absolute; saved metric arithmetic agrees to2.842e-14. First run passes,
  no errors. Original descriptor/G3000 assembly was separately verified.

All10 physical input directions and common-frequency outputs were retained.
Physical reconstruction satisfies original hidden equations and angle
kinematics. The speed-equation residual is the imposed reaction, NOT zero;
raw pu frequency conversion and M scaling are included. Per-frequency
reaction/torque/COI/common discrepancies, matrices and all solutions remain
in `private_data/task2_harmonic_reaction_0090/{arrays.npz,evidence.json}`.
Independent source, solve and metric evidence is in
`private_data/task2_harmonic_reaction_0090_independent/independent.json`.

## Scope and next implication

Use of the original699 pencil was essential.170 nonzero time constants do
not justify a170-state ordinary reduction:0035 already documented extra PSS
algebraic-state constraints. We did not replace its150 finite-mode descriptor
evidence or attempt an unverified zero-T Schur reduction. The proposed
instantaneous/memory attribution was withdrawn BEFORE registration after
independent challenge; no such attribution has been measured here.

G3000's local physical response is substantially different on this sampled
grid, consistent with0089's missing modal matches but not an explanation of
large-disturbance trajectory errors. Finite harmonic solutions do not prove
properness, stable hidden transients, a causal memory realization or a unique
missing mechanism. No H5, VAL, TEST, model loading, new FOM eigensolve or
training occurred. Accepted/default models and old rejected evidence remain
unchanged; no rollback needed.

Root next question, under independent review: can a real frequency-affine,
static-angle-gauge-preserving correction explain most of the cached torque
discrepancy? This is a bounded descriptive test of an angle/speed-like response
signature, not permission to deploy an arbitrary dense correction, assert
passivity or add latent states. All fitting/leave-one-frequency-out rules and
budgets must be preregistered before inspecting that new fit.
