# Loop0068 — frozen-S sparse storage: local family rejection

Classification: **rejected**, for the specified storage family on S only.
Authority after66; preregistration `reports/task2_sparse_storage_0068.md`.
Execution checkpoint fec89bc; S/0058/M3 were not modified or promoted.

Question: can H3 plus the224 registered corrections support the required
local strict Lyapunov curvature, before selecting<=24 terms or fitting a
nonlinear region? The local necessary-condition check permits ALL224 terms;
failure is therefore not caused by the final sparsity budget.

Evidence:7 unit tests pass. Independent actual-source H3 value/gradient
agreement with MechanicalROM is1.11e-16/1.78e-15; library value/autograd
Jacobian/Hessian errors are4.44e-16/4.44e-16/8.88e-16. Fixed Clarabel run
returns infeasible in8 iterations, with unchanged1e-6 matrix margins and
1e-8 solver tolerances. Its two positive-definite dual matrices provide a
numerical separating witness: after a5.57e-10 linear-residual projection,
minimum eigenvalues .00978925/.00167645, coefficient residual7.11e-15,
and constant inner product -.999999957. Thus this is strong numerical
evidence of incompatible local constraints, not merely a timed-out fit.
It is not an interval-arithmetic proof or a nonlinear stability certificate.

No TRAIN rollout/energy fit/validation energy selection was needed. No rho,
coverage or candidate energy is claimed; stored H without coefficients is
explicitly labeled the fixed-H3 reference, not a fitted solution. No TEST
payload was read. Evidence: `private_data/task2_sparse_storage_local_0068/`
(`evidence.json`, `local_matrices.npz`, `dual_check.json`, `dual_check.npz`);
formula check `.loop-work/storage68_formula_qa.json`.

Interpretation: S's previously verified local stability is unchanged.
H3's fixed physical curvature plus these q-cross terms and two global
physical cross terms is too restrictive for the registered local screen.
Generating more samples or adjusting a regional threshold cannot repair
this local obstruction. Stop this family's unexecuted regional fitting;
retain its reproducer/evidence and the unchanged accepted checkpoints.

Next implication: continue0067 joint closure. Before another energy fit,
test one physically motivated correction to the physical curvature (the
conservative differential component of learned instantaneous electrical
feedback), under a separate preregistration; do not enlarge q or build a
general certification framework to evade this result.
