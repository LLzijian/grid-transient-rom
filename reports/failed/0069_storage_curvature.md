# Loop0069 — one instantaneous-force storage curvature term

Classification: **rejected**, for this second fixed-S storage family only.
Preregistration `reports/task2_storage_curvature_0069.md`; run10eb62b.
Only one scalar multiplies the fixed, gauge-preserving conservative
differential curvature R'*sym(Cq*ad)*R; ODE/encoder/M3 remain unchanged.

Five synthetic tests pass. Actual-source chi autograd gradient/Hessian
errors5.55e-17/0; gauge residual4.44e-15. The registered225-variable,
zero-objective Clarabel screen returns **infeasible_inaccurate** after8
iterations. Preserve that raw status; no solver/tolerance/budget change.

Independent verification strengthens the numerical evidence: projecting
the returned dual witness onto the coefficient-annihilating subspace changes
it by2.02e-8 in norm; both matrices remain positive definite with minimum
eigenvalues .228533/.0716008, coefficient residual1.14e-13 and negative
constant inner product -.999998583. This supports local incompatibility of
the expanded family, rather than drawing rejection from solver status alone.
It remains a floating-point witness, not a formal interval/nonlinear proof.

Evidence: `private_data/task2_storage_curvature_0069/evidence.json`,
`storage_curvature_matrices.npz`, `dual_check.json`, `dual_check.npz`.
No fitted energy, regional trajectory generation, rho, validation coverage or
TEST access. Fixed-H3 matrices saved without coefficients are not a solution.

The single conservative instantaneous-feedback curvature does not remove
0068's obstruction. Do not add further curvature terms in this B1 campaign
or confuse this with rejection of S's local stability. Accepted checkpoints
remain unchanged; local reproducers/evidence are retained, not deployed.
Continue the already-running0067 C/J campaign, then use its prediction and
coupled-dynamics evidence to decide whether B2 is scientifically justified.
