# Loop 0044 rejected: origin stability does not prevent large-amplitude runaway

Baseline2a6576c; frozen implementation/preregistration38fd14e. Scientific venv
command `experiments/test_topology_stable_driven.py --output
private_data/task2_topology_stable_0044`, BLAS/OMP1. Ruff/compileall/48tests pass.
Fixed0042 coordinate,0043 three-speed support; only four coefficients changed.
Neither test read. Full-state initialization is diagnostic, not deployable.

One feasible-start SLSQP succeeded in20 iterations. Coefficients (decay then
edge speeds8/43/0) [.139341159,-11.3678655,7.47512210,-2.80105697]. Weak SSE
.245087927 vs free-decay .427099527 passes5% gate; coupled gauge-free spectral
abscissa -.009999999988 meets the frozen-.01/s constraint/tolerance.

Nevertheless,109 validation autonomous rollouts fail dramatically:95/95 stable
events newly classified unstable; 8s mean2.7513e22, pooled6.0339e22, CVaR1.3448e23.
M3 pooled8s .437194. All2/5/8s trajectory gates fail. Floats remain finite, which
is not physical validity. Unstable-prefix error passes but cannot rescue this.
Half-step/group acceptance tests are skipped after screen failure.

Revert implementation; retain private `fit.npz`, `rollouts.npz`, `evidence.json`
with optimizer result, affine Jacobian, individual errors, data/model/code IDs.
This is an active feedback defect, not evidence that short-history observation
is the limiting issue: initialization already used full states.

Diagnosis: the linear speed/Pm/memory subsystem has positive growth even though
the equilibrium's angle-restoring stiffness temporarily stabilizes the full
linearization. The sine force is bounded at large angles and cannot suppress
an unstable homogeneous speed-feedback subsystem. Before another fitted ODE,
test the capacity of the fixed spatial direction with known future residual
forcing on TRAIN only, explicitly as a nondeployable diagnostic. This separates
output-geometry/state-coordinate capacity from autonomous-dynamics failure and
prevents spending effort on observers or more modes without evidence.
