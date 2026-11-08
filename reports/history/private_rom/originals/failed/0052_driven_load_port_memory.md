# Loop 0052 rejected: bounded cosine forcing still pins recovery to its floor

Baseline08c6906; frozen implementation/preregistration1b3273f. Scientific venv
`experiments/test_driven_load_port_memory.py --output
private_data/task2_driven_load_port_0052`, BLAS/OMP1. Ruff/compileall/48tests pass.
All369 TRAIN events, fixed0051 load coordinate/allocation, no validation/TEST.

Selected one cosine drive on zero-based edge1. SLSQP converged in7iterations:
kappa5.30169051,lambda.01,g23.53846284. Weak SSE.91797640 improves over0051
.99196676 by7.46%; local spectral abscissa-.03427966/s and homogeneous
speed/Pm/a abscissa-.15632033/s pass the stability screens. But lambda again
hits its forbidden lower bound, failing the preregistered interior-recovery
criterion. No rollout or observer is justified; no rate-bound relaxation.

Revert implementation; retain `fit.npz` and `evidence.json`, including exact
selected weak equations, affine Jacobians, optimizer and code/data identities.

The proposed chain-rule continuation was not executed. The user's new
`docs/TASK2_after_loop_52.md` supersedes it: stop defining load power as a
dynamic coordinate; jointly discover a history encoder, latent sparse ODE,
and full-network spatial action. Start with r=1, 100 ms observations and
1 s autonomous rollout. Load power becomes an auxiliary algebraic output.
