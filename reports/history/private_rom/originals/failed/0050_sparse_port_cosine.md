# Loop 0050 rejected: strong pointwise cosine fit does not close dynamics

Baseline48ff6fa; frozen implementation/preregistration2215776. Scientific venv
command `experiments/test_sparse_port_cosine.py --output
private_data/task2_port_cosine_0050`, BLAS/OMP1. Ruff/compileall/48tests pass.
All369 TRAIN prefixes, fixed M3 mechanics and equilibrium. No TEST payload.

Three positive current-angle cosine terms on existing zero-based edges[1,8,4]
with coefficients[2.96700399,4.41821552,3.45548192] explain42.2178% of full
residual force energy. Their shared-endpoint power vectors reconstruct through
the actual generator-port incidence to2.22e-16. This is not pairwise conservative
force and was not claimed to preserve monotone H3.

One autonomous109-event validation screen fails:2/5/8s mean/pooled/CVaR gates
all fail;8s mean2.917112,pooled10.267207,CVaR24.877626 versus frozen M3
.370718/.437194/.874259. Six new false-unstable events and one new false-stable
event (pilot_0366) increase errors to6/95 and3/14 respectively. Unstable-prefix
mean passes, but cannot rescue the trajectory or safety failures. More expensive
acceptance gates are skipped. No test or observer was used.

Revert implementation; retain immutable `model.npz`, `fit.npz`, `rollouts.npz`,
`evidence.json` under the private output. Keep r_inst=0. The evidence rejects
this static-angle approximation, not the physical load-power mechanism0048.
Next smallest hypothesis: a single measured/full-state load-power deviation
coordinate, distributed over actual generator ports, supports reciprocal
recovery dynamics. Its force allocation, stable ODE and autonomous contribution
must pass before any short-history observer is trained. This uses nonzero-net
physical load exchange, unlike the rejected generator-only zero-sum channels.
