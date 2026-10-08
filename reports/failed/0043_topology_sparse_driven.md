# Loop 0043 rejected: sparse fit destabilizes the complete reduced ODE

Baseline8d1a393, frozen implementation/preregistrationcfd5ba8. Scientific venv
command `experiments/test_topology_sparse_driven.py --output
private_data/task2_topology_driven_0043`, BLAS/OMP1. Ruff/compileall/48tests pass.
All369 TRAIN prefixes; fixed0042 spatial direction/full-state coordinate and
r_inst=0. No validation or test payload read after the early local screen.

One decay plus at most3 drives from26 active-edge speeds,26 centered sines,
26 centered cosines,10 p components; event-balanced weak windows1/5/20, max40
each, fixed normalized ridge1e-6, decay bounded[.01,50]. Selected equation:

    a_dot = -.0770582875 a -23.0727445 edge_speed_8
            +19.9326636 edge_speed_43 -2.42301632 edge_speed_0.

Weak SSE .20408248 versus .42709953 free-decay null (52.22% improvement) passes
the>=5% fitting screen. But the full gauge-free30x30 Jacobian has leading real
eigenvalue +5.77998665/s and another +.05623350/s, failing <=-.001/s. Positive
isolated decay is plainly insufficient. No validation rollout was performed.

Revert implementation, retain private `fit.npz` (all weak equations, selected
support/coefficient vector, node direction, full Jacobian) and `evidence.json`
(eigenvalues, code/data/model identities). Do not use this ODE as a closure.

The first node direction is dominated by generator1 (v1=.9360); the fitted
signed speed feedback creates a severe interconnection instability. Next
smallest physical hypothesis keeps the same rank1 coordinate and3 speed terms
but imposes full coupled spectral stability during the4-parameter fit. This is
a new constrained hypothesis, not a lowered acceptance threshold or rescue of
this failed loop; it must independently pass the same autonomous rollout gates.
