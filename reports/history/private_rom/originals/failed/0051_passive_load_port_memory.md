# Loop 0051 rejected: load coordinate is repeatable but needs additional driving

Baselineee80844; frozen implementation/preregistrationdd5b169. Scientific venv
`experiments/test_passive_load_port_memory.py --output
private_data/task2_load_port_memory_0051`, BLAS/OMP1. Ruff/compileall/48tests pass.
All369 TRAIN events; no validation or TEST read by the early screen.

Physical coordinate a=Pload-Pload_star; nonnegative unit-sum generator-port
allocation v=[.235902,.074809,.094693,.079421,.080374,.086237,.064970,.058911,
.106607,.118076] explains69.1484% of total residual force energy. Port
reconstruction error2.78e-17; leave-group alignment gate passes. This is a
full-state/all-bus voltage coordinate, not a history observer or yet a proven
memory closure. r_inst remains0.

Reciprocal model a_dot=kappa v^T omega-lambda a: constrained fit kappa3.29825508,
lambda.01000000000065 hits the forbidden lower rate boundary. Weak SSE.99196676
versus free decay1.05710514 (free rate.24328428); unconstrained fit
kappa4.77294475,lambda-.31340720. Positive reciprocal speed coupling is now
supported, unlike0042, but a stable speed-only relaxation is not supported by
this fit. Rate-interiority gate fails; no autonomous validation is run.

Revert implementation; keep `fit.npz` and `evidence.json` with full weak equations,
physical direction and code/data identities. Do not accept negative recovery or
relax the bound. Next smallest physical hypothesis: add one bounded network
cosine drive so this voltage/load coordinate relaxes toward an angle-dependent
level. Retain positive reciprocal speed coupling, stable recovery and full
coupled local stability; unlike0044, its speed/Pm/memory homogeneous subsystem
must remain stable so bounded angle forcing cannot create exponential runaway.
