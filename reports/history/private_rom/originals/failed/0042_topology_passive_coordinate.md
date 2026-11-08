# Loop 0042 rejected: discovered force coordinate is not reciprocally passive

Frozen implementation/preregistration `90be88b`, baseline `25cab2e`.
Scientific venv command `experiments/test_topology_passive_coordinate.py
--output private_data/task2_topology_passive_0042`, BLAS/OMP 1 thread.
Ruff, compileall and all48 tests passed. All369 TRAIN events and24 groups;
neither validation nor either test payload was needed by the early screen.

Fix v to atlas0040 first balanced force direction and a=-v^T r_e; r_inst=0.
Greedy event-balanced full-state regression (no rotor delta/omega, no intercept,
fixed1e-6 ridge, at most3 states) selected deviations from equilibrium:

    a_hat = 8.25095240 GENROU_2/e2q
           -19.38469341 IEEEST_1/LL1_x
           -33.47222696 IEEEX1_1/LL_x.

This explains79.9843% of uncentered amplitude energy; coefficient signs persist
in24/24 leave-one-group refits. Coordinate screen (.5 explained energy and.8
sign consistency) passes. This is useful TRAIN state-association evidence, not
an observable/deployable state or an accepted closure.

The passive amplitude equation a_dot=kappa v^T omega-lambda a requires kappa>0.
Unconstrained weak fit gives kappa=-11.0563375, lambda=.305256917. Constraining
kappa>=0 and lambda in[.01,50] gives kappa=5.46e-19, lambda=.573324277:
exactly the free-decay null's weak SSE .4270995273. Thus both positive-coupling
and>=5% driven-contribution gates fail. No validation rollout was justified.
Flipping coordinate sign cannot repair the reciprocal product's wrong sign.

Revert implementation and retain immutable `evidence.json`, `coordinates.npz`,
`fit.npz` in `private_data/task2_topology_passive_0042`. These preserve the state
coordinate, basis, per-event traces, weak equations, model/data hashes and code
identity. Do not retain the passive candidate as a model.

Interpretation is limited: this particular atlas-leading coordinate includes
PSS/exciter dynamics and is inconsistent with a single reciprocal speed-driven
storage channel. It does not disprove all passive representations. Next smallest
hypothesis under the new plan: retain this observed rank-one spatial/state
coordinate and identify a sparse signed driven ODE from a small topology/state
library, first checking full coupled local stability, then autonomous rollout.
