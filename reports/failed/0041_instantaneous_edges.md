# Loop 0041 rejected: sparse static correction trades away a severe event

Frozen implementation/preregistration `75e1032`; scientific baseline `f4335cb`,
overview erratum `a084298`. Command: scientific venv Python
`experiments/test_instantaneous_edges.py --output private_data/task2_instantaneous_0041`,
BLAS/OMP 1 thread, NumPy 2.5.2. Ruff/compileall and all 48 tests passed.

All 369 TRAIN prefixes fit up to three existing-edge conservative corrections,
event-balanced least squares with fixed 1e-6 ridge and K+DeltaK>=0. Forward
selection picked zero-based complete-incidence edges [8,0,1], DeltaK
[-2.99143978,-2.29665592,-1.47975456]. M3 mechanical parameters and damping
unchanged. Pointwise full-residual energy explained: 8.79694%.

Frozen screen: combined validation at 50 ms (109 events, 95 stable/14 unstable),
2/5/8 s mean <=0.95 M3, pooled <=M3, CVaR <=1.05 M3, unstable-prefix mean
<=1.05 M3, no new false-stable/false-unstable events, finite autonomous 8 s
rollouts at RK4 .005. Original Pilot TRAIN scales fixed. Neither test read.

| Horizon | M3 mean | Candidate mean | M3 pooled | Candidate pooled |
|---|---:|---:|---:|---:|
| 2 s | .567316 | .547660 | .690290 | .655978 |
| 5 s | .448127 | .436357 | .531099 | .511806 |
| 8 s | .370718 | .360197 | .437194 | .420925 |

All three mean-gain gates failed (<5% gain). CVaR and unstable-prefix gates
passed, but candidate newly misses `pilot_0155`. False-stable count rises from
2/14 to 3/14, false-unstable stays 0/95. These are development observations,
not safety guarantees or independent OOS results. More expensive acceptance
gates were correctly skipped after the screen failure.

Reject and revert the implementation; keep immutable `evidence.json`,
`fit_statistics.npz`, `candidate.npz` and both per-event autonomous trajectories
in the private output directory. The candidate is not an accepted model.
The evidence does not justify sacrificing instability to slightly improve
average fit. Keep r_inst=0 for subsequent memory discovery; do not subtract
this rejected static fit. Next smallest hypothesis: the atlas's repeatable
zero-sum mode has a full-state electrical coordinate with reciprocal passive
driven dynamics, unlike a static reduction of restoring coupling.
