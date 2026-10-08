# Loop 0045 accepted: force-energy fit is not a reliable coordinate criterion

Baseline1b4a9a3, frozen implementation/preregistration69776ac. Scientific venv
command `experiments/diagnose_topology_force_capacity.py --output
private_data/task2_force_capacity_0045`, BLAS/OMP1. Ruff/compileall/48tests pass.
351 TRAIN events (311stable/40unstable),24groups,18 pre50ms outcomes excluded
only from this start-time diagnostic. Neither validation nor test read.

All forced arms consume known FUTURE information. No autonomous-memory,
observer, classification or OOS claim is made. Each actual available prefix is
integrated without padding. Full-force+true-Pm numerical control has prefix
RMSE median .00024159,p95 .00076208,max .0252226; representative half-step
maximum relative discrepancy9.99e-7. All registered diagnostic gates pass.

| Future forcing arm | 8s mean | 8s pooled | 8s CVaR10 |
|---|---:|---:|---:|
| none (M3) | .729112 | 6.252634 | 4.389570 |
| fixed rank1 true force | .651609 | 4.323214 | 3.270780 |
| fixed rank1 three-state coordinate | .868388 | 5.057603 | 5.165907 |
| all Euclidean zero-sum force | .902640 | 4.318781 | 5.773923 |
| full residual, autonomous Pm | .142239 | .169103 | .330304 |
| full residual and true Pm (control) | .000295 | .000373 | .000766 |

TRAIN M3 pooled loss is dominated by stable expansion_0094 (bus30), whose
8s RMSE is109.964. Do not remove it, but do not let its improvement imply
cross-group success. Rank1 true forcing passes the preregistered aggregate
capacity screen, yet improves mean in only3/24groups and worsens median event
error by18.2%. Three-state forcing likewise improves only3/24groups and fails
the aggregate gate. All Euclidean-balanced forcing improves4/24groups; full
residual improves24/24groups, median event ratio .4112. Unstable-prefix mean
M3/rank1true/rank1hidden/balanced/full =1.3471/1.3962/1.4291/1.5521/.2362.

Conclusion: full residual has strong corrective capacity, but neither raw-force
energy dominance nor79.98% amplitude explanation establishes a useful closure.
Current rank1 geometry and state coordinate both need scrutiny before another
ODE/observer. A Euclidean common force is NOT irrelevant to relative acceleration
when inertias differ. Next smallest physical hypothesis: use mass-consistent
COI acceleration geometry, r_rel=r-M*sum(r)/sum(M), which remains zero-sum and
incidence-representable, rather than subtracting the arithmetic mean force.
This changes neither frozen M3 mechanics nor the scientific objective.

All per-event forcing/time/predictions are saved in immutable
`trajectories.npz`; `evidence.json` includes raw records and code/data identities.
