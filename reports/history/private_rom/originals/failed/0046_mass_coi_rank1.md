# Loop 0046: correct COI geometry, rejected rank-one capacity hypothesis

Baseline715093e; frozen implementation/preregistration8671b1f. Command:
scientific venv `experiments/diagnose_mass_coi_geometry.py --output
private_data/task2_mass_coi_geometry_0046`, BLAS/OMP1. Ruff/compileall/48tests pass.
All369 TRAIN moments define the basis;351 eligible TRAIN forced prefixes,
no validation/test and no autonomous-memory/classification claim.

The identity A=diag(1/M)-11^T/sum(M), r_rel=M A r, is numerically verified:
relative-acceleration discrepancy5.68e-14, incidence reconstruction2.46e-16,
representative half-step discrepancy2.83e-7. First acceleration-energy mode
explains65.6819%, second16.9659%. This is not sufficient dimension evidence.

Full COI-force injection passes all descriptive capacity gates:8s mean.396849,
pooled4.244037,CVaR2.752014 versus M3 .729112/6.252634/4.389570. Unlike arithmetic
force subtraction it preserves the relative acceleration; omitted uniform
frequency can still influence the autonomous governor and severe events.

Rank-one COI force fails5/8s trajectory gates:8s mean.948394,pooled7.593100,
CVaR6.941800. Its2s, cross-group>=18/24 and unstable-prefix gates pass, but those
cannot rescue the tail/long-horizon failure. No rank-one direction is adopted.

Revert the attempted implementation, retain the mathematical diagnosis and
immutable private `geometry.npz`, `trajectories.npz`, `evidence.json` with all
group flags, forcing, trajectories and code/data/model identities. Do not
increase r solely because the second variance fraction is substantial. Next
state-discovery work must establish a second independent reproducible full-state
dynamic component and diagnose repeated rank1 defects before testing r=2.
The pending raw-data-only frozen-test duration supplement is also required:
current fresh-test predictive cohort has no unstable events.
