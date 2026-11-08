# Loop0077 — three-condition numerical probe: rejected

## Question / hypothesis

Does the fixed .0025/.00125/.000625s refinement make all three selected
conditions meet the preregistered trajectory/LOS criteria, with reproducible
old results? Protocol: `reports/task2_boundary_probe_0077.md`.

## Change and evidence

Exactly12 attempts from implementation cfd90439940e1d73e9038a152ff06ec339140bf3;
nine complete, three tol1e-9 arms fail dynamic initialization before any
time step. No replacement runs. All three .0025/tol1e-7 repeats reproduce
old0075 fine trajectories, labels, grids and stop times with zero error.

| Case | Last-pair COI error | Last-pair common error | COI ratio to previous pair | Last-pair gate |
|---|---:|---:|---:|---|
| bus2 / boundary_0000 | .0000427412 | .00000423022 | .630925 | pass |
| bus5 / boundary_0002 | .000338594 | .000231969 | .195084 | pass |
| bus28 / boundary_0015 | .0564899 | .00281045 | .582252 | fail |

All errors decrease, and labels agree at all completed steps. Bus28 LOS
times since clearing are1.2301/1.2101/1.1901s; each adjacent difference is
.02s (floating arithmetic handled as in0075). Its trajectory errors still
exceed .001, so the ALL3 hypothesis fails irrespective of LOS tolerance.

Actual fault/clearing callbacks occur exactly at scheduled times; shortened
event-bracketing steps are recorded, not hidden adaptive failure recovery.
All tight-tolerance attempts have initial algebraic residual max4.21569e-9,
above1e-9, and zero attempted steps. The initial differential residual is
the stored quantity after ANDES masks no_check_init rows, NOT an unmasked
raw derivative. Thus this arm cannot identify tolerance effects during
integration. Pre-fault/event-neighborhood residuals remain in the artifacts.

Evidence: `private_data/task2_boundary_probe_0077/report.json`, twelve arm
JSONs, nine trajectory NPZs, and `independent_qa.json`; logs in
`.loop-work/boundary0077.log` / `boundary0077.stderr.log`. Independent
selected-TRAIN-only recomputation matches decisive metrics exactly (max
discrepancy0). Source hashes unchanged;24 related tests pass. No TEST read.

## Classification / interpretation

Rejected local convergence hypothesis, with useful diagnostic evidence.
Reproducibility and event timing checks do not explain away the bus28
trajectory discrepancy. Its decreasing differences suggest step sensitivity,
but do not establish an asymptotic convergence order or a ground truth.
The tighter-tolerance arms are numerical UNKNOWN, not physical instability.

## Next implication

0075 remains rejected; no sample enters TRAIN and no accepted model changes.
Keep the observational generator/probe as reproducible diagnostic code;
there is no rejected predictive implementation to revert. Main work continues
with0076 then the matched original-TRAIN structure campaign0078. Any later
boundary study must separately preregister consistent initialization and
integration tolerances; do not append a fifth arm to this closed probe.
