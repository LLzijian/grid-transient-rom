# Loop0055 rejected: two-mode closure passes, hard-event prediction fails

Baselineaf6011a; preregistratione112f99; frozen implementationa77644d.
Scientific venv command `experiments/train_joint_latent.py --output
private_data/task2_joint_0055`, seed53,1500 fixed steps,188.26 s. Ruff,
compileall and54 tests pass.346 TRAIN/107 validation,100 ms, no TEST read.

Independent artifact arithmetic reproduces1 s/electrical/latent metrics to1e-12.
96 complete1 s validation prefixes: mean=.38388496, pooled=.64170101,
CVaR10=1.18339356 versus M3 .51755937/.66446164/1.44633155. All8 groups
improve mean; .2/.5 s pooled=.21672634/.32553737. Yet1 s pooled improves
only3.43% versus M3, below5%, and is worse than the best rank-one model.
CVaR also exceeds the rank-one preservation gate. Reject, no event removal.

Both q moment/consistency gates now pass: means[.25556163,-.24982662],
variances[.20024090,.44230617], autonomous/encoder RMS[.36344692,.47340545].
Force singular values[4.402335,1.684345], ratio.3826;24 Xi and12 algebraic
terms total. Memory-off mean/pooled=.51521418/.76153176, so memory provides
real short-horizon contribution. Pe/|V|/Pload MSE1.29875/.165833/.046271
all beat equilibrium output. Max numerical discrepancy8.93120e-6.

Worst event pilot_0366 (validation bus26,unstable) has1 s RMSE5.016696,
versus M3 2.874194 and memory-off5.060472. An exploratory FROZEN algebraic-
off ablation reduces it to2.846122, but worsens overall mean to.4270865
(pooled.5549753,CVaR1.1585564). This diagnoses an algebraic/dynamic allocation
tradeoff, not a deployable post-hoc rescue. Both components off reproduces
M3 to1e-10. Evidence is in private_data/task2_joint_0055_diagnosis.

TRAIN q covariance eigenvalues[.02512776,.61741931] reveal strong coordinate
correlation despite independent force columns. Do not overclaim independent
physical latent states or final order discovery from the passed marginal gates.

Retain model, normalization, per-event predictions/truth/encoded coordinates,
electrical diagnostics and identities in private_data/task2_joint_0055.
Revert implementation/preregistration. Next is the prescribed r=3 from-scratch
joint comparison: test a third electrical direction against the observed tail
tradeoff, with explicit nonredundancy and stronger-than-lower-rank prediction
requirements. Do not delete the hard event or extrapolate1 s to stability.
