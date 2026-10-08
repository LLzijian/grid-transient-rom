# Loop0057 rejected: coordinate conditioning succeeds, tail contribution fails

Baselinef247247; preregistrationd13be30; implementationbea6787. Scientific
venv `experiments/train_joint_latent.py --output private_data/task2_joint_0057`,
seed53,1500 steps,581.70 s. Ruff/compileall/56 tests pass.346 TRAIN/107
validation,100 ms. No TEST or validation-derived reference statistics.

Frozen TRAIN conditioning succeeds: q means[-.030418,-.051124,.003696],
variances[1.003872,.999521,.980252], covariance eigenvalues
[.971306,.996427,1.015912]. Autonomous/encoder RMS[.876071,.870875,.629849]
passes. Force singular values[2.917278,1.771135,1.661109]. These are coordinate
conditioning checks, NOT proof of three necessary physical memory states.

96 complete1 s validation prefixes: mean=.33027932, pooled=.63116409,
CVaR10=1.10725072. Relative to M3 (.51755937,.66446164,1.44633155) the
basic gains pass, but0056 capacity-preservation and lower-rank pooled gates
fail. Worst event RMSE5.332709 again exposes a tail problem. Electrical
Pe/|V|/Pload MSE1.120821/.213905/.049038 each beats equilibrium-only.

Frozen individual memory-force removals (mean,pooled):
mode1 off(.521953,1.046460); mode2 off(.367751,.578845);
mode3 off(.488184,.804869). Thus mode2 improves average prediction while
worsening pooled error: the required each-mode contribution gate fails.
All-memory-off(.613373,1.142939). Do not delete mode2 post hoc or claim a
successful state-order selection from normalized covariance.

Independent saved-array audit verifies all headline/ablation/latent/electrical
values, force singular values and G_L V^2 reconstruction. A no-fit force
sanity check finds no negative implied total electrical torque on any1 s
rollout (minimum5.16704); pilot_0366 spans26.4369..92.9876. Do not invent
a negative-total-power explanation for its error.

Reject and revert implementation/preregistration; retain complete immutable
artifacts under private_data/task2_joint_0057. Next fixed-rank hypothesis:
mean-only rollout optimization underweights rare extreme rotor excursions;
add a modest TRAIN-only event-risk component to joint fitting/support discovery,
without labels, validation weighting, new terms or changed acceptance gates.
