# Loop0056 rejected: strong prediction capacity, correlated latent coordinates

Baselinee2e6ca7; preregistrationbe3d11c; implementationa196841. Scientific
venv `experiments/train_joint_latent.py --output private_data/task2_joint_0056`,
seed53,1500 steps,523.17 s. Ruff/compileall/55 tests pass.346 TRAIN/107
validation at100 ms; every eligible event retained; no TEST payload read.

Independent saved-array audit verifies prediction/electrical/latent metrics,
force singular values, covariance and the decoded G_L V^2 identity.
96 complete1 s prefixes: mean=.30593176, pooled=.40097654,CVaR10=.82450047
versus M3 .51755937/.66446164/1.44633155. All prediction and memory-
contribution gates pass, including improvement beyond the best lower-rank
figures. This supports THREE-MODE SHORT-ROLLOUT CAPACITY, not final order,
physical independence, stability or prospective generalization.

q means[.37024726,-.07884917,.47308169], variances
[.44316819,.11550816,.50331993] pass marginal gates. Autonomous/encoder RMS
[.28353016,.25583753,.34125450] passes each std threshold. Force singular
values[3.741095,2.434258,2.285646] show independent available spatial action.
But TRAIN q covariance eigenvalues[.00716453,.03216749,1.02266425] have
min/max=.0070057 <.05. This is the ONLY failed gate. Reject, rather than
relax the nonredundancy threshold or whiten the finished model to claim it passed.

Interpretation is limited: correlated excitation of an observed state cloud
does not by itself prove an ODE state is mathematically removable. Conversely,
marginal q variances and independent force columns do not establish independent
electrical coordinates. Both limitations are retained with the positive
prediction evidence. No claim that r>3 is needed; do not enlarge the model.

Next smallest experiment holds r=3 and the physical library fixed, but uses
TRAIN-reference coordinate whitening DURING joint optimization, followed by
the same sparse-support selection. Freeze normalization for deployment.
This tests whether poor coordinate conditioning obstructs sparse independent
discovery; whitening by construction is not evidence of physical order, so
individual mode-removal tests must accompany the existing gates. Retrain from
scratch; do not transform this rejected checkpoint into an accepted model.

Retain immutable model.pt,normalization.npz,trajectories.npz,evidence.json in
private_data/task2_joint_0056; revert implementation/preregistration. Model
SHA2568c6e666bb4683311c89d6c564273d9d3ab5f0f081cfdc268dd3c2663f65c061d.
