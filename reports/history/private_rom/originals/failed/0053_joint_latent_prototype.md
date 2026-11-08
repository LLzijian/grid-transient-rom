# Loop0053 rejected: joint prediction works, latent centering gate fails

Baseline c1169a6; preregistration1adae8a; implementation976cbb1. Seed53,
Torch2.13.0+cpu/NumPy2.5.2, single-thread scientific venv. Command:
`experiments/train_joint_latent.py --output private_data/task2_joint_0053`.
1500 fixed Adam steps,338.57 s. Ruff/compileall/52 tests pass. User-owned
untracked loop_52.zip untouched; tracked worktree clean before fitting.

100 ms history leaves346 TRAIN and107 validation events;23/10 predecision
events excluded explicitly. TEST not read. Frozen M3 unchanged. One GRU8
coordinate, actual rank10 network-port action, one shared12-term ODE.
Independent arithmetic from saved trajectories reproduces headline metrics
and TRAIN latent moments to1e-12.

96 complete1 s validation prefixes (95 stable,1 unstable):

| Metric | M3 | Joint |
|---|---:|---:|
| Mean RMSE | .51755937 | .41157418 |
| Pooled RMSE | .66446164 | .52159648 |
| CVaR10 | 1.44633155 | 1.09852257 |

Mean/pooled improve20.48%/21.50%,8/8 validation bus groups improve. Twelve
unstable available prefixes improve mean1.197556 to.921856 and pooled
1.924730 to1.619810. These are development trajectory metrics, NOT8 s
classification, final M4 adoption or prospective evidence.

TRAIN first-second event-balanced q mean=-.532953802, variance=.779277195.
The mean violates preregistered abs(mean)<=.5. This is a centering failure,
not latent collapse and not evidence that joint discovery cannot work.
No threshold adjustment or post-hoc coordinate recentering rescues the run.
Autonomous-versus-sliding-encoder q RMS=.594464494 <std=.882766784 passes;
max numerical dt.01/.005 discrepancy1.04633e-5 passes. Every gate except
latent centering passes. Reject and revert implementation/preregistration.

Interpretation: learned electrical force direction has all-positive generator
entries; its mean negative q therefore carries a persistent reduction in
electrical load, consistent with0048. There is no instantaneous correction
(r_alg=0), so q must represent both algebraic load offset and dynamic memory.
The retained ODE has no q recovery term; this is an additional concern for
later long-horizon work, not an unregistered basis for this rejection.

Next Loop B: jointly separate an angle-dependent algebraic port from dynamic
q, with Pe/voltage/load reconstruction. Keep r=1,100 ms, fixed M3 and the
same short-rollout gates. Do not increase dimension or normalize this failed
model post hoc. Retrain the new physical decomposition from scratch.

Immutable model.pt,normalization.npz,trajectories.npz,evidence.json preserve
all predictions, true prefixes, encoded q, support, losses, identities and
failed gates under private_data/task2_joint_0053. Model SHA256
d28b9aacfa9e86764cd4fb588ad0e735256fc1a95d29d085f794ebc39dcbbc2b.
