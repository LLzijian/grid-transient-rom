# Loop0058 accepted: fixed TRAIN tail-risk short-rollout screen

## Question and hypothesis

Can the preregistered TRAIN event-tail mixture recover the tail/contribution
deficits of0057 without losing0056 short-rollout performance?

## Change

Only Lroll=.8 mean(event_loss)+.2 mean(largest ceil(.1 batch) event_loss).
No validation-weight tuning; all0057 gates unchanged, including covariance.
Preregistration2115271, documentation checkpoint5609b0d, implementation
4f07e78065ad1f6e5bc5aa61c30255885848b3cd. Frozen r=3,100ms GRU8,
conditioned coordinates, libraries/topology/M3, seed53 and1500-step schedule.
Command: scientific venv Python `experiments/train_joint_latent.py --output
private_data/task2_joint_0058`. Runtime213.69s; six-stage smoke, Ruff,
compileall and57 tests passed. No TEST payload read.

## Evidence

All22 preregistered gates pass.346 TRAIN/107 validation events retained under
the unchanged100ms eligibility rule;96 complete1s trajectories,95 stable and
one unstable. Per-event short unstable prefixes remain separately reported.

| 1s arm | Event mean | Pooled | Worst10% CVaR |
|---|---:|---:|---:|
| M3 | .517559371 | .664461645 | 1.446331546 |
|0058 full| .297089599 | .377884500 | .764601689 |
| All memory off | .638514083 | 1.286655402 | 2.919934661 |
| Component1 force off | .549306134 | 1.146151775 | 2.574922548 |
| Component2 force off | .357418679 | .482658396 | .970639050 |
| Component3 force off | .441625339 | .559934206 | 1.248335558 |

All8 validation-group means improve over M3. Compared with0056, mean,
pooled andCVaR improve; comparison is development evidence, not a fresh test.
TRAIN q means[-.016704,-.058394,-.007726], variances
[.993365,.998184,.979032], closureRMS[.872751,.889325,.645797].
Force singular values[3.367754,2.111979,1.776848]; covariance eigenvalues
[.971287,.994923,1.004371]. The latter satisfy the frozen gate by conditioning
but are NOT evidence of coordinate-invariant dynamical order.
Pe/voltage/load event-balanced MSE[1.043638,.204627,.045875] versus
equilibrium[1.678784,1.494429,1.424725]. Numerical max-event normalized
RK4.01/.005 difference2.7547944e-6 <.001. Independent saved-array arithmetic
audit reproduces all arm metrics, moments, force singular values and G_L V^2.

Decisive artifacts: `private_data/task2_joint_0058/{model.pt,normalization.npz,
trajectories.npz,evidence.json}`. Model SHA256
d82b1d8d75528eb7e55c28f5f5438270fd95eca2499b5afec5486b6894545f67.
M3 SHA2564928f40e54656fa05a5d478969f4d53f63e7551a824f761312efc793bc0621b4;
network SHA2561ad53be9faa158766f72558a461d2b5a7a53d8ac9a028b4a7ce7e26ed9172484.
Full per-event, group, horizon, loss-history, sparse coefficients and split
identities are retained in evidence.json, not duplicated here.

## Classification and interpretation

Accepted for its preregistered1s development screen only. Retain implementation;
no rollback is justified. This does not accept final M4, establish three
independent dynamic modes, imply long-horizon validity, or certify stability.
The single complete1s unstable event and repeated use of this development set
limit generalization claims. Historical0056/0057 failure conclusions are unchanged.

## Next implication

Per the latest user instruction, pause all model-complexity increases. Next
loop is only the coordinate-invariant order audit of unwhitened0056, with0057
as a secondary diagnostic. Do not switch the audit target to0058 based on this
new result. No r>3, new losses, encoder/library/topology changes or TEST access.
