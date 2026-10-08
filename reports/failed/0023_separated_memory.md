# Loop0023 rejected: static fit does not transport into autonomous rollout

Start81bc7d0; experiment71d87ef; preregistration in that commit's
`reports/task2_m4_separated_q1.md`. Command:
`python experiments/run_private_m4.py --output private_data/task2_m4_0023`.
48tests/Ruff pass; both scientific and memory-ablation screens fail.

Training instantaneous coordinate explained99.1285%, but the static-only ablation
pooled2/5/8s is2.943954/11.370539/23.701831, versus acceptedM3
.740507/.565149/.464055. FullM4 is3.643486/13.678155/26.378908; six new false-
unstable events. q observer correlation falls to.572576 after removing the
instantaneous component. q decay again hits.01/s. No implementation adopted.

The leading training residual mode is neither a sufficient electrical closure nor
a reliable memory coordinate; adding an unconstrained rank-one angle map fits
on-trajectory force but damages the free vector field. Its99% teacher-state fit
must not be reported as cross-disturbance prediction success. Preserve888 full/
ablation per-event records in `0023_m4_evidence.json`, revert all rejected code.

Next smallest physically motivated hypothesis: fit a reciprocal passive scalar
memory directly against event-balanced causal rollouts, allowing its loading and
history map to target predictive inter-area dynamics rather than selecting the
largest raw-force SVD mode. Freeze all acceptedM3 parameters, omit the rejected
static correction, keep qdim1 and a40-feature observer. This is not evidence that
more hidden dimensions or a larger encoder are necessary.
