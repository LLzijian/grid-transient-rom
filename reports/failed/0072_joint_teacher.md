# Loop0072 — joint nonlinear-memory probe: rejected

Question/hypothesis: can the single width16 joint teacher resolve the
common-motion versus COI/severe-prefix tradeoff? The fixed screen is
`reports/task2_joint_teacher_0072.md`; no thresholds or budget changed.

Change: independently start from S; jointly train the twelve J tensors and
1264 zero-initial-output teacher parameters, retaining r3/36-term support,
100ms initialization, exact0067 objective/sampler and800 Adam steps.
Implementation checkpoint aa32f17;3188.14s; model SHA
a52074b39e2f45b1bedb58b3fc2595e8643eb7ece78c4774261bfe7a1e7213c0.

Evidence: complete5s mean/pooled/CVaR=.297628/.340979/.622101.
Mean/pooled ratios to matchedJ0067=.994466/.995324, failing both .95 gates.
Against S,1s pooled/CVaR ratios1.094037/1.061154 and unstable-prefix
pooled/CVaR ratios1.124105/1.158365 fail1.05 noninferiority. Common5 pooled
improves J.309449 to .301263rad/s; this does not compensate safety-tail harm.
8s mean/pooled/CVaR=.246963/.282795/.515698 (diagnostic).

All107 eligible/117 raw validation events retained (10 latency-uncovered).
11/12 eligible unstable detected (11/22 raw); miss expansion_0155;
falseLOS0, false-window-settled1, unresolved0. All488 genuine-score rows
and107 LOS numerical intervals pass; maximum genuine discrepancy7.60749e-6.
Unobserved fixed-horizon tails are not scored truth or convergence evidence;
no finer run was triggered. Matched800 draws, frozen/inactive/source checks
and f(0)=0 pass. Full teacher Jacobian/autograd error4.26326e-14;
32-state maximum real pole=-.151862 (local only). Independent cached metric
recalculation:225 comparisons,0 discrepancies, maximum difference5.69e-14.

Classification: rejected finite-budget algorithm hypothesis, not proof that
r3 or nonlinear closure cannot work. The teacher also changes local linear
dynamics; its failure does not isolate nonlinear representational necessity.
No confirmation seeds, distillation, width/weight/budget sweep or promotion.
S/0058/M3 never changed; rejected checkpoint quarantined under its experiment
directory, reproducer retained, no shared deployed ODE change to revert.

Artifacts: `private_data/task2_joint_teacher_0072`, corresponding
`task2_joint_teacher_0072_eval`, `task2_joint_teacher_0072_l1_original`,
`task2_joint_teacher_integrity_0072`. Fresh TEST remains unopened.
Next implication: proceed to a bounded B2 dynamics–energy compatibility
experiment; do not add another teacher or repeat fixed-S curvature search.
