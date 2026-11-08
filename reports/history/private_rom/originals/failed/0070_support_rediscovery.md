# Loop0070 — original-library joint support rediscovery

Classification: **rejected**, for the fixed J-S shortlist hypothesis.
Preregistration `reports/task2_support_rediscovery_0070.md`; implementation
checkpoint0e72bb8. Campaign A is now closed: C, J and J-S received their
fixed screens; no further support/weight sweep or extension.

J-S independently starts from S and completes800 steps in2372.97s:200
open, one TRAIN-standardized top48 prune,600 refit. All36 inherited terms
plus12 new terms survive. Independent800-step sampling/phase/frozen/source
QA passes;512 TRAIN pruning windows,225 scores and exact ranking verified.
Final f(0)=0; analytic/autograd Jacobian error2.84e-14; all32 gauge-reduced
poles have negative real part (maximum-.157440/s), a local diagnostic only.

| Metric | S | C0067 | J-S0070 |
|---|---:|---:|---:|
| Complete5 mean | .288326 | .291780 | .298201 |
| Complete5 pooled | .333571 | .337179 | .341565 |
| Complete5 CVaR | .628208 | .628538 | .623425 |
| Unstable-prefix pooled | 1.230200 | 1.415804 | 1.298018 |
| Unstable-prefix CVaR | 1.667156 | 1.947420 | 1.835287 |
| Common5 pooled, rad/s | .452087 | .390790 | .310493 |

The mandatory5s mean/pooled ratios versus C are1.022006/1.013007, not<=.95.
S noninferiority also fails1s pooled (ratio1.066865), unstable pooled
(1.055128) and CVaR (1.100849). J's common-frequency benefit is retained:
.310493<=1.05*.309449. No other fixed metric/safety comparison fails.
Thus sparse support rediscovery reduces J's severe-prefix harm but does
not resolve the common-motion/COI-tail tradeoff within this budget.

All107 eligible/117 raw validation events remain, including12 eligible and
22 raw unstable events;10 latency-uncovered events stay in raw denominators.
Detection11/12 (11/22 raw), miss expansion_0155, false LOS0, false window-
settled1, numerical unresolved0. All488 genuine-score rows converge
(maximum7.60e-6) and all LOS-required intervals pass; no new finer run.
Unobserved8s extrapolations are not truth or stability evidence. Complete8
mean/pooled/CVaR .247631/.283461/.516834 (95 events), diagnostic only.

Artifacts: `private_data/task2_joint_JS_0070`, `task2_joint_JS_eval_0070`,
`task2_joint_JS_l1_0070_original`, `task2_joint_integrity_0070`.
Model SHA f802607f0886c24e1a40a3499ada88e90e80555b994348101c06fdfccb81ac84.
Independent cached metric differences<=2.85e-14, zero discrepancies.
The reused comparison helper calls this candidate J and intentionally
leaves training verification unset; separate integrity evidence supplies it.
Root applies0070's additional J common5 bound above, not0067's alternative gate.

S/0058/M3 are unchanged; rejected checkpoints/reproducers remain evidence,
not deployed replacements. This finite search does not falsify r3 or the
entire library. Next is one small joint nonlinear-memory capacity probe
under after66: the original q RHS is linear in q with constant Aq, while
the earlier0060 teacher froze the encoder/mappings. Test the missing joint
nonlinear closure before adding dimensions or undertaking B2 corrections.
