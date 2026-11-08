# Loop0074 — rejected single r4 sparse-memory candidate

Question: can one extra electrical-memory channel resolve the common-motion
versus COI/severe-prefix tradeoff? Preregistration: `reports/task2_rank4_0074.md`.
Independent S initialization; implementation ca8766d; seed67,800 steps,
200 open/600 refit, one top48 pruning and Adam restart. No rescue/sweep.

Integrity:68 targeted tests and two-step TRAIN smoke pass. Formal2315.93s;
model SHA c7abb069fa9087cf3846102468a052d1bacf7d539c6e4a52db085be1eb35eabd.
All800 matched draws,512 pruning draws,12 allowed tensors, frozen/inactive
parameters, source hashes and f(0) pass independent checks. Support48 with
columns16/12/13/7; q4 self-decay is pruned, seven angle-driven terms remain.
Full34 Jacobian analytic/autograd error1.15e-15; quotient33 max real-.025086.
Cq singular values4.56572/3.35136/2.06732/.703964. Aq alone includes zero;
it is not the coupled-system spectrum. No regional stability certificate.

Final VAL evidence (mean/pooled/CVaR10):

| Cohort | n | Mean | Pooled | CVaR10 |
|---|---:|---:|---:|---:|
| Complete1s |96|.312652|.396226|.790666|
| Complete2s |95|.340563|.391600|.732554|
| Complete5s |95|.298463|.341536|.622196|
| Complete8s |95|.248027|.283844|.517828|
| All unstable genuine prefixes |12|.710154|1.280582|1.818313|

5s mean/pooled ratios to J-S0070 are1.000880/.999917, failing<=.95.
S1s pooled ratio1.061669 and unstable-prefix CVaR ratio1.090667 fail<=1.05.
Other registered noninferiority conditions pass, including common5
.311185rad/s (1.005612 times J0067). These do not compensate failed gates.
L1 detects11/12 eligible unstable (11/22 raw); miss expansion_0155,
falseLOS0,false-window-settled1,unresolved0.107eligible/117raw/10latency
remain explicit. All488 genuine-score rows and107LOS intervals pass
numerics, max genuine step discrepancy7.5774e-6; no finer run. Independent
metric differences<=1.12e-16, common-frequency differences<=5.69e-14.

Registered direct-q4-output ablation keeps cached initial states and all
internal dynamics. Genuine numerics pass. Removing this output changes5s
mean/pooled/CVaR by-.000038/+.000365/+.001909 and unstable pooled/CVaR by
+.008310/+.010072; common5 IMPROVES from.311185 to.305874rad/s. Thus this
channel makes small, mixed contributions, not the required tradeoff resolution.
This is not structural state removal or proof of a minimal latent order.

Classification: rejected fixed-budget algorithm hypothesis, not r4-family
impossibility. Valid implementation/training but failed prospective gates.
No3-seed confirmation, second r4 candidate, dimension increase or promotion.
S/0058/M3 were never overwritten; candidate remains quarantined as evidence.
No shared physical ODE changed; retain the isolated reproducer and tested
rank-aware evaluator. TRAIN alone enters fit/pruning; inherited reader loads
unused VAL records. No fresh TEST access. All jobs for0074 are complete.

Evidence: `private_data/task2_rank4_0074`, `task2_rank4_0074_eval`,
`task2_rank4_0074_l1_original`, `task2_rank4_integrity_0074`,
`task2_rank4_ablation_0074`. Next: diagnose physical/data limitations under
after66, beginning with existing TRAIN boundary coverage rather than more capacity.
