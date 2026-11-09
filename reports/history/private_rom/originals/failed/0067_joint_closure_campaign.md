# Loop0067 — matched joint long-horizon closure

Classification: **rejected** for the fixed-budget J shortlist hypothesis;
not rejection of joint closure as a model family. Preregistration:
`reports/task2_joint_campaign_0067.md`; trainer16a197a, final L1 fix a09813e.

C (active Xi only) and J (12 parameter tensors jointly) each complete the
registered800 steps from S, seed67, unchanged objective and1/2/5s schedule.
Their800 sampled batches/endpoints match exactly; all frozen/inactive
parameters and source hashes pass independent checks. No nonfinite training,
f(0)=0 exactly; full33-state analytic/autograd errors<=5.69e-14. Quotient32
dominant real parts C-.139077/J-.160708 are local diagnostics, not regional
certificates. Training wall times1921.37/2151.48s are not inference timings.

| Validation quantity | S | C | J |
|---|---:|---:|---:|
| Complete1s pooled / CVaR | .373210 / .767441 | .404008 / .816487 | .411881 / .818630 |
| Complete5s mean / pooled | .288326 / .333571 | .291780 / .337179 | .299285 / .342581 |
| Complete5s CVaR | .628208 | .628538 | .624103 |
| Unstable prefix pooled / CVaR | 1.230200 / 1.667156 | 1.415804 / 1.947420 | 1.383506 / 1.942903 |
| Common-frequency5s pooled, rad/s | .452087 | .390790 | .309449 |
| Eligible misses / false LOS / false settled | 2 / 0 / 2 | 1 / 0 / 1 | 1 / 0 / 1 |

All107 eligible events retained; complete5/8s cohort95, eligible unstable12.
Raw denominator117 includes10 latency-uncovered unstable events (raw unstable22).
C/J each detect11/12 eligible,11/22 raw. Both have0 numerical-unresolved;
all488 genuine-prefix/complete numerical rows and LOS-required intervals
pass .005/.01, so no new finer run is triggered. No TEST access.

J fails clause(a):5s mean/pooled are1.02572/1.01602 times C, not<=.95.
It also fails S noninferiority at1s pooled/CVaR (1.10362/1.06670) and
unstable-prefix pooled/CVaR (1.12462/1.16540). Clause(b) fails: no fewer
misses than C. Safety counts pass, but cannot compensate these fixed gates.
J8s mean/pooled/CVaR .248274/.284003/.516908 adds no promotion basis.
The worst prefix pilot_0366 worsens S1.770545 -> J2.327760, despite
pilot_0155 improving1.130533 -> .632853; neither event is excluded/reweighted.

Evidence under `private_data/`: `task2_joint_{C,J}_0067`,
`task2_joint_{C,J}_eval_0067`, `task2_joint_{C,J}_l1_0067_original`,
`task2_joint_integrity_0067/{qa_joint67.json,joint67_jacobians.npz,comparison.json}`.
Independent reproducers `.loop-work/qa_joint67.py` and `compare0067.py`.
The comparison helper intentionally leaves training prerequisites unverified;
the separate checkpoint/trace QA above verifies them. Rejection is based on
the actual metric gates, not that missing helper input.22 targeted tests pass.
Independent cached-trajectory recomputation and count/cohort checks have0
discrepancies at1e-10 (actual maximum2.84e-14 for both); see
`task2_joint_integrity_0067/independent_metrics.json`, reproduced by
`.loop-work/independent_metrics0067.py`.

No candidate is deployed; S/0058/M3 are unchanged and C/J evidence retained.
J's20.8% common-frequency gain over C, with other metrics within5% of C,
supports exactly one third candidate J-S in the ORIGINAL library. Test whether
support rediscovery retains this gain while recovering S's short/severe-event
accuracy. No extension of0067, new loss, r>3, covariance gate or new energy
curvature term. B2 waits for this bounded Campaign A result.
