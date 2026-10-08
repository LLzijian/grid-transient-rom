# Loop0086 — aggregate improvement did not preserve severe-event safety

Work type: diagnostic intervention/model screen. Result: **rejected**.
Model promotion: **none**. Gates unchanged; no further0086 candidate authorized.

Preregistration c3f3471/4696c24; implementation70d394d. Command:
Python experiments/fit_guarded_regression_0086.py --output
private_data/task2_guarded_regression_0086. Root session79622 exited0.
One scientific SDP, one candidate, one conditional rebuild and one forecast
pair. Main311.0795s (solve1.2296s, rebuild231.4083s, forecast40.4562s);
six synthetic tests passed (root2.62s). Original TRAIN only:346 eligible,
311 stable/35 unstable,24 groups;46514 genuine conditional windows.
Reader materialized unused VAL; no validation scoring or TEST access.

Clarabel returned optimal after15 iterations, internal solve.2700s.
Primal/dual -.272829644609/-.272829644944; stationarity6.61e-10,
complementarity/gap3.35e-10. Minimum primal/dual eigenvalues
7.31e-11/1.86e-12. All registered optimality and0085 feasible-reference
comparison gates pass. This establishes numerical adequacy of this one convex
surrogate, not the original nonconvex joint optimization.

Actual reloaded field matches the affine Jacobian to8.24e-18 relative.
Fixed-P decay maximum -.510000000072, spectral abscissa -.1404271414/s;
origin/gauge/kinematic/export/power/frozen-state QA pass. This is local,
not a finite-amplitude stability region or full-order certificate.

| Autonomous quantity | Baseline | Candidate | Ratio | Gate |
|---|---:|---:|---:|---:|
| Average1/2/5s risk | .893819553 | .782855270 | .875853820 | <=.90 pass |
| Unstable5s-prefix mean | .665464698 | .700168168 | 1.052149227 | <=1.05 fail |
| Unstable5s-prefix worst4 | 2.303873239 | 2.682967397 | 1.164546448 | <=1.05 fail |

Candidate1/2/5s risks .590907250/.990665125/.766993435. All346 events
retained per horizon; no artificial tails. Numerical maxima COI3.643e-6,
common2.408e-7/Pm1.632e-6 pass. Conditional total/physical/unstable RMS
ratios .977232207/.976990159/.977174043; old descriptive.90/1.05/1.05
references remain unchanged (total fails). These do not reopen0082.

Artifacts: `private_data/task2_guarded_regression_0086/`; candidate model SHA
f47fc37d09a1776e8946ad22e6518b160f2a65df73a318bbe5a28c7c974ac60a.
Independent evidence: `private_data/task2_guarded_regression_0086_independent/`.
`solve_independent.json`: KKT/conditional/guard/mapping checks pass, maximum
relative3.19e-16/absolute2.84e-14,6.51s,8 synthetic checks.
`forecast_independent_v2.json`:1038 event metrics per arm, aggregates/severe
and346 numerical rows per arm reproduced to7.11e-15,2.19s,15 synthetic checks.
First forecast checker exited before metrics on an inapplicable inherited
0082 audit-output path; `checker_attempt1_failure.md` preserves the issue.
Root removed only that dependency; no formula/tolerance/production change or
scientific rerun. Independent arithmetic is not independent confirmation data.

Stage closed: its sole candidate and all scientific budgets are spent.
The stable direction improved aggregate prediction by12.4%, but cannot trade
away severe-event gates. Fixed-P conservatism remains; failure does not reject
the model family or stable coefficient refinement. No accepted/default model
changed; rejected standalone evidence retained, no rollback needed.

Root disposition: close the conditioned-regression branch rather than tune
its ridge, stability margin, weights or scalar blend.0082/0085/0086 have
distinguished an unstable unconstrained solution, an overly small stable
step, and a stable aggregate improvement with severe harm. Original causal
joint-rollout optimization adequacy remains unresolved. A new finite,
same-complexity direct-objective stage requires independent challenge and
prospective registration; it is not an extension or rescue of these loops.
