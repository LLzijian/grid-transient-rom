# Loop 0020: explicit Pm passes the development screen

Accepted against the prior `reports/task2_m3_pm.md` criteria, not a final model.
Start4138577; immutable experiment55d3c63; command:
`python experiments/run_private_m3.py --output private_data/task2_m3_0020`.
Same218train/70prospective validation, no test payload, unchanged electrical K/D.
Fit once in2.23s. All10 T=.197–.459s, positive droop K=.284–.552 in systempu per
rad/s; no time-constant guard bound reached. P_ref fixed by unique equilibrium.

| Horizon | M2 pooled trajectory | M3 pooled | M3 mean | M3 CVaR10 | True-speed Pm mean / ZOH |
|---|---:|---:|---:|---:|---:|
| 2s | 1.085943 | .740507 | .611103 | 1.536925 | .120800 / .436915 |
| 5s | 7.493971 | .565149 | .478969 | 1.118418 | .086092 / .619394 |
| 8s | 27.492233 | .464055 | .395093 | .908403 | .069855 / .604085 |

Both predeclared identification and coupled contribution screens pass. All70
fixed8s predictions finite; false-unstable0/61 and false-stable0/9. This is a
development-split observation, not a certified classifier. Only five independent
disturbance groups: even an ideal iid-group binomial calculation gives one-sided
95% upper failure-probability bound45.1% for0/5; the fixed groups are not a random
population sample. All8s-complete truth here is stable; short unstable prefixes
remain in the separately reported70-event prefix metric (pooled .496619).

Gates: Ruff/compile pass,47unit tests pass including synthetic identification,
exact equilibrium and independent DOP853 integration. Recomputed444 component
trajectory scores plus888 ZOH/joint-Pm scores directly from private predictions
and original truth; maximum relative SSE discrepancy3.61e-16. Independent
DOP853 checks for pilot_0048/0150/0367 differ from RK4 by at most1.40e-5 in
state components, negligible relative to observed prediction errors.
All444 trajectory and1332 Pm sufficient-statistic records, code/data identities
and parameters retained in `0020_m3_evidence.json`. Full per-machine errors and
private rollouts/model retained in the command's output directory.

Residual diagnosis on TRAIN only: event-balanced, uncentered electrical-force
residual SVD gives leading fractions .670587/.191838/.073776. The first direction
explains67.1% of force residual norm, motivating one global electrical/control
memory coordinate. This does not prove temporal closure or history observability;
those are the next falsifiable M4 screen. No extra mechanical term is warranted
before testing this minimum memory hypothesis.
