# Loop0027 accepted diagnostic: observer gap and structural mismatch coexist

No M4 model is accepted here. This deliberately noncausal experiment uses future
development truth; neither its parameters nor scores are deployable/test results.
Startf7db6f1; code7885f5f; command with OPENBLAS/OMP threads1:
`python experiments/diagnose_private_memory.py --output private_data/task2_oracle_0027`.
Runtime226.23s;47tests/Ruff/compile pass; no frozen test payload.

## Initial-state oracle

All288 prospective development events searched with fixed0026 a=1.916890/s.
Zero competitor reproduces frozenM3 exactly; every event's final prefix MSE is
no larger than zero competitor (largest difference −9.79e-7). Search is numerical
and not certified globally optimal. Independent validation SSE recomputation
discrepancy4.27e-16; all444 validation records plus288 oracle initial conditions
retained in `0027_oracle_evidence.json`; full train records remain private.

| Horizon | M3 mean / pooled / CVaR | Oracle mean / pooled / CVaR |
|---|---|---|
| 2s | .611103 / .740507 / 1.536925 | .521788 / .603895 / 1.126203 |
| 5s | .478969 / .565149 / 1.118418 | .425957 / .485836 / .885556 |
| 8s | .395093 / .464055 / .908403 | .353905 / .402750 / .730643 |

Oracle meets the predeclared trajectory-potential screen. Prefix mean/pooled/CVaR
is .393934/.408863/.961645. However, optimizing prefix loss yields false-stable
pilot_0316 and0366 (2/9), versus0/9M3; no false-unstable. A potentially useful
state family is not thereby a safe classifier or an accepted memory model.

## Time-resolved force inversion

Derivative uses50ms preceding true speed, omits final one-sided endpoint. Force
norm² cutoff4.09682e-5 is training-derived and fixed for validation. Projection
orthogonality residual max1.55e-13. No invalid log modifier clipped into validity.

| Validation diagnostic | Centered | Uncentered direction |
|---|---:|---:|
| Time points with valid positive strength | 40601/49122 | 49122/49122 |
| Nonpositive required strength | 8521 | 0 |
| Event-balanced force residual norm explained | .228757 | .262564 |
| Events with both q signs beyond±.05 | 66/70 | 36/70 |
| Median fixed-rate exponential relative SSE | .995367 | .878412 |
| Median best-rate exponential relative SSE | .992807 | .755002 |

Explained norm is unconstrained scalar projection, an optimistic diagnostic,
not a physical fit; centered log is undefined for17.3% of active points. Centered
best-rate median4.7713/s varies strongly across events. Even optimizing decay
rate leaves nearly all q-trajectory norm unexplained. Uncentered direction removes
invalid-positive-strength failures and modestly improves force coverage, but a
single freely decaying global modifier still misses substantial dynamics.
Centered/uncentered explained fractions by validation group: bus4 .281/.291,
10 .241/.332,16 .265/.248,22 .223/.267,26 .149/.189. Thus improvement is not uniform.

Conclusion: oracle q0 demonstrates room for better initialization, while inversion
rejects interpreting that fitted scalar as a sufficient single-exponential
electrical state. Do not jump to GRU/q2 or call all remaining error observer error.
Next execute required M3 energy baseline. A bounded compact oracle-supervised
observer test is now permissible, as is testing uncentered/driven structure based
on these explicit physical diagnostics; each must retain the full prediction and
classification gates. Private per-time oracle arrays remain at the output path.
