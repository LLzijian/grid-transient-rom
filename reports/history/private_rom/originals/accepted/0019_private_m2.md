# Loop 0019: frozen M2 comparison baseline

Accepted as a **correctly constructed baseline, not an accurate model**. Scientific
purpose and gates were fixed in `reports/task2_m2_baseline.md` before fitting.
Starting checkpoint 436b721; immutable experiment fd8717c; command:
`python experiments/run_private_m2.py --output private_data/task2_m2_0019`.

218 train events and 70 validation events remain prospective at 50 ms. Nine train
and four validation events had already reached LOS and are explicitly listed;
none were removed from the dataset. All remaining short unstable prefixes score.
No test trajectory loaded. Event-balanced weak fit: 244890 scalar equations,
27 nonzero coefficients; no tuning. Runtime 2.96 s, NumPy 2.5.2.

| Horizon s | Complete truth n | Mean normalized RMSE | Pooled | CVaR10 |
|---|---:|---:|---:|---:|
| .2 | 66 | .272117 | .336456 | .677575 |
| .5 | 63 | .492867 | .607921 | 1.283393 |
| 1 | 62 | .740406 | .930160 | 1.958501 |
| 2 | 61 | .925002 | 1.085943 | 2.108342 |
| 5 | 61 | 4.703987 | 7.493971 | 17.194468 |
| 8 | 61 | 18.947773 | 27.492233 | 55.484768 |

Prefix mean16.692714, pooled27.402799 for all70. Fixed8s rollouts are finite but
physically poor: false-unstable33/61, false-stable0/9, balanced accuracy .729508.
This is **not** boundedness or stability validation. At8s COI angle RMSE18.1594rad,
speed7.82464rad/s. Complete8s cohort contains only stable truth, a survivorship
restriction disclosed rather than a population-wide accuracy claim.

Gates: Ruff/compile pass; 46 tests pass, including exact equilibrium,
synthetic weak-fit recovery and RK4 versus independent DOP853. All coefficient
constraints pass. Independent recomputation from stored predictions and original
validation arrays reproduced every per-event component SSE within floating-point
roundoff. Data/code hashes and all444 per-event/horizon sufficient statistics are
in `0019_m2_evidence.json`; private model and predictions remain under the command's
output directory. No raw trajectories committed.

M2 is frozen. Next question: how much of the long-horizon failure can a directly
identified effective governor explain, without changing this electrical backbone?
