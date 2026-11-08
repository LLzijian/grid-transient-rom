# Loop0034 — rejected: nonlinear observer does not generalize robustly

Checkpoint a10ccf3; implementation66c8068. Command:
`python experiments/run_small_memory_observer.py --output
private_data/task2_gru_observer_0034`. One729-parameter GRU8, seed34, six causal
samples, fixed uncentered free-decay ODE and frozen M3. No test payload read.

L-BFGS reaches the fixed200-iteration budget (not a convergence claim).
TRAIN regularized objective .0408113 → .0006140995; directional error3.05e-13.
TRAIN oracle-q0 R²=.984983, RMSE=.024642; validation R²=.395169, RMSE=.194501,
correlation=.678066. No clipping. Zero history gives exactly q0=0; zeroq ODE
matches saved M3 exactly. All48 tests, syntax and Ruff pass.

Short horizons improve, but the unchanged adoption gate fails: validation
pooled2/5/8s=.938661/2.225808/3.750489 versus M3 .740507/.565149/.464055.
8s mean1.009481 and CVaR6.437467 also fail. Two new false-unstable events;
zero false-stable. All70 rollouts finite. Better TRAIN initialization does not
resolve cross-group observer error or the free-decay family's tail sensitivity.

Reject and revert, preserve all444 records and private trajectories/weights as
reproduction evidence only. No observer width, seed, training-budget or ODE
extension after this result. User-owned kiki.zip excluded from changes.

The reasonable one-dimensional screening now spans additive/passive closures,
centered/uncentered recovery, TRAIN oracle supervision, sparse driven closure,
explicit local spectral constraints, and one conditionally triggered small GRU.
None passes the full prediction/tail/physics gate. This supports choosing frozen
M3 as the main development model under TASK2_after_loop_26 sections6/25, not the
universal claim that physical memory is absent. No second independent stable
mode has been established; q2 is not justified. Next: freeze this model choice
and complete M3 modal, nonlinear energy-boundary and margin analysis before the
one-time frozen test. These remain open scientific work, not a stop condition.
