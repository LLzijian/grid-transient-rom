# Loop 0033 — rejected: local stability does not fix nonlinear closure

Starting checkpoint `47be9e5`; attempted implementation `b04b77c`. Command:
`python experiments/run_stable_driven_memory.py --output
private_data/task2_stable_driven_0033`. TRAIN-only four-parameter constrained fit,
frozen0032 support and observer, frozen M3, no test payload read.

SLSQP succeeds in16 iterations/evaluations: regularized weak objective
1.0275321 → 0.6401755, directional gradient error1.04e-11. Recovery rate
0.2438299/s is interior; the dominant mode is -0.0100000+3.8316278i/s, with the
preregistered spectral constraint active. Equilibrium residual0, q-field
derivative error1.67e-13. All48 tests, syntax, Ruff pass.

The local physics gate passes but the prediction gate fails decisively:

| Horizon | Mean | Pooled | CVaR10 |
|---|---:|---:|---:|
|2s|1.781895|3.430541|9.216852|
|5s|7.368097|13.094021|30.896940|
|8s|14.228542|24.156264|51.200974|

M3's8s pooled is0.464055. There are22 newly false-unstable events among61
stable validation events (no false-stable among9 unstable). All70 predictions
are finite. Observer unchanged: validation r=.957675, RMSE=.037630, no clipping.
The weak oracle-state fit and local stability constraint are insufficient for
nonlinear feedback validity. This is a closure failure, not optimizer failure.

Reject and revert; preserve per-event evidence and private trajectories only.
Do not increase the spectral constraint, alter support, or tune against these
validation outcomes. User-owned `kiki.zip` is not part of the experiment.

Next smallest separately triggered test: the0030 uncentered free-decay oracle
has demonstrable trajectory capacity without classification errors, while its
compact observer failed0031. Test one small causal GRU observer with that same
fixed, locally stable free-decay ODE. Do not attach it to rejected driven
dynamics, refit M3, or add a second memory state.
