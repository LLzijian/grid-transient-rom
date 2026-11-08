# Loop 0021 rejected: relaxing coordinate is not a stable coupled closure

Start6272e94, attempted implementationea41aeb. Preregistered in that commit's
`reports/task2_m4_q1.md`; no rescue or criterion change. Command:
`python experiments/run_private_m4.py --output private_data/task2_m4_0021`.
Ruff/compile and48tests pass; scientific screen decisively fails.

Training force rank fraction .675929; validation q0 correlation .971605 and
RMSE .629794 against target RMS4.146657 indicate useful history observability.
This does not validate the future q law. Its positive decay hits the lower bound
.01/s while unconstrained speed couplings create active feedback. Loading·drive
=−5.554624; finite-difference Jacobian adds an unstable pair .040188±5.381363i,
whereas M3's non-gauge eigenvalues have negative real parts. The largest predicted
state magnitude is2.82015e15. Gauge rotation's zero mode is not called instability.

At.2s pooled improves .334328→.285834, but at2s .740507→12.139104 and at8s
.464055→4.79260e11;32 new false-unstable events. All floating-point results are
finite, but these trajectories are unambiguously nonphysical. Finite arithmetic
is not a stability gate. No difficult event removed or score censored.

Retain all444 per-event trajectory sufficient statistics, classifications and
identities in `0021_m4_evidence.json`, useful negative result and observer evidence.
Reject/revert memory class, runner and experiment-only tests; retain accepted M3.
The next smallest hypothesis is reciprocal passive coupling of a driven scalar
relaxation to swing speed, preventing q from injecting unbounded feedback energy.
This is a physical constraint on the ODE, not a larger observer or second q.
