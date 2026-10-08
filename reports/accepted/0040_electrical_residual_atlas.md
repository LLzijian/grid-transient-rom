# Loop 0040 accepted: expanded TRAIN residual atlas

Implementation/preregistration: `14874a6`, baseline `c1e9f8a`.
Command: scientific venv Python `experiments/diagnose_electrical_residual_atlas.py
--output private_data/task2_residual_atlas_0040`, NumPy 2.5.2, BLAS/OMP 1 thread.
Ruff, compileall and 48 unit tests passed. All 369 TRAIN events, 24 fault-bus
groups, including 18 pre-50 ms LOS events; neither validation nor test payload
was read. Full per-event evidence, data/model hashes and residual traces are in
that immutable output's `evidence.json`, `residuals.npz`, and `moments.npz`.

Torque-derived versus central-difference acceleration discrepancy median/p95
0.00139149/0.00447591 passes the frozen 0.01/0.1 gate. Maximum is 0.0202995.
Rank(Bg)=9; relative energy-decomposition error 1.09e-15; edge reconstruction
error 7.77e-16. Independent reconstruction of the full second moment from
saved per-event residuals differs by at most 5.68e-14 absolute (summation order).

Only 45.3626% of event-balanced uncentered residual force energy is zero-sum
and representable by BgCq (stable 67.1007%, unstable 42.6911%). Do not silently
discard the common/net force remainder. Leading balanced fractions are
59.6851%, 24.1397%, 5.9470%; these are force-energy fractions, not identified
memory dimensions. First-mode group alignment median 0.8731, minimum 0.2165;
leave-one-group global alignment minimum 0.99817. Thus the global direction
is repeatable but is not a universal single-mode description of each group.

Leading-mode correlation with GENROU_2/e2q is -0.6976 (same sign in 23/24
groups), GENROU_1/e2d -0.6760 (24/24), IEEEX1_1/LS_y -0.6731 (24/24).
These are TRAIN associations, not causality or proof of memory after current
state conditioning. First-three mode median 1/e decorrelation lags are
0.23/0.35/0.30 s. Median Hann peak is the lowest nonzero bin 0.12516 Hz in
all three; this indicates slow/nonstationary content, not a resolved oscillatory
mode or a fitted recovery rate. Short-event unavailable frequencies remain null.

Accept the diagnostic, not an M4. Next: constrained sparse instantaneous
electrical correction with autonomous rollout gates; only then discover dynamic
coordinates. The nonzero-net remainder remains an explicit limitation of the
proposed incidence-only memory family.
