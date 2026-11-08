# Loop0063 preregistration — equilibrium-anchored initialization

Starting checkpoint da01b51. Authority docs/TASK2_after_loop_59.md; no increase
in model complexity.0062 found E(H_eq)!=0 although the frozen ODE equilibrium
is x=0,q=0; actual spurious equilibrium-history force max.306116pu. This
justifies testing a deterministic observer calibration, not redesigning the
encoder or assuming the initialization bias explains all group inconsistency.

## Hypothesis / one allowed change

Let q_eq=E_theta(11 zero incremental physical states), computed once from the
frozen sparse5s checkpoint. The prospective estimator is

q0_anchor=E_theta(H_0:100ms)-q_eq.

Hypothesis: this restores equilibrium-consistent initialization while
preserving complete5s and unstable-prefix physical prediction within the
fixed retention screens below. This is a change to the observer calibration,
not an affine coordinate relabeling: the ODE/force map is NOT translated.
All trained parameters, dimensions, losses, libraries, masks, algebraic
correction and topology mapping remain exactly frozen. No retraining/search.

## Inputs / arms

Model `private_data/task2_long_0060_5s/model.pt`, SHA256
d34d9f5c36fb296e1d49e59f5f7b18a5da33b56bf371a16169f82bc25664995c.
Use the same107 cached causal initial33-states and genuine aligned truths in
`private_data/task2_long_0060_5s_eval/trajectories.npz`. No H5/TEST reads.
The baseline arrays at.0025/.005/.00125 may be reused exactly from0061
`private_data/task2_long_sparse_rollout_0061/predictions.npz` after model,
normalization, event-ID and actual field/initial-state checks.

Fixed arms:

1. Full original initialization (baseline).
2. Equilibrium-anchored initialization above (only candidate).
3. q0=0 (diagnostic control, not automatically eligible for promotion).

Only the final3 initial-state entries may differ. All three arms then run
the SAME33-state ODE for a fixed8s, never re-encoding or using future truth.
Memory/algebraic forces stay enabled throughout; q0=0 is not memory-off.
Also run the same arms from exact physical equilibrium history for8s,
reporting full physical drift, force, and q. Anchored and zero-q arms must
remain at equilibrium to max absolute1e-10 in force and physical states.

## Metrics and numerical contract

Reuse verified0061 COI/scales/metrics: complete1/2/5/8 cohorts96/95/95/95,
all107 genuine prefixes and12 unstable prefixes; errors exclude initial.
Also report common synchronous frequency and per-group forecast metrics
descriptively, because COI error alone removes common motion. For common
frequency use mass-weighted omega, no change to the primary score.

Primary RK4.0025; companion.005; sample every.01. Every eventwise physical
normalized step-discrepancy must be<.001 on a claimed cohort/horizon,
including initial in this numerical comparison. Keep fixed-all, complete,
available-prefix and unstable-prefix scopes separate. If any arm's fixed
comparison fails, run the one fixed.00125 diagnostic; no adaptive tuning or
reinterpretation of failed primary gates. Persist nonfinite arrays/IDs and
invalidate that arm; do not clip/drop samples or calculate finite-only means.
Reused baseline convergence failures remain failed, not silently waived.

Registered candidate screen:

- exact-equilibrium force/physical-drift maximum<=1e-10;
- complete5s anchored/full ratios mean<=1.02, pooled<=1.02, CVaR<=1.05;
- unstable-prefix anchored/full ratios mean/pooled/CVaR each<=1.05;
- numerical validity for complete5s and all genuine unstable prefixes.

These are prospective retention screens, not requirements for one metric to
improve.1/2/8 and group/common-frequency tradeoffs are reported without new
post-hoc thresholds. Passing is a candidate for further confirmation, not
automatic M4 acceptance or completion of energy/stability/group robustness.
The zero-q control is interpretive; do not choose/calibrate intermediate
offsets after seeing validation or fit a validation-dependent mixture.

## Verification / failure meaning / continuation

Verify exact constant subtraction, physical initial-state identity, identical
frozen RHS, cached baseline compatibility, single initialization/no feedback,
genuine variable truth lengths, and equilibrium preservation with synthetic
tests before scientific runs. Independently reproduce saved metrics.
Store separate small arm files as they finish, not repeated rewrites of a
large cumulative archive. No new schemas/frameworks are needed.

Failure rejects this one deterministic calibration, not physical equilibrium
consistency as a requirement or every possible observer. Preserve useful
evidence and original estimator; do not rescue it with a fitted offset,
extra loss or enlarged encoder. Diagnose whether representation/initial-state
information or physical rollout-trained coefficient robustness is the next
smallest question. Frozen TEST remains unopened and work continues.

## Closure

Executed implementation5bb483f; all preregistered retention screens pass.
See `reports/accepted/0063_equilibrium_initialization.md` for independent QA,
common-frequency tradeoffs and retained failed all-fixed8s numerical gates.
Screen pass is not final model promotion. Continue physical-force interpretation
and closure diagnosis without changing trained equations or opening TEST.
