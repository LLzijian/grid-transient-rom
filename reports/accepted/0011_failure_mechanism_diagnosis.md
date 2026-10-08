# F1/F5 failure mechanism diagnosis

Date: 2026-09-07. Frozen experiment commit: `8ad0fcc`.

Decision: accept the diagnostic evidence. No fitted coefficients, adoption
thresholds or empirical populations were changed. This is a post-selection
diagnostic, not a confirmatory experiment or an algorithm performance fix.

## Registered questions and controls

1. Can F1's failed equilibrium searches be explained by a necessary physical
   capacity constraint of the fitted model? Enumerate every nontrivial node cut
   for all 212 validation events and compare with equilibrium residuals.
2. Can numerical time discretization or held-constant Pm alone explain F5's
   large errors? For all H100-eligible validation events in F1/F5, compare the
   frozen model using the original 0.01 s step, a 0.0025 s step, and observed
   future Pm with linear interpolation. The latter is an oracle sensitivity,
   not deployable prediction.

The capacity hypothesis would fail if the ten modal-failure events passed all
necessary cuts. A numerical explanation would require materially different
predictions at the smaller step. An input-only repair hypothesis would require
observed future Pm to resolve the large errors under the frozen coefficients.

## Algebraic capacity test

All five final checkpoints have pure sinusoidal conservative coupling. At a
zero-speed equilibrium the node, real-memory and resonant-memory states exert
zero steady force. With incidence B, nonnegative coupling K, reference angle
delta_eq and frozen clearing power, the equilibrium equation becomes

    B diag(K) sin(B.T delta) = b
    b = Pm_clear - Pm_equilibrium + B diag(K) sin(B.T delta_eq).

Sum this equation over any node subset S. Internal edge terms cancel and each
crossing edge has magnitude at most K_e. Thus a necessary condition is

    abs(sum(b[S])) <= sum(K_e over edges crossing S).

The experiment enumerates all 511 cuts modulo complement for ten generators.
It reports a violation only above a scale-dependent 1e-9 numerical tolerance.
The test uses the fitted model's power units and generator array indices, not
original IEEE bus numbers. Passing the test is not sufficient for equilibrium
existence, phase consistency or dynamic stability. It cannot be applied to a
model with additional nonlinear conservative terms; the script refuses those.

## Results

- F1 has 19 capacity violations, all at the cut containing generator indices
  `[0, 1, 5]`. There are ten stable and nine unstable events. All 19 also fail
  the 1e-7 equilibrium residual criterion. The ten stable events are exactly
  the ten failed modal cases in the previous experiment.
- Across the ten stable cases the excess required cut flow is 0.03930–0.66534.
  These are algebraic infeasibilities of the fitted model with frozen clearing
  Pm, not merely unsuccessful local least-squares searches. Changing initial
  guesses or increasing solver iterations cannot remove these violations.
- F2/F3/F4/F5 have no cut violation and their returned equilibria meet the
  residual tolerance for every validation event. This establishes only the
  stated static checks; it does not establish accurate dynamics or stability.

| Failure fold | Eligible cases | Original ZOH pooled RMSE | Step / 4 | Observed future Pm |
|---|---:|---:|---:|---:|
| F1 | 34 | 0.316955464 | 0.316955467 | 0.410825023 |
| F5 | 39 | 1.579711313 | 1.579711308 | 1.778835047 |

All 219 control rollouts succeeded. Maximum absolute prediction differences
between step sizes are 2.873e-7 (F1) and 1.458e-6 (F5), across all time points
and state components. The larger errors are therefore not explained by the
tested time discretization. Observed Pm increases pooled error in both folds,
so supplying perfect future Pm alone does not repair these frozen models.
This does not prove that a model trained and evaluated consistently with
observed Pm would fail; retraining is a separate hypothesis.

## Implications for next experiments

F1 has a static model/constant-input feasibility mismatch. F5 has large dynamic
prediction errors even where static equilibrium is feasible and numerical
integration is converged at the tested scales. These require separate work.
For F5, compare model force balance along observed trajectories and inspect
initial hidden-state assumptions and omitted electrical dynamics before
increasing library complexity. For F1, assess whether feasible learned static
coupling can fit the training dynamics without erasing required capacity.
Validation labels must not be used to tune the model to make the ten known
stable cases feasible. Repeated validation diagnosis remains exploratory.

The experiment is reproducible with
`experiments/diagnose_refit_failures.py`; output is
`results/e13_failure_diagnosis/summary.json`. It records 212 event signatures,
model hashes, data build, code commit, individual cuts, residuals and rollout
controls. Tests cover two-node feasible/infeasible systems in both signs and a
multi-node bottleneck that passes every singleton test. Output acceptance also
checks the declared validation population, original-run reproduction, all
model hashes, strict JSON and full test suite.
