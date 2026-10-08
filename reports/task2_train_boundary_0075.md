# Loop0075 preregistration — one TRAIN transition-midpoint batch

Starting checkpoint: `207b36d`. Authority: after66 section7. No simulation
results, new model fit or boundary search on VAL/TEST were used to set this plan.

## Question / hypothesis

After the matched joint, teacher and single r4 screens failed, can a small
real-simulation batch resolve useful gaps in TRAIN transition coverage without
changing the operating point, restored system or labels? Hypothesis: one
midpoint in each of20 eligible TRAIN bus groups yields at least16 numerically
qualified, forecast-eligible events, including at least4 stable and4 unstable.
This tests data readiness, NOT whether augmentation improves a model or finds
a critical clearing time.

## Reference and fixed selection

Existing Pilot+expansion TRAIN:369 accepted events (311 stable/58 unstable),
346 forecast-eligible.24 bus groups,48 bus/reactance curves;384 configured
attempts include15 absent/UNKNOWN durations. The metadata-only inventory
`.loop-work/train_boundary_inventory0075.{py,json}` finds29 adjacent
stable-to-unstable intervals,28 with both endpoints beyond100ms in20 buses.
Do not bridge unknown/invalid grid slots or combine different reactances.
For each bus choose one eligible interval by ascending
`(fault_xf, lower_duration, upper_duration, fault_line, fault_location)`.
Simulate only its arithmetic midpoint; sort buses ascending for boundary IDs.
Expected buses:2,3,5,6,8,11,12,14,15,17,18,20,21,23,24,28,29,31,32,33.
Save selected endpoints and source index/split hashes before running.

## Allowed change / budget / frozen components

20 new physical conditions, each simulated once at .005s and once at .0025s:
40 runs total, no adaptive bisection, replacement, third step or label rescue.
Reuse `generate_private_dataset.simulate`, ANDES2.0.0, IEEE39 full case,
original controllers,1s fault start, temporary endpoint fault,15s stable
horizon,.01s export, Newton1e-7/30, and all existing physical/solver/LOS checks.
Reuse the existing dataset format; retain coarse and fine evidence separately.
New IDs start `boundary_`; all are TRAIN. Old data and S/0058/M3 stay immutable.
No model, loss, library, encoder, normalization or topology changes in0075.
Workspace-local ANDES code generation is an execution-only path change.

## Numerical and data gates (fixed before new trajectories)

Each pair must pass the existing generator validity/label checks; same
equilibrium (absolute full-state difference<=1e-9), inertia/state mapping,
case and restored-system signature as the frozen source; no duplicate old
physical conditions; no held-out group. All exported fields finite, with
unmodified true physical prefixes. Both solver steps must agree on label and
have >100ms future coverage. For unstable pairs, absolute LOS times measured
from clearing must differ by<=.02s. Stable pairs must each cover15s.

Record origins must agree within1e-8s; compare the actual shared .01s grid
through the earlier physical stop (full15s for stable pairs), including the
initial state. Do not extrapolate either stopped FOM. With omega converted
to rad/s, use the existing scales s_delta=.47675314588683515 and
s_omega=1.0033667249602924 and inertia-weighted COI projection:
`sqrt(.5*mean((COI(delta_c-delta_f)/s_delta)^2 +
(COI(omega_c-omega_f)/s_omega)^2)) <= .001`.
Also require RMS mass-weighted common-frequency difference/s_omega<=.001.
Report separate delta/omega/common errors, shared duration and LOS difference.
Reject numerical disagreements as UNKNOWN, never as physical instability.
Run the existing dataset QA only on these new ALL-TRAIN roots, never on an
old mixed-split H5. Its `pilot_ready` VAL-count heuristic is not a gate for
an intentionally TRAIN-only supplement.

## Decision / failure meaning / next implication

Pass requires all isolation/physical provenance checks plus>=16 of20 pairs
meeting the above per-event gates, with>=4 qualified events of each label.
Publish all20 attempt outcomes, failures, raw labels and latency counts.
Only qualified coarse events may enter a later, separately preregistered
matched-data refit; fine simulations are numerical QA, not extra samples.
No difficult old sample is removed. Failing the batch readiness gate rejects
this20-midpoint acquisition hypothesis, not the model family or global
duration monotonicity. Preserve useful/unknown evidence and diagnose before
the next smallest physical experiment; do not extend0075 to rescue its gate.
Stop only for genuine authority/split/physical ambiguity or AGENTS conditions.
