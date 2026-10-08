# Scientific Engineering Loop — Idea-Level Capability Studies

Authority: [AGENTS.md](../AGENTS.md). State: [current_state.md](current_state.md).
The active roadmap supplies scientific models and task-specific release requirements.
This replaces future execution mechanics, not historical results, gates or budgets.
A documentation/review request does not authorize training or remove an explicit pause.
During an already authorized continuous research task, intermediate preregistration is
part of execution; it is not a new documentation-only request. The continuous-execution
rules below are prospectively merged from TASK2_continuous_execution_patch.md (2026-09-23),
without changing the then-active after97 roadmap, existing scientific protocols or their
resource caps. The prospective after101 roadmap now supplies the active priorities.

## 1. Unit of work: one coherent idea, not one target number

A research loop studies what one modeling/algorithmic idea achieves within declared data,
optimization and resource conditions. It may include 2–3 controls, coupled model/encoder
changes and bounded nuisance-parameter adaptation. A coherent architecture change need
not be split into a loop for every loss, parameter, smoke test or coefficient repair.
A bundled result supports that bundle; do not claim an isolated causal contribution.
Diagnostics and implementation QA are subruns unless they change a separate major decision.
The goal is an informative capability profile, not repeated attempts to exceed a chosen
5%/10% gain. Quantitative definitions remain essential; quantitative hopes are not truth.

## 2. Recover and write one short idea card

Read current_state, the active roadmap and the evidence needed for this decision only.
Record starting commit, data/split identity, source assets and unrelated working-tree edits.
Use one preregistration for the loop, with these fields (plain text; no new registry):

| Field | Required content |
|---|---|
| Question | Idea being tested; plausible benefit, tradeoff and bounded negative outcome |
| Comparison | Controls, information parity, scientific changes and unavoidable confounds |
| Fitted objects | What learns jointly; fixed physics; permitted nuisance settings |
| Evaluation | Common metrics/cohorts; expected effects; any actual claim/release gates |
| Budget | All trials/seeds/diagnostics, staged extensions, wall time, stop rules |
| Adequacy | How optimization/resource limitation will be distinguished from a weak idea |
| Next decision | Which result would justify continue, compare, redirect or shelve |

Distinguish scientific choices from nuisance optimization settings. Allocate a small,
comparable adaptation opportunity to each architecture; do not require the same learning
rate to be optimal for different parameterizations. Changes outside the card require a
prospective amendment for future runs, retaining old outcomes; never relabel past failures.
Never retroactively promote exploration to confirmation.

## 3. Keep three standards separate

**Evidence validity (mandatory):** causal inputs; train/validation/test isolation; units,
state definitions and equations; genuine labels/time intervals; appropriate numerical checks;
reproducible configurations, denominators and retained adverse results. A failure invalidates
or limits the affected evidence, not automatically the entire idea or unrelated results.

**Exploration value:** improvements, regressions, useful structural simplicity, new capability,
training behavior, cost and uncertainty. A candidate can be informative without beating every
metric. Report all predeclared metrics; do not hunt for an unregistered favorable subgroup.
An unexpected benefit is exploratory and needs prospective confirmation.

**Claim / deployment eligibility:** precise thresholds for the stated capability, safety,
noninferiority and independent confirmation. These remain binding once registered. A missed
threshold means the claim is unsupported, not that all scientific value is zero. Worse safety
cannot be averaged away; such models may remain explicitly non-deployable research artifacts.
Exploratory Pareto retention is not promotion. Historical gates remain unchanged.

## 4. Implement the idea and establish meaningful optimization

Run changed-code tests, dimension/unit/sign checks, finite gradients and a short smoke.
Reuse valid unchanged checks; no per-subrun full audit ceremony. For a new parameterization,
use a small fixed TRAIN capability panel when needed; it is not a generalization benchmark.
Then use the full declared training pool. Preserve every event and its eligibility status.
Train representation, dynamics and structural objects jointly when the hypothesis requires
it. Short regression/oracle/conditioned fits may initialize or diagnose; they cannot replace
autonomous training and evaluation. Report warm-start mismatch and information advantages.
Record per-task learning curves and cost, not just the last scalar training loss.

## 5. Run bounded staged exploration

Use the card's extension rules, based on TRAIN or a declared inner-TRAIN group split.
Do not eliminate a new parameterization solely because its absolute loss starts above a
mature incumbent. Show fixed-compute comparisons AND the best observed results obtained
under each declared total budget. Equal steps are not evidence of equivalent optimization.
At the cap, a still-improving or poorly optimized run is optimization-limited, not proof of
model-family impossibility. A plateau under a finite search is also not a global optimum.
Count tuning, retries, diagnostics and all seeds; renaming a loop cannot reset the budget.
Use limited development evaluation at planned stages; it remains selection evidence.
Freeze confirmation configuration/checkpoint rules separately. Keep TEST entirely closed.

## 6. Limit diagnostic chains; return to the scientific experiment

Before a diagnostic, state which model decision it can change and the autonomous comparison
that will follow. If it cannot change a decision, skip it or defer it to a finalist.
After at most TWO consecutive proxy/conditioned/frozen-block diagnostics on the same branch,
return to a joint autonomous experiment OR close/shelve the branch with its known limitation.
Do not evade this by renaming the metric, coefficient block or loop. More analysis requires
an explicit stage review and a genuinely decision-changing question, not a routine rollover.
Leakage, invalid units or irreproducible numerics must still be resolved before affected
results are used; quarantine blocked work and continue independent valid work where possible.
Never force an unsafe fit merely to satisfy the two-diagnostic limit.

## 7. Close with a capability profile, not a single pass/fail stamp

Use one short conclusion report: what was compared, valid results, limitations, cost and the
next decision. Keep trial traces and detailed evidence in artifacts. Record fields separately:

| Field | Values / purpose |
|---|---|
| Work type | idea study / diagnostic / implementation QA / confirmation |
| Evidence | valid / valid within scope / inconclusive / implementation-data-numerical issue |
| Capability | Exact gains and harms per L1/L2/L3 or the active task's comparable dimensions |
| Optimization | evaluated at declared budget / improving at cap / insufficient / unresolved |
| Decision | advance / retain complementary candidate / redirect / shelve / repair |
| Promotion | none / scoped development baseline / final confirmed model |

A QA or diagnostic success is not model success. A model can support one capability and
fail another. Preserve at most 2–3 substantively complementary candidates, considering
uncertainty and complexity; no arbitrary weighted score that hides severe-event harm.
State “best observed under this protocol,” not “maximum achievable by this family.”

## 8. Confirmation and autonomy

Only shortlisted candidates receive additional seeds, honest end-to-end held-out-group
refits, physical interventions or expensive certification appropriate to the claim.
All data-fitted preprocessing, encoders, support selection and warm starts respect the
confirmation split. Repeated development use is not independent confirmation. Keep original
safety definitions and expose latency-uncovered, unresolved and adverse cases.
For final TEST, obey the active release configuration and
[confirmatory release runbook](confirmatory_release_runbook_v1.md); this file grants no access.
Obtain the bounded independent challenge required by AGENTS.md before major claims/promotions.

A stage ends after its finite budget and decision review. The root must select and execute
the next authorized stage unless the mission is complete, explicitly paused, genuinely
blocked, resource-limited or interrupted by the runtime. Ordinary registration is not an
extra approval gate. Never silently prolong an old protocol until it passes.

Recover current_state, actual jobs, authorization and cumulative costs. While the mission
is incomplete: honor pause/runtime boundaries; inspect pending work and all legal branches;
choose the action that best resolves a current capability gap; allocate only verified
unassigned parent resources; register a short protocol; implement/check/run/evaluate it;
preserve outcomes and costs; update capability gaps and return to this outer decision.
Do not end an authorized task merely by delivering its next protocol or recommendation.
This describes root's execution duties, not an already installed scheduler.

| Event | Scope stopped | Required outer action |
|---|---|---|
| NaN, gradient or implementation failure | Affected process/evidence | Use permitted repair, valid control or shelve that arm; continue independent work |
| Numerically unresolved horizon | Affected claims/runs | Preserve raw rows; use registered checks or another valid scope; do not treat unresolved scores as valid |
| Arm/stage step or cost cap | That experiment | Close it; select a separate authorized stage only if parent resources permit |
| An idea is shelved or one capability gained | That idea/layer | Preserve gains and harms; continue the active roadmap's remaining objectives |
| Verified parent resources exhausted | Work needing new resources | BLOCKED with reason BLOCKED_RESOURCE, spent/remaining totals and next actual action |
| Every defensible branch needs unavailable external input/permission | Whole task | BLOCKED with reason BLOCKED_EXTERNAL and alternatives already considered |
| Explicit user pause | User-specified scope | PAUSED_BY_USER; no automatic wakeup |
| Host/session interruption | Current execution | INTERRUPTED_RUNTIME with checkpoints/commands; do not claim scientific completion |

Under the active after101 roadmap, mission completion requires linked evidence for an
exportable low-complexity autonomous ODE and100ms initializer, high-accuracy L1
transient/classification with qualified cost, and the claimed L2 structural stability-domain
scope in one model. L3 system-feature correspondence is report-only and optional; retain
the original one-to-one match definition and historical outcomes. Candidate regions and
strict certificates remain distinct. Negative local poles, a power identity, an eigenvalue
export or a short-time gain cannot substitute for the relevant L1/L2 evidence. A finite
unsuccessful search may leave the mission blocked within authorization; it is neither
completion nor model-family impossibility.

Parent authorization, stage allocation and per-process cap are separate. The root may
allocate future stages only from genuinely authorized, transferable parent resources.
Closed-stage leftovers and nontransferable categories are not revived. Count all seeds,
repairs, monitors and evaluations across stages. Specifically,0099 remains capped at20000
new process-seconds;0098 ancestry remains8623.47 charged seconds and its unused3376.53
seconds remain retired. This rules patch grants no further E2/E3/confirmation allowance;
verify any parent balance from actual user authorization, never assume unlimited continuation.
The prospective after101 rule may return a newly closed stage's unused allocation to the
parent pool only for a separately registered next stage and with explicit budget authority
and an adjustment record. Previously retired 0099/0100/0101 balances do not revive merely
because the roadmap changes; the original cumulative charge and parent ceiling persist.
“No F run” or “no added confirmation” in0099 limits0099, not a separately authorized stage.

Loop0099's registered work is closed and remains historical evidence. For a new after101
stage, register feasible milestones and complete the authorized comparison without waiting
for a message at ordinary milestones. Numerical, data, TEST, pause and resource boundaries
still bind. After closure select and execute a next active-roadmap branch only within verified
scope and remaining resources. If none is legal, record the concrete global blocker.

Honor external/high-risk permission boundaries. A saved next_action is not a running job;
runtime continuation requires an actual host/scheduler. Never fabricate background work or
use automation to bypass a pause or budget. Stage reports are progress updates; a final stop
must identify completion, pause, resource/global blockage or actual runtime interruption.

## 9. Preserve and maintain

Keep historic report locations/classifications; add new idea reports under the existing
reports directory without reorganizing history. Preserve candidate artifacts and default
checkpoints. Revert unsafe integration changes, not useful tested infrastructure merely
because one scientific idea underperformed. Record exact active/default model explicitly.
Replace stale current_state entries with a concise snapshot: active idea, retained models,
capability gaps, budgets, evidence, pause/job status and TEST state. Do not append a diary.
Include mission status RUNNING / COMPLETED / BLOCKED / PAUSED_BY_USER /
INTERRUPTED_RUNTIME separately from a loop's closed/rejected/retained state, plus next
action, recovery checkpoint/command and cwd, cumulative spend, verified remaining resources
and real job IDs only. Keep these fields in the existing snapshot, not a new registry.
Use one current capability table in the report or state, not a new framework or dashboard.

## Long-running process monitoring

For long-running training, simulation, or evaluation jobs, avoid frequent polling that
does not change a scientific or safety decision.

- Do not poll progress on every few training steps or at sub-minute intervals merely
  to report that a process is still running.
- Default progress polling interval for a healthy long-running job is no more frequent
  than once every 3–5 minutes, unless the process is expected to finish sooner.
- If step counters are available, normally wait at least 50–100 optimization steps
  between lightweight progress checks.
- Full scientific monitoring/evaluation is performed only at preregistered milestones
  (for example 0/200/400/600/800 updates in Loop0099), not at arbitrary intermediate steps.
- Between milestones, lightweight checks should only verify:
  process alive / exit status, non-finite failure, memory/resource limit, or an explicit
  safety-stop condition.
- Do not repeatedly deserialize checkpoints, recompute metrics, read large evidence files,
  or launch Python subprocesses only to print incremental step counts.
- Prefer waiting for the next meaningful milestone or process completion over repeated
  status narration.
- Increase polling frequency only when:
  1. approaching a wall-time/resource cap,
  2. a failure signal has appeared,
  3. a process is expected to terminate imminently, or
  4. a preregistered decision point is near.
- Progress messages to the user should correspond to meaningful milestones, decisions,
  failures, or completion—not routine liveness polling.
