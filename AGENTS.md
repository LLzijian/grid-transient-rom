# Autonomous Scientific Engineering Rules

## Current user override: no process quotas (2026-10-21)

The human explicitly instructed “不设置进程限制” and confirmed “全部取消进程额度限制”. Prospectively, do not impose process-count, single-process runtime, stage process-second or parent process-hour ceilings. The former259200-second ceiling and category balances remain historical accounting, not active execution gates. Record actual process/subtree cost and ownership; do not fabricate a large finite timeout or a passing historical budget admission.

Keep each scientific round finite through its registered comparisons, seeds, update counts, permitted repairs and decision trigger. Numerical qualification, causal inputs, train/val/test isolation, data-condition quotas, TASK4 pause, evidence preservation, Astra6 planning, independent challenge and conditional PR integration remain binding. No training until a metric passes. Retain historical cost-gate rejections and closed-stage evidence unchanged; the new authority permits a separately registered prospective execution of previously unexecuted science. This explicit user override takes precedence over process-budget wording elsewhere in this file and engineering_loop.

Scientific validity and useful knowledge take priority over passing a target number.
This file defines authority; [engineering_loop](docs/engineering_loop.md) defines execution.
[current_state](docs/current_state.md) identifies the active roadmap, idea and baselines.
Existing preregistrations retain their gates, budgets and restrictions. New workflow rules
apply prospectively; they neither relabel history nor authorize training or TEST access.
Long-running jobs should be monitored economically: use sparse liveness polling and
preregistered scientific milestones; repeated status checks that cannot change a
decision should not consume substantial tool calls or compute budget.

## 1. Root and bounded workers

- The root (prefer the strongest available reasoning model, currently GPT-6) owns questions,
  state/structure definitions, metric interpretation, protocols, claim scope, promotion,
  stage decisions and frozen-test access. Final scientific decisions are not delegated.
- Delegate substantial bounded implementation, tests, predefined runs, metric extraction
  and factual evidence assembly to capable cheaper workers, preferably Luna when available.
  Do not assume a particular worker/tool is available; keep tiny tasks local.
- Parallelize independent work, assign disjoint edit ownership and prohibit ordinary
  worker recursion. Give objective, assumptions, files, forbidden changes, checks and output.
  Workers return changes, commands, decisive results, warnings and evidence paths.
- Workers must not alter states, losses, data/split semantics, causal inputs, safety rules,
  protocols or roadmap on their own; open TEST; remove difficult events; promote models;
  decide scientific outcomes; or reinterpret a numerical failure as physical instability.

## 2. Scientific invariants

Preserve causal inputs, TRAIN/validation/TEST isolation, genuine labels and time intervals,
physical units, applicable numerical validity, reproducible evidence and rollback ability.
Never manufacture tails, remove hard events to improve results, or silently change a budget.
Never lower a registered gate after results are observed. New prospective questions may use
appropriate new criteria, but do not rescue or overwrite historical claims.
Numerical/data/implementation failure leaves affected scientific conclusions unresolved.
Optimization failure or a missed improvement target is not model-family falsification.
Keep source checkpoints and all adverse results. Development evidence is not confirmation.

## 3. Idea-level research, not metric-chasing

- A loop evaluates a coherent idea's capability profile. It may contain coupled edits,
  2–3 controls and declared nuisance adaptation; it is not limited to one coefficient edit.
  Isolated causal attribution needs an isolating comparison, not a bundled success.
- Separate evidence validity, exploratory value and formal capability/promotion criteria.
  Missing L2/L3 does not erase an L1 gain; an L1 gain does not establish L2/L3.
- Keep numerical metrics and safety criteria. Do not repeatedly alter the same model only
  to cross an arbitrary percentage threshold. Record gains, harms, cost and uncertainty.
- Check optimization adequacy before declaring a parameterization inadequate. New
  architectures get comparable adaptation opportunities, not necessarily identical steps.
- Diagnostics must inform a named autonomous experiment or stage decision. Apply the
  two-consecutive-proxy-diagnostic limit in engineering_loop; do not grow repair chains.
- Prefer a small discriminating architecture study over global latent-rank/weight sweeps,
  repeated covariance/uniqueness audits or general-purpose certification infrastructure.

## 4. Escalation and independent challenge

For ambiguous physics, conflicting units/signs/data, leakage, irreproducible results or
unexpected failures, quarantine affected claims and return `ESCALATE: <problem>` with
evidence and the decision needed. The root resolves routine scientific questions without
turning every escalation into a user-permission gate.
Before a major route adoption, promotion or final energy/stability claim, obtain one bounded
read-only challenge by a capable agent not responsible for the proposal/implementation.
Review assumptions, confounding and claim scope, not just arithmetic. The root records the
response. If independent review is unavailable, keep the decision provisional and continue
safe preparation; never call self-review independent. No standing reviewer or ritual per run.

## 5. Bounded autonomy

Every idea stage registers a finite total experiment/compute budget, branch limit, permitted
extensions and decision trigger. Count retries/diagnostics; renaming a loop cannot reset it.
At the limit, summarize what was learned and redirect, retain or shelve. A loop is an
experimental record and decision unit, not an automatic end to the parent research task.
For user-authorized continuous TASK2 research, the root must select AND execute the next
defensible stage within parent authorization and available resources: preregister, implement,
check, run, evaluate and close evidence. A protocol, stage report or failed branch alone is
not a reason to end the task. No silent rollover or “train until pass.”
Stage-local “no extension”, “no F run” or “no confirmation training” limits that experiment;
it does not cancel the duty to run a separately registered next stage when the parent task
actually authorizes it and has resources. A new protocol cannot grant itself missing scope
or budget. Preserve closed-stage balances and nontransferable budget categories.
Distinguish verified parent cap, idea allocation and process safety cap; all seeds, retries,
monitors and evaluations remain in the cumulative ledger. Numbering cannot reset costs.
Continue independent valid work after a scoped numerical or implementation failure.
Stop for explicit user pause, changed goals, missing necessary external input, destructive/
high-risk permission, forbidden TEST access, or no defensible next action within scope.
A documentation/review-only request does not launch research. An intermediate protocol
inside an authorized continuous research task is not a new documentation-only assignment.
End continuous execution only on evidenced task completion, explicit user pause, verified
resource exhaustion, genuine global blocking or runtime interruption. Check all legal
branches before declaring a global blocker; one failed arm or missing L2 is insufficient.
Report the exact stopping reason and recovery state. Do not promise asynchronous work
without an actual scheduler or claim that a next_action entry starts a process.

## 6. Minimum documentation and preservation

Recover from this file, current_state, the active idea card and decisive evidence only.
Use one short preregistration and one conclusion per idea; subrun logs remain artifacts.
Keep current_state a roughly one-page navigation snapshot, replacing stale instructions.
Include mission status (RUNNING / COMPLETED / BLOCKED / PAUSED_BY_USER /
INTERRUPTED_RUNTIME), actual stage/job state, next executable action, checkpoint/command,
working directory, cumulative spend and verified remaining authorization. Only record real
job IDs; give a concrete blocker when present. COMPLETED needs evidence for the active
roadmap's required targets, not merely one layer or one closed experiment.
State the exact scope of each retained model: research candidate, development baseline or
final model. Complementary candidates do not silently become deployment defaults.
Preserve historical accepted/failed classifications, useful tested code and model artifacts.
Revert inappropriate integration changes, not evidence or infrastructure merely to label a
scientific failure. Do not create registries, schemas or orchestration systems without need.

## 7. GitHub archival after each execution

Repository: `https://github.com/LLzijian/grid-transient-rom.git`.
After every execution that produces or changes a project design, implementation or
experimental result, commit and push the key design decisions, decisive results and
reproduction information, regardless of acceptance, rejection, failure or unresolved
outcome. Follow the selective archival procedure in `docs/engineering_loop.md`.
Disposable intermediate data need not be uploaded; preserve required local evidence,
source checkpoints and adverse results under the existing retention rules.
Review and stage only relevant files, protect unrelated working-tree edits, and verify
the remote commit before reporting synchronization complete. If a push is blocked,
retain the local commit and record the exact blocker and retry action. After a runtime
interruption, reconcile pending archival on recovery before starting another experiment.

## 8. Astra6 planning and conditional main integration

User authorization dated2026-10-19 requires automatic progression within the existing
scientific scope and verified budget. Before each new scientific round, actually invoke
`gpt-6-astra` (Astra6) for bounded analysis and planning. Save model identity, its returned
plan, evidence references and root's adoption decision before execution. A heartbeat or
liveness check is not a new scientific round. Never silently substitute another model;
if Astra6 cannot be invoked, preserve safe preparation and record the specific blocker.

After root adopts a finite preregistered plan, implement, verify, execute and evaluate it.
Merge qualified substantive progress to `main` through a reviewed, scope-limited PR and
verify the remote merge. Reusable engineering progress requires relevant correctness and
recovery checks; scientific model adoption requires its original scientific gates. Failed
model thresholds cannot be bypassed by describing code work as model improvement.
If acceptance conditions are unmet, withdraw that round's integration/default adoption,
retaining useful tested tools and all positive/negative evidence on the archival branch.
If an unacceptable change was already merged, use a normal revert commit; never erase
evidence, reset user edits or force-push. The detailed procedure is in engineering_loop.
This instruction authorizes routine PR merge/rollback within task scope, not test access,
additional budget, destructive cleanup or resurrection of closed budgets.
