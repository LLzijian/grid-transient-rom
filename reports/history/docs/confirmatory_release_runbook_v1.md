# Runbook: arming & executing the confirmatory holdout release (`test_release`)

Date: 2026-09-05 · Status: **operational procedure** (release currently BLOCKED — see §0).
Source of truth: `configs/test_release_v1.json` + `experiments/run_test_release.py` +
`src/kiki/evaluation/release.py` + `configs/dataset.json`.

## 0. Current status

`configs/test_release_v1.json` has `status=blocked_requires_fresh_holdout_and_refit_models`;
build `canonical_rebuild_20260722T022739` is in `contaminated_dataset_build_ids`. Running
`python experiments/run_test_release.py` now must **exit 2 before any store/payload access** and
create nothing. A confirmatory result does not exist yet.

## 1. Non-negotiable semantics

- The holdout event payload is read **at most once per event** (`SinglePassHoldout.load_event_once`).
  A claim is recorded **before** the backing store is called, so a failed load **burns that event**:
  retrying an uncertain read would violate the at-most-once contract.
- A dataset build already read by any legacy/exploratory script **can never** be re-labeled fresh.
- All endpoints are fixed by `evaluation_contract` **before** any payload read; editing them after the
  first read invalidates the release.
- Post-release angle/threshold sweeps on the confirmatory holdout are **prohibited**.
- Outcome labels only audit; they never fit or select.

## 2. Prerequisites checklist (ALL must hold before arming)

1. **Genuinely new dataset build**: a build id NOT in `contaminated_dataset_build_ids`
   (`canonical_rebuild_20260722T022739`), pointed to by `configs/dataset.json` (`dataset_dir`,
   `view`, `split_name`), with a real test population that contains unstable events
   (the screen audit `grouped_missed_instability_audit` raises if there are none).
2. **New preregistration**: bump `release_id` (e.g. `test_release_v2`), set
   `output_directory` to a **nonexistent** path, keep `evaluation_contract` as the frozen endpoint
   spec (trajectory 100-step H100=1 s ZOH; screen angle `0.5π`; modal ≥4.0 s population, all-pair
   macro, `best_pair_selection=false`).
3. **One final model per fold**, refitted under the corrected total module-adoption gate, with
   `"module_adoption_gate"` present in `model.diagnostics`; their paths committed under
   `model_paths` (set must equal `F1..F5`).
4. **Clean Git worktree** and a known HEAD; the run records `git_commit` into the ledger/report.
5. Only after 1–4: flip `status` to `armed` in the SAME commit that registers models/output.

## 3. Execution sequence (single invocation)

```
python experiments/run_test_release.py
```

Checked in order by the script (pre-payload checks fail the run BEFORE any event is opened):
1. `status == "armed"` (else exit 2, nothing created);
2. open store; fold test populations are **disjoint** (else RuntimeError);
3. `SinglePassHoldout.authorize` — rejects a contaminated build id;
4. `model_paths` covers exactly F1–F5; every model carries `module_adoption_gate`;
5. output dir **does not exist** (else `SystemExit`, nothing read);
6. worktree is clean; output dir + **durable access ledger** (`holdout_access.json`) written
   before the first payload read.

Then per event (F1..F5 × fold test ids): `load_event_once` (claim-before-read) → rollout trajectory
(+ `trajectory_error` captured, not fatal) → candidate-energy screen (`stability_margin`,
`certified` stays False) → modal endpoint for eligible stable events (+ `modal_error` captured).
Per-event endpoint failures are preserved in the record and surface in `trajectory_summary.
failure_rate`; they do not abort the run.

Finally `guard.finalize()` (exact population: no missing, no unexpected) → write
`release.json` (strict finite JSON, `allow_nan=False`) → ledger status `complete` → exit 0.

## 4. Failure modes

| Situation | Behavior |
|---|---|
| status not `armed` | exit 2 before store; nothing created |
| contaminated build id | authorize raises before any load |
| model missing adoption metadata | raises before any load |
| output dir already exists | `SystemExit` — this is the restart guard: an interrupted run leaves the output+ledger, so the same holdout **cannot be silently re-run** |
| dirty worktree | raises before any load |
| event payload load fails mid-run | that event is burned (already claimed); run continues to completion or aborts on the first uncaught error |
| zero unstable events in population | `grouped_missed_instability_audit` raises at summary |
| exception inside an endpoint | captured into the record; surfaced via `*_error` / `failure_rate` |

## 5. Post-run verification & promotion (BEFORE quoting any number)

1. `access_audit.single_pass_complete == true`; `claimed_events == expected == population`;
   every event appears once.
2. `release.json` is strict finite JSON (no `NaN`/`Infinity`); G6 gate passes on it.
3. Record `release_id`, `dataset_build_id`, `git_commit`; treat the output dir as **immutable**.
4. Only then may a manuscript/claim quote the confirmatory numbers; the safety statement must use
   the group-level Clopper–Pearson bound (already wired via `grouped_missed_instability_audit`).

## 6. Hard rules (anti-patterns)

- Never open the same build with a second script; `run_test_release.py` is the **only** confirmatory
  reader.
- Never re-run against a build whose events were claimed.
- Never compute best-pair modal numbers; the modal endpoint is the all-pair event-macro.
- Never run angle/threshold sweeps on the confirmatory holdout.
- `formal_stability_summary` in the report is intentionally all-zero
  (`refused_without_verified_global_lower_bound`); do not "promote" it to a certificate.
