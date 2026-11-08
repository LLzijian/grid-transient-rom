# Accepted loop 0003: holdout lifecycle and modal reporting

Date: 2026-09-05

Starting commit: `61ac5be`

Decision: **accepted infrastructure and claim correction; confirmatory test remains blocked**

## Registered claims

1. A confirmatory holdout event payload may be loaded at most once, after every
   endpoint and model has been preregistered, and all endpoints must be computed
   during that one in-memory pass.
2. A dataset build already read by exploratory scripts can never be re-labelled
   as a fresh confirmatory release.
3. Modal reporting must use the final pruned model and all pairs from a fixed
   complete Hungarian assignment. It must not select the lowest-error pair in
   each event, and selection validation is not called confirmatory OOS.

The loop fails if any old standalone test reader remains, if the contaminated
build can be armed, if a duplicate payload read reaches the store, if legacy
checkpoints pass the new modal entry point, or if the modal aggregate can still
use per-event best-pair selection.

## Accepted implementation

- Added `SinglePassHoldout`, which requires explicit authorization, rejects a
  contaminated build, rejects out-of-population and duplicate event reads
  before reaching the store, persists each claim before touching the payload,
  treats an uncertain/failed load as consumed, and requires exact population
  completion.
- Added `configs/test_release_v1.json`. It preregisters the trajectory,
  candidate-screen, and modal endpoints but marks build
  `canonical_rebuild_20260722T022739` as consumed and the release as blocked.
- Added the sole future confirmatory reader, `run_test_release.py`. Before any
  event payload access it checks the preregistration status, contaminated build,
  disjoint population, final-model adoption metadata, output nonexistence, and
  a clean Git worktree. It then creates a non-overwritable output directory and
  durable access ledger before the first read; each loaded event supplies every
  endpoint in memory and is claimed in the ledger before its payload is opened.
- Deleted `run_e6b_test.py`, `run_e6c.py`, and `run_e8.py`, which independently
  reopened test payloads. Post-release angle/threshold sweeps are prohibited on
  the confirmatory holdout.
- Replaced E5/E5b/E5c with one `run_modal_validation.py`. It uses final pruned
  models, requires the total module-adoption metadata, labels the result a
  post-selection validation diagnostic, uses mass-weighted speed shapes, keeps
  every Hungarian-assigned pair, and macro-aggregates event medians.
- Added shared modal evaluation/summary functions and removed all per-event
  best-match code paths.
- README, result, report, and manuscript status files now demote the legacy test
  and modal conclusions.

## Claim-specific evidence

- The committed test-release configuration exits with code 2 before constructing
  the dataset store or loading a test payload; no output directory is created.
- `run_modal_validation.py` rejects the first legacy checkpoint before loading
  validation events because it lacks `module_adoption_gate`; no output directory
  is created.
- The holdout unit tests prove an unauthorized read, a contaminated build, an
  out-of-order duplicate read, an uncertain backing-store failure, and incomplete
  population finalization all fail. The claim callback precedes the backing-store
  call, and a failed load cannot be retried.
- The modal regression uses one event with pair errors 0.01 and 0.49 and another
  with 0.20. A best-pair headline would be 0.105; the accepted event-macro,
  all-pair result is 0.225, proving the attractive pair cannot be cherry-picked.
- Static search leaves `run_test_release.py` as the only experiment that both
  iterates `split.test_ids` and loads event payloads. `check_data.py` inspects
  test split identifiers only and does not open test event payloads.

## Gate record

| Gate | Result |
|---|---|
| G1 syntax | pass: `compileall -q src experiments tests` |
| G2 unit | pass: 26 tests |
| lint | pass: Ruff on all changed Python files |
| G3 single-pass lifecycle | pass: claim-before-read failure injection; real config blocked with exit 2 and no output |
| G3 modal reporting | pass: all-pair event-macro regression; legacy model rejected before event access |
| G6 JSON | pass: 75 files, 0 errors, 92 inherited non-finite warnings |
| diff integrity | pass: `git diff --check` |

## Scientific boundary

No new holdout result is produced or claimed. The current canonical build has
no untouched confirmatory test population, and the final models must first be
refitted under loops 0001--0002. A real test release requires a genuinely new
dataset build, committed model paths, and a clean preregistered checkpoint.
