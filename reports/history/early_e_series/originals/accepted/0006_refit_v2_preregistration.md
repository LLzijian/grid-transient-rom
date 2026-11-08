# Accepted loop 0006: refit-v2 runner and preregistration

Date: 2026-09-06

Starting commit: `6457f43`

Decision: **accepted infrastructure; no new numerical result is claimed**

## Frozen refit contract

1. Refit all five folds using only each fold's train and validation payloads.
   The consumed test payloads remain unopened and cannot support a new
   confirmatory claim.
2. Use the existing E2/E6 training settings: seed 324, 100 epochs, four ridge
   candidates, weak windows 1/5/20, at most 120 weak windows per case and
   horizon, causal ZOH mechanical power, and 100 rollout intervals for model
   selection.
3. Run `backbone -> full -> prune` for each fold. Pruning uses 200 grouped
   bootstrap draws, the corrected physical 1/5/10 s source-stratified deletion
   audit, and the frozen 2% total candidate-versus-fresh-backbone adoption gate.
4. Treat the model's mixed-available-length validation loss only as a selection
   diagnostic. Report H100 separately and include only events with at least 101
   endpoint-inclusive samples; retain every excluded event and rollout failure.
5. Write only to `results/e10_refit_v2/{fold}/{stage}`. Each stage refuses to
   overwrite, checks a clean Git start, verifies that tracked files and HEAD did
   not change during execution, reloads and re-evaluates the saved model, and
   publishes its directory only after all checks pass.
6. Record the Git commit, runtime versions, full train/validation populations,
   ordered-ID hashes, per-event signatures, dataset index/schema/split hashes,
   upstream artifact hashes, and final model NPZ/JSON hashes. Model sidecars and
   summaries use strict JSON; non-finite diagnostic values are represented by
   JSON `null` rather than non-standard tokens.

The loop fails if any stage accesses a test payload, silently shortens a fixed
H100 endpoint, accepts a pruned model without `module_adoption_gate`, publishes
a partially verified directory, or overwrites an existing artifact.

## Split and endpoint audit before fitting

- Within every fold, train/validation/test fault-line groups are disjoint.
- The five validation populations contain 212 unique events with no repetition.
- Strict H100 eligibility counts are F1=34/43, F2=32/37, F3=34/42,
  F4=38/43, and F5=39/47. The remaining validation trajectories are explicitly
  excluded from fixed-H100 aggregation, not relabelled.
- The canonical build remains `canonical_rebuild_20260722T022739`, already
  consumed for legacy exploration. This refit can support corrected
  post-selection validation evidence, not fresh out-of-sample confirmation.

## Gate record

| Gate | Result |
|---|---|
| runner helper tests | pass: strict H100 exclusion/failure accounting, strict JSON, ordered population hashes |
| lifecycle regression | pass: only `run_test_release.py` contains a test-payload read path |
| full unit suite | pass: 31 tests |
| syntax | pass: `compileall -q src experiments tests` |
| lint | pass: Ruff on the new runner and tests |
| diff integrity | pass: `git diff --check` |

## Next acceptance step

Execute and inspect every fold/stage under this frozen contract. A numerical
result is accepted only after its strict JSON, hashes, reload evidence, physics
audit, endpoint accounting, and Git diff have passed and the artifacts are
committed.
