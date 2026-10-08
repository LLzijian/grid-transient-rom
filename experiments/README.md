# Experiment entry points

Development experiments use train/validation data only. Validation metrics are
post-selection diagnostics when the validation fold selected hyperparameters or
modules; they are not called confirmatory out-of-sample results.

- `run_refit_v2.py`: corrected five-fold backbone/full/prune experiment.
- `summarize_refit_v2.py`: recompute trajectory/modal evidence from saved event
  rows with model and dependency hash checks, without dataset payload reads.
- `run_e1_m0.py`, `run_e2.py`, `run_e6.py`: historical development entry points.
- `run_modal_validation.py`: modal diagnostic on final pruned models; complete
  Hungarian assignments and event-macro aggregation, with no best-pair picking.
- `run_test_release.py`: the only allowed confirmatory holdout payload reader.
  It is currently blocked by `configs/test_release_v1.json` because the legacy
  dataset build has already been consumed.

The removed E6b/E6c/E8 entry points independently reopened the same test data
and therefore could not support a one-time-test claim. Their historical outputs
remain under `results/` as audit inputs only.
