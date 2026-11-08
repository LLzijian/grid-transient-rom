# Accepted loop 0001: physical pruning horizons and module adoption

Date: 2026-09-05

Starting commit: `f764fe0`

Decision: **accepted**

## Registered claim

The pruning implementation must evaluate each declared horizon at its true
physical time and must retain optional modules only when the complete modular
candidate improves validation loss by at least 2% over a freshly fitted
backbone without increasing rollout failures.

The change is falsified if a short trajectory is reused for a longer horizon,
if H100 omits the 1.00 s endpoint on a 0.01 s grid, if a worse modular model is
accepted, or if any required repository gate fails.

## Accepted change

- Horizon endpoints are derived from the actual monotone time vector. Events
  that do not reach a horizon are excluded from that horizon rather than
  relabelled; a missing source/horizon stratum fails closed as not assessable.
- `horizon_steps` now consistently means integration intervals, so evaluation
  includes `horizon_steps + 1` samples.
- After backward deletion, the selected modular candidate is compared with a
  freshly fitted no-module backbone. Adoption requires at least 2% lower
  standardized validation loss and no higher failure rate.
- E2 and E6 record the explicit 0.02 threshold in their output.
- The previously missing JSON structural gate was implemented. It detects
  malformed files and duplicate keys, reports inherited non-finite tokens, and
  exposes a strict non-finite mode covered by unit tests. Its deliberately
  limited repository policy is stated in the script.
- Historical results are explicitly classified as pre-fix audit inputs in
  `results/README.md`.

## Claim-specific evidence

The deterministic tests cover exact 1/5/10 s endpoint selection, exclusion of
unobservable horizons, H100 endpoint semantics, the 2% adoption threshold, and
the failure-rate guard.

A real-data smoke gate used frozen F1 model
`results/e02_prune/model_pruned.npz` and one validation event of at least 10 s
from each source (`ncunstable_0014`, `scenario_002`). Comparing the model with
itself produced six separately assessed rows: both sources at 1, 5, and 10 s,
each with one event, zero degradation, and `status=assessed`. This establishes
that the corrected path actually reaches 10 s; it is a wiring gate, not a
performance claim.

The adoption regression reconstructed two historical comparisons:

| Fold | Backbone loss | Full loss | Relative improvement | Gate decision |
|---|---:|---:|---:|---|
| F1 | 5.554214 | 1.894030 | +65.9% | accept |
| F4 | 5.152878 | 9.904378 | -92.2% | reject |

These rows test the decision contract only. They do not promote the inherited
models as newly reproduced results.

## Gate record

| Gate | Result |
|---|---|
| G1 syntax | pass: `python -m compileall -q src experiments tests` |
| G2 unit | pass: 14 tests (including six independently added causal-contract checks) |
| lint | pass: Ruff on the changed core, verifier, and tests |
| G5 physical horizons | pass: six real-data source/horizon rows assessed |
| G5 module adoption | pass: threshold and failure guard unit regressions |
| G6 JSON | pass: 74 files parsed, no errors; 92 inherited non-finite token warnings reported |
| diff integrity | pass: `git diff --check` |

An initial verifier-test attempt depended on pytest's inaccessible system temp
directory and failed during fixture setup. The test was redesigned as a pure
in-memory contract, then all gates were rerun. No failed implementation or
generated result was retained.

The gate currently accepts Python's non-standard `NaN`/`Infinity` tokens. Ten
inherited files containing those tokens are listed as technical debt in
`results/README.md`; strict schema, manifest, hash, and manuscript-traceability
checks remain future G6 work and are not claimed here.

## Scientific boundary

This loop corrects selection semantics but does not rerun model training or
validate the manuscript's stability-certificate, OOS lifecycle, modal, or
headline performance claims. Those remain unaccepted until their own loops
pass.
