# Failed loop 0007: F1 pruning exposed non-comparable validation scales

Date: 2026-09-06

Starting commit: `b76ded8`

Decision: **rejected; generated prune artifacts removed**

## Registered contract violated

The total module-adoption gate must compare the modular candidate with a fresh
backbone using the same train-derived normalization. A model saved after the
`full` stage must therefore retain its fitted `validation_scales` when loaded by
the `prune` stage.

## Failure evidence

- The F1 prune process completed in 2113.12 s and reported an apparent 96.23%
  improvement: fresh-backbone standardized loss 5.654913531971033 versus loaded
  candidate loss 0.21297905991431465.
- `PrunableEnergyModel.save` did not persist `validation_scales`.
  `PrunableEnergyModel.load` therefore restored the constructor defaults
  `{"omega": 1.0, "edge_angle": 1.0}` for the candidate, while the freshly fit
  backbone used scales estimated from the F1 training population.
- The two losses were dimensionless numbers under different denominators and
  could not support the reported adoption decision.
- The final H100 trajectory values happened to match the accepted `full` model,
  but that does not rehabilitate the invalid total-gate comparison.
- Rejected model hashes were NPZ
  `2b8592a4bc20febdffbc12c74b6436ca6314fcc5932f728e9459dc6cbc455dcc`
  and JSON
  `182ac006766619b6b2a58979fe9fcf337b9dfd91f1b94d9edadf0e16d8b91277`.

## Corrective action

Model format v4 persists both positive finite train-derived validation scales;
loading validates and restores them. A regression test now proves exact scale
round-trip. The rejected `results/e10_refit_v2/F1/prune` directory is removed,
and F1 pruning must be rerun from the accepted full checkpoint after this fix is
committed.
