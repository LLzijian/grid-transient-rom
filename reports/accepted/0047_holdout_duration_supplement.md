# Loop 0047 accepted: prospective unstable coverage in the frozen holdout

Baseline06d3bb4, frozen generator/grid47dd093, frozen raw QA b9a98d1.
Generation: installed ANDES2.0.0 via miniconda Python,
`experiments/generate_private_dataset.py --stage holdout --config
configs/private_dataset_holdout_supplement.json --output
private_data/ieee39_holdout_supplement --workers 4`, BLAS/OMP1.
42 fixed-grid attempts completed in157.44s:41 accepted (24stable/17physical LOS),
1 numerical_TDS_failure discarded whole. No adaptive duration changes or rescues.

QA: scientific venv `experiments/audit_holdout_supplement.py`. All raw checks
passed: full220dynamic states,479algebraic columns, finite/unpadded trajectories,
physical outcomes, normal solver at accepted LOS, exact equilibrium/case/system/
state-map identity, no duplicated bus-duration-xf case, all37/38/39 groups.
MaxCOI discrepancy1.14e-13. Every accepted event is TEST. No predictive model,
observer, parameter or threshold was applied. Generic QA's `pilot_ready=False`
is expected for this TEST-only shard, not a raw-data failure.

| Observation window | New stable | New unstable | Combined fresh stable | Combined fresh unstable |
|---|---:|---:|---:|---:|
| 50ms | 24 | 12 | 67 | 12 |
| 100ms | 24 | 9 | 67 | 9 |
| 200ms | 24 | 8 | 67 | 8 |

Combined fresh TEST has86 raw accepted events,79/76/75 eligible at50/100/200ms.
Unstable eligible events come from only2 independent buses; all test events
span3buses. This removes the empty denominator but does not establish precision
or model safety. Keep expansion+supplement frozen for one final predefined
test evaluation. Old Pilot TEST remains consumed and forbidden for selection.
Total physical corpus632 accepted; TRAIN/validation unchanged.

Immutable `private_data/ieee39_holdout_supplement/coverage_evidence.json` contains
all coverage, code/generator/data identities. Index SHA256
6edcddcf90cca620d7acb0b2629d179a066ce3dde7eda3a24947aae816b1ac58;
split SHA256 d1f3df87964e048501a8a257eda57f2584f19631747fc194ce41b8c659a60a7a.
Ruff/compileall and48unit tests passed. No dataset/reporting infrastructure was
introduced beyond the unique stage prefix, fixed config and small identity QA.
Return now to physically grounded electrical state/geometry discovery.
