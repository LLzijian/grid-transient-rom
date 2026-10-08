# Loop0039 accepted: physical corpus expands to591 events; holdout caveat

Frozen generator/config9666105;240attempts,4workers,1250.189s. Commands:
`python experiments/generate_private_dataset.py --stage expansion --config
configs/private_dataset_expansion.json --output private_data/ieee39_expansion
--workers 4`, followed by existing `audit_private_dataset.py` and the small
cross-shard `audit_private_expansion.py` check. No model fitting in this loop.

230accepted (192stable,38physicalunstable);9numerical TDS failures and1unconfirmed
outcome discarded in full. All230 pass finite/core/full-state/algebraic mapping,
uniform sampling, no frozen/padded states, solver-at-LOS and exact physical
outcome checks.220dynamic states; maxCOI residual1.17e-13; stable Pm variation
up to4.43368systempu. The same physical signature and equilibrium match Pilot
exactly. All15new busgroups/incident lines are represented and disjoint from
the Pilot; combined physical corpus361+230=591.48tests/Ruff/syntax pass.

New splits: TRAIN142(115stable/27unstable),validation43(34/9),fresh frozen
test45(43/2). At50ms, prospective eligible counts are133/39/43. Original TRAIN
andvalidation retain their roles; original consumed test remains excluded from
any fitting. Expanded development therefore has369TRAIN events(351eligible)
and117validation events(109eligible). Raw arrays remain in immutable shards.

Non-test coverage increases: using only Pilot TRAIN normalization, new TRAIN
nearest-neighbor median distance to Pilot TRAIN is1.47668 versus Pilot TRAIN
leave-one-out .73446; newvalidation median1.45605. All15newgroups enlarge
physical clearing-state coverage; this is descriptive, not predictive success.

Important QA limitation: both unstable events in fresh test terminate before
50ms. Thus its43prospective events are allstable; this holdout alone cannot
test missed instability for a50ms observer. Do not present a zero false-stable
rate with an empty denominator. Keep these groups frozen; before final
confirmation, a separately preregistered raw-data-only duration supplement will
be needed to obtain prospective unstable cases. Do not inspect model scores to
select that supplement, change labels, or reuse the consumed Pilot test.

All preregistered0039 data gates pass; preserve the holdout adequacy caveat.
The user's new authoritative plan is TASK2_after_loop_39.md. M3 is a fixed
physical scaffold, not the final research endpoint. Next: TRAIN-only electrical
residual atlas, then rollout-validated instantaneous correction, topology-aware
dynamic mode discovery, sparse/energy-compatible ODE, and only then observer.
User-owned loop39.zip is excluded and untouched.
