# Loop0039: formal clearing-state coverage expansion, no model refit

Before generation, checkpoint f16e7d4. TASK1 formal-stage trigger is satisfied:
Pilot QA and M2/M3-Pm milestones passed. The first frozen test0038 exposes
boundary errors and only4independent groups. Hypothesis: the15remaining IEEE39
faultbusgroups materially enlarge clearing-state coverage while keeping the
same autonomous system and the exact accepted data-quality contract.

Use existing generator/format/QA. Only add an optional config path and expansion
event prefix; no new schema,manifest,registry or data framework. The original
Pilot file/config are immutable. New config private_dataset_expansion.json:
15buses[11,18,24,27,28,30,31,32,33,34,35,36,37,38,39] x same8durations x
same2reactances=240attempts,4workers,seed339. Original controllers/PSS, operating
point, solver tolerances,15s stable horizon, physicalLOS hold rule and whole-event
numerical failure rejection unchanged. Faults remain documented bus endpoints.

New TRAIN groups:11,18,24,27,28,30,31,32,33; validation:34,35,36;
new frozen test:37,38,39. All15 chosen incident lines are distinct and unused by
the Pilot. Old testgroups7/13/19/25 are consumed and are never training data.
New frozen groups will not enter model selection. This expansion is not a
post-hoc rerun of the first confirmatory experiment.

Acceptance: all saved trajectories pass existing mechanical QA; same physical
signature/state mapping/equilibrium; at least139 new accepted events so the
combined physical dataset reaches500; new stable and physicalunstable outcomes;
all15newbusgroups represented; at least8newTRAIN and3newvalidation groups.
Report numerical failure rates and prospective50ms eligibility separately.
Record coverage changes using only non-test initial states and Pilot TRAIN
scales (nearest-neighbor distances/PCA descriptive), not model accuracy or
labels as filtering rules. Do not discard hard events or change the240attempt
budget if a gate fails. No polling/write access to HDF5 while generator writes.

Physical corpus consists of the immutable Pilot and expansion shards in the
same format; no consolidation/copy framework is necessary. Fresh modeling
development may use originalTRAIN+validation with their roles preserved plus
newTRAIN+validation; neither old consumed test nor fresh frozen test may tune.
While simulation runs, only independent read-only diagnosis or preparation on
already-approved development data is allowed. Continue monitoring to result;
a running process is not a blocker or task completion.
