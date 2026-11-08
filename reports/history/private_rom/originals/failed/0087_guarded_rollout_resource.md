# Loop0087 — resource screen did not authorize training

Work type: implementation/timing QA. Result: **inconclusive (resource screen)**.
Model promotion: **none**. This is not optimization or scientific rejection.

Preregistration78f2dac, implementationb1684f0. Command: Python
experiments/train_guarded_rollout_0087.py --preflight. Root session20558 exited0.
Artifacts `private_data/task2_guarded_rollout_0087_preflight/evidence.json`.
Five0087 plus six0086 synthetic tests passed (root4.41s).

The single registered seed87/batch32 no-update preflight completed in35.8327s.
Only three forward/backward evaluations, no optimizer update, projection SDP,
candidate or autonomous evaluation. All gradients finite, entire model state
and input/source hashes unchanged. Original TRAIN only; unused VAL materialized
by inherited reader, TEST not read.

| Horizon | Forward/backward seconds | Theta gradient L2 |
|---|---:|---:|
| 1s | 3.860103 | 16.983463 |
| 2s | 6.989050 | 16.824536 |
| 5s | 16.655204 | 17.154101 |

Registered estimate100*sum(times)+300*.5 =2900.4357s (~48.34min), above
the2400s launch gate. Thus **0087 training is NOT authorized and will not run**.
No lowered gate, shortened selectable run, intermediate model or silent retry.
Both the no-update preflight and its resource conclusion are retained.

Stage closed; actual cost35.8327s plus synthetic QA. The direct-objective
scientific question is untested. Accepted M3/0058/S and G3000 unchanged.
Root is reviewing whether one separately justified compute allocation can cover
the same full300-update experiment, based solely on measured cost before any
optimization result. No such run is authorized by this conclusion alone; budget
review must explicitly count this preflight and avoid automatic rollover.
