# Loop0038 — strict confirmatory gate failed; Pilot test is now consumed

Checkpoint85b4f3d; frozen release implementation16771ff. Command:
`python experiments/run_private_confirmatory.py`. Single-pass ledger complete:
57eligible test events read exactly once;3predecision physical-LOS events
(0188,0190,0270) listed as ineligible, never padded. No uncertain retry. This
new private test is distinct from the historical contaminated canonical data.

|8s,48complete stable truths|M2|M3|
|---|---:|---:|
|Mean normalized RMSE|12.557695|0.435344|
|Pooled normalized RMSE|20.865083|0.626424|
|CVaR10|49.252440|1.295133|

The trajectory contribution thresholds pass at2/5/8s andprefix. However the
joint preregistered claim **fails**: M3 has1false-stable/9unstable versus M2's0,
and1false-unstable/48stable (M2:18). M3balanced accuracy=.934028 does not override
the no-new-missed-instability gate. The two frozen counterexamples are:

- pilot_0269: physical LOS at1.10s after clearing, M3 noLOS in8s; not a horizon
  censoring artifact. Bus19,duration.4s,xf.01 are reporting metadata only.
- pilot_0348: physically stable15s, M3 predictsLOS at.49s after decision.
  Bus25,duration.4s,xf.001. These events cannot be used to tune a revised model.

Only4independent faultbusgroups are represented. Two contain a classification
error; the one-sided95% group-any-error probability upper bound is.902389
(exchangeable groups assumption). This does not support an engineering safety
guarantee. Fixed paired-group bootstrap8s pooled-ratio interval [.02082,.03829]
supports relative trajectory improvement only. Margin ranking AUC=.99537, but
the conservative sufficient bound covers0/57events, not useful safety coverage.

G1/G2: syntax,Ruff,48tests pass. G3: hashes,disjoint IDs,causal current state,
fixed8s prediction and durable claim-before-read ledger pass. G4: independent
recomputation from saved canonical truths and predictions checks all353records
per model (706total), maximum relative SSE discrepancy0; no source test reread.
QA script `.loop-work/check_confirmatory_0038.py`; saved truth/rollouts/ledger
remain in private_data/task2_confirmatory_0038. G6JSON finite, exact code/model/
split identities included. The scientific G5 failure is genuine.

Revert the release implementation/preregistration from the experiment branch;
preserve this report and all numerical evidence. M3 remains the previously
accepted development baseline, not an accepted no-missed-instability predictor.
Do not refit M3 or revive rejected M4 using this test. No threshold modification,
case deletion or post-test rescue. User-owned kiki.zip is untouched.

Next: TASK1 formal expansion after the already-passed Pilot/M2/M3 milestones,
using new faultbusgroups and the same full-order system/data definitions.
The four original testgroups remain consumed; newly segregated groups supply
fresh development evidence and a genuinely new holdout for any later revision.
More independent clearing-state coverage is needed before claiming robust
boundary prediction. This failed confirmation is not completion of the Goal.
