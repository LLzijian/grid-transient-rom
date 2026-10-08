# Loop0096 — feasible conditioned point fails the autonomous/modal screen

Work type:model screen. Result:rejected. Model promotion:none.
Preregistratione1bc4e4; implementationa5d9097. Exactlyone frozen0095 returned
point, no newoptimization.0095 remains inconclusive/optimal-status failure.

| Registered quantity | Result | Gate |
|---|---:|---:|
| Coupled local alpha /s | -.141100065500479 | <=-.01, pass |
| FD/eigen sensitivity bound | 6.137761224e-8 | <.1abs(alpha), pass |
| Physical modal matches | 0/13 | >=7, fail |
| Autonomous average risk ratio | 2.529585369128 | <=.90, fail |
| Severe35 mean ratio | 3.195003012686 | <=1.05, fail |
| Severe worst4 ratio | 4.751068811094 | <=1.05, fail |

AverageTRAINrisk .893819553223 ->2.260992864473; severe35mean
.665464697750 ->2.126161714148; worst4 2.303873238747 ->10.945860289325.
All346events/1038rows, unchangedinitialstates/frozenqenc/realp0 and genuine
prefixes retained.12oscillatoryROM representatives still produce no eligible
one-to-one target matches; a stable spectrum is not modal fidelity.

Reload/masks/only-two-tensor changes, structuralpower/J/R/B/quotient QA pass.
Two actual AD Jacobians and132FDfieldcalls confirm analytic omega-row change.
FDrelativeerrors1.69406e-11/1.70033e-11; eigenresidual2.37111e-16.
Both registered autonomousintegration grids pass all finite-substep/prefix/
pairedmetric gates. Main changed only network/dampingrawparameters in the
standalone candidate; source and candidatehashes stable.

Independent cached audit passes: maxarrayrelative2.177085457e-16,
maxscalarabsolute1.421085472e-14. It verifies exact95coefficient sums, state
tensors, actualJacobian/FD/eigen/modal arithmetic and alltrajectoryrows; no
modelconstruction/optimizer/eigensolve/forecast. The failed95status is retained.

## Scope and stage review

Conditioned-balance nonworsening and57.16% harmonic-discrepancy capture do not
suffice for autonomousprediction or physicalmodal matching of this point.
These are verified failures of this frozen deployment, not proof of inadequate
latentorder, failed model family, or globally optimized training. Near-floor
damping is not robust positivity. No accepted/default update; no rollback
needed because only a fresh standalonecandidate was created.

Main49.4327851s + independent7.15735s + metadata20s =76.5901351s <=1400s.
Stage closed, no actualretry/refit/tolerancechange.19main/reusedsynthetics
pass1.81s and12independent/reusedsynthetics pass1.179s.
Commands: `python experiments/audit_witness_deployment_0096.py --output
private_data/task2_witness_deployment_0096`; `python
private_data/task2_witness_deployment_0096_independent/check_witness.py
--input-dir private_data/task2_witness_deployment_0096 --output-json
private_data/task2_witness_deployment_0096_independent/independent.json`.

Next question before another fit: does imaginary-frequency matrix capture
leave large directional balance defects at the actual COMPLEX FOM eigenvalues
and physical mode shapes? A bounded cached-operator diagnostic can separate
that mismatch from successful complex balance with sensitive pole/output
matching. It cannot by itself establish a cause of autonomouserror or family
inexpressivity. Independent route challenge requested/adopted conditioning,
eigenvector-phase invariance and exact eigenpair alignment; common-frequency
components must remain in the physical balance, not be projected away.

Artifacts: `private_data/task2_witness_deployment_0096` and `_independent`.
SHA256:
- evidence: `04e8b5700b6271e983436f36f707c2b2240f41e114aca8d1084b720eace057c0`
- arrays: `ad90932963116b973753305aba7db2ca0149168cb2308088d10db49fe215d783`
- trajectories: `b2347fa36bc4e14056ae7ad3c196de1ab866267bfe6d5c0bb7fac822c0238f81`
- candidate: `20fc8fda87f01f5addd7564eda1535d3d253615db900de937e3e58eb84f6ddb0`
- independent: `73190689f5d6c661520c9e13ceca06f1383eeff866ccf57d98c6420e56f7d290`
- checker: `70bd3b2bee8d3ba98a1fab17a6d46e9f044117ca4d0a2e0695b8176582fc99f1`

Accepted M3/0058/S remain unchanged. FOM/conditionedTRAIN-informed development,
not independentconfirmation. FreshTEST unopened; consumedPilotTEST excluded.
