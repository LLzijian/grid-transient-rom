# Loop0073 — bounded dynamics–energy joint correction: rejected

Question/hypothesis: can the registered B2 bias remove the origin storage
obstruction with the original224-term family and <=24 final energy terms,
while retaining prediction/safety? Protocol `reports/task2_joint_energy_0073.md`.
Independent S start; implementation5f734ab;800 matched J steps,2498.01s.
Model SHA a2be5b650ee37f4da8b98effa41268dda33985fb632a4e094eccfd6b8aeb2bdb.

Change: exact J0067 data objective plus the single first-TRAIN-calibrated
lambdaH=.755443329777 energy bias, with ten registered soft storage solves
and one step200 energy pruning. No teacher, r/library expansion, Xi support
change, optimizer reset, weight/budget adjustment or validation selection.

Evidence: all800 draws and frozen/source/inactive/f(0) contracts pass.
Independent recomputation reproduces all ten H/D/beta/residual records
exactly; full learned Jacobian relative error1.23e-15. Final24-term H has
min eig(VHV)=1.3099481e-5, but min eig(VDV)=-7.6164651 fails the required
1e-6 gate. Final solver status optimal uses t=7.6164751; it is NOT a strict
dissipation solution. Actual solver H/D residuals0/2.42e-9 pass numerical QA.
All32 coupled poles remain negative (dominant -.149489+/-10.136555i/s),
which does not establish compatibility of the restricted energy expression.

Diagnosis: initial t8.213469 -> step200 full8.071789 -> immediately pruned
8.176107 -> final7.616475 (7.27% reduction from start, still far from0).
Pruning adds .104317 at step200; the obstruction was already large before
pruning. This does not prove all224 terms infeasible on the final learned
ODE, nor prove that a different optimization/energy family cannot work.

Classification: rejected fixed-budget algorithm hypothesis, not a failure
of the Jacobian/solver implementation or a rejection of r3/local stability.
No VAL rollout/L1 screen, regional energy fit, rho, coverage or certificate
is claimed because the necessary first gate failed. The inherited reader
loaded TRAIN and validation records; only TRAIN entered sampling, gradients
and calibration. No validation metric selection or TEST payload access.

Artifacts: `private_data/task2_joint_energy_0073/{evidence.json,model.pt,storage.npz}`,
`private_data/task2_joint_energy_integrity_0073/{qa_energy73.json,qa_energy73_jacobians.npz}`.
S/0058/M3 remain unchanged; rejected checkpoint is quarantined and its
reproducer retained, not deployed. No shared scientific ODE was changed.
Next: do not sweep energy weights/steps/support budgets. Use after66's single
prospective r4 sparse candidate to test an additional electrical-memory
channel, motivated by the persistent common/differential-motion tradeoff
and the known nonzero force residual outside the three-channel span.
This is a bounded representation probe, not a declaration that r3 is invalid.
