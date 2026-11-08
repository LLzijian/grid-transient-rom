# Loop0061 — long sparse latent-order audit

## Question / hypothesis

Does frozen Loop0060 sparse5s retain three effectively driven/observable latent
directions with independent long-autonomous predictive value? Preregistration:
`reports/task2_long_sparse_audit.md`, checkpoint b68e4f1; implementation
cb12c14/c11bce9. No retraining, whitening, library/topology/encoder changes,
or TEST access. This is an audit of the provisional long candidate, not M4
acceptance or a global minimum-order proof.

## Evidence

Sources: `private_data/task2_long_sparse_audit_0061/{evidence.json,matrices.npz}`
and `private_data/task2_long_sparse_rollout_0061/{evidence.json,predictions.npz}`.
Model SHA256 d34d9f5c36fb296e1d49e59f5f7b18a5da33b56bf371a16169f82bc25664995c.
All346 eligible full TRAIN trajectories and10,486 representative points:
reference/regional O/R/PBH ranks3 at1e-12/1e-10/1e-8. Cq singular values
3.367754/2.111979/1.776848. O singular values5.859743/4.331921/2.354443,
condition2.488802; TRAIN-scaled reference R7.742963/4.322829/1.770414,
condition4.373533. Full spectra and regional conditions remain in evidence.

Aq poles -.709166050 and -.950450845 +/- .273284666i: real decay time1.410s
and a strongly damped conjugate block (decay1.052s, frequency.043495Hz,
damping ratio.961061). These are not three distinct slow oscillations or the
full closed-loop modes. T5 regional HSVs6.51426/4.05638/1.39986; neither
real block has negligible actual TRAIN forcing or initial output excitation.
Raw singular values depend on coordinates/units; no covariance gate is used.

All8 fixed interventions completed with finite predictions at all required
steps. Primary RK4.0025, companion.005, prescribed diagnostic.00125 for each
arm after an all-event numerical comparison failed. Initial state is cached
once; no future observations enter rollout. Complete5s cohort95, all genuine
prefixes107, unstable prefixes12. Primary5s metrics (mean/pooled/CVaR):

| Arm | Mean | Pooled | CVaR |
| --- | ---: | ---: | ---: |
| Full | .288326 | .333571 | .628208 |
| Memory off (alpha retained) | 1.589443 | 6.103085 | 12.034830 |
| Output component1 off | 1.383346 | 6.029400 | 10.690430 |
| Output component2 off | .347155 | .401677 | .750724 |
| Output component3 off | .376752 | .445548 | .884809 |
| TRAIN T5 balanced rank2 | .306828 | .360238 | .711826 |
| Real mode removed, rank2 | 3.269980 | 4.851534 | 10.996500 |
| Conjugate block removed, rank1 | 2.751062 | 8.087084 | 21.072144 |

The two rank2 ratios are1.064171/1.079943/1.133106 and
11.341253/14.544236/17.504554. Both fail all three predeclared retention
limits1.02/1.02/1.05. All complete5s comparisons are numerically valid:
max discrepancy is3.8414e-4 across arms; full/balanced/real-mode-removal
maxima4.7515e-7/5.2337e-7/5.9906e-6. Thus the primary rejection is not a
step-size failure. No projection horizon search or coefficient refit followed.

Balanced rank2 also worsens complete1/2/8s and unstable-prefix metrics:
1s .343717/.533665/1.058122 versus full.298393/.373210/.767441;
2s .350327/.414868/.831408 versus.328373/.381051/.734655;
8s .262785/.307988/.606057 versus.240233/.277769/.522296;
unstable-prefix.877593/2.041120/2.637579 versus.724623/1.230200/1.667156.
Component3 removal slightly improves unstable CVaR1.651041 versus1.667156
while worsening unstable mean/pooled and all complete5s metrics. Component
importance does not mean every intervention damages every event/metric.

## Numerical boundaries / implementation evidence

87 repository tests pass. Actual field/one-step errors3.55e-15/0; same-dt
full1s physical replication9.88e-13. Independent matrix arithmetic reproduces
O/R/Q/Gramians/HSVs exactly. Full.0025 cache8s maximum absolute7.44e-8
is confined outside observed prefixes; observed-prefix maximum1.05e-13.
The erroneously added8s absolute replica gate was corrected to match the
preregistration before intervention execution, with raw evidence retained.

Finite predictions do NOT imply convergence or stability. Full all-fixed8s
discrepancy.026908 (.0025 vs.005) and.001672 (.00125 vs.0025) both exceed
.001; genuine-prefix maxima5.45e-7 and3.42e-8 remain valid. No quantitative
claim for all unobserved unstable8s continuations is made.

Memory-off, output1-off, real-mode-removal, and conjugate-block-removal fail
the primary complete8s convergence screen. The one finer diagnostic still
fails for memory/output1/conjugate-block removal (.003079/.002531/.021248);
real-mode-removal gives.000416. Their raw8s error values are not promoted
to primary numerically confirmed magnitudes; the decisive comparison is5s.
Original0060 gates remain failed. All per-event numerical scopes are saved.

## Classification / interpretation

Classification: diagnostic; root accepts the preregistered order-audit
conclusion, not a new final model. Independent rollout QA passes with zero
mismatches at1e-10 across per-event errors, aggregates and numerical scopes:
`.loop-work/long0061_rollout_independent_checks.json`. The checker reads
saved arrays only and does not reuse evaluator metric functions.
The primary evidence supports three locally effective directions and rejects
the two specified rank2 projections. It does not prove that no differently
trained or nonlinear two-state realization exists. No reduced model is
promoted, no accepted weights are overwritten, and no rollout/stability
claim is inferred from Aq alone. Full-system storage and basin work remain.

## Next implication

Retain r3 for the current long candidate; do not add dimensions or losses.
Proceed to TRAIN group coefficient/support identifiability and consistency:
the mask was frozen during0060, so identical masks cannot demonstrate
rediscovered robust support. The next smallest diagnostic must distinguish
coefficient identifiability from consistency with the frozen encoder, whose
latent reference is not physical ground truth. Formal full-system stability
and final TEST remain after these unresolved model-structure checks.
