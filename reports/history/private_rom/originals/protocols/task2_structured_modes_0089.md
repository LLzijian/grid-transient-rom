# Loop0089 — cached physical-output modes of two frozen structured systems

Startc6f526f; authority after75. Work type diagnostic; promotion none.
0087/0088 direct-coefficient stage remains CLOSED. No optimization extension.
Independent route challenge obtained before registration; disposition below.

## Question, objects and finite budget

Do G3000 or the final0088 system reproduce a majority of the SAME13 verified
FOM electromechanical physical-output modes? Each object separately needs
>=7 distinct valid partners under unchanged0071/0078 rules. Their difference
is descriptive, not a causal effect of the guard or optimization procedure.
Failure does not reject state order, model family or all structured models.
Passing is only an L3 diagnostic screen, not an L1 shortlist or promotion.

Use ONLY0083 `baseline_A33` (G3000, NOT0083 candidate belonging to0082) and
0088 `actual_A33` (the exported/reloaded final candidate). Same equilibrium,
external coordinates(delta10,omega10,p10,q3), fixed inertia and units.
No models loaded, H5/VAL/TEST, fit, rollout, new derivatives, FOM eigensolve,
parameter sweep or new targets. Two32x32 eigensolves and fixed assignments
maximum in ONE main invocation,10min cap. One independent cached arithmetic
check,5min cap; synthetic QA separately counted, one preexecution repair.
Source/QA failure leaves science unresolved; preserve evidence, no silent retry.
Close/review at completion or budget limit; no automatic branch rollover.

## Cached evidence inheritance (explicit change from0078 recomputation)

This stage authorizes reuse of the validated0078 FOM cache instead of another
699-state descriptor eigensolve. Hash original descriptor/evidence, verified
metadata, concrete model/norm/export/source files and caches before/after.
Verify all applicable historical hash links, unchanged source/equilibrium,
and the prior successful unit/residual/target-reliability/independent checks.
Report these as INHERITED, not newly recomputed derivatives/FOM eigenpairs.

Exact cache anchors:

- 0083 matrices.npz:46bcaa3480fdb422f3f98ca069243ca8f911bd44ae7c061f4eba75b77199d212
- 0083 evidence.json:acb8adf7a42a1d6857ae8c24e6fb736dec80e9daa20c7105c524a1a6ef1cde6a
- 0088 fixed_arrays.npz:9725ee435f9a75600917dac3a90c5621d5d4237f8e157c7b3a1d450cd77407d6
- 0088 evidence.json:647d287fcaef90edc009de5f4225ec3fd623d91d726fedc0bb21cb18864cc92f
- 0078 modal_arrays.npz:96ee08704a5dde68e8e65206109c993e0768dc378e4d3c344e0bdd43bb5025da
- 0078 evidence.json:cb69655ad6ce4bcb79b90d56e479ed5b5164f699d90472ce4a5452fb79d801ac
- 0078 independent_check.json:7ebd8ce0c1718e6d56d78eedff0306f43aad7deda27d06860a20901a39b37525

Original G3000 model SHA2667480a75d7672ad49ed66c79731e754676bae52574229dbb5f98c581ac890b;
0088 final SHA5bab5fba84fb6261821c8654ffa6e33b4e015d12f472114028d8fe0e119577ba.
0083 valid baseline origin/gauge/AD/two-stepFD checks and0088 validated affine
tensor plus actual AD/export/frozen-key checks establish derivative provenance.
Require0088 independent training audit pass and unchanged anchors recorded there.
FOM cache retains150 finite/549 infinite/0 indeterminate, residual<=1e-8,
150-pole matching<=1e-6, exact13 reliable targets; inherit from verified0078.
Do not substitute an algebraically reduced160-mode matrix.

## Physical output and modal analysis

Use same Q33x32, removing ONLY common angle, retaining all10 frequencies.
Check finite/shapes, Q'Q=I, QQ'=diag(I10-11'/10,I23), A33*common_angle=0,
and agreement with saved reduced matrices to1e-10 maxrelative. Inertia equals
cached0078/M3 mass to1e-10. Confirm generator1..10 order and FOM rad/pu
conversion in inherited source. Output y=[Pcoi*delta/.47675314588683515;
FULL omega_rad_s/1.0033667249602924], Pcoi=I-1*m'/sum(m). FOM omega uses
2*pi*60; ROM uses1. Recompute cached13 target physical outputs from their
saved descriptor right vectors (not a new solve), agreement<=1e-10.

Reuse ordinary_modes/assignment in compare_physical_modes.py unchanged.
Each fresh32-state eigenpair residual<=1e-8. Retain full spectra, eigenvectors,
physical outputs, condition(V) and minimum relative pairwise pole separation.
Flag condition(V)>1e8 or relative separation<1e-6 as individual-mode sensitivity
concerns; do not drop modes, change gates or introduce eigenspace matching.

Targets remain62,64,66,68,71,73,77,79,84,91,98,100,104. Positive-imaginary
ROM poles>1e-6 only for pairing; output norm>1e-12. Damping ratio=-Re/abs,
frequency=abs(Im)/(2*pi), FOM-denominator frequency difference<=.15,
absolute damping-ratio difference<=.05, complex phase/scale-invariant MAC>=.8.
Same cost=(normalized frequency^2+damping^2+MAC-loss^2)/1000 on eligible
edges; invalid1e6 and13 dummy columns1; Hungarian cardinality-first matching.
Save all pair metrics and unmatched lists. Report each gate's availability
per target to describe frequency/damping/shape limitations, not new selection.

## Verification, interpretation and review

Reuse unchanged0071 phase/scale, assignment and unit tests; add targeted
synthetics for wrong cached object, SHA mismatch and common-frequency removal.
Root reviews implementation before ONE real run. Independent NumPy arithmetic
reconstructs physical outputs, all pair metrics and assignment from saved
arrays to1e-11 absolute, exact eligibility/matched sets. No new eigensolves.
Fresh `private_data/task2_structured_modes_0089`; refuse overwrite. Preserve
full provenance and timings; conclusions in one separate report.

Root adopts independent challenge: this explicitly changes only FOM cache
execution, not scientific matching. Original/candidate identities must be
unambiguous; inherit QA honestly and inspect conditioning. Neither a low count
nor a count change identifies why TRAIN gains were small, a unique missing
mechanism, guard causality or required latent dimension. No regional energy
claim follows. Results inform the next smallest physically motivated question,
not additional0088 updates or automatic model complexity.

Preexecution QA record: root rejected the initial unexecuted drafts. Main used
nonexistent FOM cache keys, misread old independent-QA schema and omitted
required provenance checks; checker used unnormalized assignment costs and
incorrect array orientation, without complete per-pair comparisons. No real
eigensolve/audit invocation occurred. The ONE allowed coherent preexecution
repair is now used to implement the already registered formulas/schema checks
and meaningful synthetic failure tests. No scientific gate or budget changed.
Root completed that still-unaccepted repair when the worker's revision left
schema/provenance gaps; no additional real invocation or repair allocation.
Final main implementation has16 combined targeted/inherited tests passing
(root1.92s), including real validators for missing QA, wrong SHA/object and
common-frequency preservation. Historical source links are checked explicitly.
