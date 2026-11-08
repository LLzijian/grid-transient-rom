# Loop0078 attempt1 — implementation failure, not a scientific test

Question and gates remain `reports/task2_structure_campaign_0078.md`.
The fixed1200 three-arm screen is **not completed or adjudicated**.

The first G-auto growth at400 added unit-v directions19 and18, i.e. p9
and p8. Both already have independently trainable base Rpp coordinates.
The exported coefficients were base3.029513388135643 and12.277515011433193,
with an additional.001 on each same diagonal. These are exactly duplicate
matrix/vector-field directions, not new structure. The selector excluded
previous extra R groups but omitted the existing base damping/Rpp groups.
This violates the preregistered exclusion of existing/duplicate groups.
An independent export-only check confirmed the algebra; no new dataset or
validation evaluation was needed to discover or diagnose this defect.

All three attempt1 processes were stopped to retain one matched source-code
version for the corrected screen. Last emitted update numbers were C600,
G-auto425, G-fixed275; these are last logged, not exact interruption steps.
They had not triggered nonfinite-loss/gradient/field checks. No1200 endpoint,
validation screen, hypothesis rejection, promotion, or extension is claimed.

Preserved artifacts:

- `private_data/task2_structure_0078_{C-free,G-fixed,G-auto}`: initial
  checkpoints and all346-event risk records; C/G-auto400 snapshots.
- `private_data/task2_structure_0078_G-auto/step0400_after_growth`:
  numeric export, state, optimizer, RNG and group metadata proving the issue.
- Original no-update smoke artifacts remain valid for their source version.
- Attempt source implementation7cd5c68; launchc3786a7. The running trainer,
  core and grammar were not modified while those processes were active.

No files were deleted. Unsaved in-memory updates after the last checkpoint
and full partial-run loss traces are not recoverable; only emitted logs and
saved checkpoints are evidence. Accepted M3/0058/S and dataset/splits remain
unchanged. No TEST payload or new candidate validation metric was read.

Correction: include nonzero base damping/Rpp coordinate directions in the
same existing-R exclusion set. Enumerate the same23 candidate directions;
skip exhausted/non-descent slots without substitution. This enforces the
original rule; it does not add groups or change any loss, coefficient,
optimizer, threshold, schedule, seed, data or step budget. Rerun fresh seed78
smokes and all three1200 fits in distinct attempt2 directories. Loop0078
continues, and no scientific conclusion is drawn from attempt1 performance.

Correction verification (attempt2, source6382d3b): all43 targeted checks and
three no-update smokes pass; initial losses and all recorded gradient norms
match attempt1 exactly. At growth400 the corrected G-auto adds7 groups:
4 H,2 J,1 R at v22 (q2, zero-based); the other R slot is unfilled. Its
effective-coordinate upper bound is136. Independent duplicate audit passes.
All other G-auto400 state tensors and the entire C-free400 state are bitwise
identical to attempt1, isolating the change to the defective R selection.
These are implementation/replay checks, not a scientific screen pass.
