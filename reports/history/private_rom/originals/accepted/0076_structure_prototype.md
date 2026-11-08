# Loop0076 — shared-gradient prototype: accepted for implementation QA

## Question / hypothesis

Can one explicit shared-gradient generator preserve the required identities,
contain M3 exactly, and represent both recovery/slip and hidden exchange with
nonconservative supply? See `reports/task2_structure_prototype_0076.md`.

## Change

Minimal analytic storage grammar and constant-J/PSD-R/low-rank-port model,
internal(delta,pi,p,q) with pi=M*omega, plus independent numeric export and
an origin-anchored20->8->3 GRU. Gamma/kappa are signed; dissipation is
nonnegative with exact absent-term masks. No independent Xi field in G.
No fit, event data, validation or TEST was used. M3 model asset only.

## Evidence

Implementation87d0203; command:
`python experiments/check_structure_prototype.py --output private_data/task2_structure_prototype_0076`.
Seed76,64 random float64 states, nonunit mass/scales and all six primitives.

- Analytic/autograd gradient max relative9.02e-15; finite differences at
  1e-5/1e-6 give2.36e-8/6.17e-7, both below1e-6.
- Kinematics and f(0) exact; gauge normalized1.05e-15; power-balance
  normalized9.51e-16; no positive dissipation violation.
- Exchange zero-power2.50e-16 and paired-removal residual1.17e-15.
- M3 physical-field normalized difference8.83e-15; independent NumPy
  export difference6.66e-16, Torch reload0. All active gradient checks pass.
- Both RK4 steps recover with final relative angle .0086054 and speed
  magnitude .104086 (<.15); the separate slip initial condition crosses pi.
- Hidden case both finite, shared-grid max difference1.18e-10 (<1e-4).
  Fine-step signed exchange transfer .0297313 and port work .00495522 are
  nonzero; instantaneous power residual<=1.20e-17 across both steps.

All16 preregistered QA gates pass.29 targeted core/boundary/data tests pass;
tests also check export after an optimizer update. Numeric export reports
104 raw trainable coordinates,96 effective-coordinate upper bound and556
uncollected scalar RHS contributions for the random QA fixture. The bound
is not an identifiability claim; centered trig expressions count as one atom.
Artifacts: `private_data/task2_structure_prototype_0076/evidence.json`,
`model_numeric_export.json`, `synthetic_trajectories.npz`.

## Classification / interpretation

Accepted implementation and template-expressivity QA only. Same-template
synthetics do not demonstrate structure discovery, IEEE39 prediction,
physical mechanism recovery, a stable region, or final model promotion.
Implementation errors found during review were repaired before this screen;
they were not treated as scientific rejection or repaired by changing gates.

## Next implication

Proceed immediately to the preregistered original-TRAIN matched0078 screen.
0077 is separately rejected; no boundary augmentation. Accepted M3/0058/S
assets remain unchanged. Fresh frozen TEST remains unopened.
