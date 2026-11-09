# Loop0078 follow-up — C3000 physical modes rejected

Question: does retained C-free3000 reproduce a majority of the13 fixed FOM
electromechanical targets under the unchanged0071 matching rules?
Preregistration: `reports/task2_control_modes_0078.md` (59c50d5/597ff77).
Implementation: e6023e3. No model changes, fitting or H5/TEST access.

## Evidence and decision

Only1/13 targets matches, below the registered >=7 gate. **Rejected** for
the majority-modal claim. The sole pair is FOM84/C14: C frequency
.6657434758 Hz, damping ratio .2051227127, relative frequency difference
.0379310045, absolute damping difference .0198806730, physical-output MAC
.9208404050. S also matches only FOM84 (1/13); M3 matches0/13. Unmatched
targets remain62,64,66,68,71,73,77,79,91,98,100,104; no target reselection.
Post-hoc decomposition of the already saved156 pairs finds10/13 targets
with no candidate MAC>=.8 (only84,91,104 have any). All first eight targets
62..79 have a frequency-qualified candidate but none passes MAC. Target91
has MAC-qualified candidates but none meets the damping bound;98/100/104
have no frequency-qualified candidate. These are descriptive failure modes,
not new gates or a demonstrated physical cause; tuning frequency/damping
alone cannot fix the missing output-shape matches in this fixed eigensystem.

All numerical/source QA passes: f(0)=0, gauge residual<=1.43e-14,
two central-difference relative Frobenius errors<=2.45e-11. The original
699-state descriptor reproduces150 finite/549 infinite/0 indeterminate
pairs, pole-match error0 and residual1.04e-15. Units/equilibrium and
before/after source hashes agree. Nine targeted tests passed before execution.
Independent cached NumPy arithmetic and cardinality-first assignment repeat
all156 pairs (13 targets by12 oscillatory candidates), maximum difference
1.67e-15, identical single match. No nearest-frequency or many-to-one rescue.

The full coupled32-state quotient removes common angle ONLY, retaining
common frequency. Its maximum pole real part is -.3443885875/s. This
supports local linear stability, not modal fidelity or a nonlinear region.

## Interpretation and next implication

C3000's trajectory-retention pass remains valid; better transient errors
and a stable local Jacobian do not establish faithful FOM output modes.
The evidence does not identify a unique cause or reject the whole closure
family. No small-perturbation confirmation, seed sweep, model promotion or
regional certificate follows from this failed modal screen. No implementation
rollback is needed: this was a read-only check and accepted M3/0058/S were
never replaced. Preserve C as a trajectory-control candidate.

The main0078 campaign remains OPEN pending the already authorized G-fixed
3000 endpoint and fixed retention evaluation. Do not alter that run or
rescue G-auto's failed1200 primary. Decide the next targeted hypothesis
after the matched-budget evidence is complete.

Evidence and reproducible arrays/scripts:
`private_data/task2_control_modes_0078/{evidence.json,modal_arrays.npz,
independent_check.py,independent_check.json}`; executable audit
`experiments/check_control_modes_0078.py`.
