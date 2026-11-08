# Loop0035: frozen M3 local modal structure, with full-order comparison

Preregistered before implementation; checkpoint bc3a43c. M3 remains unchanged.
Claim: the frozen explicit M3 has a locally asymptotically stable equilibrium
modulo uniform angle gauge; its modal frequencies, damping, participation and
parameter sensitivities can be quantified. Compare against the unchanged
ANDES full-order post-fault equilibrium without forcing modal equivalence.

Compute analytic30-state Jacobian, remove exactly one rotational angle to get29
states, compare centered field finite differences (1e-6 and5e-7) to relative
error<1e-6. Require equilibrium residual<1e-10 and all29 real parts<-1e-6.
Report eigen residual<1e-8, all positive-imaginary modes, and normalized absolute
left/right participation. Dominant-mode derivatives for each positive coupling,
each governor a/b and existing nonzero damping use analytic eigenderivatives;
check centered relative parameter steps1e-4 against matched closest eigenvalue
to relative error<1e-4. Do not retune any parameter.

Full-order comparison uses the same ANDES2.0.0 ieee39_full.xlsx, all original
controllers/PSS, timed faults disabled, identical physical signature and stored
equilibrium (maxstate error<1e-8). Run EIG's documented algebraic Schur reduction,
save reduced state matrix, eigenvalues and state names; independently recompute
eigen residual<1e-8. Record zero/time-constant/algebraic eliminations. Report
full-order dominant nonzero eigenvalue and oscillatory modes with>=10% normalized
GENROU angle/speed participation; nearest frequency comparisons are descriptive,
not a fitted mode assignment or proof of structural fidelity.

Acceptance is numerical validity of the local M3 stability claim and honest
comparison, not an accuracy claim for full-order damping. Any quantitative
mismatch limits the eventual spectral margin's scope to the reduced ODE. No
test trajectories, parameter selection, full-order modifications, plots, or new
general modal framework. Nonlinear energy boundaries belong to a later loop.

Numerical-audit amendment before acceptance: initial ANDES output contains a
nearly defective eigenvector matrix and positive modes involving zero-parameter
PSS filter realizations (all A1..A6=0). Save original DAE Jacobian blocks and
cross-check the descriptor pencil independently before any full-order stability
inference. This verifies reduction correctness; it does not alter physical
parameters, remove controllers, relax gates or accept the provisional summary.

Independent raw-pencil check finds150 finite modes with no positive real parts,
versus160 modes/6positive in the routine's reduced matrix. Therefore the latter
is rejected for physical interpretation. Use original descriptor QZ; require
finite eigenpair residual<1e-8, no indeterminate0/0 pairs, and eigenvalue agreement
<1e-6 after row equilibration. Descriptor participation uses the finite-mode
projector v*w^H*E/(w^H*E*v), not inversion of the defective ordinary eigenbasis.
Keep both outputs and explicitly report the reduction discrepancy. Do not patch
the installed simulator or alter the frozen data generator in this loop.

The raw pencil has20 repeated real modes at-50/s with near-zero individual
left-E-right overlaps. Their eigenvalues remain reported, but individual state
participations are non-identifiable and must be null rather than fabricated by
dividing tiny overlaps. This affects no oscillatory/dominant mode. Keep the
eigenvalue cross-check threshold unchanged; no modes are deleted.
