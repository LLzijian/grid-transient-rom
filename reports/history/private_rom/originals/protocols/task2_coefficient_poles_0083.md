# Loop0083 — did the coefficient intervention introduce local instability?

Authority: after75; prospective diagnostic stage after closing0082. No model
promotion. This document does not authorize execution before0082 closure.

## Question and finite budget

0082 lowers the conditional surrogate residual but worsens autonomous TRAIN
prediction. Test the smallest dynamical explanation: the fixed coefficient
intervention changes a strictly stable coupled equilibrium into an unstable one.
This is a property of two frozen learned systems, not attribution of all rollout
error, stability of the full-order system, or modal fidelity.

One model-only comparison: original G-fixed3000 and corrected0082 attempt2.
No training, coefficient interpolation, ablation, new candidate, trajectory,
operating-region sweep, VAL or TEST. Each model receives exactly one AD full33
Jacobian and two central finite-difference Jacobians, h=1e-5 and1e-6, at zero.
Main wall cap5min; one implementation QA repair before execution, no execution
retry without stage review. Synthetic tests and independent cached arithmetic
are QA, not extra model evaluations; count their cost. Close/review this stage
after the two-model result; no automatic additional branch.

Baseline: `private_data/task2_structure_0078_extend_G-fixed/step3000`, model SHA
2667480a75d7672ad49ed66c79731e754676bae52574229dbb5f98c581ac890b.
Intervention: `private_data/task2_coefficient_block_0082_attempt2/candidate`, SHA
72fce608287d9e13dd23957e860ad5d20abb2e3130cd883e0f493e1021348db5.
All source/model/export/normalization hashes before and after must agree.
Use the actual deployed field including every port, with interface coordinates
(delta,omega,p,q), not an isolated latent block or port-disabled dynamics.

## Matrices, numerical checks and gates

At the common equilibrium z=0 compute A33=df/dz. Require finite matrices,
max|f(0)|<=1e-10, gauge-shift field difference for delta+.37<=1e-10, and
max|A33*(ones10,zeros23)|<=1e-10. Central FD versus AD relative Frobenius
error norm(A-FD)/max(1e-300,norm(A))<=1e-7 at BOTH steps.

Let N be any orthonormal basis of the nullspace of ones(1,10), and
Q=block_diag(N,I23); A32=Q.T*A33*Q. Remove ONLY common angle, retain all ten
omega components (including common frequency), ten p and three q states.
Require max|Q.T Q-I|<=1e-12. Save Q and all full/reduced matrices.

Compute all32 eigenpairs of A32, save complex eigenvalues/eigenvectors, report
spectral abscissa, counts with real part>1e-6, <-1e-6 and in the intervening
band, and eigenvector condition as descriptive numerical information. Require
each relative eigen residual norm(Av-lambda*v)/[(norm(A)+abs(lambda))*norm(v)]
<=1e-9. The two FD spectral abscissae must each differ from AD by<=1e-6/s.
No selectable FD winner. A near-zero SPECTRAL ABSCISSA or failed QA leaves the strict
introduced-instability hypothesis inconclusive, rather than rescue it.

Support requires baseline spectral abscissa<-1e-6/s AND candidate>+1e-6/s,
with all QA satisfied. If both are strictly stable, reject this explanation.
If baseline is already unstable, reject the claim that instability was newly
introduced; report the observed difference without a new post-hoc gate.

Unit tests: known linear33 field with one common-angle null direction and
stable/unstable quotient spectra; finite-difference recovery; verify that Q
retains a uniform speed direction. Independent NumPy reconstruction from
saved A33/Q must reproduce A32 and spectral abscissa within1e-10 relative
(denominator max(1,abs(reference))), without evaluating another model field.

## Interpretation and next-stage trigger

Even support establishes only a local spectral change; it does not prove this
caused all finite-amplitude forecast harm. Both-stable rejects this simple
explanation, not nonlinear amplification, nonnormal transients, reference
inconsistency, or the whole structured family. No L2 certificate/L3 matching
claim and no change to accepted M3/0058/S. Preserve both rejected0082 artifacts.
After closure, root reviews this evidence before authorizing any new intervention.

## Independent challenge disposition, before model evaluation

stage81_challenge confirms the gauge quotient: A*g=0 makes the removed
direction invariant; the chosen complementary space need not itself be invariant.
Root adds the explicit check max|Q.T*g|<=1e-12, as well as orthonormality.
The near-zero rule above concerns the spectral abscissa, not every pole: a
clearly unstable candidate may also contain a separate near-zero mode.

Root accepts the sensitivity objection. Before results, additionally define a
conservative empirical sensitivity radius r=kappa2(V)*max(||A32-FD32_h||2 over
both h, eps64*||A32||2), with unit-normalized right eigenvectors V. Require
r<.1*abs(spectral_abscissa) for BOTH models to interpret the signs; otherwise
report numerical sensitivity/inconclusive. Save r and its components, do not
choose a step or relax any earlier gate. This is a robustness check against
observed Jacobian discrepancies, not a rigorous bound on all floating-point
errors, nor a formal certificate for exact equations. Even a supported result
is numerical local spectral evidence, not a nonlinear or full-order certificate.
This clarification is prospective and does not change0082 or inspect0083 poles.
