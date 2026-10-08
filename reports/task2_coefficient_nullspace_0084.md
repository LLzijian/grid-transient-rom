# Loop0084 — feasibility of preserving the equilibrium linearization

Authority after75; stage after0083, diagnostic only, promotion none.
The0082 intervention introduced unstable poles. Before fitting another model,
ask whether its EXISTING24 coefficient block has any direction that changes
nonlinear dynamics while preserving the actual coupled equilibrium Jacobian.

Budget one model-only sensitivity/nullspace evaluation,5min main wall cap;
one coherent implementation QA repair before execution, no model execution retry
without stage review. No TRAIN/VAL/TEST, fit, coefficient search, new structure,
trajectory or candidate. Close/review after this single result; no rollover.

Use original G3000 SHA2667480a75d7672ad49ed66c79731e754676bae52574229dbb5f98c581ac890b
and saved0083 matrices; use corrected0082 saved theta and theta0 only to verify
the already-known affine Jacobian difference. All input/source hashes unchanged.

Let F_j(z)=partial f/partial theta_j be the EXISTING all33-row coefficient
fields (eight storage gamma, sixteen port coefficients, all paired effects).
Use validated0082 coefficient_fields at frozen J/R/B, preserving physical units.
Compute one AD derivative tensor D_j=partial F_j/partial z at z=0. Stack each
full33x33 D_j into one column of D (1089x24). Do not delete common speed or
latent rows, or study an isolated subsystem. Common-angle direction is a
known null input, not a coefficient degree of freedom.

QA: all finite; each D_j has zero kinematic rows and annihilates common angle
within1e-10 absolute; D*(theta0082-theta0) reproduces saved A33_candidate-A33_base
within1e-10 relative Frobenius (denominator max(1,norm(reference))). Compare
the derivative tensor with central FD of the coefficient fields at h=1e-5
and1e-6: relative Frobenius<=1e-7, both required, no selectable step.
Preserve the tensor and both finite-difference tensors.

Normalize each nonzero D column by its Euclidean norm; treat zero column
norm<=1e-14*max norm as zero and report it explicitly. Use full SVD on the
normalized1089x24 matrix. Report all singular values, column scales, numerical
rank and nullity, condition and right singular vectors. Singular ratios>=1e-8
are clearly resolved; <=1e-12 are numerical-null candidates; any intervening
ratio leaves the rank-based claim inconclusive. Never infer nullity from q
covariance or whitening.

If all24 singular ratios>=1e-8 and no zero column, reject existence of a
nonzero linearization-preserving direction within this block. If numerical-null
directions exist with a clear gap, map their right vectors through inverse
column scales to coefficient directions, including explicit zero-column axes;
require ||D*d||/(max(1,||D||)*||d||)<=1e-10. Evaluate F(z)*d ONLY at the two
fixed deterministic_test_z states defined in0082. Report norms and arrays.
Support a nonlinear-only available direction only if at least one normalized
direction gives norm(F(z)*d)>1e-8*max(1,norm(F(z))) at one fixed state.
Otherwise no detectable nonlinear-only direction (or numerical ambiguity),
not a general impossibility theorem about other supports/coordinates.

Synthetic QA: known derivative tensor with full-column rank, exact null direction
that produces a quadratic nonzero field away from zero, and intermediate
singular ratio classified as ambiguous. Independent cached NumPy SVD/rank and
affine-difference reconstruction; maxrelative matrix/singular comparison<=1e-10.
No model promotion or decision to add structure follows automatically. At closure
root reviews whether a stability inequality rather than exact-Jacobian equality
is a justified next bounded intervention; no such fit is authorized here.
