# Loop0092 — existing backbone response-span upper bound

Start6f27c99, after75. Diagnostic, promotion none.0091 remains rejected.
Question: can changes ONLY to existing angle-potential amplitudes and diagonal
base damping span a majority of the entire sampled normalized torque discrepancy?
Hypothesis: unconstrained full-grid least-squares capture>=.50. This differs
from0091's full/LOO affine generality claim: no held-out prediction assertion,
no relaxed0091 gate, no candidate and no new response form/latent dimension.

## Equation and frozen inputs

Read0091 cachedZ,omega,mass,frequencies, and original G3000 numeric export.
All13 frequencies and all10 input/output channels; equal Frobenius weighting.
Export has26 supported angle rows D and10 diagonal damping amplitudes.
Base potential gradient is D' diag(k)[sin(phi+D*delta)-sin(phi)], phi=D*delta_star.
Canonical J_pi,delta=-I; thus, in endogenous torque units,

  DeltaT(s)=-D' diag(Delta k*cos(phi)) D - s diag(Delta d).

External omega_dot has an additional M^-1. Speed convention is rad/s.
Extra J/R/B have no delta rows/columns; changing basek,d changes only the
physical speed RHS, not hidden p/q equations, so their clamped response is
unchanged. Do not infer the same for other storage/port parameters.
Root rejects the reviewer's positive stiffness-sign shorthand: the actual
force is MINUS the potential gradient. Derive damping via -R gradH=-d*omega.

Let L=diag(Sdelta/(M*Somega)), using unchanged0091 scales.36 complex columns:
H_e(f)=-L*cos(phi_e)*D_e'*D_e, e=0..25;
H_(26+j)(f)=-i*omega_f*L*e_j*e_j', j=0..9.
Stack all real entries followed by all imaginary entries, yielding2600x36 X
and2600-vector y fromZ. Fit real beta with ONE column-scaled thin SVD:
scale_j=||X_j||2, Xs=X/scale; beta=(V*(U'y/sigma))/scale.
No constraints on changes; no physical-amplitude/positivity rescue fit.
Prediction X*beta, capture=1-||y-Xbeta||2^2/||y||2^2. No clipping/projection
of scored data, frequency deletion, weighting, LOO, ridge, coefficient search.
This unconstrained span is an upper bound for nonnegative actualk,d fits;
a pass does NOT establish their attainability or justify deployment.

Report all36 coefficients, singular values/scales/rank/condition, full and
per-frequency residual norms; descriptive k0+Delta k and d0+Delta d signs.
No parameter significance, device naming, trajectory attribution or stability claim.

## Validity and budget

Anchors0091 arrays e38c779c3d80f059c184c60e7c69aebeabfdd3fc4d74aa8ce55713738271ffcb;
evidence4e96d916630ba1a3234ded0d4c33654645aa772e8c61577fd7ffbba65e2dfc40;
independent898f06fddf969a2bfe5938a4e7b90142b612cd2797103d62921f34eebf2ebce7.
G3000export4407f45bbafc367a912c09f5f4fd5757b8722e5418a06eb46c0f6adee60d78e9;
structured_rom.py921b6dedb2e8b5218c5f4ec84fc6a35ac4844ebc0e310f8a6639bb0c7bcf88f3;
structure_grammar.py0325a0a1f50d7f22510cd3d7da5dbc28d8c3e186ea7288e7717f7a3a8704d28d.
Strictly preserve/reverify every inherited0091source hash (including code90
andcode91) before/after,0091 independentpass14fits/errorsnone; verifyexport
hash equals0083 recorded baseline_export. Separate newcode92 key.

Finite arrays, positive mass,26x10 D with zero row sums<=1e-12;
kinematic/canonicalJ andzero-delta-extraR/B export checks<=1e-12.
Use original exported nonlinear evaluator at three deterministic physical
states with delta_j=.01*t*sin(j+1),omega_j=.01*t*cos(j+1),p=q=0,
t=1,2,3, internalpi=M*omega. Perturb basek and d by+1percent in a COPY
(also update only base diagonalR by.01d). Compare six total RHS evaluations
against exact nonlinear Deltaf_pi above, allotherrowszero, maxrelative<=1e-10.
No model serialization or real trajectory generated.

Column norms>1e-12; rank uses max(X.shape)*eps*sigma_max. Require full36rank
and condition*eps*2600<=1e-8, otherwise numerical-inconclusive (no rank/ridge
rescue). Check X'(Xbeta-y) relative to X'y and X'Xbeta<=1e-10;
projection objective<=zero+1e-10*max(1,zero), finitecapture, ||y||>1e-12.
Synthetic knowncoeff recovery, nonrepresentable error, common-speed retention,
correct stiffness/damping signs and missing/nonfinite/rankfailure checks.

ONE main fit/oneSVD/sixexportRHS evaluations<=120s; ONE independent cached
fit<=120s, aggregate<=245s including5s conservatively charged metadata.
Independent uses separately assembled basis and QR (not production helper),
checks normal equations, source hashes, masks/grid, full/perfrequency scores,
array maxrelative<=1e-10 and metricabsolute<=1e-10. No realretry; one coherent
preexecution repair; synthetics separately counted. Fresh directories
private_data/task2_backbone_span_0092 and _independent; refuse overwrite.
No descriptor/ROM solve, eigensolve, newfrequency, rollout/H5/VAL/TEST, model
load/training/update, nonlinear response fit or new structural group.

## Stage review and interpretation

Independent bounded read-only challenge recommends this one span upper bound,
not constrainedcandidate deployment; root adopts fullZ scoring, condition/rank
reporting and strict local scope, corrects sign as above. No existing failed
branch is reopened. Close stage after one main+check or cap. If capture<.50,
reject majority explanation even without physical coefficient restrictions;
do not spend a backbone-only trajectory training budget on that rationale.
If supported, the remaining question is legal constrained realizability and
large-disturbance compatibility, requiring new preregistration/review.
Either outcome is a diagnostic, not the project's completion.

## Preexecution QA record

Root caught a mistyped grammar SHA and an assumed inherited `core` key
(actual inherited key `old78_structured_core`) before any real invocation.
One coherent implementation repair also completes the0083 export linkage,
0091 independent-check linkage, actual attempt counters and missing/nonfinite/
nonlinear synthetic tests. These are fail-closed implementation fixes, not
scientific changes or additional real fits. No response fit performed.
Additional source-map schema reads (including independent1.6186s) raise the
conservative metadata charge from5s to10s BEFORE any fit. The aggregate245s
cap and individual120s caps remain UNCHANGED: main+independent together have
at most235s after metadata. This accounts additional work, not a budget grant.
