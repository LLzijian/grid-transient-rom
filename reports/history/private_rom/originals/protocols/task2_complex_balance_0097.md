# Loop0097 — complex directional modal balance

Start c096b9f; active after75. Work type:diagnostic; promotion:none.
Question: does the frozen0096 point retain substantial generator-balance
defects at the actual complex FOM poles and physical mode directions despite
its .571587392 imaginary-frequency matrix capture? No model/family claim.
Hypothesis: normalized defect eta>.10 on at least7 of13 fixed targets for0096.
This engineering threshold defines substantial imbalance for this diagnostic,
not a pole-distance bound or replacement for the existing modal matching gates.
OriginalG3000 and failed0093 are descriptive controls; no candidate selection.

## Frozen inputs and equations

Use0090 cached A_F/E_F, originalG3000 A33 from0096, failed0093 and0096
candidate_A33. Same13 target IDs62,64,66,68,71,73,77,79,84,91,98,100,104.
Use EXACT aligned descriptor_eigenvalues/descriptor_right_vectors selected by
target_recomputed_indices from0078modal_arrays; no new FOM eigenproblem.
Check frozen target_outputs against [COIdelta/Sdelta,Omega*omega_pu/Somega].
Normalize each full699vector by that physical-output norm. Keep ALL10rawdelta
and fullomega, including common motion. Omega=2pi60; Sdelta=.47675314588683515,
Somega=1.0033667249602924, same positive mass as0090/0093/0096.

For each arm and target complex lambda, partition external [delta10,omega10,
p10,q3], h=(p,q). H=lambda I13-Ahh, B=Ahdelta+lambda Ahomega, X=solve(H,B).
T=diag(M)(Aomega,delta+lambda Aomega,omega+Aomega,h X).
R=diag(M)lambda^2-T; L=diag(Sdelta/(M Somega)); K=L R; a=v_delta/Sdelta.
eta=norm(K a)/(norm(L diag(M)lambda^2 a)+norm(L T a)).
Report every eta, numerator, two denominator terms, normalizedK singularvalues
and condition (descriptive), and hidden equilibratedH singularvalues/condition,
row/column scales and backward errors. No near-singular target deletion.
Hidden metrics refer to the specified equilibration, not raw-coordinate kappa.
No claimed direct/memory decomposition, modal matching or autonomous attribution.

## Numerical validity and controls

Finite arrays/shapes/IDs/sourcehashes and matching cached baseline/mass mandatory.
Physical-output norm must be positive; after normalization denominator>1e-12.
FOM eigenpair backward norm(A v-lambda E v)/((normF(A)+abs(lambda)norm2(E))
norm(v))<=1e-8. Here norm2(E) means the spectral norm of the diagonal
descriptor matrix, max(abs(E_F)); E_F is stored as its diagonal vector.
Physical kinematic check lambda v_delta=Omega v_omega_pu:
normdifference/(normleft+normright)>1e-6 fails; denominator>1e-12.
Generator-row check lambda Eomega vomega=Aomega,: v uses same relative
sum-norm rule and denominator>1e-12, tolerance1e-6. No full hiddenFOMsolve.
ROMdelta rows equal [0,I,0] and commonangle annihilation<=1e-10 maxrelative;
FOMEdelta=1, Eomega=M Omega and delta row=Omega I checked<=1e-10.
Use unchanged harmonic_solve_0090.solve_scaled complex128: both backward
errors<=1e-10; rho=kappa_eq*max(backeq,n eps/(1-n eps))<=1e-6.
All39solves must pass. K SVD is descriptive: singularK is not a numerical
failure of the hidden solve. Condition infinity stored as null with singular
flag, never dropped. Nonfinite results/control/conditioning/denominator failure
make overall diagnosis inconclusive, not rejection of the scientific hypothesis.
Valid count>=7 supports the narrow hypothesis; otherwise reject that hypothesis.

## Budget, verification and preservation

One main<=120s:39complex13x13solves,39hiddenSVD,39reactionSVD; no other
solve/eig/AD/modelconstruction/fit/rollout. One independent<=60s:39rawcomplex
solves, separate balance/control arithmetic, no SVD/eig/simulation. Compare
allprimaryarrays<=1e-10 maxrelative, allscalarprimarymetrics<=1e-10 absolute;
inspect main singularvalues/recompute their ratios, explicitly not independent
SVD confirmation. Aggregate<=200s including10s metadata (alignmentread .373s
already spent). One coherent preexecution repair; no actualretry/newtarget/
frequency/branch. Synthetics separately counted. Review at result or cap.
Test known linear block balance, phase/amplitude invariance, common motion
retention, zero denominator and wrong-unit rejection, singularhidden failclosed.
Fresh private_data/task2_complex_balance_0097{,_independent}; refuse overwrite,
save partial/failure artifacts, counts/timing and before/after sourcehashes.
Anchor0096arrays ad90932963116b973753305aba7db2ca0149168cb2308088d10db49fe215d783,
evidence04e8b5700b6271e983436f36f707c2b2240f41e114aca8d1084b720eace057c0,
independent73190689f5d6c661520c9e13ceca06f1383eeff866ccf57d98c6420e56f7d290.
Retain inherited sourcechain,0093/0090/m78 anchors; do not alter old code.

Independent route review adopted phase invariance, conditioning and restricted
claim scope. Root rejects projecting rawdelta onto COI before R action: at
nonzero lambda that removes tied commonfrequency motion and changes the mode.
COI is used only in the output norm; full10balance retained. If imbalance is
not substantial, next diagnosis concerns pole/output sensitivity, not capacity.
If supported, consider one physically constrained complex-modal intervention
only under a separate preregistration and joint trajectory safety requirements.
AcceptedM3/0058/S unchanged; no H5/VAL/TEST read, no newdimension/loss/library.

Preexecution repair: full descriptor residual/unit controls, zero-denominator
guards, source/candidate anchors and failed-solve records completed before any
scientific execution. Spectral descriptor norm clarified above. Nine main and
reused solver synthetics pass0.33s; no real retry. Latest user instruction:
finish this Loop0097, archive its evidence and pause; do not start Loop0098.
