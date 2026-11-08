# Loop0095 — positive-backbone harmonic/conditioned-balance compatibility

Startd65beef; active after75. Worktype diagnostic; promotion none.
0093 remains rejected,0094 remains supported. No new deployed candidate,
training, latentstates, losses, libraries or TEST access.

## Question and bounded claim

Can a DIFFERENT choice of the same36 backbone amplitude increments preserve
at least50% of the0092 harmonic discrepancy capture without worsening the
0094 conditioned TRAIN balance?0094 shows the exact harmonic-only optimum
worsens bothprimarymetrics; it does not establish incompatibility of the span.

Use exact0092 real X(2600x36),y and k0,d0; originalG3000 sharedq/truth/cache
from0094. Beta=(Delta k26,Delta d10). q dynamics are globally independent
of these amplitudes. Every conditioned defect is therefore exactly affine:

  r_beta(t)=r0(t)+B(t)beta.

Perunit columns are cumulative Simpson integrals of the registered0094
fk/fd on the SAME piecewise-linear physicaltruth. Construct both.005/.01
grids, score only common.01 times1..n; never integrate q or any model again.
Use all346events,311stable35unstable,all1/2/5genuineprefixes andSomega/W of0094.
Fullphysicaltruth is explicitly future-conditioned, not causal deployment.

## ONE deterministic convex optimization

Minimize J(beta)=||X beta-y||^2/J0, J0=||y||^2. No blended objective/weight.
Constraints, enforced on the predetermined FINE grid:

- actual k0+beta_k,d0+beta_d >=1e-8;
- average1/2/5(.8mean346+.2worst35) <=0094baseline average risk;
- severe35mean5s-prefix <=0094baseline severe mean;
- severe worst4of35 <=0094baseline severe worst4.

All three are NONWORSENING, intentionally not a new deployment gain claim.
Zero increment is a positive, feasible reference; failure to find feasibility
cannot establish physical incompatibility. No coefficient bound/search,
seed, alternative weighting, deleted event or tail-set freezing.

Use beta=s*v, s_j=sqrt(J0)/||X_column_j||2. Keep all36columns; rejectzero scales.
For eachgrid/horizon/event save augmented37x37 Gram of
[W B diag(s), W r0], time-mean innerproduct. Scores equal [v;1]'G[v;1].
Save a corresponding fine-minus-coarse Gram for each horizon/event, including
baseline difference. Symmetrize Grams. Fine factors use1038 symmetric eighs;
reject eigenvalue<-1e-10*max(1,largestabs eigenvalue). Clip ONLY negative
roundoff eigenvalues to0, never threshold positive directions. Require factor
reconstruction relative<=1e-10; retain eigenminimums/reconstructionerrors.
These are numerical factors, not scientific rank reduction/regularization.

Scale solver event-score epigraphs by the positive baseline average risk.
Each event e_hi >= ||F_hi[v;1]||^2 / baseline_average. Exact convex tail
epigraph: topKmean(e)=tau+sum(u)/K, u>=e-tau,u>=0. Use K35 for eachhorizon,
K4 for severe5s. Preserve .8/.2 and uniform horizon means. No frozen tailsets.
One CVXPY/CLARABEL solve, float64, max_iter200,time_limit600s,
tol_gap_abs=tol_gap_rel=tol_feas=1e-9. No fallback solver/warm retry.
Save coefficients, epigraph variables/duals, solver status and actual primal/
dual objectives,gap,residuals,iterations from the installed solver API.

## QA and classification, before seeing results

First validate0094/0092 main+independent provenance, untouched sources and
accepted checkpoints. Reproduce both-grid baseline scores and exact0093
correction scores from the new Grams within1e-10relative; reproduce all
correction accumulators on genuineprefixes within1e-10relative. This anchors
sign, scaling, columns and fixedq reuse. All shapes/IDs/labels/padding checked.
No selection by operating region/frequency; all13 fitted FOM frequencies stay.

After solve, reconstruct original unsorted event scores and all three original
risk constraints on BOTHgrids. Report ratios/slacks, active tails, minactual
amplitudes, J and capture=1-J, versus unconstrained0092capture.686483616.
Numerical feasibility tolerance for risk ratios: <=1+1e-7, not a gain margin.
Actual amplitudes >=1e-8-1e-10. Candidate paired-step fullprefix weightedRMS
<=.001 for all346events. All original0094 baseline denominator gates required.

Numerical optimality requires solveroptimal status, finite coefficients,
normalized primal/dual residuals<=1e-7 and absolute/relative objectivegap<=1e-7
(normalization max(1,abs primal,abs dual)), plus explicit postchecks above.
Save raw solver diagnostics; they are numerical evidence, NOT a rigorous
interval-arithmetic certificate or proof for the autonomous nonlinear family.
- Supported: all validity/feasibility checks and capture>=.50 on exact X,y.
- Rejected bounded conditional claim: all checks, capture<.50, and normalized
  solver dual objective>.500001, bounding the numerical optimum away fromgate.
- Otherwise inconclusive/optimizationfailure or implementationfailure as
  appropriate; never silently relax constraints/resolve/reclassify infeasible
  as a family rejection. Boundary/weak-certificate cases are inconclusive.

Independent checker: no modelconstruction, qreplay or optimizer. Recompute
known integral columns/Grams, all final scores/tails/slacks/capture, coefficient
positivity and provenance. Arrayrelative<=1e-10, scalarabsolute<=1e-10.
Audit saved solver diagnostic fields/dual arrays for finiteness and required
thresholds; do not claim an independently solved or rigorous dual certificate.
Synthetic tests: manufacturedconstant/linear integrals, sign/weightednorm,
quadratic reconstruction, tail epigraph equivalence, zero baseline feasibility,
missing/nonfinite guards and certificatefailures. Synthetic solver QA may use
one tiny known convex problem; it cannot read scientific sources or selectsettings.

## Budget, branch limit and independent review

One main<=900s including<=600s solver; one independent<=180s;
aggregate<=1100s including20s metadata. One fixed formulation/branch, no real
retry, additional fit/replay/AD/eig beyond1038 fine Gram numerical factors.
Save partial Grams and attempt counts before expensive actions, stoponcap.
One coherent preexecution repair; synthetics separately counted. Close/review
at result/cap. Newbudget justified by conditional-versus-harmonic compatibility,
not extension of0093 or0088 optimization. No automatic budget rollover.

Independent route challenge adopted: exact unblended harmonic objective,
convex original tail metrics, direct postchecks, baseline feasibility and
certificate-limited rejection. Root changed proposed.90gain constraint to
nonworsening BEFORE any new calculation so baselinezero remains feasible.
This deliberately asks compatibility, not improvement or deployment acceptance.
Fixedq may be incompatible; passing/rejection cannot certify autonomy/stability.

Source anchors:0094evidence61373d0088cdd7b0271918cbb55a0291d08db753a8344ff6a323bace16134225,
arraysd33cbb83b656e1275ac3af01aa779444db2f33e6a13bbeb285eed2dcf8dae2ec,
metricse50396f14a1d01f937ddb31f3bba8fe0ac21fc32789a73c39d04cd0278496dbb,
independent37bbb42ad09f3b23768ee747a45b256a9d9b88a9ebf75f805d1eafffe2908060.
0092arrays2b4c0725339c360ea2502e8059e03b8d4f4c9c96197a58701eccd47e26983a8f,
evidenceb5b9f7cc6f673fbd6dd4b5d59f6b3cb8d26361c68e4aad590141e72f3dc0a89b,
independent29f1d15d9c797537ada125f41baee45642184d9c98e894987864c112c8fb6535.
Carry all inherited source hashes before/after; addnewcodekeys, neveroverwrite.
Fresh artifacts private_data/task2_backbone_compatibility_0095 and_independent.
No model export/default change. FreshTEST unopened; consumedPilotTESTexcluded.

## Preexecution QA record

One coherent draft repair before any new scientific input/solve: independent
and root review caught force interpolation substituted for state interpolation,
a factor-of-two composite-Simpson coefficient, incorrect mass broadcasting,
factor orientation and an unimplemented old-increment anchor. Root replaced
the affected algebra and added exact NONZERO nonlinear/linear manufactured
values, heterogeneous masses, two-grid Gram/old-increment identities and
failure preservation tests (7pass). Main integration also removes unused tail
variables, fixes literal inherited keys, separates numericNPZ from metadata,
and enforces optimal-only status. No scientific threshold/settings changed.
Existing isolated CVXPY1.9.2/Clarabel0.11.1 from0067 are reused, no installation.
An explicit harmonic residual variable gives a zero-offset solver objective;
direct X,y objective must agree with saved native primal before using dualcost.
