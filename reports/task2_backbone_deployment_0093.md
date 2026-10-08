# Loop0093 — deploy the exact response-supervised backbone correction

Start68af7d6; after75. Work type:model screen; promotion:none.
0092 supports a response-span majority;0091 remains rejected. Question: does
the exact0092 positive-amplitude correction improve actual coupled modal and
autonomous large-disturbance behavior, without changing any other learned object?
This is explicitly FOM-response-supervised development, NOT TRAIN-only learning,
independent confirmation, a new physical mechanism or a new model-family claim.

## One fixed candidate, no optimization

Baseline original G-fixed3000, modelSHA2667480a75d7672ad49ed66c79731e754676bae52574229dbb5f98c581ac890b.
Use EXACT0092 k_plus_delta[26],d_plus_delta[10], no interpolation/refit/clipping.
Change ONLY `structured.coupling` and `structured.damping` raw softplus parameters
to the inverse-softplus of these positive actual amplitudes. Require both masks
allones and unchanged. All observer/decoder/normalization, inertia, equilibrium,
angle supports/phases, otherstorage, J, extraR, B, ports and q3 remain frozen.
Copy/save a fresh candidate, reload using existing loader; verify all state-dict
keys bitwise unchanged except those two tensors. No optimizer or data fitting.
Actual amplitudes afterreload match targets maxrelative<=1e-12; finitepositive.
J/B unchanged; R difference exactlydiag(Delta d) in internalpi block, all other
entries unchanged<=1e-10. Storage potential changes only through existingk.

Anchors0092 arrays2b4c0725339c360ea2502e8059e03b8d4f4c9c96197a58701eccd47e26983a8f;
evidenceb5b9f7cc6f673fbd6dd4b5d59f6b3cb8d26361c68e4aad590141e72f3dc0a89b;
independent29f1d15d9c797537ada125f41baee45642184d9c98e894987864c112c8fb6535.
Require0092 successfulonefit/oneSVD/sixRHS andindependentpassoneQR/noerrors,
strict existing source hashes before/after, inherited code keys retained.

## Actual coupled dynamics and conditional launch gate

Compute exactly TWO origin AD Jacobians: untouchedbaseline and reloadedcandidate,
external33-state(delta,omega,p,q). Baseline must reproduce anchored0083 A33
maxrelative<=1e-10. Candidate must satisfy A=A0+DeltaA maxrelative<=1e-10:

  DeltaA[omega,delta]=-diag(1/M)*D'*diag(Delta k*cos(phi))*D,
  DeltaA[omega,omega]=-diag(Delta d/M), allotherblocks0.

Use candidatecentralFD h=1e-5 and1e-6 only,132 RHS evaluations total. Each
FD Frobenius relativeerror toAD<=1e-6 (denominator max(1,||AD||F)). Preserve
all arrays. Existing structural_qa at deterministic_test_z tests origin,
kinematics, gauge, export/reload, poweridentity, Jantisymmetry,Rsymmetry andPSD;
all existing1e-10 gates unchanged. No parameter-perturbation affinity_qa24fit.

Use unchanged0083Q (33x32) removes ONLY commonangle; QQ'=diag(P0,I23),
Q'Q=I and A*commonangle=0 within1e-10. ONE ordinary32-state candidate
eigensystem; finite, normalizedbackwardresidual<=1e-10. Let alpha=maxRe eig.
Local screen alpha<=-.01/s. Numerical sensitivity bound:
cond(V)*max(||Q'(A-FDh)Q||2 for bothh,eps||A32||2)<.1*abs(alpha).
This is a local/numerical screen, not nonlinear basin or fullorder certificate.
If numerical/structural QA fails, classify implementationfailure or numerical-
inconclusive as appropriate; ifalpha fails, localcandidate rejected. Skip
forecast in either case; retaincandidate as rejected evidence, no rescue setting.
Do not choose favorable eigenvalues,steps or checkpoints.

## Modal and trajectory screens: both binding

Use the SAME singlecandidate eigensystem and0089 matching protocol/physicaloutputs:
13 targetIDs62,64,66,68,71,73,77,79,84,91,98,100,104; relativefreq<=.15,
absolutedampingdifference<=.05,MAC>=.80, one-to-one allgates on eachpair.
Require>=7matches. Retain allpairs, singular/nulloutputs and unmatchedtargets;
baseline0/13 from0089 unchanged. No FOM eigensolve. These targets/frequencies
supervised0092, so modal score is response-informed development performance,
not independent confirmation. A modal failure does NOT skip otherwisevalid/
locallystable forecast: joint behavior is the scientific question.

Only afterlocal/numerical gates pass, ONE paired candidateRK4 forecast at.005s
and.01s; existing_checked_rollout withfinite EVERYsubstep and genuineprefixes.
Reuse verified0088 cached originalbaseline forecasts/truth/qenc/lengths/eventIDs;
no baseline rerun, H5/VAL/TEST reader or freshdata. TRAIN346eligible=311stable+
35unstable,369raw/23excluded,24groups; identicalorderedinitialstates truth[:,0]
plus frozenG3000qenc, realp0, no futuretruth feeding. Use fixed1/2/5s scoring
through each genuine availableprefix, no paddedtail or difficult-event deletion.
Baseline metrics from cached trajectories reproduce0088 to1e-10 absolute.
Eachcandidateper-event finesteppingnumerics(coi,common,p)<=.001, existing0080
formula/scales. Nonfinite/inconsistent integration is numerical-inconclusive,
not model-family rejection, no timestepretry in this stage.

Point score=.5mean((COIdelta/Sdelta)^2)+.5mean((COIomega/Somega)^2)
+(commonomega/Somega)^2+mean((p/p_scale)^2), Sdelta=.47675314588683515,
Somega=1.0033667249602924; frozenp_scale inoriginalcache. Eventrisk .8mean+
.2worst35; average1/2/5risk candidate/baseline<=.90. Severe35genuine5s-prefix
mean andworst4 ratios BOTH<=1.05. Allthree trajectorygates AND modal>=7
binding for jointscreen support; report individual outcomes, no endpoint choice.
Baselineaverage.8938195532229601,severemean.6654646977500597,worst4 2.303873238746948.

Cached0088trajectoriesSHAc466a551740404390ef967f82dd116df1543e8ee166356d348c3c9cc02ab0c5e;
forecastindependent67940d769feb3aa15b58127bee73cdbde75ccc8e329444e7f43d4b32688f2da5.
Verifyexistingindependentpass/linkedsourceandcachebeforeuse; no data regeneration.
Helpers fit_coefficient_block_0082.py SHA1c680b4705a67ffa9999dc6de2651a4b8f5419d22f558469b4b8f7c06eb0fe1e;
audit_force_initialization_0080.py SHA4617bfd1ea3e90468d74ba7fc10854a8a7f748da58c327fe13965554b2fbb1d9.

## Budget, checks and review disposition

ONEcandidate/main<=1200s; includes construction/reload/structuralQA(oneR PSD
eigensolve),twoAD,132FD RHS,onecandidate32eig,norm/SVDconditioningQA,one paired
candidateforecast andcachedbaseline scoring. No training,seeds,extraforms,
newfrequencies,parametersearch orrealretry. One coherent preexecution repair.
ONEindependentcachedcheck<=180s; aggregate<=1400s including conservative20s
metadata/preparation reads. Synthetictests separatelycounted. No automaticextension.
Fresh private_data/task2_backbone_deployment_0093 and_independent; failclosed,
recordactualcounts/partialoutputs/candidatehashes, refuseoverwrite.

Independent checker: separate formula/metric arithmetic, cachedstate/array/source
checks, actualchangedstatekeys, expectedJac/FD/eigenresidual, allmodalpairs,
all1038per-event/horizonmetrics ifforecastperformed, numerics andjointgates.
No newmodelconstruction, eigensolve,training,forecast or frozenTEST. Arrays
maxrelative<=1e-10 and metricabsolute<=1e-10; preservefirstfailure,no silentretry.
Synthetics before main: inverse-softplusandmask/frozenkeys, analyticJacobian
blockunits/signs, gatebranching (includingnumericalvs scientificfailure),
metricseveregates/index/prefix and missing/nonfinite source failclosed.

Independent route challenge adopted: positive coefficients are not stability;
verify mask/reload realizability, allotherblocks/tensors unchanged, call this
physical-response supervision, treat integrator failure as inconclusive, and
make modal/trajectoryjointacceptance explicit. No globalLyapunov/ROAclaim.
At completion/cap close stage andreview next smallest physicallymotivated
hypothesis; failedcandidate is notprojectcompletion. AcceptedM3/0058/S/defaults
unchanged. Fresh expansion/supplementTESTunopened; consumedPilotTESTexcluded.

## Preexecution QA record

Root found helperdraft failures before any realcandidate/run: whole-array
finitechecks would reject legitimate NaN padding outside genuineprefixes;
point scoring assumed equal33/30 channel widths; a customRK4 path checked
only finalsteps rather than every substep. One coherent preexecution repair
uses the registered existing_checked_rollout, true-prefix QA and first30
physical channels, with1038-row/severe/padding synthetic coverage. No
scientificgate, integrator setting, dataset or realrun changed.
The same preexecution review fixes historicalcandidate cache-key collisions,
checks all1038 baseline rows and actualcandidate hashes/tensors, and separates
FD sensitivity failure from failure of the localalpha gate. Root combined
main/helper synthetics:15pass1.71s. No realcandidate/run preceded these fixes.
