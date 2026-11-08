# Loop0094 — conditioned speed-balance defect of the rejected correction

Start227fa66; after75. Diagnostic, promotion:none.0093 remains rejected.
Question: does the exactk/d correction worsen physical balance even on a fixed
replayedphysical path, without autonomousphysical feedback departure?
Hypothesis: fullcorrection/base conditionaldefect ratios>=1.25 for BOTH
average1/2/5eventrisk and severe35mean5s-prefix. Both integrationgrids must
support those inequalities. Severe worst4 is descriptive, not a third gate.
No newcandidate, training, coefficientselection or addeddeployedstate.

## One shared conditioned path, explicit truth access

Use originalG3000 and EXACT0093k/d, allotherobjects frozen. Same anchored0088
TRAINcache used0093:346eligible=311stable+35unstable,369raw/23excluded,
unchangedordered IDs, truephysicalstate30, qenc0, mass,scales,genuine lengths<=500.
NoH5/VAL/TEST reader, boundarydata merge or newfrequency. Sourceanchors below.
Replay physicaldelta,omegarad/s,p from thatcache with PIECEWISELINEAR
interpolation in.01s sampleintervals, including all genuineprefixes.
This accesses futuretruth ONLY for an explicitly conditioned diagnostic;
it is NOT autonomous forecasting or a deployable observer. Replayed angle/
speed interpolants need not satisfy exact continuous-time kinematics.

Integrate G3000qdot=fq(xphysical(t),q), q(0)=frozenqenc0. Basek/d changes
leave fq exactlyunchanged globally because gradU changesonlydelta effort,
Jdelta iscanonical andR/B have no delta rows/columns. Confirm three fixed
state probes (two deterministic_test_z rows pluszero): actualcandidatefield
minusbasefield equals the analytic omega-onlychange below, maxrelative<=1e-10.
Sharedq remains model-generated, NOT latenttruth or optimalconditionalencoding.

At eachstage accumulate I0=integral fomega_base dt and knownincrements

  fk=-diag(1/M)*D'*[Delta k*(sin(phi+D*delta)-sin(phi))],
  fd=-diag(Delta d/M)*omega,  Ik=integral fk dt, Id=integral fd dt.

Diagnostic integration vector [q3,I0_10,Ik_10,Id_10] consists ofq and
quadrature accumulators, NOT extraODEmemory in a candidate. Initialintegrals0.
OneRK4 path at.005 andone at.01, every stage/statefinite, no extrapolation
beyond genuineendpoints. Stoponerror/cap, savepartialarrays, no timestepretry.
Use fixed sampled grid forscoring: fine[::2] versus coarse; excludeinitialtime.
Four algebraic defects: r0=I0-(omega_true-omega_true0), rk=r0+Ik,
rd=r0+Id, rfull=r0+Ik+Id. These are speed-balance defects inrad/s;
equivalently inertia-normalizedmomentum defects, not unnormalized momentum.

## Scores, decomposition and numerical adequacy

Pcoi=I-1*M'/sumM, W*r=[sqrt(.5/10)*Pcoi*r/Somega;
(M'*r/sumM)/Somega], Somega=1.0033667249602924. Per-eventscore is mean
||Wr(t)||2^2 over genuine available1/2/5s prefix. For eacharm/grid, risk
=.8mean346+.2worst35, averageover1/2/5; severe35mean andworst4 at5s.
Report all1038rows/arm/grid. No deletion, weightingchange, favorablegrid choice,
extra conditionalinitialization, GRUreestimation or replayphysicstuning.

For each event/horizon retain SIX quadratic terms mean||Wr0||2^2,
mean||WIk||2^2,mean||WId||2^2,2mean<Wr0,WIk>,2mean<Wr0,WId>,
2mean<WIk,WId>. Their sum equals fullscore within1e-10relative to
max(1,maxabs operands). Report fixedall346/severe35 means, retain negative
cross terms. Risk/worst35/worst4 sets canchange, so these terms must NOT be
presented as an additive decomposition of risk or autonomouserror.

For eacharm,event,fullgenuineprefix require RMS||W(rfine-rcoarse)||<=.001.
For each primarydenominator (baselineaveragerisk andbaselineseveremean),
require >1e-12 AND >100 times the corresponding risk/mean of per-event
fine-coarse weighteddefectdifference energies. Bothgridbaseline denominators
must pass. Otherwise unresolvedratios/numerical-inconclusive, no huge-ratioclaim.
Report all fourarm/grid scores. Primary H supported only iffull/base>=1.25
for bothprimarymetrics onBOTHgrids. If bothgrids consistently do not satisfy
the jointclaim, reject thatclaim; differinggrid classifications inconclusive.
Worst4ratio onlyreported ifits baseline passes analogousdenominatorcheck.
Step agreement tests the fixedinterpolatedproblem, not unsampledFOMaccuracy.

## Sources and bounded execution

Use allinherited0093 sourcehashes before/after, exactcandidatebytes andunchanged
verified0093QA beforeanyreplay; addnewcodekeys withoutoverwritingoldones.
0093evidenceec4ab06c662dadfd0aff6dfc71c23c8db11d27af0375d0560789965665ae107f;
arrays82659802566986e76bcb5cdd11b289f5adc3110ae0895790c5c586d5c97d8f51;
independent99bd9c74d71dd0dafa727018cf32bc8f99b91a37d03b9479fc72b0b8d28c843d;
candidate8f699a7524c71529227f01a2876e9d4ec4633cea095dad3edfc93914ca33d46e;
G3000baseline2667480a75d7672ad49ed66c79731e754676bae52574229dbb5f98c581ac890b;
original0088cachec466a551740404390ef967f82dd116df1543e8ee166356d348c3c9cc02ab0c5e.
Readmetadata source maps ifnecessary; recordpreparationtime within20s allocation.

ONE mainpairedconditionalreplay<=900s, including3pairedfixedfieldprobes,
no modelupdate/AD/eig/fit/autonomousrollout; ONEindependentcachedcheck<=180s,
aggregate<=1100s includingconservativemetadata20s. Two conditionalpaths only;
fourarms are algebraicfromsameintegrals, notfourmodelruns. No realretry or
additionalbranch; onecoherentpreexecutionimplementation repair. Synthetics
separatelycounted. Close/reviewatcompletion/cap; noautomatic budgetrollover.
Fresh private_data/task2_conditional_balance_0094 and_independent; refuseoverwrite.

Independent no modelconstruction/replay: verify allsource/candidate/cacheidentities,
genuinepadding/initialstates, knownIk/Id by independentcompositeSimpson integration
of thereplayedphysicalinputs at bothfixedsteps (no q dynamics), allscores,
denominatorconditions, componentidentities and brancharithmetic. Arraysrelative
<=1e-10; scalarabsolute<=1e-10. Baselineq integration is checked by main
syntheticmanufacturedsolutions and pairedgrid QA, not asserted independently
reintegrated. Test knownlinearq/force solution, endpointinterpolation, masked
prefixes/NaNpadding, opposite-signedcross terms, common-speedretention,
small-denominator andmissing/nonfinitefailures beforemain.

## Independent review disposition

Root adopts review's speed-balance naming, interpolation qualification,
resolveddenominators and fixed-set quadraticdecomposition. Worsening shows
autonomousphysical departure is NOT necessary for worsening this conditioned
diagnostic. It does not establish fraction/causality of autonomousfailure:
sharedq may be incompatible and olderrorsmay compensate. Conversely, improved
conditionedbalance motivatesclosed-loopcompatibility work, not a unique
feedback/initialization explanation. Componentresults inform a separately
registered next hypothesis; they do not authorize an autonomousvariant here.
Accepted/defaultM3/0058/S unchanged; freshTEST unopened, consumedPilotTESTexcluded.

## Preexecution QA record

Before any realinput/replay, root identified draft issues: per-event scalar
fieldloops instead of the existing batchedmasked pattern; absoluteamplitudes
mislabelled as savedincrements; intended rather thanactual pathcounts; missing
in-progress replay preservation. One coherent implementation repair batches
activeevents, computes/stores k-k0 andd-d0 once, counts attemptsbeforeactions,
and retains partialarrays. It also completes source/candidate/checker linkage,
cacheprefix checks and metricarray serialization. No gate, budget, replay
path, numericalsetting or scientifichypothesis changes.
Root also corrected the draft metric implementation before real execution:
time-mean of the SUM of 11 weighted squares (not a coordinate mean), severe35
rather than all346, worst4 reporting, identical tail-risk numerical denominators,
relative quadratic identity, and inconclusive/unreported unresolved ratios.
Nine metric synthetics include heterogeneous event scores and numerical tails.
