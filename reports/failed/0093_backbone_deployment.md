# Loop0093 — exact response-supervised backbone candidate rejected

Work type:model screen. Result:rejected. Model promotion:none.
Prereg1a2356b, implementation0f921b8; protocol
`reports/task2_backbone_deployment_0093.md`.0091 remains rejected and0092
remains a supported response-span diagnostic, not a promoted model.

## Completed fixed test

Deployed exactly0092's26 positivecoupling and10 positivedamping amplitudes,
without fitting/interpolation/training. Only two raw parameter tensors changed;
masks, observer, otherstorage/J/extraR/B/ports, normalization and physical
coordinates remained frozen. Actualreload and nonlinearstructure checks pass.
TwoAD Jacobians,132 candidateFD RHS,one32-stateeig andtwo candidateforecasts
completed once. Original346-event baseline is cached, not rerun.

ActualcoupledJacobian matches the derivedbackbone change. FDrelative errors
1.70223e-11/1.61334e-11; eigenresidual2.99760e-16. Alpha=-.1443456409719708/s,
sensitivity8.04177e-8, so the registeredlocal/numerical launch checks pass.
This is not a nonlinear stabilityregion/fullorder certificate.

Modal matches4/13 (64,77,79,91), improved overG3000's0/13 but FAIL>=7.
Targets/frequencies informed the response fit: development, not independent
modal confirmation. Unmatched62,66,68,71,73,84,98,100,104 retained.

| TRAIN metric | OriginalG3000 | Candidate | Ratio / gate |
|---|---:|---:|---:|
| Average1/2/5 risk | .8938195532 | 2.8524347639 | 3.191287049 / <=.90 FAIL |
| Severe35 mean5s-prefix | .6654646978 | 3.7570451269 | 5.645746708 / <=1.05 FAIL |
| Severe worst4 mean | 2.3038732387 | 15.3523579534 | 6.663716430 / <=1.05 FAIL |

Candidate1/2/5 risks2.0512183874/3.3458244847/3.1602614196. Pairedstep
numerical maxima COI6.87438e-7,common4.64134e-8,p2.86577e-7 allpass.001.
All346events/1038rows and genuineprefixes retained; no fabricatedpaddingtail.
Thus this is a numerically valid rejection of the fixed jointcandidate,
not an optimization failure or evidence that the model family is impossible.

Independent firstcachedcheck passes: arrayrelative2.32722e-16,
scalarabsolute2.13163e-14; no additionalmodelconstruction/AD/eig/forecast.
Main47.9126592s + independent5.8844966s + metadata20s =73.7971558s;
stageclosed within1400s. Main/helper15synthetics1.71s; independent9synthetics
1.224s atroot beforelaunch. One coherent preexecutionimplementation repair,
no realretry or gatechange. Candidate/source hashes unchanged duringexecution.

## Evidence and scope

Command `python experiments/audit_backbone_deployment_0093.py --output private_data/task2_backbone_deployment_0093`.
In thatdirectory:
- arrays82659802566986e76bcb5cdd11b289f5adc3110ae0895790c5c586d5c97d8f51;
- evidenceec4ab06c662dadfd0aff6dfc71c23c8db11d27af0375d0560789965665ae107f;
- trajectories835159a63b046e8962d92acc4c67120f6442ff70d134e95395969883717ff91a;
- candidatemodel8f699a7524c71529227f01a2876e9d4ec4633cea095dad3edfc93914ca33d46e.
Independent `_independent/independent.json`
SHA99bd9c74d71dd0dafa727018cf32bc8f99b91a37d03b9479fc72b0b8d28c843d.

Local harmonic response improvement does not transfer automatically to
large-disturbance forecasts. Positivebasecoefficients and localstability do
not rescue poor modal/trajectory fidelity. This jointk/d intervention does
not by itself distinguish damping, finite-angle forces, initialization or
closed-loop feedback as causes. No extrapolated mechanism claim.

Rejected standalonecandidate and reproductioncode retained; no accepted/default
model was overwritten, so no sharedimplementation rollback is needed. M3/0058/S
remain unchanged; noH5/VAL/TEST read, freshTEST unopened, consumedPilotTEST excluded.
Nextproposal under independentchallenge: conditionalphysical-momentum balance
on the same cachedtruephysical trajectories and sharedlatentpath, separating
knownk/d force increments before any new optimization or model complexity.
No next execution authorized by this conclusion; failedcandidate is not Goalcompletion.
