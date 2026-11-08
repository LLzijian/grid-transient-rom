# Loop0087 — bounded direct-objective training of the same stable coefficient block

Start ae2c0e8, authority after75. Diagnostic intervention/model screen; promotion
none.0082/0085/0086 remain closed. Their conditional optimizers do not establish
adequacy for autonomous risk or severe safety. Test one fixed-budget direct
causal objective procedure without increasing states, support, observer or ports.
The question is useful reachable behavior under THIS procedure, not convergence,
model-family expressivity or attribution of0086 failure to the surrogate alone.

## Budget and preflight

One timing preflight on frozen G3000: one seed87 batch32, forward/backward at
1/2/5s, no optimizer update/model mutation/forecast scoring. Three gradient
evaluations total,10min cap. Report each time, finite gradients and parameter
hashes, not a candidate. Synthetic projection tests are separate QA. If finite
QA fails, preserve evidence and close as implementation/numerical work.

Expected training cost =100*sum(three preflight seconds)+300*.5seconds for
projection. Training authorized only if this estimate<=2400seconds. This is a
predeclared resource screen, not performance selection. Otherwise close this
stage as resource-inconclusive, no shortened selectable run or automatic retry.

Then exactly300 updates,300 single projection SDPs maximum, one final candidate,
one346-event paired autonomous forecast. Training cap45min INCLUDING all
gradients/projections/logging; projection max200iterations/300s each, no fallback.
Main cap60min excluding the separately counted preflight, forecast cap20min.
No extension, extra seed, intermediate best selection, line search or tuning.
If caps prevent completion, preserve partials and classify incomplete optimization,
not scientific rejection; do NOT select/forecast an intermediate checkpoint.
One coherent preexecution QA repair; no silent post-execution retry.

## Frozen identity and sole trainable block

Baseline `private_data/task2_structure_0078_extend_G-fixed/step3000`, model SHA
2667480a75d7672ad49ed66c79731e754676bae52574229dbb5f98c581ac890b.
Start here, not rejected0086. Only8 storage_gamma and16 entries of the two
port_coefficient tensors change. All other state_dict objects frozen, including
GRU/decoder, empirical physical backbone, J/R/B, support and all scales/buffers.
No new h/port terms, latent dimensions, losses, pruning or data.

Use unchanged original TRAIN reader/sampling:346 eligible/369raw,311stable/
35unstable,24groups;23 excluded. Exact CSV/split identities and exclusions match
0082/0079. Original reader may materialize unused VAL, disclose; no VAL score,
TEST access or boundary0075/0077 merge. Initialphysicalp0 remains true; encoder
uses11 causal angle/speed samples once for each rollout; future ODE gets no truth.

## Exact objective and algorithm

Reuse `train_structured_rom.objective` and `train_joint_campaign.sample_batch`
unchanged. NumPy/Torch seed87, batch32, new RNG/Adam at training start (preflight
does not advance training RNG). Horizons cycle1,2,5 starting1, the same mature
schedule as0078 globalsteps3001–3300. DT=.01, genuine future masks, exact physical
metric normalization and inherited group/stability-balanced sampling unchanged.
Objective =10*risk +local.2s +.1*latent_consistency +.1*(force+electrical),
complexity_weight=0 as0078 after1000. Batch risk=.8mean+.2topceil(.1*batch).
This batch tail is NOT the final unstable35/worst4 metric; final safety gates
remain mandatory. No extra severe penalty or reweighting after inspecting results.

Adam acts on original theta amplitude coordinates, lr1e-3, betas(.9,.999),
eps1e-8, weight_decay0, amsgradFalse; fresh zero moments. Clip joint theta-gradient
L2 norm at100. Freeze all other parameters before creating optimizer.
After each Adam proposal theta_hat, project beta_hat=s*(theta_hat-theta0) onto
the SAME0086 fixed-P feasible set:

  minimize ||beta-beta_hat||2^2
  subject to .49I32-sum_j beta_j L_j >=0.

Use0086 solver with K=I24,g=beta_hat (constant omitted), not a new penalty in
the scientific objective. Exact same Clarabel options/tolerances/no warm start.
L_j uses the0084 full33 coefficient tensor and0085 P,0083 Q,0082 scales as0086.
This Euclidean beta projection is NOT Adam's metric projection. Persist original
theta-space Adam moments across projections; do not silently reset/reparameterize
them. Apply theta=theta0+beta/s only after numerical projection QA passes.
No convergence claim. Projection may stall or oppose momentum: log that evidence.

Check each solve with0086 registered KKT gates (stationarity/complementarity/gap
<=1e-6, primal/dual eigenmin>=-1e-8, finite; optimal/optimal_inaccurate only).
The old beta is a feasible reference; projected objective must not exceed its
objective by1e-7*max(1,abs(reference)). Save proposal/projected beta, raw/symY,
solver status/options/versions/time, gradients, displacement/projection distance,
loss components and elapsed time EACH update, including a failing solve before
stopping. Count actual solves, don't treat300 projections as one numerical run.
No second solve on a failed projection. Check finite parameters/gradients/losses
and unchanged frozen objects. Raw trace/checkpoint is evidence, not candidate
selection. Training means/endpoint trends are descriptive, not adequacy proof.

## Inputs, final checks and autonomous gates

Preserve0086 preregistered baseline/cache/tensor/P SHA anchors, exact24 order and
theta0, fixedP residual<=1e-9 and minEigenP>1e-10. Reassemble L independently
from immutable prior tensors; no secondconditional-design rebuild. Save all
identities/source/input hashes before/after; never modify accepted/default models.
Fresh dirs `private_data/task2_guarded_rollout_0087_preflight` and
`private_data/task2_guarded_rollout_0087`; refuse overwrite.

After300 valid updates only, export/reload final explicitODE; same0082/0086
origin/gauge/kinematic/power/affinity/frozen-key QA<=1e-10. Full33 AD Jacobian
once, relativeFro match A33_0+sum_j(beta_j/s_j)D[:,j,:]<=1e-10; actual fixedP
decaymax<=-.5, spectral abscissa<-1e-6/s,eigenresidual<=1e-9. Localguard is not
a finite-amplitude/full-order guarantee. No newFD needed for validated tensor.

Use0085 baseline-cache provenance/metric verification, exact0082/0080 scales,
all346TRAIN events at genuine1/2/5s prefixes; one RK4.005/.01 pair, finiteall
stages, event COI/common/Pm numericalRMS<=.001. Save forecasts before gates and
partial outputs on failure. No invented tails, filtering or severe-event removal.
Success requires candidate/baseline average1/2/5s risk<=.90 AND unstable35
5s-prefix mean/worst4 each<=1.05 AND all optimization/numerical/structural QA.
No promotion or confirmation on this TRAIN-only result. Independent cached
reconstruction of all projection KKT/affine finalguard (1e-10maxrelative) and
forecast1038rows per arm/summaries/numerics (1e-11absolute) required.

Synthetic tests: original24theta pack/set ordering; known off-diagonal and
free projections; moment preservation after overwrite; frozen-key enforcement;
finite/shape failures; budget-incomplete behavior; no preflight parameter change.
Reuse0086 KKT and existing objective tests without building a new framework.

Independent route challenge (stage81_challenge) adopted:300 solves count/cost,
resource preflight, theta-Adam versus beta-projection geometry/moments explicit,
exact mature objective/schedule, no safety guarantee from localP. Starting point,
objective and optimization path differ from0086, so no single-factor attribution.
At completion/cap close and review this finite stage, no automatic rollover.
