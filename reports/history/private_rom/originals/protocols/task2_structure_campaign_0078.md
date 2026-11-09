# Loop0078 — matched structure-discovery screen (preregistered)

Authority: after75 stage B. Execution is conditional on prototype0076 QA.
0077 is independent; no boundary0075/0077 sample enters this experiment.
Starting accepted checkpoints remain M3/0058/S; this loop cannot itself
promote a final L1/L2/L3 model or open TEST. Implementation starts from the
closed0076 checkpoint, whose hash must be saved before the first fit.

## Question and hypothesis

Can a small automatically selected shared-gradient structure learn useful
IEEE39 autonomous dynamics at complexity comparable with ordinary sparse
closure, without transferring the old free Xi dynamics into the generator?
Screen hypothesis: G-auto meets the development retention gates below and
improves complete5s pooled error by at least5% relative to matched C-free.
G-fixed is the optimization/selection control, not an alternative way for
G-auto to pass. A fixed-budget failure is not model-family falsification.

## Immutable data and interface

Original Pilot+expansion only: explicit TRAIN and validation IDs through
read_development(full_train=True),346 eligible TRAIN/107 eligible validation.
Retain all raw/excluded denominators (369/117). No TEST payload enumeration,
whole mixed-H5 hashing, new data, future inputs or repeated truth injection.
State interface delta/omega/p/q, p=raw tm-pm_star, internal pi=M*omega in G;
r3, actual p0, eleven angle/speed observations over100ms, GRU20->8->3.
Known M, equilibrium, angle-difference support, units and normalization frozen.
Use S normalization/q_transform, but encode=(raw(h)-raw(zeros))*q_transform
in every arm; freeze the exactly cancelling head bias. No rewhitening.
Copy S observer/decoder; add Wq*old_encoded_zero to decoder bias so the
initial auxiliary output is preserved under this coordinate translation.
S is initialization/reference only, not a training truth target. Record
initial physical-field/rollout differences from S and M3 on TRAIN.

## Three arms and complexity

C-free: original S sparse support (36 Xi,18 alpha), spatial columns and
amplitudes, now with empirical active network coupling, all10 damping,
all10 mechanical rate/gain trainable. Constant Xi row remains zero. This
ordinary closure is present ONLY in C, never alongside a G field.

G-fixed/G-auto: the0076 generator, all nonzero empirical base coefficients
trainable:26 potential coefficients,10 damping,10 J_pi,p,10 Rpp,13 p/q
quadratic coefficients. Known mass/canonical J never train. Zero base
damping is initialized at1e-4 in ALL three arms; nonzero original damping
unchanged. Other absent potential coefficients stay absent. Positive base
and R amplitudes use softplus, signed H/J amplitudes remain signed.

For this first bounded search, additional J/R directions are fixed sparse
unit-coordinate vectors (plus one fixed common-momentum direction in the
fixed control); indices may be selected, but these directions do not rotate
during optimization. Port directions do train on all23 nonangle coordinates,
normalized to unit length. This is deliberately not unrestricted discovery
of every possible direction, device mechanism, or topology.

Both G arms have two ports, each eight linear centered terms at v indices
(0,3,6,10,13,20,21,22), v=(pi,p,q). Scales are M*speed feature scales for
pi, original p feature scales for p, and1 for q. Port initial directions are
the first two left singular vectors of S's physical latent-force mapping,
embedded in momentum rows, with deterministic largest-entry-positive sign;
this is model-only initialization, not a forced physical interpretation.
Initialize all16 port coefficients at1e-3. No future electrical inputs.

G-fixed has16 added groups from the start (zero-based indices):
H: p0*q0,p3*q1,p6*q2,q0*q1,q1*q2,q0*sin_diff(eta0),
q1*sin_diff(eta8),q2*sin_diff(eta16); angle indices refer to the26 active
M3 columns in their existing order. Normalize p by its feature scale, q by1.
J: (pi0,q0),(pi3,q1),(pi6,q2),(p9,q0). R: q0,q1,q2 and the unit direction
proportional to M on the ten momentum coordinates. H/J initial1e-3,
extra R initial1e-3. These generic positions carry no equipment labels.

G-auto starts with no added H/J/R groups and the same base/ports. Exactly
two opportunities for at most8 new groups occur after steps400 and800.
No dimension/encoder/library expansion. Continuous dynamics parameters at
maximum budget: C about143, G at most145 effective scalar coordinates,
excluding the shared observer/decoder and accounting for normalized-column
radial redundancies; record actual counts, masks, and expanded RHS scalar
contributions. Count discrete searched indices separately; do not claim
equal optimization difficulty from similar parameter counts.

## Lazy selection (TRAIN only)

At each growth use the next fixed32 balanced TRAIN windows from a separate
seed780 stream; these do not alter the fit sampling stream. At their start,
form an Euler one-step integral residual x+.01*f(x)-x_next (truth next
physical state and encoder of the shifted11-point window for q). Convert
physical omega targets to pi=M*omega before this residual. Use the gradient
of this normalized one-step squared residual w.r.t. the starting vector field; this is a
selection surrogate, not latent truth or an extra final prediction loss.

Propose exactly8 legal new groups sequentially: H product, H product,
H state*sin_diff, H state*cos_diff, J coordinate-pair, J coordinate-pair,
R coordinate, R coordinate. Choose the largest absolute normalized
directional loss derivative for each signed group; for R use largest
negative derivative only. Evaluate candidate derivatives as streamed scalar
reductions, never materialize a full Cartesian feature library. H products
range over distinct p/q indices with at least one q; state/trig groups use
q and active angle columns; J pairs use v with at least one q; R uses all
23 unit coordinates. Exclude existing/duplicate groups; lexicographic ties.
Normalize each unit-amplitude field direction by its RMS in state/time
units on these32 windows (floor1e-12). If sensitivity magnitude<=1e-12,
skip that slot; do not substitute another kind. Signed coefficient starts
at1e-3 against the derivative; R at1e-3. RMS normalization is for ranking
sensitivities only, not division of the inserted physical coefficient.
Directions remain fixed. Save all
eight proposed selections, scores and missing slots, not a large registry.

At step1000 prune whole extra groups/port columns whose normalized RMS
field contribution is below1e-3 of total field RMS on the second fixed
seed780 selection batch (the growth800 batch); use this identical batch
in all arms, not a new third draw. No row-wise pruning and no reactivation. In C apply the
same relative contribution threshold to its existing active Xi/alpha terms,
never grow support. Freeze the resulting support and use the remaining200
steps for debiasing (complexity penalty zero). Log both sides of pruning.

## Matched optimization and physical objective

Seed78 for all arms, same balanced32-event/group sampling stream and actual
prefix masks; no padded, clipped or fabricated targets. Adam with the fixed
prefit parameter-block rates below, global gradient-norm cap100, float64,
RK4 dt.01. Fixed1200 updates each:
steps1..100 horizon1s;101..300 alternate1/2s;301..1200 cycle1/2/5s.
Local integral horizon.2s and one-step encoder consistency as in0067.

Point error = .5*(mean(COI_delta_error/.47675314588683515)^2
+ mean(COI_omega_error/1.0033667249602924)^2)
+ (common_omega_error/1.0033667249602924)^2
+ mean((p_error/original_p_feature_scale)^2).
The first two means denote means of squared componentwise errors, NOT
squares of component means. Event-balanced risk=.8*mean+.2*worst10% mean.
Total=10*risk+local+.1*latent_consistency+.1*(force+electrical)+1e-4*C.
Keep latent consistency normalization .1, electrical Pe/V/load averaging
and original TRAIN RMS normalizations. No energy-identity or global-negative
Hdot loss. Origin conditions are exact template/encoder properties.

Force supervision is now the TOTAL physical electrical torque deviation:
prediction=p-M*omega_dot; truth=old_force+E0_frozen_M3+D_frozen_M3*omega.
This algebra uses the audited raw te convention; never train G against the
old memory(q+alpha) output. The decoder remains auxiliary, not an ODE input.
C is the mean RMS normalized field contribution of learned closure groups;
G counts extra H/J/R groups and port columns (ports weighted5), C-free
counts Xi/alpha scalar terms and memory columns (same port weight5).
Normalize derivatives by [S_delta,M*S_omega,p_scale,1] in internal momentum
coordinates; omit the common canonical/base field from this penalty.

Save losses/gradients/parameter and wall-time counts; abort an arm on
nonfinite forward/loss/gradient, failed f(0)/kinematic/structural check or
nonfinite parameters. Preserve failure checkpoint, no silent retry/LR change.
Before fit: targeted tests and real TRAIN one-batch forward/backward smoke;
smoke uses no optimizer update and cannot change any threshold or support.

Prefit optimizer clarification,2026-09-21, before any IEEE39 fit: GRU/head/
decoder lr1e-4; signed closure/storage/exchange coefficients and normalized
directions lr1e-3; all softplus raw nonnegative base and dissipation amplitudes
lr1e-2. Apply the same parameter-type rule to C and G. A data-free scalar
check explains this clarification: Adam1200 from softplus coefficient .001
toward target .1 reaches only .001130864 with raw lr1e-4, but .1 with lr.01.
This check uses no project trajectory or validation result. No LR sweep,
schedule, adaptive rescue, threshold or group-budget change is allowed.

## Screening, budget extension, interpretation

Evaluate fixed1200 endpoints, not best-validation checkpoints. All genuine
prefixes and complete1/2/5s physical metrics; dt.01/.005 agreement<=.001
normalized COI/common and no disagreement in physically scored LOS interval.
Use at most the existing one finer diagnostic if its preregistered numerical
trigger is met; unresolved numerical rows cannot count as scientific pass.
The reporting reference remains the .005s forecast even when the one
.0025s diagnostic is triggered; compare both on .01s outputs and audit
ALL original substeps. Refinement verifies accuracy, not selection of a
better-scoring forecast. Missing long truth is ineligible for that complete
cohort, whereas a nonfinite eligible forecast invalidates its aggregate.

G-auto primary gate: complete5s pooled <=.95*C-free. Retention versus S:
complete1/2/5s mean,pooled,CVaR and unstable available-prefix pooled/CVaR
each<=1.05*S; common5<=1.05*S. L1 eligible unstable detections>=10/12,
false LOS<=0, false window-settled<=2. Counts and raw denominator are always
reported separately. A primary pass is a shortlist, never final promotion.

TRAIN-only extension rule fixed now: evaluate all346 events at step0,800,
1200 with the identical1/2/5s physical risk (no stochastic comparison).
For this deterministic monitor, give each of346 events equal weight, compute
the global worst35-event mean separately at each horizon, and average the
three horizon risks. Do not average minibatch CVaRs. Fit sampling remains
stable/unstable/group-balanced as above.
The step800 monitor is recorded immediately after update800, before the
second structure insertion; before/after insertion snapshots are both kept.
Eligible arms must be numerically finite, improve step0 risk by>=10%, and
improve800->1200 risk by>=2%. At most the two lowest1200 TRAIN-risk eligible
arms extend once to3000, same horizon cycle/optimizer, support frozen and
no complexity penalty. No extension based on validation failure. Repeat
the retention gates at3000. The5% superiority claim requires matched step
budgets: compare3000 G against3000 C only if C independently qualified for
extension. If C was not extended, a retained3000 G is an optimization/
retention candidate, not a rescue of the matched1200 primary hypothesis.
Only the top1–2 screen-passing arms receive seeds79/80
with the entire selection/fitting procedure repeated; define confirmation
and independent L1/L2/L3 follow-up before those runs.

If G-fixed works but G-auto fails: selection/optimization diagnosis. If both
G fail and C works: bounded template diagnosis. If all fail similarly:
shared target/data/optimization diagnosis. Preserve evidence and distinguish
implementation/optimization failures from scientific rejection. No frozen
TEST, parameter-budget expansion, or post-hoc threshold change is authorized.
