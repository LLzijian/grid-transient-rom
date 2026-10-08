# Loop0060 preregistration: temporal identification campaign

Starting checkpoint94fa382. Authority: docs/TASK2_after_loop_59.md.
Champion0058, model SHA256
d82b1d8d75528eb7e55c28f5f5438270fd95eca2499b5afec5486b6894545f67.
No fresh TEST access. This is adaptive development, not confirmatory testing.

## Question and first falsifiable hypothesis

Does exposure to the complete TRAIN reachable region and longer autonomous
time scales improve closure within the champion's existing sparse equations?
First challenger isolates coefficient identification: retain0058 encoder,
conditioning, normalization, topology C/amplitudes, algebraic alpha, decoder,
M3 physics and support masks exactly; optimize only existing nonzero Xi.
This is the smallest temporal-identification probe, not an assertion that
support or representation should remain fixed forever. No artificial damping,
new library terms, new states or covariance gate.0056's zero pole is not
assumed to exist in0058; inspect each candidate's actual masked Xi.

Pre-fit artifact inspection (no new validation evaluation):0058 Aq eigenvalues
are -.58306160 and -.98270392 +/- .19328395i s^-1, unlike0056's integrator.
Each0058 Xi column has12 active terms. The new temporal experiment therefore
does not presume a missing negative diagonal term or a zero-pole cure.

## Data and loss

Read explicit TRAIN/validation IDs only from the existing Pilot/expansion
splits. TRAIN: every valid stored sample, without the811-point loader or7s
sampling caps. Validation: retain up to8.1s for a fixed8s forecast after100ms.
Keep the pre-existing100ms eligibility rule; record every shorter exclusion.
Never remove eligible short/unstable events. Event labels are scoring only.

Each step samples32 TRAIN events uniformly with replacement. For each event:
(A) draw a local endpoint uniformly from10 through length-2 over its ENTIRE
available trajectory; initialize from its genuine11-point history; forecast
up to .2s of genuinely available future;
(B) encode genuine initial0--100ms history ONCE and autonomously forecast the
stage horizon. Average each event's losses over its available observed points,
then average events equally. No padded targets, frozen tails or future inputs.
Training may stop each sampled segment at its actual end because it defines
the supervised loss; deployment/evaluation never uses truth length to stop.

Use existing normalized COI delta/omega squared loss. Long risk remains fixed
R=.8 mean(event MSE)+.2 mean(worst ceil(.1 N) event MSE), TRAIN only.
Total=10 R_long + 1 mean(local event MSE) + 1 L_dyn + .0001 mean(abs(masked Xi)).
L_dyn is the existing .01s latent-step versus next-history-encoder consistency
divided by .1, squared mean, evaluated at sampled local endpoints. It is a
reference consistency term, not a latent truth claim. Existing electrical
heads/mappings are frozen in this first coefficient-only probe, so their
local force/Pe/V/load supervised errors are reported (constant w.r.t. Xi),
not misleadingly counted as extra training gradients. No loss-weight search.

## Fixed budget and curriculum

Adam lr=.0003, seed60, float64,32 events/step,700 steps at2s. Warm-start0058
weights, never random reinitialization; only final step is evaluated/retained,
not validation-based within-stage early stopping. RK4 maximum .02s for TRAIN,
including endpoint .01s for very short prefixes; no state/gradient clipping.
Reject nonfinite loss/gradient/parameter as optimization/numerical failure,
preserve evidence, do not silently skip that event or step.

If2s retains a >=5% mean AND pooled advantage over M3 and its2s numerical
convergence check passes, continue the SAME coefficients for700 steps at5s
with identical settings. This is a curriculum advancement criterion, not a
multi-metric model-acceptance gate. Report tail tradeoffs explicitly and do
not promote systematic severe-event deterioration.8s is evaluation only in
this first probe. Any later8s refinement/support/joint/teacher intervention
requires a prospective addendum before its execution, within this campaign.

## Shared development benchmark and numerical validity

Evaluate0058, each final challenger and M3 at1/2/5/8s from the SAME initial
100ms, all107 eligible events, fixed8s integration independent of truth length.
RK4.005 reference and .01 companion. Score complete horizons separately from
available unstable prefixes; record count, mean event RMSE, pooled RMSE,
worst10% CVaR, every group/event, maxima of physical/q/force states and
autonomous-versus-encoder q discrepancy on available truth. No truth padding.
Existing SCALES and COI convention unchanged. Verify0058's saved1s trajectories
to maxabs1e-8 and aggregates1e-10 before interpreting new comparisons.

Numerical hard constraints: finite loss/gradients/weights/forecasts, no runtime
clipping and eventwise .01/.005 normalized physical discrepancy <.001 at a
horizon before asserting that horizon's quantitative validity. Report both
all-event fixed forecasts and complete scored cohorts; an unobserved-tail
failure is still disclosed. If convergence fails, a fixed .0025 diagnostic
may distinguish integrator sensitivity from model failure; never change
weights/selection thresholds to rescue the outcome. Finite values alone do
not certify physical long-time boundedness or stability; inspect growth and
state/force magnitudes in the reported evidence before any long-time claim.

Hard gates otherwise only: split isolation, causal initialization, autonomous
future, same physical system, no artificial samples/deletion, frozen TEST.
Performance: preserve all nondominated candidates in the1/2/5/8 mean/pooled/
CVaR vector (lower is better); report tradeoffs rather than add veto gates.
Coefficient-only temporal hypothesis is supported if a numerically valid
challenger improves >=5% in at least one2s/5s metric relative to0058 with no
increase in its other two same-horizon aggregates; mixed results are diagnostic.
This hypothesis screen does not equal final champion or M4 acceptance.

Record support, signed coefficients/scales, Aq eigenvalues at1/2/5 stages,
topology-force spectrum, q drift and group-level residual/consistency evidence.
Fixed support makes absence of support changes tautological: do not call it
discovered robustness. No extra group refits in this first small probe.

## Interpretation and next campaign branch

Failure of coefficient-only refinement does not reject r3, joint discovery
or the model family. Diagnose optimization, support limits and latent-reference
compatibility. Follow the roadmap's small r3 neural teacher upper-bound test
after the curriculum evidence; change only latent RHS and preregister the
small architecture/budget before running. Expand equations/distill only if
teacher evidence warrants it. Keep0058 immutable and preserve rejected
challenger artifacts; revert experimental changes where appropriate.
Full reduced Jacobian/energy/stability work is deferred until long ODE freeze.

## Stage B observation and registered Stage C execution

2s completed all700 steps at b2a38d2,1291.58s, finite gradients/parameters;
all non-Xi tensors and normalization remain exactly0058.69 repository tests
pass using a workspace temporary directory (the first default-temp attempt
had3 permission-related fixture errors, not test assertion failures).
Shared evaluator41965a0 reproduces0058 trajectories and1s aggregates exactly.

2s challenger mean/pooled/CVaR at2s=.327516847/.379742568/.728080163 versus
M3 .552234464/.667677861/1.397653010. Maximum eventwise .01/.005 discrepancy
across all107 fixed2s forecasts=.000323488 <.001. The preregistered Stage C
advancement criterion is therefore satisfied. This is not model promotion:
1s and8s regress relative to0058;5s complete-cohort metrics improve. All-event
5/8s convergence is not established, despite small complete-cohort discrepancies.

Execute the already registered700-step5s stage from
`private_data/task2_long_0060_2s/`, same seed60/lr.0003/loss/frozen components.
The standalone stage runner creates a fresh Adam optimizer; source model
coefficients, not optimizer moments, transfer. TRAIN integration uses.01s
steps (within the registered maximum.02s); the long loss scores1/even/actual
odd-final offsets. Do not choose intermediate steps or alter tail weights.
Only the final5s model is evaluated;8s remains an audit horizon.

## Prospective Phase 2 teacher specification (before Stage C results)

Question: with the same frozen r3 representation and electrical output map,
can a small nonlinear latent RHS improve long-time closure beyond the current
sparse support? This is an empirical capacity/optimization diagnostic, never
an accepted final model or a guaranteed universal capacity bound.

Prepare implementation during Stage C but do not train it before Stage C
finishes and its evidence is reviewed. If Stage C produces a finite valid
checkpoint, use that final sparse checkpoint as the fixed teacher anchor
regardless of whether it dominates0058. If Stage C has an implementation or
optimization failure, diagnose first; do not silently choose another anchor.

Let u=the existing feature vector without its bias column (q3, scaled omega10,
p10, sin26, cos26;75 dimensions). Keep all sparse anchor coefficients frozen:

fq_teacher(x,q) = fq_anchor(x,q)
                 + W2 [tanh(W1 u + b1) - tanh(b1)].

W1 is16x75, b1 length16, W2 is3x16, no output bias. Initialize W1 using
PyTorch Linear's default seed61 initialization, b1 from the same constructor,
and W2 exactly zero. The teacher therefore initially reproduces the anchor
field/rollout. Its neural correction is zero at u=0 and does not introduce a
constant equilibrium forcing. It neither forces negative poles nor adds
state damping. Only these1264 neural parameters train; encoder, conditioning,
Xi/alpha/supports, C/amplitude, electrical decoder and M3 remain bitwise fixed.
This changes only latent RHS expressivity, not observations or coupling.

Reuse exactly the registered full-TRAIN local and initial-history long loss,
fixed20% TRAIN tail risk, event balancing and integration. Xi L1 is constant
in this teacher comparison; do not add a new regularizer/loss. Adam.0003,
seed61,700 steps2s then700 steps5s if the existing Stage C advancement condition
is met; stagewise optimizer reset, final checkpoints only. No8s fit in this
probe. No gradient/state clipping or nonfinite-event skipping.

Use the identical1/2/5/8 benchmark. Report local teacher Aq as the actual
derivative of the complete latent RHS at the reference (base Xi alone is NOT
the teacher Jacobian). No formal full-system Jacobian/energy audit yet.
Teacher expressivity evidence is strong if its numerically valid5s mean/
pooled/CVaR vector has at least10% improvement in one entry without increasing
the other two versus the final5s sparse anchor. Otherwise retain tradeoffs
and optimization uncertainty; a weak teacher does not prove the library is
adequate. This is a hypothesis interpretation rule, not a set of new model
acceptance vetoes. All candidates remain on the common performance Pareto
comparison. Equation expansion/distillation requires subsequent root judgment
and preregistration from this evidence; no larger library/r>3 is opened now.

## Stage C evidence and Phase 2 authorization

Final5s fit completed700/700 steps in3179.73s with finite loss/gradients/
parameters. Model SHA256
d34d9f5c36fb296e1d49e59f5f7b18a5da33b56bf371a16169f82bc25664995c.
All non-Xi tensors are bitwise identical to both0058 and the2s source;
normalization remains4e27ea9018655ff31931566b7b61c754592ba41d14baaec086867930f46f8b3a.
Independent saved-array checks reproduce all complete/prefix/M3/numerical
rows exactly.77 repository tests pass. Evidence: `private_data/task2_long_0060_5s/`,
`_5s_eval/`, `_5s_step0025/`, and `.loop-work/long0060_5s_independent_checks.json`.

| Horizon | Final5s mean / pooled / CVaR | Complete events |
|---|---|---:|
|1s|.298393264 / .373210293 / .767440577|96|
|2s|.328373414 / .381051143 / .734655202|95|
|5s|.288326166 / .333570888 / .628207978|95|
|8s|.240232550 / .277769306 / .522296005|95|

The5/8s gains over0058 are substantial without changing encoder/topology/
library/support. Small1/2s tradeoffs remain, so keep nondominated candidates
rather than call this universal dominance. All95 complete8s stable forecasts
now have max absolute COI angle <=1.882rad, versus >18rad in four2s-challenger
forecasts. This is finite-time development behavior, not a stability proof.
The12 unstable genuine-prefix mean/pooled/CVaR=.724623255/1.230200075/1.667156312,
versus M3 1.197555958/1.924729736/3.042979053; no difficult samples were removed.

The original .01/.005 ALL-event discrepancy gate still fails at5s (.001577610)
and8s (.418035981); do not rewrite it as passing. Its complete-cohort maxima
are only7.517e-6/8.627e-6. The preregistered .0025 diagnostic confirms the
recorded .005 forecasts: ALL-event .005/.0025 maxima at1/2/5/8s are
5.280e-7/1.848e-6/.000114849/.026907664. Every genuine observed prefix converges
to max5.446e-7; complete5/8s maxima4.752e-7/5.446e-7. Thus the measured
long-horizon gains are numerically supported on the observed cohorts, but
unobserved unstable8s forecasts are still sensitive. No dt, event or weight
was selected to rescue model performance, and no full8s all-event claim is made.

Aq poles=-.709166050 and-.950450845 +/- .273284666i. They remain distinct from
0056's zero/slow modes; do not narrate a removed integrator. Fixed support is
not discovered support robustness. One small coefficient changes sign from
2s to5s: Xi[35,0]=-.009814449 -> .020875370 (zero-based existing feature index).
Coefficient norms at0058/2s/5s=2.308829/2.379928/2.385950. Group refit/support
robustness and final modal interpretation remain unresolved.

Root classification at this internal campaign stage: useful positive temporal
identification evidence and a strong sparse long-horizon candidate, not final
model promotion/M4 acceptance. The finite, faithfully implemented5s checkpoint
is the registered teacher anchor. Proceed with the already specified Phase2
2s teacher run,700 steps, seed61/lr.0003, then its preregistered advancement
check. Keep8s as audit only; no sparse-library expansion or formal energy work.

Saved-array group/trajectory extraction during teacher training (no new fit
or raw data access): at5s the sparse candidate's mean improves in8/8 validation
groups versus0058; pooled/CVaR improve in7/8. Group34 is the exception:
pooled .2817 -> .2828 and CVaR .5219 -> .5311; at8s its CVaR also rises
.4236 -> .4279. Preserve these localized tradeoffs, not a claim of uniform
tail improvement. Complete95-event8s maxima are COI angle1.8816rad,
COI speed7.1951rad/s, |q|3.2721 and memory force5.2019. All107 fixed8s
forecasts instead reach COI angle190.9075rad and speed54.6194rad/s, including
unobserved unstable continuations; numerical sensitivity there remains above.
Encoder discrepancy is diagnostic, not latent truth: maximum eventwise8s
RMS on the complete cohort falls from2.0453 to.9920, whereas maximum genuine
unstable-prefix RMS rises from.9667 to1.0482. These do not change teacher
losses/thresholds. Factual tables: `.loop-work/long0060_sparse_group_diagnostics.md`;
primary per-event quantities remain in the immutable evaluation artifacts.

## Teacher2s evidence and registered5s advancement

Teacher2s completed700 steps in1388.94s, finite, with1264 trainable neural
parameters. All sparse-anchor tensors are bitwise unchanged; normalization
hash remains4e27ea9018655ff31931566b7b61c754592ba41d14baaec086867930f46f8b3a.
Model SHA256 d7ec8037d89e20dec6e8713b4ff63a56d634a59fec4f7368c3884040a153242b.
Independent saved-array fine/M3/complete/prefix/numerical checks reproduce
exactly. Artifacts: `private_data/task2_long_0060_teacher2s/`, `_teacher2s_eval/`,
`.loop-work/long0060_teacher2s_independent_checks.json`.

| Horizon | Teacher2s mean / pooled / CVaR | Complete events |
|---|---|---:|
|1s|.292148725 / .390265491 / .778048196|96|
|2s|.307021660 / .357261366 / .689657113|95|
|5s|.269590735 / .314555893 / .600397337|95|
|8s|.223124844 / .259693122 / .494121963|95|

At2s mean and pooled are below.95 times M3; all107 fixed2s numerical maximum
.0000356113301 <.001. The preregistered curriculum advancement criterion
passes. Continue700 steps5s, SAME sparse5s anchor, resuming only the teacher2s
neural parameters; fresh Adam.0003, seed61, unchanged loss and tail risk.
8s remains evaluation only. This is not teacher acceptance or strong-expression
classification, which is reserved for the registered final5s comparison.

Important tradeoff: genuine12-event unstable-prefix mean/pooled/CVaR
.749552612/1.398819561/1.927852473 is worse than sparse5s anchor
.724623255/1.230200075/1.667156312. The1s worst event also rises to2.323601103
from1.770544634. No difficult event is dropped, and no post-hoc advancement
veto or loss-weight adjustment is added. Original all107 numerical maxima
fail at5s.001683262 and8s.069612184, while complete-cohort maxima are
5.970e-6 and5.776e-6; no all-event long-time claim is supported. Actual
complete teacher Aq poles=-.712219808 and-1.078311816 +/- .369398080i;
these include the neural derivative, not only the frozen sparse anchor.

Registered fixed.0025 teacher2s diagnostic: every genuine observed prefix
converges to max3.65002e-7; complete5/8s maxima3.77189e-7/3.65002e-7.
All107 fixed5/8s maxima are.0000913272/.0038205054; the unobserved8s-tail
uncertainty remains. Artifact: `private_data/task2_long_0060_teacher2s_step0025/`.
No parameter/dt/threshold was selected to rescue the comparison.

Per-event unstable-prefix comparison against the sparse5s anchor (NOT M3)
has six positive and six negative RMSE changes. Positive total .6293839312,
negative total -.3302316441; net .2991522872 equals12 times the mean change.
pilot_0366 accounts for+.5530564683 (87.88% of positive changes), while
pilot_0155 improves by.246006. Thus the aggregate tail regression is strongly
concentrated, not uniform across unstable events; it remains real measured
evidence, not numerical error. `.loop-work/long0060_teacher2s_tail.md` contains
all12 rows. An initial extraction mistakenly used M3_records; root caught the
inconsistent aggregate-difference arithmetic and corrected the comparator
before interpretation. The primary evaluation/independent QA were unaffected.

Read-only teacher2s parameter diagnosis while5s trains: for the existing
non-bias feature vector u, neural residual derivative at u=0 is
Jres=W2 diag(1-tanh(b1)^2) W1. Its squared Frobenius norm is.6814443541,
of which.1230926359 lies on the frozen sparse support and.5583517182 outside.
This is coordinate-dependent allocation in the fixed normalized feature
basis, NOT an order/independence gate or proof of necessary interactions;
the angular features are not physically independent coordinates. It shows
the teacher probe can adjust couplings excluded by the current mask as well
as add nonlinearity. If the final teacher meets the registered expression
criterion, first distinguish existing-library support limitations from
genuinely new interactions using TRAIN-only evidence. Do not infer a need
for a larger nonlinear library from this reference-point calculation.
Details: `.loop-work/long0060_teacher2s_reference_couplings.md`.

## Final teacher evidence and campaign decision

Teacher5s completed700 steps in4613.81s. Model SHA256
34f8dc19ecda7fcdd3737b6bb27dbc043e83bdb5e664fee93851677c43c52066.
All anchor tensors/norm remain bitwise fixed; independent complete/prefix/M3
and numerical saved-array arithmetic reproduces exactly. Evidence:
`private_data/task2_long_0060_teacher5s/`, `_teacher5s_eval/`,
`_teacher5s_step0025/`, `.loop-work/long0060_teacher5s_independent_checks.json`.

| Horizon | Teacher5s mean / pooled / CVaR | Complete events |
|---|---|---:|
|1s|.294220798 / .383999365 / .767027243|96|
|2s|.307315326 / .357286364 / .689871152|95|
|5s|.261733951 / .305473069 / .586381271|95|
|8s|.217384648 / .252127520 / .479828697|95|

Compared with the sparse5s anchor,5s improvements are approximately
9.22%/8.42%/6.66%. None reaches the preregistered10% condition. Root decision:
the strong teacher-expressivity claim is NOT supported at this fixed budget.
Do not round9.22% into10%, extend training, change the comparison horizon,
or modify tail-risk weights. These modest gains are real development evidence,
not proof that the existing library is optimal or that representation is the
unique bottleneck. Teacher remains a diagnostic, never the accepted sparse ODE.
Genuine unstable-prefix mean/pooled/CVaR=.734855184/1.338713414/1.863387641,
still worse than the sparse anchor;1s worst event2.177180002 remains higher.

Original all-event .01/.005 maxima at5/8s=.010130919/.589674803 fail the
registered gate. Complete-cohort maxima6.056e-6/5.763e-6 are small. Fixed
.005/.0025 diagnostic gives all-event5/8s maxima.000783328/.094151283;
all genuine prefixes converge to max3.642e-7. Thus no all-event8s quantitative
or stability claim is made. Actual teacher Aq poles=-.727450316 and
-1.129083743 +/- .394444341i; this is not the full reduced spectrum.

Additional saved-array center-frequency check (not a new acceptance gate):
teacher2s mean synchronous mass-center omega RMSE is lower than sparse5s at
all complete horizons and on unstable prefixes. At8s it is.336691 versus
.347972 rad/s (M3 .391427); considerable individual errors remain. The initial
hand-extracted1s drift table used later-horizon cells; root caught violation
of the delta_dot=omega integral bound. Corrected executable arithmetic is in
`.loop-work/check_common_frequency.py`, with table in
`.loop-work/long0060_common_frequency_diagnostic.md`. No model or primary
benchmark artifact was affected. Do not confuse M3 field input Pm_raw with
the joint model's p=Pm_raw-Pm_star in subsequent physical audits.

Close0060 as diagnostic: coefficient-only long training yields a strong sparse
candidate; the preregistered strong teacher condition fails. No model family
is rejected. Preserve all nondominated sparse candidates (0058,2s,5s). Select
the sparse5s checkpoint as the provisional top LONG-HORIZON audit object:
same36 active coefficients/encoder/topology complexity, much better5/8s
errors and restored bounded complete-stable forecasts, with disclosed short
and group34 tradeoffs. This is adaptive development selection, not final M4
acceptance or prospective TEST confirmation. No library expansion is triggered.
Teacher code/artifacts remain isolated diagnostic tools; no teacher was made
the default and no accepted model was overwritten. Next: fixed-model scientific
audit. Group-wise coefficient/support robustness and formal stability/energy
remain unresolved. Fresh TEST stays unopened.
