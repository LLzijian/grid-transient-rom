# Loop0082 — one existing-coefficient solve and autonomous transfer

Startca9f43a. Authority: after75, new finite stage after the0079–0081 review.
Work type: diagnostic intervention/model screen; no model promotion in this
stage. The old0078 screen remains rejected and its training budget closed.

## Question and bounded claim

Can recoverable coefficient error in the EXISTING G-fixed3000 observer/support
be reduced by one tractable integral regression, with useful autonomous TRAIN
transfer? Freeze coordinates, encoder, decoder, all base coefficients, J/R/B,
normalizations, support, units and data. Change ONLY the eight existing signed
storage_gamma scalars and sixteen existing port_coefficient scalars. Preserve
all paired field effects of each H coefficient. No extra state/group/port,
direction rotation, new output head, qF target or independent RHS is allowed.

This is not optimization of the original full nonconvex rollout objective.
It tests one explicit conditional surrogate and its autonomous transfer, not
the best attainable template, convergence of Adam, or true latent coordinates.

Budget: one24-coefficient fit, one fixed candidate, one full346-event autonomous
TRAIN comparison at two steps; reuse0080's verified baseline cache if identities
match exactly. At most one implementation repair before scientific execution;
if a later correctness fault is found, quarantine evidence and review the stage
before rerunning. No fit variants, lambda search, line search, extra epochs,
support search, finer-step rescue or validation screen in this stage. Main
execution wall cap60min (fit/design30min; forecast30min); save partial/failure
evidence on exhaustion. Tests and independent arithmetic do not create further
scientific candidates; record their cost. Close/review after this candidate's
result regardless of success. A further stage requires a new justified budget.

## Data and conditional integral objective

Original full346 TRAIN events,311 stable35 unstable,24 groups,369 raw; no
boundary additions. Use the46514 windows and exact0079/0081 per-event mapping:
[10,20],[20,30],..., retain genuine short final windows. No TEST/H5 enumeration
or whole mixed-H5 hash. Inherited read_development may load unused validation;
disclose it; no validation quantities enter fitting or scoring.

For each window [a,b], form z_ref(t)=(x_true(t),qenc(t)) from the saved0079
G3000 causal encoder at each genuine endpoint and the measured physical30
state. Both are linearly interpolated between stored .01s samples for THIS
conditional regression only. qenc increments are constructed observer targets,
NOT measured latent dynamics. qF is not used. Autonomous evaluation receives
only one100ms initialization, never these future reference samples.

At fixed J/R/B and physical base, the full interface field is affine in the
24 signed amplitudes theta: f(z;theta)=f0(z)+F(z)(theta-theta0). Derive H
columns via shared gradient and (J-R); port columns are their frozen direction
times the existing centered basis. Convert momentum derivatives to omega via
M^{-1}. All33 rows must match direct coefficient perturbations, not just force.

For each window of duration tau, r(theta)=[integral f(z_ref;theta)dt-
(z_ref(b)-z_ref(a))]/tau. Evaluate the integral by composite Simpson on each
stored interval, using linear-reference quarter/midpoint states: reference
two Simpson panels of width .005 per .01 interval. Comparison is one Simpson
panel of width .01, SAME reference interpolation. No new q encoder at synthetic
times and no interpolated sample beyond the genuine terminal endpoint.

Use a fixed linear output W_b on r, giving31 residual components:

- sqrt(.5/10)*COI(r_omega)/1.003367 (10 components);
- mass-weighted common r_omega/1.003367 (1);
- r_p/(sqrt(10)*frozen p_scale) (10);
- sqrt(.1/10)*L_b*r_q/frozen force_scale (10).

The scales are exact state_dict buffers of
`private_data/task2_structure_0078_extend_G-fixed/step3000/model.pt`, SHA256
`2667480a75d7672ad49ed66c79731e754676bae52574229dbb5f98c581ac890b`:
force_scale is scalar1.2613572464007323; p_scale is the generator-wise vector
[.3992764927880838,.5194290668735476,.5511126892482401,.5345526891567607,
.5240140125801449,.5173564423363574,.4652199303177382,.3292289724162482,
.7433608362339378,.3559398305391048], applied elementwise in the saved generator
order. These were read model-only before any0082 fit. The second independent
preregistration review requested this explicit scale/source clarification;
root incorporated it before execution. Quadrature tolerance is .001/s in
this fixed normalized residual metric, not dimensionless.

L_b is the ORIGINAL0079 G3000 force map and stays fixed in the fitting metric
even as H coefficients change the candidate's actual force map. The first21
components are the physical residual; last10 are the mapped latent residual.
Angles obey fixed kinematics and add no coefficient-dependent fit rows. This
physical scaling follows existing speed/Pm metrics; .1 is a prospectively fixed
latent-surrogate weight, not a claim of equivalence to the old latent loss.
No q-coordinate covariance/Euclidean norm criterion is introduced.

Write W_b r(theta)=c_w+A_w DeltaTheta. Weight each window by
w_w=1/(346*n_windows_in_its_event), with sum w=1. No duration weighting or
stable-tail weighting. Define s_j=sqrt(sum_w w_w sum_k A_w[k,j]^2). Columns
with s_j<=1e-12*max_j(s_j) are numerically inactive: DeltaTheta_j=0, report them
without deleting support. If every column is zero, record that result, do not
invent an update. For other columns let beta_j=s_j DeltaTheta_j and solve ONCE:

  min_beta sum_w w_w ||c_w+A_w diag(1/s) beta||^2 + 1e-4 ||beta||^2.

This strictly convex ridge subproblem fixes the collinearity/tie-breaking rule
prospectively. No post-hoc damping or blending with the old coefficients.
Report unregularized residuals separately from the penalized objective,
design singular values/condition, ridge condition, coefficients/deltas/scales,
and normal-equation stationarity relative residual<=1e-9. Ridge uniqueness
does not imply physical identifiability or overall optimization adequacy.

## QA, gates and transfer

Before the one fit: root reviews equations/code; synthetic affine-field,
Simpson polynomial/interpolation, unequal-window event weighting and exact
ridge-solution tests; finite shapes and complete identities; direct all-row
finite coefficient-perturbation columns and full candidate-affinity check
<=1e-10 relative. No original model/source file is overwritten.

For baseline and candidate, every window's norm of W_b*(fine-integral minus
coarse-integral)/tau must be<=.001; all quadrature values finite. Fixed reference
remains fine Simpson. Failure is numerical/integration evidence, not rejection
of the scientific hypothesis; no alternative quadrature or rescue candidate.

Conditional-fit gates: total unregularized equal-event RMS candidate/baseline
<=.90; physical-only RMS<=1.05; unstable35 total RMS<=1.05. Report physical
and latent RMS separately overall,311/35,24 groups, and initial windows.
Run the one autonomous comparison after fit and numerical QA even if these
conditional gates fail, because transfer itself is the question. Save that
failure as well; never turn a subgroup into the primary result.

Autonomous protocol: original G3000 causal qenc at endpoint10, identical true
physical initial state; no future truth input. Full genuine available TRAIN
prefixes to1/2/5s, all346 events at each horizon, masks exactly as0080. RK4
reference .005 vs .01; all genuine substeps finite; per-event COI/common/Pm
numerical RMS<=.001 using0080 definitions. No artificial tails or finer rescue.
Use0080/0078 point errors, equal-event .8mean+.2worst35 risk and average across
1/2/5s. Candidate/baseline average risk<=.90; unstable35 5s-available-prefix
mean and worst4 mean each<=1.05 baseline. All gates required for support.
No developmental VAL selection, LOS/L3/L2 promotion or TEST in this stage.

Cache window c/A/weights, exact mapping and both-step baseline/candidate
residuals, solver statistics, coefficients and model export, forecasts, raw
per-event metrics, original/candidate hashes and allowed-change verification.
Independent NumPy cached arithmetic must reconstruct the fit and summaries;
reloaded exported field must match in-memory field, kinematics/origin/gauge
and energy identity remain valid. An energy identity is QA, not model validity.

## Interpretation and independent challenge disposition

Supported gates show useful recoverable error in this fixed coefficient block,
not template superiority or final validation. Better conditional fit but failed
autonomous transfer rejects this surrogate intervention, not the structured
family or r3. Numerical/implementation failures remain separately classified.
No accepted M3/0058/S or C-control status changes. Preserve failed artifacts;
revert no accepted model because none is overwritten.

The independent stage81_challenge review supports the distinction between a
tractable subproblem and the original nonconvex objective, not this hypothesis's
outcome. Root adopts its cautions: explicit qenc target assumption, separate
physical/latent residuals, fixed ridge tie-break, H effects in every field row,
and mandatory autonomous transfer. This single branch replaces more oracle
diagnostics; it does not silently roll over0078's training budget.

## Prospective correction stage2 (2026-09-22, before attempt2)

Stage1 is closed as implementation failure against the literal W scale;
see `reports/failed/0082_attempt1_normalization.md`. Its one fit and133.84s
execution remain counted, not reset or promoted to scientific rejection.
Root authorizes ONE further corrected execution, same hypothesis and gates,
with the same60min total/30min design/30min forecast caps. Thus cumulative
authorized0082 executions are two (one invalid, one corrected), not unlimited.
No further retry is authorized without stage review.

Preserve the original text's conditional W speed denominator EXACTLY1.003367.
Keep inherited0080/0078 autonomous scales EXACTLY(.47675314588683515,
1.0033667249602924); do not use the conditional literal for autonomous metrics.
Force/Pm scales, coefficients, ridge, data, windows, all thresholds and all
other implementation choices stay fixed. Record both scales in artifacts.
Only this normalization implementation is corrected; no numerical rescue,
new model structure, alternate loss, post-hoc gate or parameter tuning.

Use a fresh `private_data/task2_coefficient_block_0082_attempt2` output and
separate independent outputs; never overwrite attempt1 evidence. Existing
passing tests remain reusable; rerun affected scale tests and model-only QA.
Original source d49b106 remains the reproducible attempt1 checkpoint. Keep
the strict internal1e-11 resummed-coefficient flag if it recurs; independently
verify the registered1e-9 stationarity and actual candidate metrics, reporting
the distinction instead of changing either threshold. Close/review after this
one attempt, irrespective of result. No VAL/TEST or further fit is authorized.
