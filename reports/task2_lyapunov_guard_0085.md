# Loop0085 — one analytic locally guarded coefficient update

Authority after75, new finite stage after0084; diagnostic intervention/model
screen, promotion none. Prior0082 is CLOSED/rejected, not extended or rescued.
0083 found newly unstable local poles;0084 found no nonzero exact-Jacobian-
preserving direction in the same24 amplitudes. A weaker local stability
inequality can allow movement without another state, loss term or structure.

## Question, intervention and budget

Does ONE explicitly guarded version of the existing conditional-ridge proposal
improve autonomous TRAIN prediction while preserving a specified local quadratic
decay inequality? No causal fraction of0082 error is claimed. No optimization,
line search, candidate screening or threshold selection occurs here.

Baseline original G3000 model SHA
2667480a75d7672ad49ed66c79731e754676bae52574229dbb5f98c581ac890b.
Proposal is corrected0082 theta-theta0, model SHA
72fce608287d9e13dd23957e860ad5d20abb2e3130cd883e0f493e1021348db5.
Everything except these same eight H and sixteen port scalar amplitudes frozen.

Budget one analytic guard, ONE candidate and ONE full346-event autonomous
comparison at.005/.01 steps; main wall30min, forecast20min. No alpha/margin/P
search, retry or extension without stage review. Synthetic and cached arithmetic
QA counted separately, no additional candidates. One coherent preexecution
implementation repair allowed; finish/review this stage after its sole result.
No H5, new TRAIN materialization, VAL or TEST: reuse immutable0082 corrected
TRAIN caches with verified provenance, masks and initial states. No new support.

## Guard defined before computation

Use saved0083 Q (only common angle removed), A0=Q.T*A33_baseline*Q and
DeltaA=Q.T*(A33_proposal-A33_baseline)*Q. Fixed interface coordinates and Q;
do not whiten, rescale states, optimize P, omit ports or remove common speed.

Solve A0.T*P+P*A0=-I32 once with scipy.linalg.solve_continuous_lyapunov(A0.T,-I).
Symmetrize P to remove roundoff; require min eigen(P)>1e-10 and relative
residual norm(A0.T*P+P*A0+I)_F/norm(I)_F<=1e-9. Record full P/eigenvalues,
condition and residual. This fixed P is restrictive, not all stable systems.

Let S=DeltaA.T*P+P*DeltaA (symmetrized), mu=max eigen(S).
Set alpha=1 if mu<=.49, otherwise alpha=.49/mu. No alternate alpha, margin,
line search or post-hoc damping. Define theta_guard=theta0+alpha*(theta82-theta0).
The.49 leaves a prospective.01 slack relative to the required half-baseline
decay: (A0+alpha*DeltaA).T*P+P*(A0+alpha*DeltaA)<=-.51I theoretically.
Save alpha/mu before any candidate forecast or its scores.

After exporting/reloading the candidate, compute its actual full33 AD Jacobian
ONCE. Require relative Frobenius difference from A33_baseline+alpha*DeltaA33
<=1e-10 (denominator max(1,norm(expected))). Require actual reduced symmetric
decay maximum eigenvalue <=-.5, local spectral abscissa<-1e-6/s and
eigenpair residual<=1e-9. No separate full candidate FD: existing validated
0083/0084 affine Jacobian tensors and direct candidate AD provide the check.
Numerical failure leaves this intervention inconclusive; no guard rescue.

Check original theta matches cache exactly, alpha finite and0<alpha<=1,
only allowed state_dict keys changed, frozen causal observer/buffers unchanged.
Reuse validated0082 all-row affinity and structural/export/origin/gauge/power QA
at its two fixed synthetic states, threshold1e-10; don't rerun a full design.
Source/model/caches before/after hashes unchanged. Preserve candidate separately.

## Conditional description and primary autonomous gates

Because the field is affine, compute guarded fine/coarse conditional residuals
exactly as r0+alpha*(r82-r0) from corrected0082 saved residuals. Preserve all
46514 windows,346 events,311/35,24 groups; unchanged equal-event aggregation,
fixed L_b and exact0082 scales. Report all previous conditional ratios and
their ORIGINAL reference gates (.90/1.05/1.05); do not relabel0082 as passing.
These are descriptions of the new guarded update, not the primary0085 question
of useful autonomous transfer. Every-window fine/coarse norm<=.001 remains a
mandatory numerical gate, not merely descriptive. No new weighting or fitting.

Autonomous protocol and success gates are unchanged from0082: one saved causal
100ms qenc and true physical x0, then no future injection. All346 TRAIN events
at genuine available1/2/5s prefixes, including severe truncated events; no
artificial tail/filtering. Actual RK4.005 reference/.01 comparison with finite
every-stage checks; per-event COI/common/Pm numerical RMS<=.001.
Exact autonomous scales(.47675314588683515,1.0033667249602924), frozen p_scale;
point=.5mean(COI delta/dscale)^2+.5mean(COI omega/wscale)^2+
(common omega/wscale)^2+mean(Pm error/p_scale)^2. Average genuine offsets1..k.
Risk=.8event mean+.2worst35; average across1/2/5s. Required candidate/baseline
average ratio<=.90; unstable35 5s-available-prefix mean and worst4 each<=1.05.
All these gates and the guard/QA are required for diagnostic support. No model
promotion in this stage, even if supported. In particular local stability alone
does not support the prediction claim. No change to accepted M3/0058/S.

Reuse0082 corrected baseline trajectories ONLY with exact hash link, event IDs,
labels/lengths/physical initial state/qenc/mass/p_scale and metadata checks;
recompute baseline1038 metrics and numerical rows against0082 evidence<=1e-11.
The original causality/coverage verification is inherited from0082, not newly
established by reading unused future data. Read cached truth only for scoring.
Cache new forecasts BEFORE numerical gates; partial/failure evidence on error.

## QA and interpretation

Synthetic scalar/matrix cases must reproduce the analytic alpha, P residual,
PSD guard, unchanged24-amplitude interpolation and rejection of invalid inputs.
Root reviews implementation before its sole real execution. Independent cached
NumPy recomputes P/guard/alpha, actual candidate decay, conditional summaries and
all autonomous event/aggregate/numerical metrics to1e-10 (1e-11 for forecasts).
For clarity before execution, independent guard/conditional comparisons use
max-absolute discrepancy divided by max(1,maxabs(left),maxabs(right))<=1e-10;
record absolute discrepancies too. Forecast comparisons remain absolute1e-11.
This avoids imposing absolute1e-10 accuracy on a potentially large condition
number while retaining the separately stated physical/numerical guard gates.

Independent stage81_challenge cautions, adopted by root: fixed P can reject
otherwise stable changes and depends on declared coordinates; verify actual
candidate inequality, not just analytic alpha. Scalar contraction confounds
stability with update size. This tests one guarded proposal, not why all error
occurred, optimal stable coefficients, the full nonlinear region or full-order
fidelity. A failed guarded forecast rejects its useful-transfer claim only.
No0082 threshold or classification is changed; this is a new, explicitly bounded
intervention with a different primary question, not an after-the-fact rescue.

Preexecution independent contract review: stage81_challenge found no blocking
conceptual objection and checked the.49/.51 guard algebra. Root adopts its
clarification that reusing0082 QA means reusing the PROCEDURES on this new
candidate, not inheriting the old candidate's results. In particular f(0)=0,
gauge, export and actual port-inclusive power checks must run for0085.

Preexecution cache anchors, matched to the existing independent0082/0083
checks without computing0085 quantities:
fit_arrays.npz b23928f3075dc88966e2f1e51d6b689056ff72ea54b7e64ce88ff90b3dbe5294;
0082 trajectories.npz 5b09a22e8c5cdadf8a6710db3164d97fc0ad15051d7397ffcdb178e00d73997d;
0083 matrices.npz 46bcaa3480fdb422f3f98ca069243ca8f911bd44ae7c061f4eba75b77199d212.
0082 evidence hash1fb5a1146f90bfe24455021914525cf69d7366e24a318f6384cb37b10ab0adad.
Baseline numerical rows are in0080 evidence.numerics.baseline, SHA
38b206d5316846596da8670489f3714fedb866f9f92dd2f9759cf1fb8e931209;
0082.autonomous_numerics describes its candidate, not baseline. Reuse that
verified0080 reference for the baseline-numeric check; metrics remain0082.
