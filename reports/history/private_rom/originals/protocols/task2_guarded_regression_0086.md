# Loop0086 — optimize the coefficient direction inside one fixed local guard

Start43bc08e, authority after75. Diagnostic intervention/model screen; no
promotion.0082/0085 remain closed and rejected,0083 local finding unchanged.
0085's analytic truncation preserved local decay but allowed only.5591% of the
proposal and missed the autonomous gate. Test whether the SAME conditional
objective has a useful autonomous solution when its direction, not just length,
is optimized within this one fixed-P feasible set. No mechanism attribution.

## Finite budget and invariants

ONE convex24-coefficient solve, one candidate, one conditional quadrature QA
rebuild, one346-event autonomous comparison at.005/.01. Total main60min,
solve5min/max200 iterations, conditional rebuild30min, forecast20min. No second
solver, altered tolerance/margin/lambda, alpha blending, line search, retraining,
new support/GRU/state/port/loss term or validation screen. One coherent
preexecution implementation QA repair; after scientific execution any defect
requires preserved evidence and stage review, not silent retry. Close after
this result. Synthetic/cached audits counted separately, no new candidates.

Only the same eight existing H and sixteen port amplitudes may change. Same
G3000 baseline/units/frozen observer/base/J/R/B/scales and causal initialization
as0082/0085. Fresh output `private_data/task2_guarded_regression_0086`.
No accepted/default model overwritten. TEST never read. Original TRAIN only,
346 eligible/311 stable35 unstable/24 groups, no boundary additions.

## Fixed convex problem

Use corrected0082 cached K,g,s,theta0 (all24 active), with K_s=(K+K.T)/2;
require K_s positive definite. K already includes the SAME ridge1e-4. No
new penalty. beta=s*(theta-theta0), so the objective, apart from constant,
is beta.T*K_s*beta-2*g.T*beta.

Use0085's P and0083's Q/A0, not a new Lyapunov function or state scaling.
Check A0.T*P+P*A0=-I residual<=1e-9, min eigen(P)>1e-10. From0084 saved
tensor D[row,coefficient,input], form E_j=Q.T*D[:,j,:]*Q/s_j and
L_j=E_j.T*P+P*E_j. Symmetrize each L_j for roundoff only.

Solve exactly one convex quadratic/semidefinite problem:

  minimize beta.T*K_s*beta - 2*g.T*beta
  subject to F(beta)=.49I32-sum_j beta_j*L_j >=0.

This is the same prospective half-baseline local decay with.01 slack used
by0085, not a fitted energy function. Baseline beta=0 strictly feasible;
the guarded0085 point is a reference feasible point. Signed amplitudes remain
signed; don't silently constrain their signs or prune any support.

Use the EXISTING isolated `.loop-work/research_deps_0067` solver libraries,
not an installation or new framework. Load existing main NumPy/SciPy first,
then import CVXPY1.9.2/Clarabel0.11.1 from that path; record runtime versions
(preflight NumPy2.5.2/SciPy1.18.0). CLARABEL only, max_iter200,time_limit300,
tol_gap_abs=tol_gap_rel=tol_feas=1e-9; no warm start, fallback or tuning.
Preserve beta, dual PSD multiplier Y, K_s/g/s/P/L, objective and solver status.

Numerical optimality gates, independently recomputed (not status-only):
finite beta/Y; min eigen(F)>=-1e-8; min eigen(sym(Y))>=-1e-8;
stationarity ||2(K_s beta-g)+[tr(Y L_j)]_j||2/max(1,2||g||2)<=1e-6;
absolute complementarity |tr(YF)|/max(1,|primal|,|dual|)<=1e-6.
Here primal=beta.T*K_s*beta-2g.T*beta and
dual=-(g-.5ell(Y)).T*K_s^{-1}*(g-.5ell(Y))-.49tr(Y).
Require |primal-dual|/max(1,|primal|,|dual|)<=1e-6. Solve status must be
optimal or optimal_inaccurate with ALL numerical gates passing. Report residuals.
Also require primal<=objective(beta85)+1e-7*max(1,abs(objective(beta85))),
where beta85=s*(theta85-theta0); this is QA for this same convex problem,
not selection of another candidate. Failure is optimization/numerical evidence,
not scientific rejection or reason to try another solver. No forecast on
failed prerequisites. All success statements remain numerical, not exact proof.

## Input provenance and necessary quadrature QA

Keep0085 preregistered cache/model SHA anchors. Additional inputs:
0084 derivatives.npz SHA8817cff75d4d6308c561f8eee4f2362292ba40c119a096f10bb611dcf58aaf53;
0085 guard_arrays.npz SHAeeca4bce9c3233f6bdcc81e94ac8f24a8e2f5805ad4d20be151abd092800d640.
Verify theta/scales ordering, same24 block, original baseline/model/source hashes
and P/Q/tensor links to prior evidence, before/after unchanged. Never mutate caches.

0082 saved fine A but NOT coarse A. A new direction therefore cannot infer its
coarse quadrature residual by scalar interpolation. Reuse validated0082
_table/_design ONCE on original TRAIN and0079 saved qenc/L mapping, rebuild
fine/coarse c/A, and require fine c/A/weights plus IDs/labels/groups/windows/L
to reproduce the corrected0082 cache to1e-10 maxrelative (denominator max1).
This QA rebuild is counted, not another fit or model candidate. The inherited
explicit development reader may materialize unused VAL; disclose this, never
score it or enumerate TEST. No whole mixed-H5 hash. Save coarse A/c for this
correctness check and possible repeated use; no schema/framework redesign.

Compute r_candidate=c+A*(beta/s), at both steps; every-window fine/coarse
norm<=.001/s and finite required. Preserve46514 windows, genuine final shorts,
all346 events and unchanged conditional metric/scales/L_b/equal-event weights.
Report all conditional RMS/components/subsets/groups/initial windows and old
.90/1.05/1.05 reference-gate statuses descriptively; not an0082 pass or a
new conditional threshold. Save solve/conditional evidence before forecasts.

## Candidate QA, autonomous gates and interpretation

Export/reload same explicit structured field. Apply0085/0082 structural,
origin/gauge/kinematic/power and all-row coefficient affinity PROCEDURES to
this candidate, <=1e-10. Full33 AD Jacobian ONCE on reloaded candidate;
match A33_0+sum_j(beta_j/s_j)*D[:,j,:] to1e-10 relative Frobenius (max1).
Actual quotient decay maximum<=-.5, spectral abscissa<-1e-6/s and eigenpair
residual<=1e-9. Freeze all other state_dict objects and verify hashes/reload.
No separate FD beyond validated0083/0084 tensors, no local-to-global inference.

Reuse0085's verified baseline-cache procedures (including0080 baseline numerical
rows, not0082 candidate numerical rows). Check exact IDs/labels/lengths/masks/
x0/qenc/mass/p_scale and reproduce all1038 baseline metrics<=1e-11. Causal
100ms initial state only; autonomous future never receives reference truth.
All346 TRAIN events at genuine available1/2/5s prefixes, no artificial tails,
finite filtering or severe-event removal. RK4.005 reference vs.01 comparison,
finite every stage; per-event COI/common/Pm numerical RMS<=.001. Preserve
new forecasts BEFORE numerical gates and partial/failure artifacts on errors.

UNCHANGED autonomous success gates: exact0080/0082 physical metric and scales,
.8event mean+.2worst35 risk averaged over1/2/5s; candidate/baseline<=.90.
Unstable35 5s-prefix mean and worst4 each<=1.05 baseline. All numerical/guard
and autonomous gates required for diagnostic support; promotion remains none.
No three-seed/fold/final-test claim from one TRAIN intervention.

Synthetic tests: analytic unconstrained-feasible and binding scalar/diagonal
quadratic/LMI solutions with dual/KKT checks; tensor coefficient order and
actual affine map; nonfinite/shape failures. Independent cached arithmetic
reconstructs KKT/conditional/actual guard to1e-10 maxrelative, forecast summaries
and1038 per-arm event metrics to1e-11 absolute. No second scientific solve.

Independent prior challenge adopted: fixed P is restrictive and coordinate-
dependent; failure cannot reject stable refinement generally. The constrained
fit changes direction and stability simultaneously, so no causal attribution.
This gives adequacy only for the stated convex surrogate/subspace, not original
nonconvex joint-rollout optimization or template expressivity. Root reviews
the one result before a further scientific stage; no budget rollover.

Preexecution independent contract challenge (stage81_challenge): the SDP,
Lagrangian sign, dual objective and single ridge factor are correct. Root adopts
two requested clarifications: the P residual is explicitly
||A0.T*P+P*A0+I||_F/||I||_F, as0085; verify min eigen(F(beta85))>=-1e-8
in the NEW assembled L_j before using its objective as a feasible comparator.
Failure is numerical QA, not grounds to silently choose another reference.
Use sym(Y) consistently in KKT/dual/PSD checks, preserving the raw returned Y.
Y is the dual of the exact constraint F>=0, with Lagrangian minus tr(YF).
No new experiment, relaxation or result inspection accompanies these clarifications.
