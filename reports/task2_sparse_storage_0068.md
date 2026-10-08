# Loop0068 preregistration — B1 sparse storage on frozen S

Authority after66; may execute alongside0067. Source S and its normalization,
M3/topology/encoder/ODE remain immutable. This is a candidate-criterion screen,
not a continuous-domain proof or model promotion. No TEST or new dynamics.

## Question / family / fixed budget

Can H3 plus <=24 scalar correction terms retain local strict Lyapunov
curvature and give useful initial-state coverage without admitting unstable
development events? Use z=[delta1-delta10,..,delta9-delta10,omega10,p10,q3].
H3 uses p+pm_star at its existing API. Its coefficient is fixed1, H(0)=0
and gradient H(0)=0 exactly; no free overall energy shrinkage.

Library: six symmetric .5*q'Pq*q coefficients; all30 q_i*omega_j,
30 q_i*p_j,78 q_i*(sin(eta_e)-sin(eta*_e)),78 analogous centered cosine
terms on existing26 angle edges; two physical terms delta'K0*omega and
omega'p. K0 is M3 equilibrium angle stiffness, gauge invariant. Total224
coefficients; six Pq coordinates are retained, then at most18 others.
No polynomial/library expansion or different state coordinates.

Two convex fits only: L1 fit in this library, select the18 largest absolute
RMS-standardized non-Pq contributions (index tie-break), then refit fixed
support with L1 weight0. Clarabel max200 iterations, tol_gap_abs/rel and
tol_feas1e-8; no solver/weight sweeps. Infeasible or inadequate numerical
solution is reported, not silently relaxed. Keep full first fit as evidence.
Before trajectory generation, a data-free feasibility check of these same
hard local matrix constraints may screen out an infeasible family. It uses
the same Clarabel budget/tolerances and zero objective, not a third fitted
energy candidate; no coefficient or threshold selection from validation.

## TRAIN points / objective

Frozen-S trajectories from all eligible stable TRAIN initial100ms histories,
sampled0,.5,..,8s, RK4 .005 with .0025 companion. No re-encoding. Preserve
every event and numerical/LOS failure; no falsely padded or clipped states.
Only finite points strictly before the earlier LOS of the two integrations,
with cumulative-prefix normalized COI step discrepancy<=1e-3, may enter
fit; every omitted point/event remains in coverage and failure accounting.
Finiteness is checked at every integration substep through each sample.
Data-interval QA and existing LOS implementation are reused.

Let s_j=max(TRAIN-point95th percentile abs(z_j),1e-3). Declared D is the
intersection of |z_j|<=s_j and all absolute restored generator-angle differences<=pi.
Add2048 seeded68 interior probes along random box directions, radially
limited by D and multiplied by Uniform(0,.95); add1024 boundary probes on
the exact same ray/polytope intersection. No validation point sets scales.
Each direction is s*Uniform([-1,1]^32); solve its positive ray intersection
with every box and affine angle halfspace exactly. Boundary directions are
fresh draws. After setting scales from all eligible trajectory fit points,
only in-D trajectory points enter the regional fit and calibration; retain
out-of-D counts. Independent check points use the same domain restriction.
Independent TRAIN check times .25,.75,..,7.75s plus1024 fresh seed6801
interior probes are verification-only, not refit inputs.

For each library column use RMS over fit points, floor1e-8, as its coefficient
scale. Reference Href=a*z'P65*z, with a fixed by matching physical-block
Hessian trace to H3 at0. P65 is the saved0065 Lyapunov matrix. Let hscale be
max(median positive Href on fit points,1e-6); W=diag(diag(H3 Hessian physical),
ones(3)). Local constraints in W-normalized coordinates:
Hessian H >=1e-6*W; -(J32'Hessian H+Hessian H*J32)>=1e-6*W.
These constrain the full curvature, not just Pq. Primal residuals independently
checked; minimum W-normalized eigenvalues must be >=.5e-6 after solving.

Minimize mean(((H-Href)/hscale)^2)+.001*sum(abs(standardized coefficients))/hscale
+10*mean(pos((1e-6*z'W*z-H)/hscale)^2)
+10*mean(pos((dotH+1e-6*z'W*z)/hscale)^2).
dotH is exact grad(H)'f of the frozen ODE, affine in coefficients; derivative
and origin Hessians checked against autograd on synthetic/local points.
Nonlinear inequalities are penalized, NOT assumed proven by the fit.

## Candidate boundary / evaluation / meaning

Calibrate rho=.95 times the minimum H among sampled D-boundary points,
in-D TRAIN fit/check positivity/dissipation counterexamples and all eligible
unstable TRAIN initial states inside D. Counterexample: H<=0 or
dotH>1e-7*hscale away from0. A nonpositive limiting value gives rho=0 and
zero coverage; no rescue by discarding it. Keep limiting IDs/points.
The same D/calibration rule applies to Href as the dense local baseline.

Report in-D and H<rho membership, rho, support/equation, violations, stable
coverage and unstable inclusion, with all raw/eligible/latency-uncovered
TRAIN and validation counts. Membership alone does not verify connectedness,
continuous positivity/dissipation or forward invariance: label CANDIDATE,
not certificate/stable truth. No outside-region 'unstable' classification.

Screen support requires local matrix checks, <=24 terms, no violation on
independent sampled TRAIN checks within the selected sublevel, >=20% eligible
stable validation initial coverage and0 unstable validation inclusions.
Frozen rho/scales/support before validation evaluation; no threshold tuning.
Regardless of screen result, continuous-region certification remains open.
Small coverage triggers after66's family-vs-dynamics diagnosis/B2, not project
completion or a general32D certification framework. Root decides next step.
