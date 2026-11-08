# Loop0073 — B2 joint sparse-storage compatibility screen

Starting checkpoint5fdf457. S/0058/M3 remain unchanged; J0067 is the matched
data-only algorithmic control. Authority: after66 B2, following the fixed-S
0068/0069 local separation witnesses and the closed0072 capacity probe.

Question/hypothesis: can one small dynamics–energy joint optimization remove
the necessary origin storage obstruction in the ORIGINAL224-term family,
with at most24 final energy terms, without sacrificing S safety/tail fidelity?
This is a necessary first B2 screen, not completion of regional L2: its
declared energy constraint concerns the equilibrium tangent system only.
No finite-radius domain, coverage, global convergence or certificate follows.

One allowed algorithmic change: add an alternating sparse-storage bias to
the exact0067 J objective. Start independently from S; same twelve trainable
tensors,36-term Xi mask, alpha mask, physical parameters/ports, r3, GRU,
100ms causal history, fixed normalization/q buffers, sampler seed67,
batch32,800 draws,1/2/5s schedule, fresh Adam1e-4, one CPU. No teacher,
new ODE term, dimension, pruning of Xi, loss reweighting or optimizer reset.

Storage H=H3+sum gamma_k psi_k uses exactly0068's224 terms (including6 Pq
terms and its two physical cross terms), not0069's added curvature. H3
coefficient is fixed1. Use its exact origin Hessians H0,Hk and gauge P/E;
W=diag(diag(H0 physical29),ones3), V=W^(-1/2), fixed throughout. This is
physical unit scaling, not latent covariance whitening. Let
s_k=max(||V Hk V||_F,1e-8), beta_k=s_k gamma_k,
H=H0+sum gamma_k Hk, D=-(J32(theta)'H+H J32(theta)).

At optimizer steps0,100,200,...800, freeze theta and solve ONCE for gamma,t:
min t+1e-4 sum|beta|, subject to t>=0, VHV>=1e-5 I and
VDV+tI>=1e-5 I. Clarabel max_iter200, gap/feasibility tolerances1e-8,
no alternative solver or tolerance sweep. At step200, solve full224 first,
keep6 Pq plus18 largest |beta| others (ascending index breaks ties), then
one immediate refit with that support. Keep this energy support thereafter;
inactive gamma exactly0. All other solves use current support. Save all
solver states/coefficients, t, independently computed constraint residuals.
Finite optimal/optimal_inaccurate solutions require residuals<=1e-6;
otherwise stop and classify solver/optimization failure, not family rejection.

Between solves, gamma is fixed. Define LH=mean[relu(1e-5-eig(VDV))^2].
J32 is the exact differentiable full-coupled origin Jacobian, not Aq.
On the FIRST matched TRAIN batch at unmodified S, compute Euclidean gradient
norms gL and gH over the SAME twelve allowed tensors (unused gradients=0).
Fix lambdaH=.1*gL/gH for all800 steps. Require finite gL,gH>1e-12 and
finite lambda; otherwise stop as unusable-bias optimization evidence without
choosing another weight. This single TRAIN-only calibration allocates10%
of the initial data-gradient norm to energy; it does not tune against VAL.
Train L+lambdaH LH; no clipping, rescaling, offset correction or budget rescue.
Run all800 steps unless a numerical/source/frozen contract fails.

Cheap preflight: analytical J32 versus full-autograd, gradient versus finite
differences through energy penalty, physical Hessian/library dimensions,
solver residual checks on synthetic systems, exact support/tie selection,
f(0)=0/frozen/inactive checks, then two-step real-TRAIN smoke. Initial data
loss and both batch draws must match J0067. Smoke is not a formal warm start;
formal execution starts again from S after root diff review and checkpoint.

Screen final step800 ONLY. First independently require <=24 energy terms
(6 Pq slots retained), both actual minimum eigenvalues of VHV and VDV>=1e-6,
f(0)<=1e-10 and all source/frozen/finite/sampling contracts. The1e-5 fit
target is a numerical cushion, not a relaxation of this1e-6 physical-unit
screen. Do not run expensive VAL when this necessary local screen fails.
If it passes, run existing107 eligible/117 raw VAL and L1 numerical policy:
all genuine-score and LOS-required intervals must pass; no unresolved gain.
Misses, falseLOS and false-window-settled cannot increase versus BOTH S/J.
Complete1/2 pooled/CVaR, complete5 mean/pooled/CVaR, unstable-prefix
mean/pooled/CVaR each<=1.05*S (floor1e-8); common5 pooled<=1.05*J0067.
Report complete8s, per-event errors and full coupled poles diagnostically.
No required RMSE gain: genuinely new sparse energy compatibility is the
prospective benefit, but cannot compensate violated safety/tail gates.

Pass only authorizes immediate TRAIN regional storage/coverage and
counterexample work with a new preregistered finite domain, then limited
candidate confirmation; do not end B2 at a tiny local ellipsoid. A finite
budget failure rejects this algorithm, not existence of another H or r3
closure. Preserve evidence and diagnose the specific obstruction before
another candidate; no lambda/steps/support-budget sweep. Fresh TEST remains
unopened. No final promotion, formal regional proof or project completion.
