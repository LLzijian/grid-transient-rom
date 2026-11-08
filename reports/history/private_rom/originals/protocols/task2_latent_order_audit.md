# Loop0059 preregistration: coordinate-invariant latent-order diagnostic

Starting accepted checkpoint c159dd8.0058 is closed with all22 unchanged gates
passing; its accepted1s screen does not change the audit target. The user's latest instruction supersedes the old covariance-based
order criterion for this NEW diagnostic, never retroactively for0056/0058.

## Question and hypothesis

Does frozen unwhitened0056 contain three excitable, observable dynamical
directions with distinct dynamics and independently useful autonomous action,
or is it dynamically reducible to two states despite its three coordinates?
Hypothesis: the third direction is not merely coordinate redundancy. This is
a diagnostic, not acceptance of a newly fitted predictor.0057 receives a compact
secondary local-linear audit; it is not a coordinate transformation of0056.

## Frozen baseline and allowed work

M3 remains accepted. Primary source0056 implementationa196841 and model SHA256
8c6e666bb4683311c89d6c564273d9d3ab5f0f081cfdc268dd3c2663f65c061d.
No training, r>3, extra losses, encoder changes, library changes, topology
changes, threshold tuning, TEST access, or model promotion. Only frozen-system
derivatives, input/output diagnostics, coordinate changes and no-refit projections.

## Derivatives and representative operating region

In column convention fq=Aq q+g(x), x=[delta,omega,p]. Aq is exactly constant
for this retained library, Xi[1:4,:]^T. Bq=dg/dx is evaluated at the zero
physical reference and at available TRAIN endpoints .1,.3,.6,1.1s, uniformly
weighted within each event and then across events. Never pad missing points.
Zero is not called a learned equilibrium unless the full vector-field residual
is below1e-8; report the residual regardless. Aq is the latent subsystem,
not the full feedback-coupled ROM's stability Jacobian.

Actual output Cq=(S_G B_net C_normalized) diag(amplitude), with generator
electrical-power/torque correction as output. Also report unit-direction C.
Report Bq in physical rad,rad/s,pu units, and input-normalized using TRAIN
event-balanced first-second RMS of [COI delta, synchronous omega,p], floor.01.

## Dynamic and input/output diagnostics

Report Aq eigenvalues, decay/growth rates, frequencies, pole separations and
eigenvector conditioning. For O=[Cq;Cq Aq;Cq Aq^2] and R=[Bq,Aq Bq,Aq^2 Bq]
report all singular values, condition and relative-rank sensitivity at
1e-12,1e-10,1e-8, including representative-region reachability and PBH checks.
Their raw singular spectra/conditions depend on coordinates/physical units;
rank, poles and transfer maps are similarity invariant. Do not replace a
covariance gate with another coordinate-dependent singular-ratio gate.

Compute finite-horizon Gramians and Hankel singular values at fixed1,2,8s;
do not presume infinite-horizon Gramians exist. Main input metric is
Q=event-weighted mean(Bscaled Bscaled^T); output metric Cq^T Cq. These are
pooled local-linear input/output diagnostics, not a nonlinear controllability
or stability theorem. Report every horizon, never select a favorable one.

For simple modes/conjugate blocks P_i, report invariant residues Cq P_i Bq
and actual TRAIN excitation: RMS(Cq P_i g(x)), RMS(Cq P_i[g(x)-g(0)]) and
RMS(Cq P_i q_initial). Distinguish available input directions from actual
physical forcing and initialized latent action. Preserve complex pairs;
report defective/repeated subspaces without inventing individual eigenmodes.

## Frozen autonomous evidence and structured reduction

Reproduce all1070056 validation predictions at1s first. Then report all three
original-coordinate force ablations and all-memory-off (algebraic term retained).
These coordinate ablations are not invariant modal order tests.

Construct ONE rank2 balanced Petrov-Galerkin projection from the TRAIN-pooled
1s Gramians, keeping the two largest Hankel directions. Fixed V,W with WV=I:
q=Vz, z_dot=W fq(x,Vz), z0=W q0; actual output Cq(Vz+a(x)). No refit,
re-centering, coefficient tuning, or validation selection of the basis.
If simple real eigenmodes exist, additionally report all three single-mode
removal projections; for complex modes remove only complete conjugate blocks.
Do not report the best posthoc ablation as an accepted predictor.

Primary comparison1s: a rank2 reduction is near-preserving only if mean and
pooled errors are each <=1.02 times full, and CVaR<=1.05 times full. An individual
mode has the existing strong contribution evidence if removing it worsens both
mean and pooled by >=1%. Report tradeoffs/mixed evidence, not forced binary order.
There is no covariance or whitening acceptance condition in this diagnostic.
Exact rank3 alone does not prove three important modes in the operating region;
failure of one projection does not prove no rank2 representation can exist.

Fixed2/5/8s autonomous rollouts are secondary stress evidence for all predefined
arms; all107 events are integrated for the fixed horizon regardless of truth
length. Score complete horizons and available unstable prefixes separately.
No padded truth, truth-based stopping or feedback. Use unchanged COI normalized
angle/speed scales and event mean/pooled/worst10% CVaR; preserve group/event rows.

## Engineering validity

Analytic Aq/Bq vs central differences at zero and three fixed TRAIN points:
max error<1e-7. Actual topology force singular values reproduce0056<1e-10.
Full1s predictions reproduce saved0056 maxabs<1e-8, metrics<1e-10.
Fixed nonorthogonal similarity matrix [[2,.4,-.2],[0,.5,.3],[.1,0,1.5]] must
preserve transfer samples at1+i,3+2i,10+3i and Hankel values to relative1e-8,
and physical autonomous trajectories to1e-8. Check RK4.01/.005 discrepancies
for full and balancedrank2 at each horizon;1s threshold remains.001. Long-horizon
discrepancy is reported, never silently attributed to physics or optimized away.

## Interpretation and next implication

Use the combined invariant dynamics, physical excitation/output and autonomous
evidence. Distinguish exact linear order, practical regional order, predictive
importance and long-horizon validity. If evidence supports three relevant modes,
remove covariance orthogonality from FUTURE prospective r3 acceptance and design
that validation separately. If a nearly unobservable/unreachable direction and
near-preserving reduction support lower effective order, pursue structured
reduction instead of increasing dimension. Mixed evidence stays mixed and
requires a smaller diagnosis, not new complexity. Fresh TEST remains sealed.
