# Loop0064 preregistration — physical-force representation and closure audit

Starting checkpoint c14c1b2. Authority docs/TASK2_after_loop_59.md, Phase5.
0062 showed group-dependent encoder-reference closure;0063 restored exact
equilibrium initialization cheaply but exposed common-frequency tradeoffs.
Neither identifies whether spatial output capacity or within-span latent
representation is the main physical-force limitation. No new model is fitted.

## Hypothesis and fixed model

Primary hypothesis: more than half of the event-balanced squared physical
force error of the frozen sliding encoder lies WITHIN the existing Cq span,
rather than in directions that this fixed output map cannot represent.
This is a diagnostic attribution, not a prediction-performance assertion.
Reference is the original sparse5s encoder and ODE, model SHA256
d34d9f5c36fb296e1d49e59f5f7b18a5da33b56bf371a16169f82bc25664995c.
All36 active coefficients, all other tensors, normalization, topology, r3,
library and dataset/split semantics remain frozen. No TEST access or tuning.

## Physical quantities and exact decomposition

At each genuine TRAIN endpoint i>=10 use the causal history x[i-10:i+1].
Read the full346 eligible TRAIN events using existing full_train reader;
no time truncation/padding/drop. Its unused validation loading is disclosed
but contributes nothing to this audit. Group/label are summaries only.

In incremental x=[delta_sync,omega_sync,p=Pm_raw-pm_star], the subtracted
physical electrical force target is

F=te-pm_star-E0(delta)-D*omega_sync.

Thus p-E0-D*omega-F=Pm_raw-te. te is raw electromagnetic torque in the
established system-pu swing convention, not Pe alone or omega*Pe.
Use actual Cq=(ports @ column_normalized_c)*amplitude and frozen algebraic
coordinate a(delta). Set P=Cq Cq^+, with SVD rcond1e-12 and verified rank3.

qF=Cq^+(F-Cq*a), Fperp=F-Cq*(qF+a),
e_parallel=Cq*(qF-qenc), e_total=F-Cq*(qenc+a).

qF is an instantaneous force-projection oracle using hidden physical truth;
it is NOT the true latent state, a causal measurement, or a forecast input.
Both algebraic and memory force share span(Cq); consequently Fperp is also
(I-P)F and cannot be fixed by changing only latent dynamics at fixed Cq.

Verify e_total=Fperp+e_parallel, Cq^T Fperp=0 and the squared-norm
Pythagorean identity. Euclidean generator-force norm uses equal system-pu
units, not COI projection or mass-scaled acceleration. Latent similarities
q'=Tq, a'=Ta, Cq'=Cq inv(T) must leave all force quantities invariant;
do not inspect q covariance as an order criterion.

## Metrics / declared decision

For every event compute mean over all genuine endpoints and ten generators
of F^2, Fperp^2, e_parallel^2 and e_total^2, then average equally over events.
Primary share=E_parallel/(E_parallel+E_perp), with zero denominator reported
undefined (hypothesis not tested), not replaced with a tuning floor.
The primary hypothesis is supported iff share>.5; otherwise it is rejected.
This is a finite-TRAIN descriptive attribution, not a statistical significance
test or independent group generalization claim. Do not count endpoints as
independent experimental replicates.
Report raw RMS and denominators, event distributions, groups and stable/
unstable subsets; also initial endpoint and time bins [0.1,1),[1,2),[2,5),
[5,8),[8,infinity) seconds from clearing. Per-bin event weighting includes
only events with genuine observations there, reporting counts explicitly.
No inference that late bins describe the same population or unstable tails.

Report net-force-error RMS and common angular acceleration error:
sum_generator(error)/sum(M), where M is ROM swing mass, separately for
Fperp, e_parallel, e_total. Their cross term generally does not vanish in
this scalar common-motion projection; do not claim a Pythagorean partition
of common-frequency error or causal attribution of autonomous rollout error.

Secondary descriptive closure diagnosis, no fitting: on the same consecutive
<=.1s windows as0062, including final short windows, compute

D(q)=Cq*((q_b-q_a)/duration - trapz(Theta(x,q),dt)/duration * Xi).

Compare event-balanced RMS force/s of D(qF) and D(qenc), overall/group/label.
Raw latent coordinates and equation defects are not ground truth. Derivative
noise/quadrature and the different physical-rollout objective limit the
interpretation; no new coefficient/support acceptance gate is inferred.
Use source0062 window X/y to reproduce D(qenc), after event/window/hash
compatibility checks. Because current features are affine in q, X_F equals
X_enc with ONLY columns1:4 replaced by integrated qF mean. Save those new
means, yF and exact endpoint mappings rather than duplicating feature archives.

## Verification / persistence / next decision

Before execution test the projection with synthetic in-span/out-of-span
forces, coordinate similarity, exact decomposition, event balancing and
linear-window integrals. Numerical QA identities require relative error
<=1e-9 with max(1,norm) scaling, not a scientific fit threshold. Reject
nonfinite arrays as implementation/data failure, preserving offending IDs;
no finite-only aggregation. Verify frozen hashes before and after execution.
Save per-endpoint F/a/qenc/qF, event indices/endpoints, per-event statistics,
window qF integrals and source hashes. Independent saved-array arithmetic
must reproduce the decisive share and decomposition before root closure.

Support would prioritize investigation of existing-coordinate physical
meaning/closure, not authorize more states. Rejection would identify a larger
fixed output-span floor, not prove that increased rank or a new topology is
the solution. In either case compare physical projected closure and group
evidence before choosing the next smallest change. No model promotion,
formal full-system stability claim or frozen-test opening occurs in this audit.

## Closure

Implemented7264949; primary share .739474>.5, independently reproduced.
Accepted diagnostic report `reports/accepted/0064_physical_force_projection.md`
retains stable/unstable closure asymmetry and interpretation limits. No model
was fitted or promoted. Continue conditional coupled-Jacobian/storage analysis
of the same frozen equations; no TEST access or automatic complexity increase.
