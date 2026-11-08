# Loop0079 — frozen learned electrical-force span (preregistered)

Starting checkpoint bd6a520. Authority: after75 stage B failure diagnosis.
0078 is closed: both bounded G screens fail while C3000 retains trajectory
performance. Its G TRAIN error was dominated by relative angle; local
origin instability and mechanical-power overweight were not simple causes.
Do not refit/add capacity before locating a more specific limitation.

## Question and hypothesis

Does the frozen G-fixed3000 instantaneous electrical-force representation
leave a majority of its causal sliding-encoder force error outside the
directions reachable by changing q? Primary hypothesis: the event-balanced
perpendicular share defined below is >.5. This is one finite-TRAIN
descriptive attribution, not a model-selection or statistical test.
C-free3000 is a matched-budget descriptive control, with no separate gate.

Objects: `private_data/task2_structure_0078_extend_G-fixed/step3000` and
`private_data/task2_structure_0078_extend_C-free/step3000`. All tensors,
observer, coordinate transform, supports, normalizations and physical units
stay frozen; accepted M3/0058/S stay unchanged. No G-auto refit, oracle
rollout, coefficient fitting, new loss, new state, TEST or validation tuning.

## Exact physical quantity and decomposition

Use original346 eligible TRAIN events (369 raw;311 stable/35 unstable),
all genuine .01s endpoints i>=10, with causal histories x[i-10:i+1]. Reuse
read_development(full_train=True); disclose unused validation materialized
by that reader. Never enumerate/read TEST payloads or hash a mixed H5 file.
No endpoint sampling, artificial tails or removal of difficult events.

x=(delta_sync,omega_sync,p=Pm_raw-pm_star). The common true subtracted
electrical torque deviation is T=te-pm_star, in system-pu torque units.
Recover it as event.force + E0_frozen_M3(delta) + D_frozen_M3*omega; this
reverses the reader's audited old-force subtraction. te is not Pe or
omega*Pe. Do not subtract the newly fitted backbone from the true target.

For each frozen deployed model define its actual total force prediction

  h(x,q) = p - M * f_omega(x,q).

This includes ALL learned backbone/closure/ports and uses actual p. It is
not a claim that fitted G coefficients are individually physical torques.
Both current fields are affine in q at fixed x; verify this numerically,
do not assume that all future templates will be. Define

  h0(x)=h(x,0),  L_j(x)=h(x,e_j)-h0(x),  u=T-h0,
  qF=L^+u,  e_perp=u-L*qF,
  e_parallel=L*(qF-qenc),  e_total=u-L*qenc.

qenc is each model's own anchored causal encoder. L may vary with x in G;
never replace it by the origin map. Use per-endpoint SVD pseudoinverse with
relative cutoff1e-12. Report ranks/singular values; do not require rank3,
whiten q, or use covariance/conditioning as a latent-order acceptance gate.
No eigenmode independence claim follows from this instantaneous rank.

qF is an omniscient pointwise force projection, not a true latent state,
causal forecast input, or dynamically consistent trajectory. Only the
physical error decomposition is interpreted; it is invariant under an
invertible constant change of latent coordinates.

## Metrics and fixed decision

For each event average squared e_perp/e_parallel/e_total over ALL its
genuine endpoints and ten generators, then average events equally. Set

  share_perp=E_perp/(E_perp+E_parallel).

Accept the primary diagnostic iff G share_perp>.5; otherwise reject it.
Zero denominator is undefined/not tested, not a tunable numerical floor.
Do not reinterpret a C/G difference or a time bin as the primary gate.
Report raw force RMS, event distributions, all24 groups and stable/unstable
subsets for both objects, plus initial endpoint and record-relative time
bins [.1,1),[1,2),[2,5),[5,8),[8,infinity). Report counts in every bin;
later bins do not describe missing unstable tails or the same population.

Also report net-force-error RMS and common-acceleration-error RMS
sum_generator(e)/sum(M) for all three components. Their scalar common-motion
cross term need not vanish; do not assert Pythagoras in that scalar output.
These force metrics are not an attribution of autonomous rollout error.

## Engineering QA before scientific execution

Synthetic tests: exact affine in-/out-of-span examples, deficient and zero
rank, equal-event weighting, fail-closed nonfinite input, and coordinate
similarity. On actual endpoints verify affine equality at qenc and fixed
q=(.7,-.4,.2), decomposition, L^T e_perp orthogonality and Pythagoras with
relative max residual<=1e-9 (max(1,norm) scaling). Test physical projection
invariance under q'=Tq, L'=L inv(T), with fixed
T=[[1,.2,0],[0,2,.3],[.1,0,.5]], same1e-9 bound. Report maxima, not only flags.
Nonfinite arrays, wrong dataset identities or QA failure are implementation/
data failures; preserve the offending event and do not produce a scientific
pass. Missing/zero-rank modes are reported, not removed.

Before actual full-TRAIN execution, root reviews the implementation and
targeted synthetic tests. Hash concrete checkpoint model/meta/norm/export,
accepted base/network, dataset CSV/splits and relevant execution sources
before/after. Verify original0078 split identities and model hashes. Retain
minimal per-endpoint inputs to independently reconstruct the projection
(T,h0,L,qenc, event/endpoint IDs), per-event summaries and QA results.
Independent saved-array arithmetic must reproduce the primary shares and
decomposition before closure. No generic framework or schema is needed.

## Failure meaning and next implication

Support identifies a majority floor in this learned frozen force map on
observed physical states: changing only q cannot remove that pointwise
floor. It does not prove the entire allowed H/J/R/port family is incapable,
justify more q, or establish the cause of all rollout errors. Rejection
instead prioritizes within-span state initialization/closure diagnosis.
Either outcome leaves model selection, energy/stability claims and TEST
access unchanged. Choose only the next smallest hypothesis from the result.
