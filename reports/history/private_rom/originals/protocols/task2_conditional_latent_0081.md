# Loop0081 — frozen conditional latent consistency (preregistered)

Start84bb5f2; after75 diagnosis following rejected0080. G3000's one-shot
force-projection initializer worsens autonomous prediction despite improved
instantaneous force. Do not add capacity or another fitting loss yet.

Question: is the pointwise projected qF trajectory MORE consistent with the
existing frozen latent dynamics than the causal encoder qenc trajectory,
when accumulated physical-state rollout error is removed diagnostically?
Hypothesis: qF's event-balanced physical-force-mapped short-window defect
RMS is<=.90 qenc overall, with unstable-subset RMS<=1.05 qenc. Both required.
This is a conditional descriptive diagnostic, not independent validation,
autonomous prediction, latent truth or a deployable physical-input path.

## Frozen objects and exact experiment

Use only G-fixed3000 and its0079 saved qF/qenc/L, original full346 TRAIN
events/24 groups/311 stable35 unstable (369 raw). Same model/source/data
identity checks as0080; no new model, parameter update, output map, encoder,
state, support, threshold, TEST access or validation computation. Reader
may materialize unused validation; disclose it. No whole-H5 hashing.

Partition each genuine trajectory from endpoint10 into consecutive windows
[10,20],[20,30],... with last short window retained, never dropping a
difficult event. No zero-duration window. Expected46514 windows over the
465317 saved endpoints (465317-346 genuine increments); verify mapping.
For each reference r in {qenc,qF}, set qhat(a)=q_r(a). On [a,b] integrate
ONLY dqhat/dt=f_q(x_true(t),qhat) from the SAME frozen deployed full field.
Interpolate actual physical30-state x linearly between its .01s samples at
RK4 stages. Do not integrate/reset physical x as if this were autonomous,
and do not reset q within a window. Reset only at the next diagnostic
window's declared start. Use reference endpoint q_r(b) only for scoring.

At the true physical endpoint b use0079's saved actual10x3 force map L_b:

  d_r = L_b * (qhat_r(b)-q_r(b)) / ((b-a)*.01).

This is the force effect of a finite-window latent consistency defect,
in torque/s, NOT a derivative of true electrical force (L varies with x).
It is invariant under constant invertible q-coordinate transforms. No
latent covariance/whitening/coordinate norm enters the criterion.

## Metrics and numerics

Per window take mean over10 generators of d_r^2. Average windows equally
within each event, then events equally; RMS=sqrt(that mean). All346 events
must remain. Report35 unstable and311 stable, all24 groups, initial window
and window lengths/counts, not just an aggregate. This equal-window measure
includes final short windows; do not silently duration-weight or discard
them. No reinterpretation of a subgroup as the primary gate.

Reference RK4 .005; comparison .01; both use SAME linear interpolation of
stored physical x, not different physical data. All genuine intermediate
q substeps and field evaluations must be finite. For every window and both
references compare endpoints via
RMS_generator[L_b*(qhat_.005-qhat_.01)/duration]/16.534910857679173 <=.001.
The fixed scale is0079 true torque RMS1.6534910857679173 divided by .1s.
Numerical failure is unresolved numerical evidence, not scientific rejection;
no finer-step rescue or alternative window size is authorized.

Before full execution test affine/linear synthetic q systems, interpolation
at RK4 stages, terminal-short-window mapping, no within-window reset, zero
defect for a known exact constant drive, physical-output coordinate similarity,
event weighting and fail-closed nonfinite input. Root reviews implementation.
Verify0079 per-event endpoints exactly and original model/norm/base/network/
CSV/split identities; compare hashes before/after and frozen in-memory
tensors. Save per-window a/b/event/group/label,L_b,reference endpoints,
predicted endpoints at both steps,duration, errors and provenance. Independent
cached NumPy arithmetic must reproduce numerical comparisons and all RMS
summaries. Never call repeated truth-conditioned windows autonomous rollout.

## Interpretation and continuation

Support means the projected curve is more locally compatible under this
conditional test, not proof it is a good autonomous initializer. Rejection
means instantaneous force inversion is not a better dynamically consistent
reference for these frozen equations; do not replace encoder targets with
qF or increase dimension on that basis. Decide the next smallest physically
motivated change only after combining this result with0078–0080 evidence.
Accepted M3/0058/S and C3000's prior control status remain unchanged.
