# Loop0079 — majority out-of-span attribution rejected

Question: does the learned frozen G-fixed3000 electrical-force map leave
most sliding-encoder force error outside directions reachable by q?
Preregistration24a201e; implementation42bc7fa; baseline bd6a520. No fit or
model change. Primary share>.5 is **rejected**, not the whole G family.

## Evidence

All346 eligible TRAIN events (311 stable/35 unstable,24 groups) and465317
genuine endpoints are retained per model.369 raw TRAIN includes23 excluded
before100ms. Unused validation is materialized by the inherited reader;
only TRAIN enters the diagnostic and TEST is not read. The physical target
is te-pm_star, not Pe, with all current backbone and ports included in
h=p-M*f_omega. Both learned maps are affine in q at each physical state.

Equal-event mean-square/RMS torque errors:

| Quantity | G-fixed3000 | C-free3000 control |
|---|---:|---:|
| Perpendicular MSE | .319097240 | .115537523 |
| Within-span MSE | .441123308 | .383466368 |
| Total MSE | .760220549 | .499003891 |
| Perpendicular RMS | .564886927 | .339908110 |
| Within-span RMS | .664171144 | .619246613 |
| Total RMS | .871906273 | .706402075 |
| Primary perpendicular share | .419742982 | .231536317 |

True torque-deviation RMS=1.653491086 for both. G stable/unstable
perpendicular shares=.298515/.464140; C=.169048/.255641. At the initial
100ms endpoint G share=.317391, within-span RMS1.190806 versus total1.441302.
G time-bin shares [.1,1)/[1,2)/[2,5)/[5,8)/[8,inf) are
.312238/.242450/.450063/.416837/.576587 with346/316/311/311/311 events.
The last bin has stable events only and tiny total MSE .001511619; its
majority share does not supersede the overall gate or describe missing tails.

Both maps have rank3 at every endpoint under the fixed1e-12 SVD cutoff.
This is instantaneous force rank, not three independent dynamic modes.
Affine/decomposition/orthogonality/Pythagoras/similarity residual maxima
are below1e-14. Seven synthetic tests passed before execution. Source,
dataset CSV/split and original0078 identity checks pass before/after.
Independent batched-pseudoinverse arithmetic repeats all vectors, event/
group/label/bin/initial aggregates and exact endpoint coverage: zero
discrepancies; vector relative differences<=1.88e-15, similarity<=2.82e-15.

Common-acceleration RMS (perp/parallel/total) is G .281311/.885557/.800194
versus C .131973/.616874/.568737 rad/s^2. Scalar components interfere;
their squared errors do not form the force-space Pythagorean partition.

## Interpretation and next implication

58.03% of G's observed-state force error is within its existing latent
force span. An out-of-span floor is substantial and larger than C, but is
not the majority of G's total error. As a post-hoc comparison only, about
77.93% of the G-minus-C total MSE gap is the perpendicular-MSE difference;
that cross-model arithmetic neither changes the registered primary result
nor identifies which template restriction/optimization choice caused it.

No coefficient change, rollback or promotion: accepted M3/0058/S remain,
and C3000 keeps its prior trajectory-control status. qF is hidden-truth
pointwise projection, not latent truth or an autonomous state trajectory.
The next smallest diagnostic is to replace ONLY G's initial q at100ms by
this force-projection value, then run the SAME frozen autonomous equations.
Preregister whether that single initialization improves TRAIN rollout risk
without severe-prefix harm. Failure cannot prove that every other possible
initializer is ineffective; success cannot make the hidden-force input
deployable or establish dynamic closure. No repeated truth injection.

Evidence: `private_data/task2_structure_force_0079/{evidence.json,
projection_arrays.npz}`. Independent script/evidence:
`private_data/task2_structure_force_0079_independent/{check_arrays.py,
independent.json}`. The checker had implementation issues corrected before
its successful audit; no diagnostic model/data/gate was altered. Runtime
was approximately277s from monitoring, not a serialized scientific metric.
