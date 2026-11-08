# Loop0072 preregistration — one joint nonlinear-memory capacity probe

Authority: after66 section7; Campaign A is CLOSED. Starting checkpoint19184a8
plus0070 closure. S remains unchanged, J0067 is the matched algorithmic
control (not an accepted model). No neural teacher becomes final dynamics.

Question: can a small nonlinear latent RHS, learned together with the same
100ms initialization/physical mappings, resolve the persistent joint
common-motion versus COI/severe-prefix tradeoff? The original library has
constant Aq;0070 support rediscovery did not resolve it.0060's earlier
teacher froze its anchor representation and used a different objective;
do not resume or rescue that fit. This is the single new capacity probe,
not a claim that the teacher provides a strict performance upper bound.
Interpretation clarification before validation: the teacher can also change
the effective local linear map. A pass does not by itself prove that genuine
nonlinearity is indispensable; later explicit distillation must distinguish
linear and nonlinear contributions. No gate or training change follows.

Reuse the existing `TeacherROM` exactly, independently initialized on S:
fq=Theta*maskedXi+W2[tanh(W1*u+b1)-tanh(b1)], u=Theta_without_constant.
u has75 entries (q3 and existing72 physical features), width16, output3,
1264 additional parameters; W2 starts zero. Existing constructor seed61
fixes teacher initialization; NumPy seed67 fixes the matched training draws.
No stochastic training layer. Initial field/objective must equal S exactly.
The correction vanishes at the origin; no post-fit offset subtraction.

Jointly train exactly the12 J0067 parameters plus W1.weight, W1.bias and
W2.weight. Keep inherited36-term Xi support, alpha support, all normalization
and buffers, physicalM3, ports, r3, GRU architecture and100ms observer fixed.
Inactive Xi/alpha storage stays equal to S. No pruning, new loss, weight,
state, physical input, future re-encoding or teacher supervision.

Use the EXACT0067 objective, full-TRAIN sampler, sampled batches/endpoints,
1/2/5s schedule, batch32, fresh Adam1e-4,800 steps, one CPU thread. Evaluate
only step800. Nonfinite loss/gradient/parameters, frozen/inactive contract
failure or f(0)>1e-10 stops with evidence; no extension or restart/rescue.
Use original S, not the fitted J/J-S or historical teacher, as the source.

Before formal fit: unit forward/gradient/zero-field/frozen tests and a two-
step real-TRAIN smoke with identical initial0067 loss/sampled batches.
W1 initially has zero gradients because W2=0; require nonzero W2 gradients
and a valid downstream W1 gradient after one update. Root reviews and commits.

Screen on the same107 eligible/117 raw validation events and original L1
causal/numerical/refinement policy. All genuine-score and LOS intervals must
pass; no unresolved event can count as a safety gain. Misses, false LOS and
false window-settled must not increase versus either J0067 or S.

Require complete5 mean AND pooled<=.95*J0067, complete5 CVaR<=1.05*J0067,
and allthree<=1.05*S. Require1/2s pooled/CVaR, unstable-prefix mean/pooled/
CVaR and common5 pooled each<=1.05 times BOTH J0067 and S (floor1e-8).
These prospective gates use the matched joint control and preserve severe-
event/common-motion benefits; they do not relax or reinterpret prior screens.
Report8s, per-event errors and full coupled Jacobian as diagnostics. Teacher
Aq/full Jacobian must include the neural derivative, never the sparse anchor.

Only a screen pass permits bounded confirmation (total3 seeds, protocol
registered before additional fits), then distillation into few explicit
terms and revalidation. Failure closes this one capacity probe, not the
whole project or model family. Diagnose its evidence before B2 or the single
r4 candidate allowed by after66; do not sweep teacher width/loss/steps.
Fresh TEST remains unopened. No final L1/L2/L3 or energy claim.
