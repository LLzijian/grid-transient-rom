# Loop0074 — the single r4 sparse electrical-memory candidate

Authority: after66 sections7–8. Starting checkpoint d3bf977 after0073 closure.
This uses the ONE permitted r4 candidate, not another teacher or energy
weight/support-budget sweep. S/0058/M3 remain unchanged; matched rank3
support-search control is J-S0070. No claim that r3 is generally insufficient.

Question/hypothesis: can one additional electrical-memory force channel,
initialized along common acceleration, resolve the persistent common-motion
versus COI/severe-prefix tradeoff?0064 found a nonzero out-of-three-channel
force residual, while0070/0072 did not resolve the tradeoff within three
channels. This motivates a bounded prospective probe, not an impossibility
proof or an assumption that the new state has a unique physical identity.

One complete algorithm change: rank4 extension of the0070 joint sparse
algorithm. Same M3 physical ODE,100ms history, true p0, GRU20->8 architecture,
physical feature family/normalization, ports and causal restrictions. No
neural RHS, energy loss, new physical feature, window change or r>4.

Initialize independently from S, never from a rejected fitted candidate.
Copy the old three head rows, decoder columns, spatial/instantaneous
parameters and every physical buffer exactly. Add a ZERO fourth head row
and bias, ZERO fourth decoder column, q_center4=0 and block-diagonal
q_transform=diag(S_transform,1); memory_enabled4=1. Keep old alpha support;
the entire fourth alpha column/support stays zero: this probes an extra
MEMORY channel, not an additional instantaneous force channel.

Set fourth spatial raw column to the minimum-norm least-squares solution
ports*c4=M/||M||2 and its amplitude to median(abs(S_amplitude)). Require
amplitude>1e-8, relative port residual<=1e-10 and relative distance of M
from span(S_Cq)>1e-6; otherwise classify unusable initialization and stop,
do not choose another direction after seeing results. M-proportional force
initially produces common acceleration; c4 is subsequently trainable, so do
not label the learned state a pure common mode without checking outputs.
This is the existing full-row-rank port parameterization, not a KCL claim.

Map old masked Xi coefficients to the correct new indices in the77x4 array
[1,q1:q4,omega10,p10,sin26,cos26]. Newly opened old entries and all new entries
start zero EXCEPT Xi[q4,q4]=-1/s (a neutral one-second relaxing memory).
All constant entries are exactly zero/inactive. The extra state starts and
stays zero at initialization, so initial physical autonomous rollouts,
old q trajectories and electrical outputs exactly reproduce S. The34-state
initial spectrum is S's full spectrum plus-1/s; verify the mapping numerically.

Open all304 nonconstant Xi entries for200 steps. Use the SAME0070 independent
512-TRAIN-window pruning draws (seed7000), scores |Xi_ij| RMS(Theta_i),
positive top48 entries and row-major ties. Prune ONCE and zero rejected Xi,
reset Adam ONCE exactly as0070, then600 refit steps with fixed support.
All twelve joint tensor names remain trainable; fourth alpha is masked.
Preserve source-expanded buffers/inactive alpha exactly. No pruning rescue
if the added state's dynamics are removed; record it as part of the result.

Use the exact0067 physical/local/electrical/risk objective and sampling,
seed67,batch32,800 steps,2s adaptation then1/2/5s,fresh Adam1e-4,one CPU.
For a matched rank extension, keep the ORIGINAL per-component denominators:
latent consistency and encoder-zero penalty use sum over q divided by3,
Xi L1 uses sum divided by76*3. Thus each pre-existing component retains its
weight, rather than silently weakening it when rank increases. All existing
scalar weights and tail-risk weights are unchanged. This is equivalent to
4/3 times the rank4 helper's latent/equilibrium means and the corresponding
fixed-denominator Xi sum. Initial total loss must equal J0067 first loss
plus1e-4/(76*3), solely from the new-1/s coefficient; all physical and
auxiliary prediction terms are identical initially. No VAL calibration.

Preflight: initial S equivalence, feature/parameter/index mapping, old
normalization/expanded buffers, f(0)<=1e-10,34-state full Jacobian/finite
differences, new-head/new-Xi physical gradient paths, pruning/support/reset
contracts, and rank-aware evaluator Aq(4x4) checks with unchanged r3 output.
Two-step TRAIN smoke may prune after step1 to exercise the refit path (as
0070); its draws must match the first two J-S draws. Formal fit starts fresh
from S after root review and checkpoint, never from smoke. Nonfinite loss,
gradient/parameter, source/buffer/f(0) failure stops with evidence; no rescue.

Evaluate only the final800-step candidate with the same107 eligible/117 raw
VAL events and existing L1 causal/numerical/one-finer policy. Keep genuine
complete horizons separate from all unstable prefixes; no artificial tails.
All genuine-score and LOS intervals must pass and no unresolved safety gain.
Misses, falseLOS and false-window-settled must not increase versus BOTH S
and J-S0070. Require complete5 mean AND pooled<=.95*J-S0070, complete5
CVaR<=1.05*J-S0070, and allthree<=1.05*S. Require1/2 pooled/CVaR and
unstable-prefix mean/pooled/CVaR<=1.05*S; common5 pooled<=1.05*J0067
(floor1e-8 throughout). Report8s, support, actual force singular values,
fourth-channel ablation on the same cached causal initial states and full
coupled33-state quotient poles as diagnostics, not covariance/order gates.
The ablation specifically sets memory_enabled[3]=0 in the physical force
output, leaving q4 dynamics/other latent couplings and the cached initial
state unchanged. It tests that direct channel, not full structural removal
of a state; require genuine-interval numerical checks before interpreting
its errors. It is not a model/threshold selection branch.

Only a screen pass permits bounded total3-seed confirmation followed by
explicit-equation, regional energy/L1/L3 and true end-to-end held-out-group
work. A finite-budget failure rejects this candidate, not all r4 systems.
No second r4 initialization/dimension/weight/support-budget search; diagnose
remaining physical/data limitations under after66 instead of adding states.
Development reader may load unused VAL records, but only TRAIN enters fit
and pruning. Fresh TEST remains unopened. No project completion/promotion
from this single screen.
