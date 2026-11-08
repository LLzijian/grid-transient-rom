# Loop0080 — one-shot hidden-force initialization (preregistered)

Start c15ea52. After75 bounded-G diagnosis follows closed0079:58.03% of
G-fixed3000 sliding force error is within span, and68.26% at100ms. This
does not yet say whether a better initial state improves autonomous motion.

## One question and one allowed change

On frozen G-fixed3000, does replacing only q(100ms) by0079's pointwise
force projection improve full-TRAIN autonomous risk by>=10% without
material severe-prefix harm? Compare baseline causal qenc against qF at
the SAME endpoint10. Everything thereafter follows the identical frozen
33-state ODE. No parameter update, repeated truth injection, optimization,
new loss/state/encoder or model selection. No C/G-auto fitting or TEST.

qF uses hidden instantaneous te and is unavailable to the deployed sensor
interface. This is a diagnostic intervention, not a deployable candidate
or best-initializer upper bound. Failure does not prove all initializers
ineffective; success would motivate a later causal-observer question only.

## Objects, data and integrity

Model `private_data/task2_structure_0078_extend_G-fixed/step3000` only.
Read original346 eligible TRAIN events (311 stable/35 unstable,24 groups,
369 raw), full genuine trajectories, via existing full_train reader; unused
validation loading disclosed, no validation metrics. No TEST payload access.
Use x(100ms) exactly as stored and histories0..10. From0079 arrays extract
exactly one endpoint10 qF and qenc per event; verify event ordering, source
model/norm/base/network and dataset identities against0079 and0078 before
using them. Re-encode baseline histories and require maxabs difference
<=1e-10 versus saved qenc. Only q0 differs between the two interventions.

Roll out each event up to min(5s,remaining genuine truth). Never continue
an event beyond this endpoint for scoring, fabricate a tail, reset state
from truth, or finite-filter a difficult event. Targets start at offset1,
not initialization. Use the same RK4 field and frozen p scales as0078.

## Fixed metrics and gate

For each event/horizon1/2/5s, take mean over all available genuine future
samples of0078 point error:
.5*(mean squared COI angle/.47675314588683515 + mean squared COI speed/
1.0033667249602924) + squared common speed/1.0033667249602924 + mean squared
p error divided by its original feature scale. Parentheses mean each
component is scaled BEFORE squaring. Every horizon retains all346 events,
with available length separately recorded, not a complete-only cohort.

Per horizon risk=.8*equal-event mean+.2*mean worst35 EVENT errors; average
the three horizon risks. Primary pass requires qF/baseline average risk
<=.90. Additionally, on35 unstable events' full available prefix (capped
at5s, same for both), require mean event point error and worst4-event mean
each<=1.05 baseline. All three conditions required, no compensatory gate.
Report per-event risks, all horizons and stable/unstable subsets, not just
the decision. No L1/L2/L3 or new model acceptance is claimed here.

## Numerical and implementation checks

Fixed reference=.005s, comparison=.01s, outputs aligned .01. Save both;
no finer-step rescue authorized in this diagnostic. On every event's genuine
interval require all original substeps finite and normalized COI combined
RMSE<=.001, common speed RMSE/scale<=.001 and p RMSE/p_scale<=.001 between
the two forecasts, separately for each initialization. Numeric failure is
unresolved numerical evidence, not a scientific initializer rejection.
The coarse baseline's1038 per-event errors and horizon risk summaries must
reproduce the saved3000 TRAIN monitor within maxabs/max(1,maxabs)<=1e-9.
This checks genuine-prefix/initialization/weighting conventions. Fine errors
need not equal the old coarse monitor exactly. All model tensors/hashes
remain unchanged, and both initialized physical states are identical.

Before actual data execution root reviews equations and synthetic tests
covering initial-state-only change, RK4 autonomous continuation, variable
length truth masks, event-balanced top-tail risk, nonfinite fail-closed and
numeric comparison. Hash model/meta/norm/export/base/network,0079 inputs,
CSV/splits and execution dependencies before/after; no whole-H5 hashes.
Preserve failure details and do not silently retry. Save initial q values,
genuine lengths, both step forecasts, aligned truth, per-event errors and
minimal evidence. Independent cached NumPy metrics must reproduce the
decision quantities, numerical errors and cohorts before root closure.

## Interpretation

Support shows sensitivity of this frozen model to a particular noncausal
force-based initial state, not proof that encoder training alone solves it.
Rejection indicates that fixing instantaneous initial force is insufficient
for the registered autonomous-risk improvement; inspect consistency of the
projected q trajectory with frozen dynamics before changing model capacity.
Do not use this diagnostic to rescue0078, promote a model, or open TEST.
