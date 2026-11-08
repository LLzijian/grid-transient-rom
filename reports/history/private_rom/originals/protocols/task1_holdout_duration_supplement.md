# Loop 0047: prospective frozen-test duration coverage, raw data only

Baseline06d3bb4. Existing expansion TEST buses37/38/39 have45 accepted events,
but both unstable events terminate before50ms. Thus the43 predictive-start
events cannot estimate missed-instability performance. Do not interpret an
empty unstable denominator as zero missed failures.

One data-coverage hypothesis: refining the preexisting .4-to-.6s fault-duration
interval supplies post50/100/200ms unstable trajectories in these same frozen
groups. Generate the entire fixed grid, not adaptively selected model failures:
buses[37,38,39], durations[.425,.45,.475,.5,.525,.55,.575]s, xf[.001,.01],42attempts.
Same incident-line selection, original restored system/controllers/operating
point, ANDES2.0.0, original solver and outcome rules. Separate immutable shard
with unique holdout_ prefix; every accepted event remains TEST, never TRAIN/VAL.
Existing Pilot TEST remains consumed and is not read. New TEST only receives
raw mechanical QA; no predictive model, observer or threshold is evaluated.

Freeze config/generator before launch,4workers. Whole numerical failures or
unconfirmed outcomes discarded under unchanged rules; no truncation/padding,
no deletion based on label or model difficulty, no extending this grid to pass.
Retain all42attempts' outcomes/discard reasons. Acceptance: existing mechanical
QA passes, exact physical signature/case/equilibrium/state-map equality against
expansion, no duplicate bus/duration/xf event with expansion, all3test groups
represented, >=1 new physical unstable event available after200ms. Report counts
at50/100/200ms and by independent bus group; do not promise statistical precision
from only3groups. If no eligible unstable is generated, reject the coverage
hypothesis, retain valid raw evidence and decide the next data question separately.

Only generator change is a holdout stage/prefix; no new data schema or framework.
Ruff/compileall/unit tests, then raw QA after process completion. While running,
read-only preparation for the next state-discovery hypothesis is permitted.
