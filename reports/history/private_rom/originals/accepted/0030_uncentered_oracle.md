# Loop0030 accepted diagnostic: uncentered oracle retains class behavior

No causal model adopted. Start988fe60; code9e64cd4; command:
`python experiments/diagnose_private_memory.py --uncentered --output private_data/task2_uncentered_oracle_0030`.
239.20s;48tests/Ruff/compile; same288development initial states,a=1.916890/s,
same numerical search and bounds as0027. Zero memory reproduces M3 exactly and
every prefix loss is no worse than its included zero competitor. Test untouched.

Validation oracle2/5/8s mean=.516921/.425163/.353433, pooled
.600792/.487109/.404045, CVaR1.141448/.903874/.746174. Clear trajectory-potential
gate passes relative toM3. Unlike centered oracle, no false-stable or false-
unstable in70validation events. This is diagnostic capacity, not a deployable
classifier, and does not demonstrate observer attainability or generalization.
Prefix mean/pooled/CVaR=.415493/.428566/1.268283. Per-event records retained in
`0030_uncentered_oracle_evidence.json`; full private raw outputs remain available.

q0 train quantiles[min,p10,median,p90,max]=[-2.17806,-.22430,-.06882,-.01513,.10211];
validation[-1.90738,-.26434,-.07378,-.01670,.12097]. Most useful corrections are
small negative offsets; severe-event oracle extremes remain. No clipping endpoint
is required by these optima. Centered and uncentered8s trajectory potential is
similar (.402750 vs .404045 pooled), but qualitative behavior favors uncentered
for this fixed oracle objective. Do not claim a statistically independent winner.

Next minimal causal experiment: same fixed ridge40-feature observer as0029,
trained only on uncentered TRAIN oracle q0, with identical accuracy/tail/class/
clipping gates. No GRU, driven term or additional state yet. Time-oracle evidence
from0027 still limits the single-exponential physical interpretation.
