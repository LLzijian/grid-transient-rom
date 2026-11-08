# Loop0024 rejected: additive passive q has insufficient predictive contribution

Start188866b; experimentf39487c, predeclared80updates executed once. Command:
`python experiments/run_private_m4_predictive.py --output private_data/task2_m4_0024`.
220.55s,Torch2.13.0+cpu;48tests/Ruff/compile pass, including Torch/NumPy field
equivalence and causal history contract. No validation checkpoint selection.

Training loss1.467183→1.239993, meanMSE .585803→.510549. Final a=1.785482,
g=.177440; parameters interior, no numerical failure or new classification error.
However validation mean2/5/8s=.618747/.478590/.393579 versus M3
.611103/.478969/.395093. Required>=5% improvement fails decisively; no extension.
Pooled .742560/.563832/.461945;8s tailCVaR .899590, modest changes not adoption.
All444 per-event records plus81 training observations retained in evidence JSON.

TRAIN-only physical diagnostic, not model selection: fitting one electrical
strength multiplier against actual Pm/acceleration after COI power projection
gives .372085 across all events and explains33.65% of the force SSE. Reporting
strata give .683726 on stable truth and .151155 on unstable truth; labels would
not be available as features. The difference suggests transient voltage/internal
EMF depression alters transfer strength, particularly for severe clearing states.

Revert additive-memory implementation. Next hypothesis: one history-initialized
recovering electrical-strength coordinate modulates the existing sine network,
instead of imposing a constant spatial additive force. M2/M3 remain unchanged;
same q dimension, smaller41-parameter fitting problem, no larger encoder.
