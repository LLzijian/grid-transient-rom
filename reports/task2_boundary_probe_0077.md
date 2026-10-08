# Loop0077 — fixed three-condition numerical probe (preregistered)

Authority after75 section11; start0faf3b3.0075 stays CLOSED rejected. No data
promotion or model selection. Question: for fixed passing/trajectory-failing/
LOS-time-failing TRAIN conditions, does further time refinement reduce the
discrepancy, and is Newton tolerance or event alignment a plausible contributor?

Deterministic representatives from0075: smallest passing ID boundary_0000
(bus2); smallest trajectory-failing ID with matching LOS/label boundary_0002
(bus5 stable); LOS-time failure boundary_0015 (bus28). Keep exact config,
fault conditions,15s horizon and label rules. No VAL/TEST payloads.

Exactly four new runs per condition (12 total): (.0025s,1e-7) repeat;
(.00125s,1e-7); (.000625s,1e-7); (.0025s,1e-9). Newton budget stays30,
fixed step/no shrinking, original event definitions/stop/export. Fresh paths;
no fifth arm, adaptive replacement, new disturbance or pooling into TRAIN.
The repeat is necessary instrumentation/reproducibility control, not a new
member of the numerical ensemble. Read only these3 old0075 event payloads.

Reuse original simulate with optional diagnostic capture, preserving default
returned arrays and dynamics. Record initial/pre-fault DAE f/g residuals,
finite status, attempted/converged steps, actual steps bracketing scheduled
fault/clearing, record start delay, step-size range, labels and physical-stop
times. Inspect whether scheduled event instants are present within1e-8s;
report any different event callback convention explicitly, not as instability.
Initialization residuals are diagnostic, not a retroactive generator gate.

Use0075 exact common .01s sample grid and scopes (15s stable or earlier true
stop), COI/common normalized RMSE and LOS difference. No extrapolation.
Repeat reproducibility gate: same label, grid/record origin within1e-8,
normalized COI/common differences<=1e-8, identical saved physical stop time.
If not reproducible, classify implementation/environment uncertainty and do
not attribute step trends until explained; no silent overwrite of0075.

Primary hypothesis (all3): last adjacent pair .00125/.000625 agrees on label,
COI and common errors<=.001, LOS difference<=.02s, and errors are no greater
than previous adjacent .0025/.00125 (allow1e-12 arithmetic slack). Also
report full two-level ratios and per-variable initial/trajectory errors.
Tolerance-control difference compares .0025 tol1e-7/1e-9: it is descriptive,
not a second chance to pass the step hypothesis. Neither finest step nor
tight tolerance is called ground truth. Numerical failures are UNKNOWN.

Pass supports this local numerical protocol for these3 conditions only,
not the20-event pool nor a universal convergence order. Failure is diagnostic
evidence for a subsequent root-defined hypothesis, not0075 rescue. Preserve
all12 outcomes, logs, configurations, event traces and paired metrics; one
short report. Run alongside0076; do not block structured-model development.
