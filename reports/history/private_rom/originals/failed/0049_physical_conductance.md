# Loop 0049 rejected before fitting: circuit equilibrium contract fails

Baselinef0ef585; frozen implementation/preregistrationf5aa276. Installed ANDES
command `experiments/test_physical_conductance.py --output
private_data/task2_conductance_0049`, BLAS/OMP1. Ruff/compileall/48tests passed.
The raw-equilibrium circuit assertion failed before TRAIN fitting or validation.

An unchanged rerun stopped at the same assertion; local traceback diagnosis
(`.loop-work/diagnose_kron_equilibrium.py`) found max torque discrepancy
2.22944797e-6, exceeding the preregistered1e-8 gate. Internal EMF relation
E=V+(ra+jxd2)I holds to2.37e-16, but assembled network KCL at the stored
equilibrium differs by2.2469e-5 and the reconstructed terminal voltage differs
by1.6946e-7. The source of this small network-reference discrepancy is not yet
fully established; do not claim an exact circuit equivalence or blame a sign
error that the local EMF identity does not support.

No beta fitted, no candidate rollout, no validation or test payload read. The
output directory contains no completed evidence artifact because the guard
fired early; this report preserves the exact failed code and diagnostic values.
Revert implementation; do not relax the1e-8 criterion or manufacture a current
bias to make this attempt pass. This does not falsify the independently verified
0048 load-power balance, nor show that conductance correction cannot help.

Next smallest distinct hypothesis: identify at most3 positive network-cosine
coefficients directly from TRAIN residuals, with equilibrium-centered phases
and explicit nonzero-net port power supply. This is a sparse data-identified
instantaneous model, not the rejected single scalar times a supposedly exact
physical Schur circuit. Keep M3 mechanics frozen, preserve all hard events,
and apply the unchanged autonomous contribution/tail/classification gates.
