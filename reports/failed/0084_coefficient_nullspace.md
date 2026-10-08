# Loop0084 — no exact-linearization-preserving direction in this block

Work type: diagnostic. Result: **rejected** (existence of a nonlinear-only
direction in the specified24-amplitude block). Model promotion: **none**.

Preregistration b777e98; source54d542c. Command: Python
experiments/audit_coefficient_nullspace_0084.py --output
private_data/task2_coefficient_nullspace_0084. One execution,2.2369s main.
Four synthetic tests passed (root2.26s); preexecution root review corrected
tensor stacking, zero-column bookkeeping and finite JSON reporting, and added
an actual AD/FD quadratic synthetic check. No model evaluation was retried.

The1089x24 matrix of full coupled equilibrium Jacobian sensitivities has
normalized rank24, nullity0, no zero columns. Singular values .402799439 to
1.451710052, minimum/max ratio.277465489, condition3.604052; far from the
preregistered1e-8 resolved/1e-12 null cutoffs. The affine Jacobian difference
reconstruction has relative error1.7362e-16; FD relative errors1.57294e-12 and
1.51681e-12. Kinematic rows and common-angle action are exactly zero. All
source/input hashes unchanged, no dataset/VAL/TEST read or model changed.

Full tensor/FD/SVD/mappings: `private_data/task2_coefficient_nullspace_0084/`.
Independent cached NumPy reconstruction in the corresponding `_independent/`
directory agrees exactly on recorded matrix/scale/singular/rank quantities;
qa_pass true, no discrepancies. Cached-check command took about1.92s including
shell startup. Both null-direction/nonlinear-detection flags are false because
there is no null direction, not because QA failed.

This blocks a nontrivial update under EXACT preservation of the original
linearization within the specified support/frozen objects. It does not imply
that useful coefficient updates must be unstable, or that the whole Jacobian
needs to be fixed. It does not reject structured models or original joint
optimization. The one-run stage is closed, budget spent, no promotion/rollback.

Root next chooses one analytic local-Lyapunov-guarded coefficient intervention,
not a new support or a line search. The independent challenge explains that
scalar contraction confounds stability with update size; any gain will be
claimed only for this explicitly guarded update. Exact proposal/guard/forecast
gates and finite budget are separately registered in0085. Old0082 failure and
0083 numerical local finding are preserved without reinterpretation.
