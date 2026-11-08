# Loop 0032 — rejected: sparse driving creates unstable feedback

Preregistered implementation `5e15899d9b5f823ba4fda5091cee7fd850c4f818`, starting
checkpoint `b741323`. Command: `python experiments/run_sparse_driven_memory.py
--output private_data/task2_driven_0032`. Frozen M3 and development data only.

Three TRAIN-selected trigonometric drives lower event-balanced weak integral SSE
from 0.0005361661 to 0.0003105006. Recovery rate is 0.1158203/s, but the
gauge-free dominant eigenvalue is **+0.4780512 + 3.6846521i /s**. This fails the
preregistered local physics gate; validation rollout was correctly not run.
The observer's validation correlation is 0.9576745, RMSE 0.0376299, clipping 0%.
Good state reconstruction does not establish a stable autonomous closure.

Equilibrium residual is zero; q-field finite-difference relative error is
1.67e-13 and free normal-equation gradient residual 3.04e-18. All 48 existing
tests, syntax and Ruff passed. The regression solved its declared objective;
this is a dynamics/feedback failure, not an optimization or observer failure.
Leave-one-training-group fits hit the recovery lower bound in two groups.

Reject and revert the implementation and its preregistration. Preserve the
numerical evidence here and the model/observer in the ignored private output
directory for reproduction only. They are not accepted models. No test payload
was opened. User-owned `kiki.zip` is excluded from all loop changes.

Next falsifiable hypothesis: constrain the same three-term feedback using the
full closed-loop local Jacobian during TRAIN fitting. Positive scalar recovery
alone was insufficient. Do not repair this rejected fit after the fact or relax
its gate; the constrained identification is a separate preregistered loop.
