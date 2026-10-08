# Loop0083 — coefficient intervention introduced local unstable poles

Work type: diagnostic. Result: **supported** (numerical local spectral claim).
Model promotion: **none**. This is not acceptance of the0082 candidate.

Preregistration dde1dd1, prospective sensitivity clarification5842f38;
implementation0e63af8. Command: Python experiments/audit_coefficient_poles_0083.py
--output private_data/task2_coefficient_poles_0083. One execution,0.4533s main;
five synthetic tests passed (root2.18s, earlier worker QA also recorded).
No trajectories, dataset metadata, VAL or TEST read. No fit or candidate change.

Both deployed33-state fields include all ports. Remove only common angle;
retain common frequency. Baseline G3000 spectral abscissa **-.1402759476/s**,
all32 poles strictly left of-1e-6. Corrected0082 candidate **+.1925916573/s**,
three poles right of+1e-6 and29 negative; neither has a near-zero-band pole.
Thus the registered introduced-instability claim is supported, not merely
failure to prove stability. This is numerical local evidence, not a formal
floating-point, nonlinear-region or full-order certificate.

Origin/gauge field differences zero; A*g max1.42e-14; Q orthonormality4.44e-16
and Q.T*g3.89e-16. AD/FD relative Frobenius maximum2.45e-11 versus1e-7;
FD spectral-abscissa discrepancy maximum1.63e-13/s versus1e-6. Eigen residuals
below1.61e-16. Eigenvector conditions59.8275/70.7469; empirical sensitivity
radii2.7525e-7/3.2535e-7, below the prospective limits.01403/.01926.
All source/model hashes unchanged; full/FD/reduced matrices and complex
eigenpairs retained in `private_data/task2_coefficient_poles_0083/matrices.npz`.

Independent NumPy checker in `private_data/task2_coefficient_poles_0083_independent/`
reconstructed quotient matrices, spectra, eigen residuals, FD comparisons and
sensitivity radii. Spectral abscissae agree exactly; largest recorded comparison
4.44e-16. Root corrected the worker's draft cached checker before its sole
execution (proper operator-norm eigen denominator, mandatory recorded-abscissa/
radius comparisons, FD Frobenius gate, hash checks); no production evaluation
was repeated. This is independent implementation of arithmetic, with transparent
root QA, not a second independent scientific decision. Cached execution cost
about2.09s including shell startup.

Independent conceptual review and root disposition were registered before
evaluation in the preregistration. Root retains the review's caveat: this
intervention changes the local spectrum, but no causal fraction of large-
amplitude forecast error is established. The structured template/power identity
alone did not prevent local instability. Original optimization adequacy and
template expressivity remain unresolved.

Stage budget spent: two models, each oneAD/twoFD, no retries. Accepted M3/0058/S
unchanged; rejected0082 candidate preserved. Next smallest question: does this
same24-amplitude block have any nonzero direction that leaves the full coupled
equilibrium linearization unchanged? A model-only rank/nullspace audit can
answer feasibility before proposing another fit or adding structure.
