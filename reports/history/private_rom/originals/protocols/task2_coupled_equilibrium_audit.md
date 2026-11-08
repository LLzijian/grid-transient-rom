# Loop0065 preregistration — frozen coupled equilibrium and storage audit

Starting checkpoint43763ab. Authority docs/TASK2_after_loop_59.md. The same
sparse5s ODE has stayed fixed through0061–0064; conditional force diagnostics
do not justify a larger library/output/state. Its full coupled spectrum is a
missing physical prerequisite before any further refit or final acceptance.
This audit does not waive unresolved group coefficient robustness or promote
the provisional model to final M4. No trajectory selection or TEST access.

## Primary question / fixed hypothesis

Is the unchanged33-state sparse5s ODE locally asymptotically stable at its
known zero incremental equilibrium, modulo the single physical angle gauge?
Stable Aq alone does not answer this. All tensors/encoder/calibration choices,
library, support, normalization, M3 and topology remain unchanged. Use model
d34d9f5c36fb296e1d49e59f5f7b18a5da33b56bf371a16169f82bc25664995c,
`private_data/task2_long_0060_5s`. Both0063 estimators share the same ODE.

## Exact equations / Jacobian

x=[delta(10),omega_sync(10),p(10),q(3)], p=Pm_raw-pm_star; x*=0.
Write qdot=Aq*q+u(delta,omega,p), algebraic coordinate a(delta), and force
Cq*(q+a), SUBTRACTED in the swing equation. Set

K0=B diag(k*cos(B^T delta_star)) B^T, Ka=Cq Da(0),
[Bd,Bw,Bp]=Du(0), R=diag(rate), G=diag(gain).

```
J33 = [ 0              I          0       0       ]
      [ -M^-1(K0+Ka)  -M^-1 D    M^-1   -M^-1 Cq ]
      [ 0             -G         -R      0       ]
      [ Bd             Bw         Bp     Aq      ]
```

Use masked Xi. Feature order is [1,q3,omega10/scale,p10/scale,
centered_sin26/scale,centered_cos26/scale]. For H=angle_incidence,
Gdelta=stack(diag(cos(H^T delta_star)/s_sin) H^T,
diag(-sin(H^T delta_star)/s_cos) H^T), aδ=(alpha*mask)^T Gdelta,
Bd=Xi[24:76]^T Gdelta, Bw=Xi[4:14]^T diag(1/s_omega),
Bp=Xi[14:24]^T diag(1/s_p), Aq=Xi[1:4]^T.

Compute independently actual JointROM autograd J and centered FD at steps
1e-6 and5e-7. Require equilibrium max<=1e-10, analytic/autograd elementwise
relative max<=1e-9 and both FD errors<=1e-6 (denominator max(1,abs(J))).
These are correctness checks, not fitted scientific thresholds.

Verify J*g=0, g=[ones10,zeros23], relative max<=1e-10. Use P32x33/E33x32
with coordinates [delta_i-delta_10 (9),omega10,p10,q3], PE=I32,
J32=P J33 E. Keep common frequency; never use the COI error projection to
remove an omega state. Verify P J33 = J32 P numerically.

## Spectrum / registered classification

Report all32 eigenvalues, dominant real part, frequencies/damping ratios,
eigenvector condition and residuals. Primary hypothesis supported if the
maximum real part is < -1e-6 /s and all correctness checks pass. A real part
>1e-6 rejects local asymptotic stability; the interval[-1e-6,1e-6] is
numerically/marginally unresolved, not forced to zero or counted as support.
Require eigen residual<=1e-9. A verification failure is implementation or
numerical failure, not physical falsification; retain all matrices.

For simple modes use right eigenvectors V and left rows L=inv(V), LV=I.
Report complex block participation sum_{j in block} L[k,j]*V[j,k] for
relative angles/omega/p/q; these sum to1. Also report normalized magnitudes
of the FOUR block sums, not a probability or per-q physical identification.
Flag modes separated by<=1e-7 or eigenbasis condition>1e10 as individually
unreliable. Block sums are invariant under invertible within-q coordinates.
Test this algebra on synthetic matrices. No covariance/whitening gate.

Reproduce the unchanged M3 quotient29 matrix using audit_private_modes and
compare to VERIFIED0035b M3_linearization.npz (relative max<=1e-9). For
context reuse ONLY `private_data/task2_verified_modes_0035b/evidence.json`,
SHA2568a1d3e53a1b4aeba5ceb2f1c7017378dc3ceb51be64a372a3562da8b9a7970ef.
Its original699-dimensional DAE QZ has150 finite modes,549 infinite, one
gauge and zero positive modes. The preliminary160-mode ANDES elimination
with6positive modes was REJECTED in0035 and must not be reused as physics.
Compare each positive-imaginary reduced mode descriptively to the nearest
frequency among stored verified fullorder_electromechanical_modes. Report
all matches, frequencies and damping differences, not a one-to-one match,
new fit threshold or calibrated full-order spectral/stability margin.

## Mechanical storage identity / conditional local quadratic storage

Use MechanicalROM energy_gradient with [delta,omega,p+pm_star], NOT p alone.
The exact identity for the actual joint ODE is

dot H3 = -omega^T D omega - p^T R G^-1 p
         -omega^T Cq*(q+a(delta)).

Verify on256 fixed synthetic states from numpy default_rng(65), normal
samples scaled blockwise [.1rad,.1rad/s,.1pu,.1 latent-coordinate units].
Also include zero. Save states, actual field, gradient, both sides and the
three work/dissipation terms. Relative identity max<=1e-9. These are algebra
probes, not reachable-state samples or a full-order energy validation.
Positive dotH3 is allowed energy exchange, not an automatic rejection.

Only if J32 passes the Hurwitz screen, solve J32^T P+P J32=-I32; symmetrize
roundoff only. Save P/eigenvalues/condition and residual. Positive-definite
numerical screen: lambda_min(P)>1e-10*max(1,lambda_max(P)); relative residual
<=1e-9. This is a conditional linearized local storage in declared coordinates,
not discovered sparse physical Htotal, a quantified nonlinear basin, global
energy certificate or full-order safety margin. If it fails do not repair by
changing J or forcing dissipative coefficients. If J is not Hurwitz skip it.

## Verification and continuation

Synthetic tests first: coupling signs/feature derivatives, gauge maps,
block participation, storage identity and a known Lyapunov fixture. Root
review before execution; independent saved-matrix arithmetic afterward.
Save a small NPZ and JSON, hashes before/after; no new frameworks or training.
No raw H5 or fresh TEST payload is needed. No parameter change requires
rollback. Support advances only conditional local analysis; rejection directs
a smallest coupling/storage diagnosis, not increased dimension or threshold
relaxation. Continue toward physical/group/nonlinear confirmation afterward.
