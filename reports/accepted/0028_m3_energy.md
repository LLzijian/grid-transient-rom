# Loop0028 accepted: exact M3 storage, not a full-order energy certificate

Starte8285e9; code8aad238; command:
`python experiments/audit_private_energy.py --output private_data/task2_energy_0028`.
No parameter refit or trajectory-model change.48tests/Ruff/compile pass,6.23s.

For p=Pm-Pm_star and the frozen equilibrium-anchored network potential U,
H3=.5*omegaᵀM*omega+U+.5*sum(p²/b) has exact derivative
`Hdot=-sum(D*omega²)-sum(a*p²/b)<=0`. This is normalized model storage; no claim
of a directly calibrated Joule-valued full-order electromagnetic energy.
All a,b,M positive,D nonnegative. Gradient finite-difference error9.92e-10;
gradient·field versus analytic dissipation relative error8.88e-16. H(eq)=0.
Fixing generator10 angle removes the gauge; minimum potential curvature1.728197
(other storage blocks positive) gives a strict local minimum in the quotient.

All288 reduced8s trajectories have no sampled H increase above tolerance. This
does NOT imply all are stable: the tilted trigonometric potential is only locally
positive; escaping-angle trajectories can drive H deeply negative. No global
positive-definiteness, critical energy, ROA or full-order certificate asserted.

Three highest-H0 stable cases0255/0223/0047 agree with independentDOP853 within
1.53e-5 state units. Halving integration/output steps reduces the normalized
energy quadrature imbalance from3.72–3.87e-5 to9.30–9.67e-6, approximately4x as
expected for trapezoidal quadrature. Numerical error is below the physical
full-order storage variation discussed next.

On original full-order trajectories, all196train+61validation stable events have
net H decrease, but all contain upward intervals. Median upward variation/H0 is
.1625train/.1992validation; largest validation single-step increase is.02383H0.
All9validation unstable events show net increase (median+.6072H0), whereas the
reduced escaping trajectories can dissipate the tilted H while angles diverge.
These differences expose eliminated electrical/control storage and model error,
not violation of the proved reduced-ODE identity. Do not forcibly smooth them away.

Descriptive initial-H separation AUC=.9583train/.9545validation; H0 versus fault
duration Spearman=.8383/.9269, withinvalidation groups .9433–.9763. No threshold
was fitted and this is not independent prediction evidence. Only5validation
groups and9prospective unstable cases limit generalization claims. All288 event
records retained in `0028_m3_energy_evidence.json`; private energies remain at the
output path. Frozen test untouched.

Next: the oracle q0 gap permits one small compact observer test without upgrading
the network. Time-resolved inversion and real hidden storage also motivate the
uncentered/driven family if initialization alone cannot pass. M3 remains the
accepted robust model and its energy baseline is now available to either route.
