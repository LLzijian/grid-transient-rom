# Loop0035 accepted: local M3 spectrum; full-order reduction artifact identified

Checkpoint bc3a43c; final diagnostic code26a4139. Commands:
`python experiments/export_private_fullorder_modes.py --output
private_data/task2_fullorder_descriptor_0035` (installed ANDES Python), then
`python experiments/audit_private_modes.py --fullorder
private_data/task2_fullorder_descriptor_0035 --output
private_data/task2_verified_modes_0035b` (project scientific Python).

Frozen M3 is locally asymptotically stable modulo angle gauge:29 modes have
negative real parts, equilibrium residual0. Its spectral margin is
**0.03883057197/s**, dominant mode -0.03883057 ±13.51092715i/s, frequency
2.15033084Hz, damping ratio0.00287400 (0.2874%). Dominant participation is
primarily generator1 relative angle and speed, about44.3%each. Analytic Jacobian
matches centered finite differences to7.49e-10; eigen residual1.21e-16.
All47 positive-parameter derivatives are retained with finite-difference checks.
48tests, syntax and Ruff pass. No M3 parameters or dynamics changed.

## Essential full-order audit correction

The same physical signature and stored full/algebraic equilibrium match exactly.
The installed ANDES EIG reduction reports160 modes including6positive and has
a defective eigenbasis. Those results are **not accepted as physical modes**.
The all-zero A1..A6 PSS filter realization introduces algebraic state constraints;
the routine's reduced result does not preserve the original descriptor spectrum.
No simulator patch, controller disablement or dataset regeneration was made.

Direct QZ of the original699-dimensional DAE pencil gives150 finite modes,
549infinite modes, no0/0 pairs, one rotational zero, and **no positive modes**.
Maximum eigen residual1.04e-15; independent row-equilibrated pencil agrees to
4.48e-12. The20 repeated modes at-50/s have non-identifiable individual state
participations; values remain reported and participations are explicitly null.
The full-order slowest nonzero mode is real -0.153599153/s (PSS washout dominated).

Nine conventional electromechanical modes span approximately0.641–1.687Hz with
damping ratios0.153–0.224. M3 captures some frequency scale but substantially
underestimates damping and has a2.15Hz dominant mode above that range. Nearest
frequency comparisons are descriptive, not a verified one-to-one modal match.
Thus M3's spectral margin is only a reduced-ODE property, not a calibrated
full-order stability margin or nonlinear safety certificate.

Raw matrices and both preliminary/rejected and verified outputs remain in
private_data. Evidence JSON contains all mode/sensitivity records and identities.
M3model SHA256 remains4928F40E54656FA05A5D478969F4D53F63E7551A824F761312EFC793BC0621B4.
No test trajectory was read; kiki.zip is user-owned and excluded. Next: a
conservative nonlinear cohesive-energy bound and numerical basin checks on
the frozen M3, keeping their reduced-model scope explicit.
