# Loop0037 accepted: direction-dependent numerical LOS boundary, not one Hcrit

Checkpoint37921ba; implementation9b7433c. Command:
`python experiments/audit_private_reachable_boundary.py --output
private_data/task2_reachable_boundary_0037`. All48tests, syntax and Ruff pass.

Among15 TRAIN-derived physical-amplitude directions,14 have a finite60s
stable-to-LOS transition within rho[0,2]. The fixed255-point coarse grid contains
137LOS outcomes; no nonmonotone ray is observed on this grid. Twelve bisections
give first-transition interval widths3.05176e-5. The first3 bracket pairs agree
with independent DOP853 on all6 LOS decisions. The remaining direction has no
bracket by rho2; it is not dropped or extrapolated to an invented boundary.

Transition energies span about12.985–48.615, versus Hsafe=.091513. They are
roughly142–531 times the conservative sufficient bound and depend on direction.
This confirms that Hsafe is extremely conservative and cannot be renamed the
actual critical energy. One scalar universal Hcrit is not established by these
ray scans. All423 grid/midpoint records and exact starting states are retained.
No sufficient-region violation or disallowed energy increase occurred.

These are finite60s LOS-proxy transition intervals, not an exhaustive or
asymptotic basin boundary. Synthetic radial points are not claimed reachable
by the full-order system, even though each direction starts from a real TRAIN
clearing state. Outside the sufficient region, negative margin means unknown,
not predicted instability. No test payload or validation-selected ray is used.

The dynamics choice, explicit storage, conservative energy definition, local
spectral definition and finite-time boundary interpretation are now ready for
an explicitly frozen confirmatory evaluation. Such a test must retain the
limitations above and all event/group uncertainty; it cannot convert a reduced
model bound into a full-order guarantee.
