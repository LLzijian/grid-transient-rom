# Loop 0048: physical source of nonzero-net electrical residual

Preregistered baselinebbe5eaf. Diagnostic question, not a new memory model:
does the omitted net generator force correspond to actual load/loss exchange
with the rest of the network? Installed ANDES GENROU uses Flux0,
te=Pe+ra(Id^2+Iq^2); default PQ TDS config is p2z=q2z=1,p2p=p2i=0.
Verify those settings and exact case/equilibrium identity in an unchanged model.

Export actual line admittance, PQ equivalent conductances, fixed shunt
conductances, stator resistances and generator-terminal bus mapping. Never
modify installed simulator, physical parameters or data. On all369 TRAIN
prefixes0..8s verify sum(Pe)=PQload+lineLoss+shuntLoss and the torque identity.
Then decompose sum(r_e) into negative deviations of voltage-dependent load,
line/shunt/copper loss, plus frozen-ROM damping supply. The generator-only
sine coupling sums to zero. Report event-balanced component covariances and
load-only residual explanation, without assuming orthogonal/additive energies.

Acceptance is correctness of this physical audit, not a forced dominant-load
finding. Require equilibrium state/algebraic mismatch<1e-8; torque and total
power balance event-normalized RMS median<1e-4,p95<.005; net-residual identity
median<1e-4,p95<.005, denominator max(1,RMS total generator torque/power).
No event deletion/padding; all369 TRAIN retained, finite outputs. No validation
or either TEST payload. Save per-event component traces and exact source/data
identity. Ruff/compileall/all tests required.

Also record the ACTUAL39-bus branch incidence restricted to the10 generator
terminal rows. This is a generator-port reduced incidence, not a claimed Kron
equivalent or license to invent grounded edges. Its omitted rows represent the
rest of the physical network. Merely document rank and port mapping; do not fit
or adopt a new spatial memory direction/ODE in this audit. If nonzero-net force
has a physical load/loss origin, later evaluate a port-aware low-rank channel
with explicit network supply/storage interpretation before adding more states.
