# Loop 0048 accepted: omitted net force is voltage-dependent load exchange

Baselinebbe5eaf; frozen implementation/preregistratione4ef807. Installed ANDES
Python command `experiments/audit_electrical_port_balance.py --output
private_data/task2_electrical_ports_0048`, BLAS/OMP1. Ruff/compileall/48tests pass.
All369 TRAIN prefixes, no validation/test. Exact equilibrium x/y errors are0.

The actual model uses constant-impedance TDS loads: p2z=q2z=1, other ZIP
fractions0. In system pu, equilibrium PQ load58.564, line loss.37101006,
shunt active loss0, stator copper loss.49081330. GENROU Flux0 satisfies
te=Pe+ra(Id^2+Iq^2). Generator terminal power balances voltage-dependent PQ
load plus branch/shunt loss. These are the installed case's equations, not
an assumed constant-power load model or a simulator change.

Normalized torque/power/net-residual identity p95 errors respectively
5.54e-7,4.28e-6,4.32e-6, well below the preregistered.005 gate; medians also
pass1e-4. Maximum net discrepancy5.35e-5. Component traces and nonorthogonal
second moments are saved; do not sum component energies as orthogonal shares.

Negative load-power deviation alone explains99.2969% of the event-balanced
uncentered squared net residual. Thus the large nonzero-net force in0040 is
physical voltage-dependent load exchange, not differentiation noise and not
a disposable common-angle gauge. This explains why deleting arithmetic mean
force lost useful correction capacity in0045.

Actual46-branch39-bus incidence restricted to generator-terminal buses30–39
has rank10; it is a generator PORT incidence whose omitted rows represent the
rest of the network. It is not a10-generator pairwise lossless incidence, not
a fitted mode, and not itself a Kron reduction. A future model must make its
load/network power exchange explicit rather than silently claiming zero net
generator force. No new memory state or model is adopted here.

Next smallest physical test returns to instantaneous/memory separation:
derive the unchanged network's subtransient circuit and test an equilibrium-
preserving conductance/cosine current-angle correction before assigning the
voltage-dependent net residual to hidden memory. M3 mechanics stays frozen.
Immutable `physical_network.npz`, `components.npz`, `evidence.json` retain
actual network parameters, traces, group records and code/data identities.
