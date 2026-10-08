# Loop0065 — conditional coupled equilibrium stability supported

## Question / hypothesis / change

Is the frozen sparse5s 33-state ODE locally asymptotically stable at zero
incremental state, modulo one common-angle gauge? Starting43763ab;
preregistration c0463b3; reviewed implementation481aa6a. No model, encoder,
support, topology, loss or estimator changed. This is conditional analysis
of the provisional model, not final M4 promotion or a group-robustness waiver.

## Evidence

Command: `experiments/audit_coupled_equilibrium.py --output
private_data/task2_coupled_equilibrium_0065`, project venv, exit0.
118 tests pass. No trajectory/H5 or TEST payload was needed. Source hashes
are unchanged. `jacobian_precheck.npz`, `coupled_equilibrium.npz` and
`evidence.json` preserve all matrices, parameters, modes and storage probes.
Independent saved-matrix arithmetic reproduces checked quantities with
zero mismatches (`.loop-work/long0065_coupled_independent_checks.json`). That
checker does not independently regenerate live JointROM autograd/FD; those
are separately supported by source review and synthetic/live checks.

| Registered correctness quantity | Result |
| --- | ---: |
| Equilibrium maximum |0 |
| Analytic/autograd relative maximum |8.795e-16 |
| Centered FD, h=1e-6 /5e-7 |4.307e-10 /3.588e-10 |
| Gauge residual |4.650e-17 |
| PE identity /intertwining maximum |0 /2.842e-14 |
| Eigenpair residual maximum |2.421e-16 |
| Participation sum error maximum |2.115e-15 |
| VERIFIED0035b M3 matrix difference |0 |
| Mechanical storage identity relative maximum |1.221e-15 |

All pass unchanged preregistered thresholds. The quotient retains ALL10
synchronous frequencies and removes only the common angle. All32 poles have
negative real parts. Dominant pair is
**-0.15656887475093795 +/-9.944463547665174i /s**, frequency1.582710530Hz,
damping ratio.015742375. Eigenbasis condition25.028895; no mode is flagged
individually unreliable by the fixed separation/condition criteria.

The dominant pole's four complex participation sums (relative angle,
frequency, mechanical power, latent) are approximately
(.465770-.018339i, .503438-.006024i, .020919+.019379i, .009874+.004984i).
The slow pair -.235355058 +/- .331548371i has latent block sum
.859879+.169999i (positive-imaginary pole), normalized four-block magnitude
.659132. These are modal diagnostics, not probabilities or per-q physical
labels. Coupled modes differ materially from Aq alone.

M3's verified dominant real part is -.038830572/s. Nearest-frequency matching
to verified full-order modes remains descriptive and many-to-one. All11
positive-imaginary reduced poles and their matches are saved; damping-ratio
differences range -.181639 to -.032615. The dominant pair's nearest-frequency
difference is .010094Hz, but damping is lower by .160539. This is not a
calibrated full-order stability margin. The rejected preliminary160-mode
elimination was not reused.

The exact identity dotH3 = -omega^T D omega -p^T R G^-1 p
-omega^T Cq(q+a) holds on zero plus256 seeded synthetic states.26/257 have
positive dotH3: mechanical storage can increase through electrical-memory
exchange. These probes are not reachable physical trajectories or a
stability-failure frequency.

J32^T P+P J32=-I yields symmetric P with eigenvalues .1046084475 to180.2702591,
condition1723.285866, relative residual9.527e-16. Both numerical gates pass.
This is local linearized quadratic storage in declared mixed-unit coordinates,
not identified sparse H3+Hq+Hc, a quantified nonlinear basin or global energy.

## Classification / interpretation / next implication

**Accepted conditional diagnostic:** local coupled stability is supported.
The smooth frozen nonlinear ODE consequently has a locally attracting
equilibrium modulo angle gauge, but its attraction region is not quantified.
No learned implementation changed, so no model rollback is needed.

Next distinguish0062's encoder-reference inconsistency from actual
deployment-loss coefficient sensitivity by a small matched continuation/
leave-group-out comparison, keeping the existing objective and representation.
Group31 is selected from0062's documented worst diagnostic, not fresh
validation. Group robustness, nonlinear storage/boundary and final
confirmation remain unfinished; fresh TEST remains closed.
