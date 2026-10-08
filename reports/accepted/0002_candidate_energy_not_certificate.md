# Accepted loop 0002: candidate energy is not a stability certificate

Date: 2026-09-05

Starting commit: `f21d34c`

Decision: **accepted semantic correction; formal certification remains unavailable**

## Registered claim and falsification rule

The software must never emit a formal stable verdict from a local constrained
minimum or an incomplete saddle-root search. A formal positive margin requires
a verified lower bound on the minimum escape energy plus the applicable
Lyapunov/region assumptions. The correction fails if the previous positive
candidate margin can still set `certified=True` or `predicted_stable=True`, if
the heuristic result is lost instead of being clearly separated, or if zero
observed misses is reported without independent-group uncertainty.

## First-principles finding

Let

\[
E_* = \inf_{x \in \partial \mathcal D} H(x)
\]

be the minimum storage on the relevant escape boundary. To conclude stability
from `H(clear) < E_*`, the computation must establish a lower bound
`L <= E_*` and show `H(clear) < L`. A local optimizer returns a feasible value
`U = H(x_feasible) >= E_*`, i.e. an upper bound in the wrong direction. An
incomplete root search also does not establish that a saddle belongs to the
relevant stability boundary or that the lowest relevant saddle was found.
Therefore `H(clear) < min(local candidates)` is a useful empirical screen but
not a theorem.

This boundary distinction is consistent with the original direct-method
literature: stability depends on the stability boundary and relevant unstable
equilibria, while the BCU method supplies a specific theoretically grounded
procedure for locating the controlling UEP. See Chiang, Wu and Varaiya,
[Foundations of Direct Methods](https://doi.org/10.1109/TCS.1987.1086115), and
[A BCU Method](https://doi.org/10.1109/59.336079).

## Accepted implementation

- `critical_energy_level` now returns
  `critical_energy_candidate`, never a certified critical energy. It records
  equilibrium residuals, search attempt counts, constraint/stationarity
  residuals, candidate semantics, and that global optimization and stability
  boundary membership were not verified.
- `stability_margin` always returns `certified=False`, `margin=null`,
  `predicted_stable=False`, and `verdict=refused` until a verified lower bound
  exists. The old numerical behavior is retained under
  `screening_margin`/`screening_predicted_stable` with an explicit heuristic
  status.
- The numeric storage check was renamed and narrowed to an
  `EnergyStructureAudit`. Its output says exactly what it checks: local
  centering, stationarity, COI-projected angle/speed curvature, and local
  supply-identity residual. Uncomputed hidden curvature, discrete convergence,
  global decrease, region of attraction, and escape energy are no longer
  implied.
- An unused legacy energy gate whose API still claimed a certificate was
  deleted.
- E4/E6c/E8 and figure-generation code now separate formal refusal from the
  retrospective candidate-energy screen.
- Safety uncertainty uses an exact one-sided Clopper--Pearson upper bound at
  the declared `source + fault_line` independence unit.
- README and `paper/STATUS.md` block promotion of the old certificate and
  submission claims until the missing proof and lifecycle corrections exist.

## Real-data gates

For frozen F1 event `ncunstable_dense_0006`, the corrected code found a local
candidate level of 0.128886 and a positive screening margin of 0.126537, yet
returned `certified=false`, `critical_energy_lower_bound=null`, `margin=null`,
`predicted_stable=false`, and `verdict=refused`. This is the decisive semantic
regression.

The retained local structure audit on two F1 validation events passed with
center error 0, gradient norm `2.52e-10`, angle/speed projected Hessian minimum
0.126528, and maximum local supply-identity error `1.83e-10`. These numbers
support only the audit's stated local boundary.

Recomputing the historical candidate screen over all test records gives 0/51
missed unstable events, but only 9 independent source/fault-line groups. The
one-sided 95% exact upper bound on the group failure probability is 0.283129.
Thus the evidence supports “zero observed in this sample,” not “zero risk” or a
safety guarantee.

## Gate record

| Gate | Result |
|---|---|
| G1 syntax | pass: `python -m compileall -q src experiments tests` |
| G2 unit | pass: 17 tests |
| lint | pass on every changed Python file |
| G5 semantic refusal | pass on a formerly positive real-data event |
| G5 local structure audit | pass, with explicit claim boundary |
| G5 uncertainty | pass: event and 9-group counts plus exact 95% upper bound |
| G6 JSON | pass: 74 files, no errors; 92 inherited non-finite warnings |
| diff integrity | pass: `git diff --check` |

## Remaining boundary

This loop does not create a formal region-of-attraction certificate. Doing so
would require a mathematically specified domain, verified global lower bounds
or a valid BCU/related construction for this learned model, complete assumptions,
and independent confirmation. The current manuscript remains not
submission-ready and is explicitly marked as such.
