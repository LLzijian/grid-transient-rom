# Loop0063 — equilibrium initialization retention screen passes

## Question / hypothesis / change

Can q0=E(H)-E(0) restore equilibrium consistency while retaining the registered
physical forecast performance? Starting da01b51; preregistration e4f3ac4;
implementation5bb483f. Every trained tensor, support, topology and ODE equation
is frozen. Only initialization changes, not the ODE coordinate convention.
No retraining, offset search, future feedback, new loss or TEST access.

## Evidence

Command: `experiments/audit_equilibrium_initialization.py --output
private_data/task2_equilibrium_initialization_0063`, project venv. Managed
session80431 exited0. Six arm/equilibrium NPZs and evidence.json are retained.
All107 events/12 unstable prefixes remain; invalid=[];104 tests pass.
Independent saved-array QA reproduces per-event, aggregate, group, common
frequency, numerical and screen results without mismatches:
`.loop-work/long0063_initialization_independent_checks.json`. This verifies
the actual finite run, not hypothetical invalid artifacts. Source/RHS review
and synthetic checks separately establish implementation correctness.
All source hashes remain unchanged; model d34d9f5c...64995c.

E(0)=[-.187266122,-.048734471,-.130429841]. Anchored/zero-control equilibrium
states and forces remain exactly zero through8s at both steps. Original
initialization produces maxima .460104rad delta, .223505rad/s omega,
.074342pu p, and .859827pu force. Initial/final force max=.306116/.043077pu;
equilibrium step discrepancy1.217e-8. This is observer/ODE inconsistency,
not evidence that the ODE zero equilibrium is unstable.

| Arm | Complete5 mean / pooled / CVaR | Unstable-prefix mean / pooled / CVaR |
| --- | --- | --- |
| Original | .288326 / .333571 / .628208 | .724623 /1.230200 /1.667156 |
| Anchored | .290185 / .335766 / .638222 | .733123 /1.210270 /1.582572 |
| Zero q0 | .582356 /1.216514 /2.792203 |1.265838 /2.131882 /2.862432 |

Anchored/full complete5 ratios1.006446/1.006582/1.015940 pass1.02/1.02/1.05;
unstable-prefix ratios1.011730/.983800/.949264 pass1.05. Original/anchored
complete5 numerical maxima4.752e-7/4.841e-7 and unstable-prefix maxima
2.294e-8/2.489e-8 pass. Zero q0 is not memory-off and was not an eligible
alternative candidate; it confirms useful information in the encoded start.

Tradeoffs are retained, not made into retrospective gates:

- Anchored1/2s means .308792/.335592 versus original .298393/.328373;
  anchored8s .241450/.279294/.530211 versus .240233/.277769/.522296.
- Complete5 group mean ratios range .971607(group26) to1.025921(group22);
  group16 pooled/CVaR ratios1.024944/1.037067. No group is removed.
- Common synchronous omega complete5 mean/pooled/worst errors increase
  .397870/.452087/.933162 to .481972/.528092/1.003713rad/s, worst pilot_0237.
  Complete8 pooled increases .396013 to .448407rad/s. COI error conceals this
  physical common-motion tradeoff. Unstable common-frequency pooled improves
  .958796 to .940048rad/s; worst event remains pilot_0366.
- Original/anchored/zero all-fixed8s primary convergence fails with maxima
  .026908/.117065/.093763; one finer diagnostic still fails with
  .001672/.007813/.005485. All genuine complete8/prefix comparisons pass
  (anchored complete8 max5.552e-7). Unobserved unstable8s continuations do not
  support accurate-integration or stability claims. Failed gates stay failed.

## Classification / interpretation / next implication

Accepted at the preregistered retention-screen level only. Keep anchoring as
an equilibrium-consistent candidate for confirmation, not an automatic model
replacement or final M4 acceptance. Original estimator remains available;
no deployed weights changed or require rollback. Calibration is not a Pareto
improvement and does not solve physical interpretation or group robustness.

Next: on full genuine TRAIN, separate force outside the fixed Cq span from
within-span encoder/reference mismatch, and diagnose projected integrated
closure without fitting new coefficients. Physical-force inversion is an
oracle diagnostic, not latent truth or a causal forecast input. No larger r,
encoder, library or new loss. Frozen TEST remains unopened; work continues.
