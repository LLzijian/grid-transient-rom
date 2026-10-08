# Loop0064 — within-span force mismatch is dominant

## Question / hypothesis / change

Does more than half of the frozen sliding encoder's physical force error lie
within the existing Cq span? Starting c14c1b2; preregistration e4f7315;
implementation7264949. No learned tensor, equation, estimator, support or
topology is changed. Projection of true electromagnetic force is an oracle
diagnostic, not a deployable latent observation or ground-truth memory state.

## Evidence

Command: `experiments/audit_physical_force_projection.py --output
private_data/task2_physical_force_projection_0064`; project venv; managed
session89818 exited0. All346 full TRAIN events (311 stable/35 unstable),
465,317 genuine endpoints and46,514 windows are retained. Existing reader
loads unused validation; no validation quantity enters the audit. TEST is
not read.113 tests pass. Re-encoded0062 X/y window reproduction is exactly0.
Source model/normalization/base/network hashes are unchanged. Physical force
uses te, not Pe alone, and M=physical inertia/omega_s for common acceleration.

`force_arrays.npz`, `closure_windows.npz` and `evidence.json` retain the
decisive arrays. Independent saved-array QA reproduces projection, endpoint
and window coverage, force/initial/bin/group/label/common-motion summaries
and integrated closure: zero mismatches, share .7394738348564146.
See `.loop-work/long0064_force_independent_checks.json`. Raw force acquisition
and encoder causality are supported separately by source review and the
0062 reproduction, not by that saved-array checker alone.

Equal-event mean squares, in system-pu force squared:

| Quantity | Mean square | RMS |
| --- | ---: | ---: |
| Physical target F |1.482620562 |1.217629074 |
| Unrepresentable Fperp | .165650561 | .407001918 |
| Within-span mismatch | .470180243 | .685696903 |
| Total encoder force error | .635830804 | .797389995 |

Within-span share=.739474>.5: the primary hypothesis is supported.
Stable/unstable shares=.813793/.711633; all24 group shares exceed.5,
ranging .5673(group29) to .9408(group1). This is finite-TRAIN descriptive
attribution, not independent generalization, a latent-order proof or
permission to fit using validation. Projection/decomposition, orthogonality
and Pythagorean relative errors are7.30e-18/1.28e-17/1.20e-19.

At the initial100ms endpoint the share is .588327; total/perpendicular/
parallel RMS=.897171/.575641/.688152. Across time bins:

| Time from clearing | Events / endpoints | Share | Total force RMS |
| --- | --- | ---: | ---: |
| [0.1,1) |346 /29,398 | .810903 |1.119541 |
| [1,2) |316 /31,308 | .808348 | .897630 |
| [2,5) |311 /93,300 | .756591 | .591766 |
| [5,8) |311 /93,300 | .754660 | .211620 |
| [8,infinity) |311 /218,011 | .968964 | .145797 |

Late bins contain stable events only; the rising share does not imply large
absolute late error or describe unobserved unstable tails.

Common angular-acceleration RMS for perpendicular/parallel/total errors is
.201202/.693309/.586949rad/s^2; net total-force error RMS=2.824038pu.
These scalar common-motion terms interfere: their squared RMS values do not
obey the generator-force Pythagorean partition. They do not by themselves
explain autonomous common-frequency trajectory errors in0063.

Integrated frozen-ODE defects in force/s:

| Reference | All346 | Stable311 | Unstable35 |
| --- | ---: | ---: | ---: |
| Sliding encoder |2.560702 |1.365482 |6.946566 |
| Physical-force projection qF |2.932905 | .681496 |8.994966 |

Stable/unstable window counts46,339/175 receive equal event weights, not
equal total population weight. qF is more consistent on stable trajectories
but worse on unstable prefixes and overall. Noise/quadrature, hidden physical
dynamics and a different physical-rollout training objective limit causal
interpretation. Do not replace every encoder target by qF from this result.

## Classification / interpretation / next implication

Accepted diagnostic attribution, not model promotion. Fixed output-span
limitations account for26.05% of this force error; they are real but not the
dominant observed component. This does not justify increased r or a new map.
Nor does a pointwise force projection establish a closed dynamical state.
No deployed model change requires rollback; keep original and0063 anchored
initializations as previously classified. Group coefficient robustness is
still unproved; identical supports and proxy fits do not resolve it.

The same sparse5s ODE has remained frozen across0061–0064. Before proposing
further fitting, inspect its full coupled equilibrium Jacobian, not Aq alone,
and the mechanical storage exchange identity. This is a conditional audit of
that frozen equation, not final model acceptance, a full-order stability claim,
or a waiver of group/numerical uncertainties. Final TEST remains closed.
