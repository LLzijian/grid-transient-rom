# Loop0070 preregistration — one original-library J-S candidate

Authority after66. Starting checkpoint a09813e plus0067 closure; no deployed
model change. This is Campaign A's third and final screening candidate.

Implementation preflight note (before the scientific fit): S's inactive
constant row has stale raw values of order1e-7 despite its zero mask. Clear
that masked storage row to zero when opening candidates; do not activate it.
The first smoke stopped before training on an overly strict raw-value check
(`.loop-work/support0070_smoke`); this is an implementation failure only.
Require exact initial vector-field preservation in the regression test.

## Question and fixed algorithm

Hypothesis: the inherited36-term support limits joint closure's ability to
retain improved common motion without damaging COI/severe-event prediction.
Test support rediscovery, not larger latent dimension/library or new losses.
J0067 is rejected; its common5 gain is diagnostic evidence, not a promotion.

Start independently from the SAME S checkpoint SHA
d34d9f5c36fb296e1d49e59f5f7b18a5da33b56bf371a16169f82bc25664995c.
Use exactly0067's joint12 parameters, objective/weights, normalization,
100ms observer, physicalM3, ports, alpha mask, r3 and1/2/5s schedule.
Seed67, fresh Adam1e-4, batch32,800 optimizer steps, one CPU thread.
Training sampler/random draws remain the same as0067. Evaluate only step800.

Steps1–200: open every NONCONSTANT Xi entry of the original76x3 library
(225 candidate entries). Newly opened entries start at zero, not stale
pruned coefficients. Constant row stays zero/inactive. Other source buffers
remain bitwise fixed; no q whitening or post-fit coordinate shift.

After step200 only, use a separate RNG seed7000 to sample512 local TRAIN
windows with the same label/group/event/endpoint balancing. Their current
11-point encoded q and true endpoint physical state define Theta_j; these
are contribution coordinates, not latent truth or new supervision. Set
s_j=sqrt(mean(Theta_j^2)) over these512 rows. Rank each nonconstant entry
by abs(Xi_jk)*s_j. Keep the largest48 strictly positive scores (or fewer
if fewer are positive), ties by ascending row-major index. This rank cutoff
is the fixed standardized-contribution threshold; save all225 scores,
sampling IDs/endpoints, before/after support and coefficients. No validation
selection, mandatory q-diagonal retention or hand-chosen terms.

Zero and freeze dropped entries, reset Adam once, then steps201–800 jointly
refit on the retained support. Keep the SAME objective including its original
1e-4 sparsity coefficient; no new penalty or weight tuning. A zero-support
selection is an optimization failure. Nonfinite loss/gradient/parameters,
f(0)>1e-10 or frozen/phase-support contract failure stops and preserves the
fit; no budget extension/rescue. Final support<=48 is mandatory. Failure of
this finite algorithm does not falsify the entire model family.

## Prospective screen and consequences

Reuse0067 C and S, same107 eligible/117 raw validation events and exact L1
numerical/refinement/causal rules. All scored/claimed intervals must pass;
no newly unresolved events may count as a safety gain. J-S misses, false
LOS and false settled must not increase versus either C or S.1/2s pooled
and CVaR, unstable-prefix mean/pooled/CVaR, and5s common pooled must each
be<=1.05 times BOTH C and S (floor1e-8).

Require5s complete mean AND pooled<=.95*C, CVaR<=1.05*C and allthree<=1.05*S.
Also retain the observed joint common-motion benefit: common5 pooled must
be<=1.05*J0067 (.30944867516462987rad/s reference). These are new prospective
criteria for this algorithm, not a reinterpretation of0067. No seed sweep
or8s tuning unless this screen passes. Report8s, all per-event evidence,
and complete coupled local Jacobian as diagnostics, not stability claims.

Before fit: targeted support/sampler/frozen/gradient tests and one real-TRAIN
pruning smoke, root diff review, checkpoint. No TEST. If accepted, only
bounded confirmation follows; otherwise close Campaign A without another
support/weight sweep, diagnose the remaining physical residual and decide
B2 or the single capacity probe allowed by after66.
