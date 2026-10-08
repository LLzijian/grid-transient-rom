# Loop0066 preregistration — paired deployment-loss group sensitivity

Starting2336a8a. Authority docs/TASK2_after_loop_59.md.0062's largest
conditional encoder-reference sensitivity occurred in TRAIN group31
(E=2.1320,D=1.7156).0064 shows oracle force targets do not uniformly improve
closure. Neither result establishes that physical deployment-loss fitting
requires inconsistent coefficients.0065 establishes only local stability of
the frozen sparse5 ODE; group robustness is still a prerequisite to promotion.

## One question / hypothesis

Does removing group31 from a fixed-budget coefficient-only continuation
materially change its genuine-prefix autonomous predictions compared with
a matched all-TRAIN continuation? Hypothesis: conditional omission sensitivity
passes the fixed predictive ratios and disagreement thresholds below.
Selection of group31 is adaptive to published0062 TRAIN evidence. It is NOT
a new validation-driven group search or honest held-out-group generalization:
encoder, topology and initial coefficients were already trained on group31.

## Fixed arms and allowed change

Source S: unchanged private_data/task2_long_0060_5s, model SHA
d34d9f5c36fb296e1d49e59f5f7b18a5da33b56bf371a16169f82bc25664995c.
Arm A: continue S with all346 eligible TRAIN events.
Arm B: continue S with TRAIN group31 excluded ONLY from optimization.
Every excluded event remains in diagnostic evaluation and evidence; this is
intentional coefficient sensitivity, not dropping difficult samples from
reported results. Record exact fit/exclusion/sample IDs and group counts.

Each arm:700 steps, seed60, batch32 uniform event sampling with replacement,
Adam fresh moments lr=.0003, horizon5s, existing full-trajectory local
sampling. Use the EXACT train_long_latent.campaign_loss:10*(.8 mean+.2
batch worst10%)+local_mean+dyn+1e-4 XiL1. No loss/weight changes, gradient
clipping, extra optimizer schedule, checkpoint selection or extensions.
Final step700 is the sole candidate. Default sampler RNG remains unchanged;
same seed does not imply paired event draws after the event pool changes.
Train A then B to avoid simultaneous heavy memory use.

Only Xi may update on its existing36-coefficient support. All other tensors,
inactive Xi entries, normalization, encoder/100ms initialization, r3,
algebraic mapping, topology and physical backbone must remain bitwise equal
to S. Do NOT apply0063 anchoring or0064 qF targets. No new library/loss.
No group metadata enters model inputs. TRAIN alone supplies gradients;
existing reader may load unused validation. TEST payloads remain closed.

## Optimization adequacy diagnostic

Before interpreting the primary hypothesis, compare source/final objective
on a fixed deterministic TRAIN panel. One sample per eligible event, endpoint
floor((10+len(x)-2)/2), genuine11-history, evaluated without gradients by
the SAME campaign_loss, all events in each panel at once so event balancing
and worst10% are not chunk-dependent. Panels:all TRAIN, complement31, group31.
Save total and every objective component for S/A/B on all three panels.
Adequate screen: A all-TRAIN total <= S all-TRAIN total*(1+1e-6), and B
complement31 total <= S complement31 total*(1+1e-6), finite training and
unchanged frozen tensors. If not, classify the predictive sensitivity result
as optimization-unresolved, not coefficient-family rejection. This finite
midpoint panel is a reproducible optimization check, not stationarity proof.

## Primary predictive comparison

Encode each TRAIN group31 event once from its true initial100ms history.
Evaluate S/A/B autonomously to8s, with output grid .01s, no re-encoding or
metadata. Preserve fixed forecasts separately from actual genuine prefixes.
Primary cohort: ALL eligible group31 events; score genuine available prefix
through min(5s, available duration), excluding initial output from errors.
Use established COI delta/omega metric, scales .47675314588683515 and
1.0033667249602924, equal-event mean, pooled RMSE, ceil10% CVaR. No omitted
short or unstable events and no invented truth. Save per-event scores.

Primary supported only if numerical/optimization checks pass AND
B/A mean, pooled and CVaR ratios are all <=1.10, with denominator floor1e-8;
AND pooled trajectory disagreement(B,A) divided by max(A pooled prediction
RMSE,1e-8) <=.10. Disagreement uses identical genuine group31 prefix points
and the same COI/scale/SSE formula, excluding initial. Any valid failure
rejects THIS fixed-budget conditional insensitivity hypothesis, not r3 or
all conceivable coefficient fits. Source comparisons are descriptive.

Report group31 complete1/2/5/8 and capped8s-prefix scores, stable/unstable
prefix summaries and common synchronous-frequency RMSE descriptively.
Run unchanged development benchmark on A/B for1/2/5/8s and all validation
groups, preserving Pareto/tail tradeoffs; no validation-driven reranking,
early stopping, tuning or selective extension. S benchmark already exists.

## Numerical and structural checks

For group31 evaluate RK4 h=.0025 with companion .005, both sampled .01.
Primary support/rejection requires finite outputs and maximum per-event
normalized COI step discrepancy <=1e-3 on genuine primary prefixes. If it
fails, exactly one .00125 diagnostic may refine the comparison; failure
remaining is numerical-unresolved, never masked as poor prediction.
Keep all-fixed8s convergence separate; prior unresolved unobserved tails
are not grounds to invent truth. Validation benchmark retains its existing
.005/.01 audit and, only if genuine-prefix/complete comparisons require it,
one .0025 diagnostic. Report unresolved cases without extra refinement.

Save masked Xi changes/signs/scales, Aq poles, and full gauge-quotient J32
poles using the verified0065 analytic Jacobian. Cq/encoder/support unchanged
by construction. New positive full-system poles (>1e-6) preclude promotion
but do not automatically change this sensitivity hypothesis's criterion.
No candidate is automatically promoted by one selected-group diagnostic.

## Execution / preservation / continuation

Small CLI fit-group filter with explicit recorded IDs; no data-layer change.
Small dedicated fixed-panel/group31 evaluator reusing verified functions;
do not build a fold-training or reporting framework. Synthetic filter,
prefix alignment, event weighting, missing-tail and disagreement tests,
root review, full tests and smoke BEFORE scientific runs. Commit code first.
Independent saved-array/score/hash QA afterward. Preserve model failures,
sample IDs, source hashes, train traces, panel terms and per-event forecasts.
If a fit is rejected keep it as evidence only, S remains unchanged. Record
evidence, diagnose next smallest physical question and continue. No fresh
TEST access, increased dimension or automatic library/loss changes.
