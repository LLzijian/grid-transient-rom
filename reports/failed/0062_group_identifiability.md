# Loop0062 — conditional group consistency rejected

## Question / hypothesis / change

Can the frozen encoder's integrated latent increments be represented consistently
across24 TRAIN fault-bus groups on the existing36-entry support? Preregistered
at12d7734 (`reports/task2_group_identifiability.md`), implemented fb1a237.
One pooled and24 leave-one-group-out SVD fits are diagnostics only. Every
deployment tensor, library, mask, topology and encoder remains unchanged;
no new model.pt, optimization curriculum, validation selection or TEST access.

## Evidence

All346 eligible full TRAIN events, lengths14..1501, contribute46,514 genuine
windows including short final windows. Per-event weighting is equal. Saved
arrays/results: `private_data/task2_group_identifiability_0062/`.
97 repository tests pass. Independent saved-array arithmetic reproduces
coefficients, spectra, group metrics and decisions with zero discrepancies:
`.loop-work/long0062_group_independent_checks.json` (absolute1e-8/relative1e-10
arithmetic tolerance, not a change to scientific gates).

All pooled and leave-group-out output designs have rank12 at rtol1e-10.
Pooled scaled-design conditions are141.967/63.585/102.066; LGO ranges
135.136..198.353,61.678..73.722,98.275..125.839. Thus the declared numerical
rank screen passes. Finite rank is not a noise-robust coefficient certificate;
these nontrivial condition numbers and full spectra remain reported.

The common-group consistency screen fails for10/24 groups. E is the ratio of
held-out output-projected encoder-reference defects (LGO/pooled); D is the
relative projected-field disagreement, not a trajectory-prediction error.
Registered limits are E<=1.10 and D<=.10, for every group.

| TRAIN group | E ratio | D |
| --- | ---: | ---: |
| 2 | 1.210556 | .638852 |
| 15 | 1.097817 | .121711 |
| 17 | 1.095306 | .104916 |
| 20 | 1.233272 | .786302 |
| 23 | 1.064535 | .260741 |
| 28 | 1.365151 | .534723 |
| 29 | 1.164090 | .219691 |
| 31 | 2.131965 | 1.715605 |
| 32 | 1.435054 | .268287 |
| 33 | 1.117826 | .593551 |

Group31 is not a floor-denominator artifact: projected-field disagreement
numerator2.885251 versus denominator1.681769 force/s; pooled/LGO reference
defects2.239353/4.774224 force/s. Its13 events are retained. All group and
stable/unstable contributions remain in evidence, including groups that pass.

The unconstrained proxy fit differs greatly from deployed coefficients and
has Aq poles +.826413 and -.273422 +/-1.286099i, versus the deployed stable
conditional poles -.709166 and -.950451 +/- .273285i. This is not evidence
that the deployed Aq is unstable or a full-system stability calculation.
It shows why a better encoder-reference fit cannot be promoted as a model
without autonomous validation. The encoder reference is not latent truth,
and its derivative objective differs from the deployed physical-rollout loss.

The prescribed reference check also finds a distinct consistency defect:
encode(zero physical history)=[-.1872661,-.0487345,-.1304298], although the
frozen ODE's physical equilibrium is [0,0,0,q=0]. Initial force RMS is.148069
pu and max.306116 pu; the actual full-field maximum is1.260165 (the largest
component is a synchronous angular acceleration). qdot is
[.179791,.085439,.006035]. All values are finite and reproducible. This
observer/ODE equilibrium mismatch is not removed by a coordinate relabeling.

## Classification / root interpretation

Rejected: the conditional common-group encoder-reference equation hypothesis
does not pass its preregistered sensitivity screens. Numerical identifiability
alone is insufficient. This does NOT reject r3, the successful sparse5s
autonomous model, the encoder family, or every possible support. No diagnostic
fit is promoted; no deployed implementation requires rollback because no
deployment changes were made. Diagnostic code/evidence are retained.

Group disagreement and the proxy/deployed-field gap need diagnosis, not an
immediate larger library. Identical imposed masks still do not demonstrate
robust support discovery. The separate equilibrium initialization defect is
concrete enough for a minimal physically motivated next intervention, without
assuming that it explains all group inconsistency.

## Next implication

Preregister an equilibrium-anchored initialization comparison on the same
frozen r3 ODE: subtract the encoder's response to the known equilibrium
history once at initialization, with no trained weight or future feedback.
Compare physical forecasts prospectively before any promotion. This is an
observer calibration hypothesis, not an encoder architecture change, an
affine reparameterization of the ODE, or a claim that group robustness is solved.
Keep the original candidate/checkpoint available; continue after either result.
