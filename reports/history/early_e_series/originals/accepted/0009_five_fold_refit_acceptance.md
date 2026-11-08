# Five-fold refit acceptance

Date: 2026-09-06

Baseline: `4b3bea4`; final pending artifact: `results/e10_refit_v2/F5/prune`.

Decision: accept completion of the registered five-fold refit experiment,
including the poor F5 result. This accepts the numerical record and protocol
execution, not a claim of adequate prediction or generalization.

All 15 stage model NPZ/JSON hashes and upstream dependency hashes match their
manifests. Recomputing median RMSE, pooled RMSE and pooled R2 from the stored
event rows reproduces every stage summary. All five final models persist the
same validation scales recorded in their scale provenance. Each total-gate
improvement recomputes from its two comparable losses. Physics checks pass.

| Fold | Total candidate improvement | Final optional instances | H100 eligible / declared | Final pooled RMSE | Final pooled R2 |
|---|---:|---:|---:|---:|---:|
| F1 | 66.309% | 6 | 34 / 43 | 0.316955 | 0.765788 |
| F2 | 0.042% | 0 | 32 / 37 | 0.361648 | 0.713422 |
| F3 | 59.230% | 5 | 34 / 42 | 0.297576 | 0.857565 |
| F4 | -91.670% | 0 | 38 / 43 | 0.399669 | 0.676278 |
| F5 | 24.314% | 6 | 39 / 47 | 1.579711 | -1.698347 |

F2 and F4 correctly fall back to the fresh backbone. F5 removes the 0.02 s
edge-memory instance: its recorded deletion trial passes physics, modal
noninferiority, and all six source-by-horizon checks. The final F5 total gate
compares 50.8544910771 against 38.4897137078, with zero rollout failures on both
sides. No other F5 deletion is accepted. The near-critical source has only one
fault-line group per validation fold, so its group bootstrap does not establish
uncertainty across unseen lines.

There are 177 eligible H100 events out of 212 declared validation events, with
zero numerical rollout failures among eligible events. The 35 short events
are excluded by observation length; this performance population must not be
interpreted as covering all unstable events. H100 is 101 samples, and actual
end times are recorded per event rather than assumed to be exactly 1 s.

F5's negative R2 rules out a claim of consistently accurate five-fold
prediction. Passing a relative adoption threshold does not establish absolute
accuracy. The broad positive generalization claim remains unsupported.

The F1 backbone/full artifacts predate scale persistence. Their post-reload
standardized selection summaries used unit scales and must not be used as
normalized evidence. Their stored trajectory errors remain usable; the F1
final gate and v4 checkpoint use reconstructed training scales. The earlier
statement in loop 0008 that identical final coefficients prove the scale bug
never affected the deletion path was too strong: coefficient equality proves
output equality for that run, not equivalence of intermediate decisions.

Next: recompute the corrected modal diagnostic using these final checkpoints,
consolidate trajectory and structural evidence, and revise manuscript claims.
The legacy holdout build remains consumed; no confirmatory release is claimed.
