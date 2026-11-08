# Loop0026 closed: optimization succeeds, prediction screen fails

Start e5221e0; frozen experiment689e5fa. Command:
`python experiments/run_private_m4_predictive.py --output private_data/task2_m4_0026`.
48tests/Ruff pass; directional derivative relative error1.32e-9. L-BFGS used its
preregistered max_iter80/max_eval120 configuration (121 closure evaluations as
the last strong-Wolfe search completes, no manual extension). Runtime287.21s.
TRAIN objective1.474183→1.051160, meanMSE .592381→.442738: optimization gate passes.

| Horizon | M3 pooled | Candidate pooled | Candidate mean | Candidate CVaR10 |
|---|---:|---:|---:|---:|
| 2s | .740507 | .810332 | .674480 | 1.585157 |
| 5s | .565149 | .611632 | .519003 | 1.159359 |
| 8s | .464055 | .499415 | .425436 | .939940 |

Required5% mean gain fails; pooled worsens. One new false-stable (pilot_0366),
no false-unstable. Stable prefix n61 mean/pooled/CVaR .425436/.499415/.939940;
unstable prefix n9 .847136/1.209174/1.957070. Prefix pooling is dominated by long
stable records; event-balanced results are also retained. This is development
selection, not test/OOS evidence. All444 records and training evaluations are in
`0026_m4_evidence.json`; paired differences are exactly reconstructible with
`reports/accepted/0020_m3_evidence.json` by event_id/horizon.

Final a_q=1.916890/s, not at either [.05,20] guard. q0 quantiles[min,p10,median,p90,max]:
train[-2.694764,-1.714907,-.160571,.349972,.693147]; validation
[-2.742807,-2.027249,-.044671,.574308,.693147]. Upper clipping1/218train,
7/70validation (10%); lower0. Thus observer saturation/generalization is a real
diagnostic, not hidden behind parameter positivity.

Worst genuine-prefix event pilot_0239: unstable, normalizedRMSE1.95707, truth
observed .37s after decision; truth angle spread peak5.08393rad and raw synchronous
speed13.04961rad/s. Candidate fixed8s reaches395.772rad/48.10997rad/s, outside
observed truth and not claimed accurate. Worst stable pilot_0314: RMSE1.33353,
q0 at upper clip; predicted/true max angle spread2.90401/3.31863rad and speed
6.91247/6.46133rad/s. False-stable pilot_0366: truth1.05s, candidate speed peak
7.17713 versus observed14.18970rad/s; the sustained joint LOS rule is missed.
Full physical traces remain in private run rollouts and the original HDF5.

Reject and revert the implementation; retain accepted M3. User's new authoritative
`docs/TASK2_after_loop_26.md` now requires (1) numerical oracle q0, (2) time-resolved
modifier inversion, (3) immediate M3 storage/dissipation baseline, before selecting
the next memory mechanism. No GRU/q2 or further observer fitting before these tests.
