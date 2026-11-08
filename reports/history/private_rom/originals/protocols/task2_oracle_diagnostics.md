# Loop0027: observer gap versus centered-strength family capacity

Preregistered under user `TASK2_after_loop_26.md`. Diagnostic, not model adoption.
Keep rejected0026 a=1.9168904267222429/s and acceptedM3 parameters. Use all218train
and70validation prospective events; frozen test never loaded. User-owned kiki.zip
is untracked and untouched; scientific tracked checkpoint is clean.

Oracle q0: minimize each event's genuine-prefix normalized trajectory MSE with
q0 in[-3,log2], using full future truth explicitly as diagnostic supervision.
Coarse9 equally spaced values plus0, then24 batched golden-section iterations in
the best coarse cell. Include q0=0 as a final competitor so attainable prefix loss
cannot exceed M3. This is a bounded numerical search, not certified global argmin.
Integrate all candidates fixed8s atRK4dt.005, score only original observed truth.
Clear potential: oracle mean>=5% better at2/5/8s, pooled no worse,CVaR10<=1.05x;
classifications and prefix results reported separately (oracle never deployable).

Time-resolved inversion: derive required electrical force from trueTRAIN/VAL Pm,
omega and centered finite differences using the preceding50ms samples; omit final
one-sided derivative point. Project `required-E(delta)` onto E(delta); s=projection,
q=log(1+s) only when1+s>0 and forcing norm² exceeds max(1e-8,1e-4*TRAIN median).
Never clip invalid oracle q into apparent validity. Record coverage/negative-scale
rates, sign changes (>0.05 log-strength on both sides), variation, free-exponential
fit at fixed0026 rate and best bounded [.05,20] rate, and generator-wise residual
norm explained. Compare the uncentered direction E(delta)+Pm_star as a diagnostic
of transient electrical offsets, not a trained/deployed candidate.

One script, compact JSON evidence plus private per-time arrays. Numerical gates:
q0=0 reproduces M3, oracle prefix score<=zero competitor, all events accounted for,
projection reconstruction checks, no test access. Results may be negative and still
close this diagnostic loop. Then immediately establish M3 energy baseline.
