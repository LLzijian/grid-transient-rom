# Loop 0020: first-order mechanical-power contribution

Preregistered after frozen M2, before fitting. Hypothesis: a stable first-order
effective governor explains the observed mechanical-power evolution better than
decision-time ZOH, and its causal coupling improves the frozen swing rollout.

Keep the 0019 split, decision50ms, cohorts, normalization and electrical K/D.
Fit true TRAIN Pm using event-balanced weak windows1/5/20 samples (max40 each):
`Pm_dot=-a*(Pm-Pm_star)-b*omega_sync`. Equilibrium fixes P_ref=Pm_star, no
event intercept. Constraints a in [.01,50] (T=.02..100s), b>=0; fixed ridge1e-6
after column scaling. These broad bounds are physical plausibility guards, not
validation-tuned search ranges. No test payloads.

Identification screen: with validation true omega forcing (explicitly diagnostic,
not deployed prediction), mean Pm RMSE improves>=5% over ZOH at2/5/8s, pooled
does not worsen, CVaR10 worsens<=5%. Report boundary hits and per-machine error.
This tests the constitutive law separately from inaccurate M2 omega.

Contribution screen for joint30-state autonomous rollout: mean normalized
trajectory RMSE improves>=5% at2/5/8s, pooled does not worsen, CVaR10<=1.05xM2,
no new false-unstable event among stable truth, no numerical failures. Record
all horizons and full/prefix cohort counts regardless of outcome.

If identification passes but coupled contribution is limited, retain the
identified mechanical module as an assessed component (not an accepted accuracy
claim), per TASK2 milestone1, and proceed M4. If identification fails, reject this
implementation, retain evidence, diagnose mechanical residuals and test the next
small physically motivated mechanical law before introducing electrical q.

Software gates: exact equilibrium, positive T/nonnegative gain, synthetic
parameter recovery, integration agreement with DOP853, split/causal contracts,
independent per-event score reconstruction. No framework or new metadata format.
