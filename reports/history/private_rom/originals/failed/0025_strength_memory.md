# Loop0025 rejected: fixed Adam fit did not minimize its training objective

Start6ecffa8; experiment3bf09e6; command:
`python experiments/run_private_m4_predictive.py --output private_data/task2_m4_0025`.
Fixed80steps,200.80s; no intermediate validation selection or extension.48tests
andRuff pass. Actual26-edge Torch/NumPy field discrepancy9.77e-15; exponential
q integration discrepancy9.28e-11, strength remains in[.0498,2].

Scientific screen fails: pooled2/5/8s1.244759/1.863656/3.140226 versusM3
.740507/.565149/.464055;5new false-unstable and2false-stable events. More
fundamentally, training loss1.474183→87.233491 and meanMSE .592381→29.394385.
An optimizer that increases its own objective by59x does not identify the best
model in this physical family. Do not interpret this as strong evidence against
the voltage-recovery mechanism itself. Reject the fitted implementation anyway.

Retain444 per-event records and81 training records in `0025_m4_evidence.json`;
revert implementation. Next bounded hypothesis: strong-Wolfe line-search
identification can test the same41-parameter physical family without uncontrolled
Adam steps. Keep all scientific gates, exact data and model unchanged; constrain
the optimizer using training loss only, not extra validation access.

Additional unit audit: ANDES2.0.0 genbase.py defines M*omega_dot=tm-te-D*(omega-1),
consistent with this project's effective Pm=tm convention; it is mechanical
torque on nominal-frequency power base, not tm*actual_omega shaft power. Rawomega
is retained for that conversion. TRAIN delta_dot=omega finite-difference RMSE
3.97e-4rad/s supports coordinate consistency; this is not a source of huge errors.
