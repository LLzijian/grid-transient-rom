# Loop0029 rejected: better compact observation is not safe closure

Start125c4ed; experiment2794807; command:
`python experiments/run_oracle_observer.py --output private_data/task2_observer_0029`.
One fixedridge.01 fit from TRAIN oracle targets only;48tests/Ruff/compile pass.
No test payload. Zero clipping on both splits; validation oracle-q0 correlation
.757628,RMSE .505191,R² .534731 (train correlation .772965,R² .596913).

Early pooled errors improve: at.2s .334328→.277480,.5s .519887→.404714,1s
.737084→.599391. But2/5/8s pooled=.861652/2.126319/3.645277 vsM3
.740507/.565149/.464055;2new false-unstable (0056,0150),1false-stable(0366).
All444 per-event records retained in `0029_observer_evidence.json`.

Thus removing optimization difficulty and saturation does not make the centered
model robust to attainable observer errors. Initial-state estimation correlation
must not replace end-to-end dynamics acceptance. Revert the failed observer.
Do not jump to GRU despite oracle trajectory potential: the time-resolved
diagnosis already shows missing electrical offsets and poor exponential closure.
Next, test the user's prioritized uncentered/effective-voltage form at the same
fixed decay rate using the same numerical oracle, changing only centering. If its
family has useful capacity, test a small causal observer before driven terms.
