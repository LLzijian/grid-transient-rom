# Loop 0045: TRAIN-only spatial/coordinate forcing capacity diagnosis

Baseline1b4a9a3. One diagnostic question: do the fixed atlas direction and0042
full-state coordinate have trajectory-correction capacity when their time
evolution is supplied, or did the preceding failures already arise in the
output geometry? No observer, parameter fit, or memory-dimension selection.

All351 TRAIN events still available after50ms; exclude18 already-terminated
events only for this predictive-start diagnostic. Use each available prefix
through8s after decision, no padding or extrapolation. Read no validation/test.
Compute exact r_e from te at every available grid point as0040. Compare frozen
M3 with additive future r_e forcing in: first balanced direction only; frozen
three-hidden-state amplitude only; all zero-sum force; full force. Full force
plus true future Pm is a numerical-contract control, not a candidate model.
All other arms keep autonomous frozen Pm dynamics. RK4 .005, linear interpolation
of exogenous force/Pm; initial true state at50ms. Forced arms explicitly use
future information, cannot be called autonomous, causal or deployable.

Report2/5/8s mean/pooled/CVaR with original Pilot TRAIN scales, early horizons,
available-prefix errors, stable/unstable and disturbance groups. No stability
classification from shortened forced prefixes. Descriptive capacity screen:
>=5% mean gain, pooled nonincrease, CVaR<=1.05M3 at2/5/8s. This never adopts a
model and cannot justify increasing dimension by itself.

Accept diagnostic if all351 eligible events/24groups retained, finite outputs,
full-force+true-Pm control has prefix RMSE median<.01 and95thpercentile<.05,
and representative half-step predictions relative discrepancy<1e-3 (first stable
event per shard and longest available unstable TRAIN event, metadata-only rule).
Preserve all per-event predictions/forcing/time arrays and metrics, code/data
identities; Ruff/compileall/all tests. If direction capacity is poor, revisit
physically appropriate force geometry before another memory ODE; if coordinate
is poor but exact direction helps, revisit state discovery; otherwise diagnose
closure dynamics. Never subtract rejected0041 instantaneous correction.
