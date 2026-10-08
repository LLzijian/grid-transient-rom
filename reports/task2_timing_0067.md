# Loop0067 matched-interval timing protocol

Timing only; no model selection or threshold change to the joint screen.
Before measurements select lexicographically first eligible stable TRAIN
event and first eligible unstable TRAIN event across the two existing pools.
No TEST and no choice based on observed runtime or model error.

For each event compare the interval from its stored record start+100ms to
that instant+min(8s, true remaining prefix). The true prefix length defines
the benchmark interval only; it is not a forecasting input or LOS rule.
ANDES 2.0.0 replays the same case, fault and solver settings. Verify case
SHA, restored-system signature, equilibrium and the replay against saved
TRAIN truth before using timings. Preserve numerical failure as failure.

Separate case/code initialization, PFlow/TDS initialization and fault/history
prelude from post100ms integration. Instrument converged integration steps
over the matched interval, not the whole 15s generator or HDF5 export.
Report the actual solver endpoints and boundary rounding (<=one solver step).
ROM uses the existing JointROM implementation, one encoding of11 real states,
true initial p, autonomous RK4 .005; log GRU, integration and their sum.
Both engines use one CPU thread, same host, no competing research jobs.
One unmeasured warmup then three measured repetitions per event/model;
report each repetition and median, not the minimum. S and0058 both included.
Serial event throughput is events/s from the measured totals; any batched
throughput must be labeled separately and cannot replace serial latency.
The100ms observation wait is reported separately from computation.

Repeat FOM at .0025 once for accuracy checking, not for picking fastest runs;
align on the stored record-relative grid using actual absolute solver times.
Both engines require normalized COI discretization RMSE<=1e-3 relative to
their .0025 reference on the matched interval. Also report ROM-vs-FOM
trajectory RMSE (COI and common frequency separately). A raw cost ratio is
not an accuracy-equivalent acceleration claim: such a claim for this pilot
benchmark additionally requires ROM normalized COI error<=.05 and common
frequency RMSE<=.05 rad/s on that same event. Otherwise explicitly label the
ratio cost-only, with failed accuracy checks. These timing reporting bounds
are fixed before replay and do not alter the0067 scientific shortlist gate.

All replays write to a fresh ignored private_data timing directory. Original
datasets, accepted checkpoints and the shared Python environment are not
modified; ANDES generated code and isolated dependencies stay in workspace.
