# Loop0081 — conditional latent consistency rejected

Work type: diagnostic. Result: rejected. Model promotion: none.
Preregistration764b08d; implementatione26832c; test supplement8ccc1f9.

The frozen G-fixed3000 force-projected trajectory qF is not more compatible
with its latent equations than the causal qenc trajectory, even when physical
states are supplied from TRAIN truth throughout each diagnostic window.
Overall mapped-defect RMS ratio qF/qenc=2.693877201 exceeds .90; the unstable
subset ratio3.234708188 exceeds1.05. Both registered gates fail.

| Equal-event mapped-defect RMS, torque/s | qenc | qF |
|---|---:|---:|
| All346 | 1.447304882 | 3.898861624 |
| Stable311 | .924069382 | 1.209334136 |
| Unstable35 | 3.622150182 | 11.716598852 |
| Initial window,346 | 2.176613054 | 7.859119694 |

All24 groups and46514 windows are retained:46482 full10-step windows and32
short windows, spanning464971 increments across465317 saved endpoints. The
reference is RK4 .005, comparison .01, with the same piecewise-linear true
physical trajectory. Per-window normalized numerical maxima are qenc
4.600969766e-12 and qF4.600629046e-12, below .001; no finer rescue. Runtime
198.04792s. Parameters and original identities/hashes remain unchanged.

Independent cached NumPy arithmetic reproduces per-window vectors, numerical
comparisons, event/group/subset/initial summaries, full partitions and0079
reference mappings within1e-11; recorded vector/event-MSE discrepancies are
zero. It reads saved development arrays only, not models or H5. Production's
inherited reader materializes unused validation; no TEST is read.

Five initial synthetic tests passed before launch. A supplementary nonzero
affine-q test was added during execution; its analytic reference initially
used float32, corrected to float64 after root review (not by weakening the
tolerance). Six tests then pass; affine fine/coarse errors2.06e-11/3.33e-10.
This supplemental QA ordering differs from the intended all-before-launch
sequence; production code, scientific thresholds and the single run were
unchanged. No favorable rerun or fitted scientific result was selected.

## Interpretation and inherited stage review

0079 found substantial error both inside and outside the learned force span;
0080 showed that one-shot qF initialization harms autonomous forecasts;0081
now shows worse conditional compatibility. Reject this oracle-reference
route, not r3 or the entire template. qenc is also a constructed state, not
latent truth. Neither rank3 nor these diagnostics establishes well-conditioned
inversion, unique modes, template capacity, or adequate original optimization.

An independent read-only scientific challenger (stage81_challenge, not an
implementer) emphasized these confounds, the still-falling G TRAIN objective,
and the danger of more reference/window variants without an intervention.
Root accepts these objections. Changing a learned map changes0079's floor;
no structural impossibility or original-optimizer convergence claim is made.

Close the inherited diagnostic stage here: three frozen-model diagnostics,
0079/80/81, approximately277+416.87+198.05=891.92s of main runs, plus QA,
independent recomputation and review. This excludes earlier0078 training:
the G3000 extension alone cost17271.40s. No new oracle/window variant or
automatic budget rollover follows. Preserve all failed evidence and unchanged
accepted M3/0058/S; C3000 remains only the prior trajectory control.

A justified next finite stage can test recoverable coefficient error within
the existing observer/support using a tractable affine coefficient block,
then assess transfer to autonomous TRAIN prediction. The reviewer warns that
conditional regression is a different objective: report physical/latent
residuals separately, fix the tie-breaking rule prospectively, and do not
call its optimum the optimum of the original nonconvex training objective.
Root adopts those limits; a new preregistration is required before execution.

Evidence: `private_data/task2_conditional_latent_0081/{windows.npz,evidence.json}`;
independent checker/output: `private_data/task2_conditional_latent_0081_independent/`.
