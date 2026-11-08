# Loop0061 preregistration — fixed long-horizon sparse-system audit

Starting checkpoint a38755d. Authority: docs/TASK2_after_loop_59.md, Phase4/5.
0060 selected the sparse5s candidate for audit, not final model acceptance:
`private_data/task2_long_0060_5s/model.pt`, SHA256
d34d9f5c36fb296e1d49e59f5f7b18a5da33b56bf371a16169f82bc25664995c.
No teacher extension, fitting, support change, new state, loss or encoder.
All model/norm/M3/topology weights frozen. Fresh TEST forbidden.

## Question and falsifiable hypothesis

Does the long-trained sparse r3 system retain three effectively driven and
observable state directions with non-redundant long-autonomous contributions?
Hypothesis: full-TRAIN reachable-region diagnostics support local order3,
and the two specified fixed rank2 reductions below fail to preserve5s
prediction. This is a local/finite-time learned-system claim, not a proof of
global physical minimality, full-system stability or physical latent truth.

## A. Frozen matrices and actual driving

Reuse0059's verified Aq/Bq/Cq, finite-Gramian and real-modal-block algebra.
Aq=(masked Xi[1:4]).T. Bq is the actual latent-RHS derivative with respect
to physical [delta rad,omega synchronous rad/s,p pu]; Cq is the ACTUAL
generator electrical-correction map including normalized C and amplitudes.
Report Aq poles, real decay times, oscillation frequencies/damping ratios;
a conjugate pair is one oscillator with two real state directions.

Use all346 eligible TRAIN trajectories in full, without111/811 caps. At each
event choose unique rounded linspace endpoints10..last, at most31 points.
Include the reference x=0 separately. Compute Bq at those representatives.
Physical input RMS scales: event-balanced full-trajectory samples after100ms,
COI angle plus synchronous omega and p, floor.01 as in0059. Derive no scale
from validation. Report physical and TRAIN-scaled reachability separately.
Use equal-event representative mean Q=mean_e mean_t(Bscaled Bscaled.T).

Report O=[Cq;Cq Aq;Cq Aq^2], R=[Bq,Aq Bq,Aq^2 Bq], singular values,
condition and rank at the reference and region, plus PBH/modal residues.
Retain0059 numerical rank tolerance and derivative/similarity checks; ranks
and transfer maps are invariant, raw singular values/conditions depend on
coordinates/units and are not independent-state acceptance gates.
Finite-horizon Gramians/HSVs at1/2/5/8s use the frozen regional Q.
Measure actual TRAIN forcing f_q(x,0)-f_q(0,0) per real modal block, with
equal-event RMS; encode each genuine initial100ms once to report initial
modal output excitation. No latent covariance/whitening-order test.

## B. Fixed no-refit autonomous interventions

Use the saved107 validation initial33-states and genuine aligned truth from
`private_data/task2_long_0060_5s_eval/trajectories.npz`. No new validation H5
access is needed. Same fixed8s deployment, never stop/re-encode using future
truth. Report1/2/5/8 complete cohorts, all genuine prefixes and12 unstable
prefixes separately, same COI scales/metrics and mean/pooled/CVaR.

Arms: unchanged full model; memory force q contribution off (algebraic alpha
retained); each of three output-component ablations; each invariant real
modal-block removal (never remove only one member of a complex pair); and
one TRAIN T=5 balanced rank2 projection. For any reduction q=Vz,
zdot=W f_q(x,Vz), initial z=Wq0, physical output CqVz, with WV=I.
The balanced rank2 keeps the top two T=5 HSV directions, fixed before seeing
validation intervention results. Do not search projection horizons or refit.

Primary rank2 preservation screen at5s: mean and pooled ratios<=1.02 and
CVaR ratio<=1.05 versus the SAME full model, with numerical validity. Both
the predeclared balanced rank2 and the real-mode-removal rank2 are examined;
if either preserves5s, minimum order3 is not established by these tests.
If both fail, this rejects these reductions only, not every conceivable r2
system. Report1/2/8 and unstable-prefix tradeoffs; passing5s does not accept
a reduced model. Component/block effects are quantitative diagnostics, not
additional model-acceptance vetoes.

## Numerical and interpretation constraints

Verify field/one-step equivalence and full1s physical replication to1e-8
against the saved source before intervention interpretation. Primary audit
RK4.0025, companion.005; save at.01 and include initial state in numerical
comparisons but exclude it from trajectory-error metrics. Full.0025 also
compares to the already saved0060 .0025 diagnostic. Eventwise normalized
physical discrepancy must be<.001 on a claimed horizon/cohort. Preserve
all-event and observed-cohort differences separately. If any arm fails this
comparison, run one fixed.00125 diagnostic, no weight/threshold changes.
Nonfinite forecasts invalidate that arm, with evidence preserved; no clipping,
sample removal or finite-only averages. Do not rewrite0060's failed gates.

Positive local-order evidence proceeds to outstanding group coefficient/
support robustness, then full-system equilibrium/Jacobian/energy work after
ODE freeze. A viable reduction instead motivates a separate prospective
structured-reduction test. Numerical/implementation failures are diagnosed
before scientific interpretation. No r>3 or library expansion is triggered.
This loop neither proves group-wise support robustness nor completes Goal.

## Stage-A evidence (2026-09-13; recorded before Stage-B)

Implementation checkpoint cb12c14. Command:
`python experiments/audit_long_latent_matrices.py --model-dir private_data/task2_long_0060_5s --output private_data/task2_long_sparse_audit_0061`.
Artifacts: that directory's `evidence.json` and `matrices.npz`. All346 eligible
TRAIN events, raw lengths14..1501, contribute10,486 representative points.
The existing reader also loads validation; no validation values enter these
matrices, input scales, Gramians, or fixed reduction bases. TEST is not read.

Poles are -.709166050 and -.950450845 +/- .273284666i. Their decay times
are1.410107s and1.052132s; the conjugate block has frequency.0434946Hz and
damping ratio.961061. This is a heavily damped oscillator plus a real decay,
not three oscillatory frequencies or evidence for three well-separated slow
physical time scales. These are conditional latent-subsystem poles, not full
closed-loop ROM modes.

Reference observability singular values5.859743/4.331921/2.354443,
condition2.488802; reachability physical13.023002/8.446483/4.631449,
condition2.811863; TRAIN-scaled7.742963/4.322829/1.770414,
condition4.373533. All reference and regional R/PBH ranks remain3 at the
three preregistered tolerances. Worst regional R conditions are5.35786
physical and5.30664 scaled. Raw spectra/conditions retain their coordinate
and unit dependence; they are not covariance-based independence gates.

Finite-horizon HSVs at1/2/5/8s respectively:
4.58058/3.54573/1.10534;5.96495/3.95484/1.33591;
6.51426/4.05638/1.39986;6.52259/4.05860/1.40083.
Actual TRAIN output-projected forcing RMS is1.172346 for the real block and
1.672562 for the conjugate block (force per unit time); initial modal output
RMS is1.519449/1.578584 (force units). Both blocks are genuinely driven and
initially excited. This does not establish their indispensable rollout value.

Reference full33-state field is exactly zero. Finite-difference max errors
Bq6.39e-10/Aq8.05e-11; similarity check relative error2.49e-14 and T5 HSV
absolute error8.88e-15. Frozen source hashes and Cq spectra match0060.
TRAIN-fixed T5 balanced and real-block bases are saved before Stage-B results.
Root conclusion: local/finite-time order evidence supports three directions;
the preregistered no-refit autonomous interventions are still required.
Independent saved-array recomputation (`.loop-work/check_long0061_matrices.py`,
result `.loop-work/long0061_matrix_independent_checks.json`) reproduces O/R/Q,
all four Wc/Wo/HSV sets exactly, verifies full representative coverage and
regional ranks, and modal projector identities to2.23e-16. This is arithmetic
QA of saved inputs, not an independent data acquisition or model fit.

## Stage-B engineering preflight

Saved-source checks pass: actual JointROM versus NumPy field3.55e-15,
one-step0, same-dt.005 full1s physical maximum9.88e-13 (<1e-8).
The .0025 fixed-full8s comparison with0060 has max absolute7.44e-8 at
pilot_0058; complete and genuine-prefix maxima are1.05e-13. See
`.loop-work/long0061_baseline_preflight.json` and matching NPZ.
The first implementation incorrectly imposed the1s absolute replication gate
on the full8s cache comparison too. Root removed that unregistered extra gate
before any intervention run; this is a specification correction, not a change
to the registered1s or eventwise normalized numerical criteria. The8s raw
discrepancy and scope-resolved normalized comparisons remain reported, and
0060's original all-event convergence failures remain failures. No changed
weights, rank selection, loss, or validation-dependent threshold follows.

## Closure

Loop0061 closed diagnostic at0f057c3. Both registered rank2 projections fail
the complete5s retention screen with numerically valid comparisons. All87
repository tests pass; independent saved-array rollout QA has zero mismatches
at1e-10. See `reports/accepted/0061_long_sparse_order_audit.md` for the full
decision and8s numerical limitations. No weights were changed or promoted.
One monitor lost its tool-cell identifier while the original process remained
active; process and artifact checks corrected the status. The attempted same-
directory relaunch exited before computation (FileExistsError); no duplicate
scientific experiment or rerun/revised threshold followed. Partial artifacts
are preserved alongside the completed output.
