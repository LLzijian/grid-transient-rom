# Loop0091 — frequency-affine signature of the cached feedback discrepancy

Starta9e5a38, authority after75. Diagnostic; promotion none.0090 remains closed.
Question: can a real angle/speed-like frequency-affine correction account for
most of G3000's sampled normalized endogenous-torque discrepancy?
Hypothesis: captured discrepancy energy>=.80 both in the full fit and pooled
leave-one-frequency-out (LOO). This is not a new ODE or a model promotion gate.

## Frozen data, fit and score

Use ONLY validated0090 arrays T_F,T_G,mass,frequencies at ALL13 registered
frequencies, unchanged order. Define omega_f=2*pi*f and

  Z_f=diag(S_delta/(M*S_omega))*(T_F(f)-T_G(f)),
  Zhat_f=U+i*omega_f*V,

with U,V REAL10x10, U*1=0. NO constraint1'*U=0, no symmetry/positivity on
U/V, no projection of Z or scored predictions/residuals; V retains common
speed. S_delta=.47675314588683515, S_omega=1.0033667249602924.

Equal frequency/Frobenius weighting. The exact constrained minimizer is

  P0=I-11'/10,
  U=mean_f(Re Z_f)*P0,
  V=sum_f omega_f*Im Z_f / sum_f omega_f^2.

ONE full13 fit and exactly13 LOO fits, each fitted on the other12 indices.
Full capture=1-sum_f||Z_f-Zhat_f||F^2/sum_f||Z_f||F^2.
Pooled LOO capture=1-sum_f||Z_f-(U_-f+i*omega_f*V_-f)||F^2 /
sum_f||Z_f||F^2. Do NOT average per-frequency capture or clip negative values.
Both captures must be>=.80. The GLOBAL discrepancy Frobenius NORM (not square)
must exceed1e-12; otherwise inconclusive. No frequency/channel deletion.

Report per-frequency discrepancy/full-residual/held-out-residual norms to
expose concentration; all190 real spatial coefficients (90 U+100 V), all14
fit parameters and predictions retained. No frequency-degree, ridge, symmetry,
weight, selected-subset or cross-validation scheme search.

## Validity and finite budget

Anchors in private_data/task2_harmonic_reaction_0090:
arrays.npz a997503d4c935291ba9330328e5cc19215aa94f9b35090740bb9ebb3bb4d368c;
evidence.json e94429dac4b333ec51ee267add7b574b444c2acd545a26b5b8328f45c98c6f2a.
Independent0090 report SHA85ef82af4921e50ab33177ef496613bc574eeaa1eab6686c87748867b869d836.
Verify main completed,26 solves/26 SVDs/all numerical QA; independent pass,
26 solves/0 SVDs and matching current hashes; all prior source paths match
before/after. No model deserialization, descriptor solve/eigensolve, rollout,
H5/VAL/TEST or future data. Store source/code hashes before/after.

All fits require finite real U/V, U*1=0 maxrelative<=1e-10. Constrained normal
equations (U-mean(Re Z))*P0=0 and (sum omega^2)*V-sum omega*Im Z=0,
maxrelative<=1e-10. Relative max uses maxabs/max(1,maxabs relevant operands).
Each training objective must not exceed zero-correction objective by more
than1e-10*max(1,zero objective). Finite coefficients/captures/residuals; no
claims of convergence from merely finite arithmetic: this problem has an
explicit global constrained least-squares solution.

ONE main14-fit invocation,60s; ONE independent14-fit cached check,60s.
Aggregate120s INCLUDING source-hash metadata call1.5360985s. No real retry or
new branch/form; one coherent preexecution repair if needed. Synthetic QA is
separately counted. Close/review at completion/cap. Fresh output
private_data/task2_affine_signature_0091, refuse overwrite.

Synthetic tests: known real U/V with nonzero common-speed response; U*1=0
without imposing1'*U=0; a nonrepresentable frequency component; exact LOO
index isolation; missing/nonfinite input and zero denominator failures.
Root review before actual fit. Independent arithmetic computes parameters,
normal equations, predictions, per-frequency objectives and full/LOO captures
with different code (no production fit helper), agreement maxrelative<=1e-10
for arrays and absolute<=1e-10 for scalar metrics; exact fit/index counts.

## Independent review disposition and interpretation

Root adopts the review: right-project ONLY U; retain common-channel error,
angular-frequency units, global pooled denominator and negative capture.
LOO is mostly interpolation among nearby previously inspected development
frequencies, NOT independent physical confirmation. Report concentration.
Dynamic hidden feedback can look affine over a finite band, so a pass does
not identify an instantaneous/memory split or prove memory unimportant.

These190 free spatial coefficients may be unattainable by the existing24-
coefficient structured model; they need not be passive, reciprocal or stable.
They are an exploratory response signature, not permission to deploy dense
forcing, add a loss/state/library, or select a new model. Any implementation
or physical-supervision proposal requires its own legal-structure/trajectory
and numerical tests. A failure rejects only this sampled affine signature,
not the model family or all low-order memory realizations.

## Preexecution implementation review

Root identified a provenance-key collision: registering0091 code under `code`
would overwrite inherited0090 `code`. Repair keeps the old key and adds
`code91`; the scientific source chain remains strictly checked. The same
coherent preexecution repair adds final deadline/finite-score checks and
preserves completed partial fits on failure. No real fit preceded this repair.
Normal-equation denominators use the relevant operands as registered, not
the residual itself. All repairs are implementation QA, not gate changes.
