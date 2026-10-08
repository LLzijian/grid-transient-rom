# Loop0078 fixed1200 endpoint — intermediate evidence

Authority: `reports/task2_structure_campaign_0078.md`; after75 stage B.
This is not campaign closure: the preregistered TRAIN-selected extensions
are running. Accepted M3/0058/S remain unchanged; no TEST access.

## Question and decision

At the matched1200 endpoint, does G-auto retain S's required behavior and
improve complete5s pooled error by at least5% relative to C-free?

**No. The fixed1200 screen is rejected.** G-auto/C-free complete5s pooled
ratio is5.112943, not <=.95. G-auto also fails every complete-horizon
S-retention metric, both severe-prefix metrics, common5 and false-LOS gate.
G-fixed is a control, not an alternative way to pass the G-auto hypothesis.
Neither G arm is ready for confirmation. C improves longer complete
predictions but fails severe-prefix retention, so it is not promoted.

## Evidence

All attempt2 arms use source implementation6382d3b, seed78, original346
eligible TRAIN events and fixed1200 updates. All independent integrity
checks pass. The quarantined attempt1 duplicate-dissipation implementation
failure is documented separately; none of its endpoints is used here.

| Quantity | S reference | C-free | G-fixed | G-auto |
|---|---:|---:|---:|---:|
| Complete1s pooled | .373210293 | .377863560 | .600025518 | .657602389 |
| Complete2s pooled | .381051143 | .349957871 | .716678609 | .711171818 |
| Complete5s pooled | .333570888 | .282547673 | 1.744960587 | 1.444650192 |
| Complete5s CVaR10 | .628207978 | .583870374 | 3.138320568 | 2.339845224 |
| Unstable-prefix pooled | 1.230200075 | 1.335763902 | 1.466331361 | 1.487175278 |
| Unstable-prefix CVaR10 | 1.667156312 | 1.878880062 | 1.984980149 | 2.015455415 |
| Common5 RMS, rad/s | .452087182 | .315613965 | .681030128 | .660864805 |
| Eligible unstable detected | 10/12 | 10/12 | 9/12 | 10/12 |
| False LOS | 0 | 0 | 2 | 1 |
| False window-settled | 2 | 2 | 3 | 2 |

Validation retains117 raw/107 eligible/10 latency exclusions,22 raw unstable,
and complete1/2/5 cohorts96/95/95. All new arms pass the registered scored
numerical checks; no finer diagnostic was triggered. Reporting forecasts
are always dt=.005. Independent aggregate differences are <=4.45e-16;
physical reclassification agrees for all107 events in each arm. G-auto's
false LOS is pilot_0150; G-fixed adds pilot_0056. G-fixed misses unstable
expansion_0155/0186/0189. There are no numerically unresolved L1 events.

Final effective dynamics-coordinate upper bounds are141/143/141 for C/Gfixed/
Gauto; these are not proofs of parameter identifiability. C retains34 Xi and
18 alpha coefficients. G-fixed retains H8/J4/R2/ports2; G-auto H7/J4/R1/ports2.
Shared observer/decoder trainable scalars1262 are counted separately.
All saved source hashes and frozen state checks pass. RNG-state replay and
cross-arm checkpoint consistency support deterministic matched sampling,
but no historical per-step event-ID log exists; that limitation remains.

## TRAIN-only extension decision

| Arm | Risk0 | Risk800 | Risk1200 | Extension |
|---|---:|---:|---:|---|
| C-free | .771459087 | .468692254 | .434825174 | selected |
| G-fixed | 17.359303105 | 3.840218133 | 2.927119378 | selected |
| G-auto | 17.365617946 | 14.479518752 | 4.071026581 | not selected |

All three satisfy the original >=10% total and >=2% last400-step TRAIN
improvement prerequisites. The two lowest1200 risks select C and G-fixed,
as recorded in b0172b8 before their extension launch. G-auto is not extended
even though its validation5s error is lower than G-fixed's. Validation
metrics do not allocate budgets. Resume model, Adam and RNG trees are
bitwise identical to the corresponding1200 checkpoints in both arms.

Continue once to3000 with frozen support/zero complexity penalty and no
growth, pruning, new loss or threshold changes. A later G-fixed result
cannot retroactively rescue G-auto's failed primary hypothesis. Large TRAIN
improvement and substantially different initial risks mean these finite
budget results alone do not falsify the entire structured model family.

## Exploratory local diagnostic (not a selection gate)

To check whether the false LOS predictions merely reflect an unstable
origin, differentiate the full33-state deployed field at zero and project
with Q=block_diag(null_space(ones(1,10)),I23). This removes only common angle;
common frequency is retained. All32 quotient poles have negative real part:
maximum real parts C=-.2021824613, G-fixed=-.04383924614,
G-auto=-.02390339876 /s. f(0)=0 and gauge residuals<=2.85e-14.
Independent central differences at1e-5 and1e-6 agree with autodiff to
maxabs<=2.75e-9/relative Frobenius<=2.61e-11. Thus an unstable local
linearization is not the simple explanation of these G false LOS results.
This does not establish a nonlinear stability region, physical modal
fidelity, or transient forecast accuracy, and it changes no campaign gate.
Model-only script/matrices/evidence: `private_data/task2_structure_local0078`.

Further fixed5s TRAIN decomposition retains all346 events and uses the
same worst35 events selected by TOTAL point loss for every component. The
component risks sum to the original monitor risk (all six step0/1200
checkpoints reproduce within2.85e-14). Relative-angle error accounts for
about87.0% of each G initial5s risk, and66.4%/70.6% of final G-fixed/G-auto
risk. Mechanical-power shares at1200 are only10.8%/11.7%. Thus mechanical
power dominating the registered risk is not the simple explanation of the
large G-versus-C training gap. C1200 component shares are angle10.0%,
relative speed23.3%, common speed35.6%, mechanical power30.9%. This does
not authorize reweighting or identify which structural restriction causes
angle error. Script and per-event components:
`private_data/task2_structure_risk_components0078`. The inherited reader
materializes TRAIN/validation, but this computation uses TRAIN only; no TEST.

## Artifacts

- `private_data/task2_structure_0078_attempt2_{C-free,G-fixed,G-auto}`:
  training evidence, checkpoints and `independent_integrity.json`.
- Corresponding `_eval` directories: fixed forecasts, per-event evidence,
  and `independent_evaluation.json`.
- `private_data/task2_structure_0078_attempt2_screen.json`: exact gate ratios.
- `private_data/task2_structure_0078_attempt2_sampling/replay_audit.json`:
  bounded sampler replay audit, with available-checkpoint limitations.
- `private_data/task2_structure_0078_extend_{C-free,G-fixed}`: active extensions.
