# Loop0097 — persistent complex directional modal imbalance

Work type:diagnostic. Result:supported. Model promotion:none.
Preregistration bc223cd; implementation86b5aa3 (preexecution norm clarification
and synthetic repairs included). Three frozen operators,13 cached FOM
eigenpairs each; no fit, eigenproblem, new model, simulation or TEST access.

The registered0096 hypothesis is supported: **13/13 eta>.10**, exceeding
the required7/13. Its eta ranges .135375593–.946746623. The originalG3000
also exceeds.10 on13/13; failed0093 on12/13. These controls are descriptive,
not selection of a new winner or revision of prior rejected deployment gates.

| Target | eta G3000 | eta0093 | eta0096 | sigma_min(K96) | sigma_max(K96) | condition(K96) |
|---:|---:|---:|---:|---:|---:|---:|
| 62 | .5367648312 | .3908534784 | .5492883555 | 13.02751783 | 59.75988786 | 4.587204456 |
| 64 | .1790671779 | .1050263105 | .1353755929 | 8.482965811 | 55.23980257 | 6.511850195 |
| 66 | .4163368521 | .2010775979 | .2933041197 | 5.161264420 | 51.66456743 | 10.01006018 |
| 68 | .2289262687 | .0862746460 | .1849536121 | 2.107665016 | 46.61567132 | 22.11721074 |
| 71 | .4076902547 | .1339953848 | .2062391806 | 2.908973147 | 40.42328201 | 13.89606571 |
| 73 | .2179496896 | .1307483021 | .2269122530 | 2.014496035 | 34.37248654 | 17.06257344 |
| 77 | .2546204590 | .1471326348 | .2346276297 | 2.071804316 | 30.62227532 | 14.78048630 |
| 79 | .1772024391 | .1050168089 | .1657855418 | .6388836571 | 29.01356089 | 45.41290196 |
| 84 | .2248579180 | .3927358687 | .2620640841 | .8272165399 | 34.58340517 | 41.80695561 |
| 91 | .5892204884 | .6963986248 | .6188341574 | .1220307331 | 38.23120323 | 313.2915968 |
| 98 | .9481828385 | .9464191342 | .9393995351 | .2447921941 | 39.57094923 | 161.6511889 |
| 100 | .9243348194 | .9187438758 | .9126292890 | .2596897872 | 38.57852383 | 148.5561841 |
| 104 | .9475036635 | .9454366357 | .9467466230 | .2848285796 | 38.82726387 | 136.3180055 |

K is the registered mass/unit-normalized reaction matrix, not a Jacobian.
All full singular spectra, other-arm conditions, absolute defect and inertial/
force norms, scale vectors and individual solve metrics are in arrays/evidence.
Raw10delta and fullomega retained, including tied common-frequency motion;
COI appears only in the eigenvector physical-output normalization.

## Numerical evidence

All39 hidden solves pass. Maximum original/equilibrated backward errors
9.58114e-17/7.79693e-17; equilibrated hidden condition6.03710 and rho1.74266e-14.
Minimum normalized balance denominator3.29368, above1e-12. Cached FOM
eigenpair/kinematic/generator-row maxima7.07325e-17/4.68881e-14/3.71761e-13.
Source matrices, target IDs, eigenvalue/vector alignment, physical outputs,
mass, units, common-angle identity and before/after source hashes all pass.

Independent39 RAW complex solves reproduce primary arrays to6.04286e-15
maxrelative and scalar metrics to5.32907e-15 maxabsolute. Raw backward
maximum8.18719e-17. Saved singular spectra/ratios checked, NOT independently
recomputed by SVD. No independent eig/modelconstruction/simulation/fit.

## Interpretation and stage closure

0096's .571587392 imaginary-frequency discrepancy capture is compatible with
substantial directional defects at all13 actual complex FOM eigenpairs.
Thus this point's harmonic improvement does not establish modal balance.
This does not prove the cause of autonomous error, pole-distance bounds,
insufficient latent order, model-family failure or adequate optimization.
0095 remains inconclusive on its optimal-status gate;0093/0096 remain rejected.
No candidate/accepted/default change and no implementation rollback required:
this loop only adds a diagnostic and preserved evidence.

Main3.077982s + independent1.6495104s + metadata10s =14.7274924s <=200s.
Exactly39main solves,39hiddenSVD,39reactionSVD and39independent solves;
one coherent preexecution repair, no actualretry or parameter/target adjustment.
Nine main/reused solver synthetics pass.33s; five independent synthetics pass.030s.
Stage closed at result. Independent route objections on conditioning, phase
invariance and claim limits were adopted; rawcommon-motion projection rejected
for the physical reason recorded in the preregistration.

**PAUSED at the user's request after this loop. No Loop0098 launched or
preregistered.** On explicit resumption, the next defensible question is whether
the same physical k/d directions can reduce complex directional balance under
fixed baseline weights and TRAIN compatibility constraints; any such fit and
subsequent autonomous/modal screen require a separate finite preregistration.
This is a proposed question only, not a result or authorization to run while paused.
AcceptedM3/0058/S unchanged; no final L1/L2/L3/M4 promotion. FreshTEST unopened;
consumedPilotTEST remains excluded. Broader roadmap is not complete.

## Reproduction and identities

Commands (designated project venv):
`python experiments/audit_complex_balance_0097.py --output private_data/task2_complex_balance_0097`
and `python private_data/task2_complex_balance_0097_independent/check_complex.py --input-dir private_data/task2_complex_balance_0097 --output-json private_data/task2_complex_balance_0097_independent/independent.json`.
Both refuse overwriting existing evidence; do not rerun into these directories.

- arrays SHA256: `8626c3ec9977561eb7c3a19e16111dea853fe93a830f109d8379c54d6ce8199f`
- evidence: `6c8dbd27516678027eed258fbeef5aeb07f9a88cd05d7a6b3beec9374eeff8ad`
- independent: `6c88e476a446db6b5ff3afe0a6e224a5d69155c588f04fd63424cfd8555717a3`
- checker: `aa5e9c3b8caa5cb8df07556cdbefc80d9220fe86df3b91de9e168c29a16b3c6e`

Full inherited source/candidate/accepted-checkpoint identities are preserved
in both JSON artifacts; no unrelated documentation edits were staged.
