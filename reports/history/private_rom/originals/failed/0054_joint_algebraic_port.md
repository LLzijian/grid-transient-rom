# Loop0054 rejected: algebraic output improves, rank-one latent loses closure

Baseline250d64c; preregistration4bcfd02 with pre-fit voltage representation
correctiond96fbcc; implementation03639ec. Scientific venv, seed53,1500 fixed
steps,178.07 s, Torch2.13.0+cpu/NumPy2.5.2, single thread. Command:
`experiments/train_joint_latent.py --output private_data/task2_joint_0054`.
Ruff/compileall/53 tests pass.346 TRAIN/107 validation; TEST never read.

The loader-only first smoke found signed polar voltages, before optimization.
A scoped data-quality check found13 development events with nonpositive raw
entries,3 eligible at100 ms: pilot_0254,pilot_0380 (TRAIN unstable, negative
roundoff near zero), expansion_0144 (validation stable,min-1.13483635).
Use physical magnitude abs(V), preserving G_L V^2 exactly; delete/reclassify
no events. Decoder voltage magnitude is abs(V*+linear deviation), not logV.
This representation correction was committed before formal training; no
acceptance threshold changed. No dataset infrastructure or notebook was added.

96 complete1 s validation prefixes:

| Metric | M3 | Full joint | Memory force removed |
|---|---:|---:|---:|
| Mean | .51755937 | .40957775 | .42457386 |
| Pooled | .66446164 | .53815833 | .55608614 |
| CVaR10 | 1.44633155 | 1.09928547 | 1.18100582 |

8/8 validation groups improve over M3. Memory provides3.53% mean/3.22%
pooled gain over its fixed no-memory ablation. Observed-history electrical
event-balanced normalized MSE Pe/|V|/Pload =1.36349/.190536/.0492521 versus
equilibrium-only1.67878/1.49443/1.42473. These auxiliary diagnostics do not
use future information in rollout and do not establish ODE closure.

Reject: first-second TRAIN q variance=.06658194 below.1; mean=-.35319801
passes centering. Autonomous/encoder q RMS=.64640230 exceeds std=.258035,
and pooled is3.18% worse than0053, beyond the2% preservation limit. Other
gates pass, including12 Xi terms,6 algebraic terms and numerical discrepancy
6.29236e-6. Short unstable-prefix mean1.04443,pooled1.83777 remain worse
than0053 though better than M3. No8 s classification or final M4 claim.

A TRAIN-only follow-up checks the actual training sampler (half decision,
half uniform endpoints through7 s): q mean=-.17348065,var=.18566492. Thus
the near-decision variance failure is not whole-trajectory collapse; the
population difference limits that interpretation. It does not reverse the
preregistered first-second gate or the independent closure/prediction failures.

Keep immutable model.pt,normalization.npz,trajectories.npz,evidence.json in
private_data/task2_joint_0054; independent saved-array arithmetic reproduces
all1 s and electrical metrics to1e-12. Revert implementation/preregistration.
Next: TASK2_after_loop_52 Loop C, jointly retrain r=2 from scratch with the
same100 ms inputs and algebraic port, to test whether distinct electrical
spatial/dynamic components require more than one coordinate. No PCA ordering.
