# Loop0036 accepted: conservative reduced-model bound with limited coverage

Checkpoint834b3ed; frozen implementation6c3601b. Command:
`python experiments/audit_private_energy_boundary.py --output
private_data/task2_energy_boundary_0036`. All48 tests, syntax and Ruff pass.

For the fixed gamma=1.4rad active-edge cohesive polytope, the exact-rational
quadratic comparison gives the downward-rounded bound
**Hsafe=0.09151303931750955**. The exact inverse residual is zero, all exact
pivots/resistances and cosine lower bound are positive, and outward direction
of the reported bound is checked using exact Fraction arithmetic. Exact bound
numerator/denominator and all26 edge calculations are preserved in evidence.

Proof: along the segment from equilibrium to any interior point,
HessianU>=cL_K, hence U>=c delta^T L_K delta/2. Every boundary facet requires
|b_e^T delta|>=gamma-|eta_star_e|; minimizing this quadratic under that linear
constraint gives the resistance expression. Thus no boundary point can have
H<Hsafe. Positive mechanical/kinetic storage and nonpositive Hdot prevent escape
from a compact sublevel. The zero-dissipation invariant set has p=omega=0 and
E(delta)=0; strict convexity makes equilibrium unique modulo angle gauge.
This is an analytic sufficient bound for the exact reduced ODE, **not its true
critical energy and not a full-order certificate**. Floating-point membership
tests on measured states are numerical, not interval-certified guarantees.

Coverage is very limited: only3/218 TRAIN and0/70 validation initial states
meet the conservative numerical membership check. All3 are physically stable;
zero validation coverage cannot be described as validated safety classification.
Initial-H ranking remains descriptive AUC .95826TRAIN/.95446validation; margin
decreases with fault duration (Spearman -.83833/-.92691), including each group.
Monotone ranking is not boundary calibration or a proof for the full-order data.

The fixed192-ray-point60s scan finds no cohesive exit among sufficient initial
points. Three near-bound points match DOP853 with maxrelative state error
8.71e-6 (<1e-4). All probe outcomes, exits, LOS and energies are retained,
including outside-bound cases. Finite-horizon survival is not a full ROA map.
No thresholds, rays or model parameters were adjusted after these results.

This closes the conservative-bound hypothesis, not the entire research task.
An enlarged bound would be a separate scientific hypothesis, not a relabeling
of Hsafe as Hcrit. Before any final test read, freeze the exact predictor,
initialization policy, all evaluation metrics and the reduced-only interpretation
of both energy and spectral margins. The frozen test remains untouched.
