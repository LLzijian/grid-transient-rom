# Loop0092 — existing backbone span explains a majority

Work type: diagnostic. Result: supported. Model promotion: none.
Prereg e965391, implementation59dcfb9;
`reports/task2_backbone_span_0092.md`.0091 remains rejected.

One unconstrained36-coefficient scaled-SVD fit explains
.6864836160164514 of the full13-frequency normalized discrepancy energy,
passing this distinct majority>=.50 gate. Objective23717.220427300068 versus
zero75649.06218280636; rank36, condition2.596481436636289, singular extrema
.7071067811865472/1.8359896310705084; normal-equation relative1.30238e-15.
Six exported nonlinear RHS evaluations verify the derived force change to
2.87964e-16 maxrelative. All existing26 supported couplings plus10 diagonal
damping changes are retained, with no frequency/channel removal.

The resulting actual amplitudes happen to be strictly positive:26 couplings
.7581431537–6.9690170179 and10 damping amplitudes.2401814387–.5749938828;
no constrained refit is needed merely to satisfy these coefficient signs.
This does not establish positive storage curvature, local stability, good
large-disturbance trajectories or modal fidelity. Full coefficient/spectrum/
per-frequency arrays remain in the artifacts rather than duplicated here.

Independent separate-basis QR fit passes on its first invocation, no errors:
arrayrelative1.59630e-15, scalarabsolute7.10543e-15. Main1.499523s,
independent.1328537s plus conservatively chargedmetadata10s =11.6323767s
within unchanged245s. Mainonefit/oneSVD/sixRHSevals; independentonefit/oneQR,
zeroSVD/RHSevals. Root synthetic checks5pass1.50s and7pass.123s;
one coherent preexecution implementation repair, no real retry.

Command: `python experiments/audit_backbone_span_0092.py --output private_data/task2_backbone_span_0092`.
Arrays SHA2b4c0725339c360ea2502e8059e03b8d4f4c9c96197a58701eccd47e26983a8f;
evidence SHAb5b9f7cc6f673fbd6dd4b5d59f6b3cb8d26361c68e4aad590141e72f3dc0a89b.
Independent `private_data/task2_backbone_span_0092_independent/independent.json`
SHA29f1d15d9c797537ada125f41baee45642184d9c98e894987864c112c8fb6535.

Stage closed. No candidate/model/ODE changed, no trajectory/H5/VAL/TEST read.
Fresh frozenTEST remains unopened; accepted M3/0058/S checkpoints untouched.
The supported statement concerns this local clamped response span only, not
nonlinear causal attribution or mechanism discovery. A positive-amplitude
copy using these exact cached coefficients is the next smallest prospective
test: actual coupled dynamics first, then autonomousTRAIN screen if numerically
valid and locally stable. Independent route challenge pending; no new fit or
candidate execution authorized by this conclusion.
