# Loop0028: frozen M3 storage/dissipation baseline

Preregistered under TASK2_after_loop_26 §12. No model refit, no test payload,
no critical energy or basin certificate. Accepted M3 gain b>0 permits
`H=.5*w.T*M*w+sum K*(cos(eta*)-cos(eta)-sin(eta*)*(eta-eta*))+.5*sum p²/b`.
Its exact ODE derivative is `-sum D*w²-sum (a/b)*p²<=0`.

Hypothesis: the implemented frozen M3 exactly realizes this identity and has a
local storage minimum modulo common-angle gauge; verify rather than infer from
prediction accuracy. Gates: analytic gradient versus central difference<=1e-6
relative, gradient·field versus dissipation<=1e-10 relative; all gains/masses
positive, damping nonnegative; gauge-fixed potential Hessian positive. Integrate
fixed8s for all288 prospective development initial states, score every case.
On stable-truth initial states, RK4 H growth tolerance1e-6*max(1,H0); independent
DOP853 and half-step checks on three largest-energy stable cases distinguish
integration error from an ODE defect. Unstable initial conditions not removed.

Evaluate H on original full-order genuine prefixes separately from reduced-model
rollouts. Report net change, upward variation, quadrature balance residual,
stable/unstable and group strata, rank association with severity, and initial-H
stable/unstable separation (diagnostic AUC, no tuned threshold). Full-order H is
not guaranteed monotone because eliminated electrical/control storage is absent;
do not reject an exact reduced-ODE identity merely for real hidden energy supply.

Add only direct energy/gradient/dissipation methods to the existing M3 class and
one numerical script/tests. No PH, ROA, schema or reporting framework. Physical
units: effective Pm is ANDES tm nominal-frequency torque-power, as documented.
