# Loop0076 — minimal shared-gradient prototype (preregistered)

Authority: user resumption under `docs/TASK2_after_loop_75.md`,2026-09-21.
Start0faf3b3; S/0058/M3 and all datasets remain immutable. This is stage A,
an implementation/expressivity test, NOT an IEEE39 model promotion or fit.

Question: can one small explicit generator preserve kinematics and a correct
power account, include M3 exactly, and represent both damped recovery/slip
and hidden-state exchange with nonconservative power? Hypothesis: the single
template passes all identities, export checks and two fixed synthetic cases.

State internally x=(delta,pi,p,q), pi=M*omega, rank3 for IEEE39; interface
state=(delta,omega,p,q). Only common angle is a gauge. Known M is fixed.
H=.5*pi^T M^-1*pi + U(delta) + .5*sum(a_s*s^2) + sum gamma*psi,
s=(p,q); U=sum k*(cos(phi)-cos(phi+eta)-sin(phi)*eta), eta=delta*D.
Initial a_p=1/gain,a_q=1, k=M3 coupling; phi=delta_star*D. Refit of
empirical coefficients is allowed later, not executed in this prototype.
All quantities/gradients use float64 and explicit normalization chain rules.

Minimal grammar: centered sin/cos of one eta; .5*(s_i/scale_i)^2;
(s_i/scale_i)*(s_j/scale_j); (s_i/scale_i)*(sin(phi_e+eta_e)-sin(phi_e))
or the analogous cosine difference. Center standalone sin/cos by subtracting
value and linear term at0. No pi in h. Analytic value/gradient share each
coefficient; compare against autograd/finite differences independently.
Only index lists for active primitives, not a full combinatorial library.

Canonical J_delta,pi=I. All remaining groups act only on v=(pi,p,q).
M3 initialization J_pi,p=diag(gain), R_pi,pi=diag(damping),
R_pp=diag(rate*gain), q decoupled. Rqq may be zero. Extra exchange groups
kappa*(a*b^T-b*a^T), dissipation rho*a*a^T with unit nonzero a,rho>=0.
Use positive parameterization for trainable nonnegative coefficients with
separate exact-zero masks for initially absent terms. No squared zero factor
as the activation mechanism. Port columns b are unit nonzero directions,
rank<=2, u=sum<=8 centered linear-state/sin/cos terms per column; u(0)=0.
All angle rows of extras are zero; e=grad H, xdot=(J-R)e+B*u.
External omega derivative divides momentum derivative by M. Export includes
numeric H primitives, scales, J/R/B/port terms, explicit field evaluation
and independent-parameter/expanded-RHS counts; no registry/schema framework.

Budget: two modules (`structure_grammar.py`,`structured_rom.py`), targeted
tests and one small evidence runner. No model training, support search,
VAL/TEST access or FOM event data in0076. S/M3 model assets may be read.
Optional GRU wrapper reuses20->8->3, eleven history points, COI delta/full
omega input; same differentiable origin anchoring during fit and deployment,
not a post-fit RHS bias subtraction. No old Xi RHS in parallel.

Frozen tests: seed76,64 random float64 states with nonunit masses/scales and
nonzero groups. Require max normalized identity errors<=1e-10 for kinematics,
angle shift, f(0), skew exchange power, dissipation nonpositivity, total
Hdot=-e^T R e+(B^T e)^T u, M3 physical field, and export/reload field.
Analytic gradient vs autograd<=1e-10; central FD steps1e-5/1e-6 relative
error<=1e-6. Nonzero finite training gradients through all active parameter
families and encoder. Removing one exchange group removes both paired terms.

Synthetic case1: two rotors M=(1,1), one angle difference, phi0,k1,
damping(.2,.2), p0=0 and J_pi,p=Rpp=0, no hidden feedback/port. Recover from
relative angle .2/relative speed0; a second initial relative speed6 must
cross pi (slip). RK4 .01/.005, horizon10s: recovery final relative angle
and speed magnitudes<.15; slip crosses pi; no claim of perpetual rotation.
Synthetic case2: same rotors, one q, H adds .5*q^2, exchange between pi1
and q with coefficient .3, Rqq=.1, port b=unit pi1,u=.05*q. Initial
(delta1,delta2,omega1,omega2,p1,p2,q)=(.1,-.1,0,0,0,0,.4). Fixed5s
RK4 .01/.005: all finite, max shared-grid state difference<=1e-4,
nonzero exchange/port work magnitude>1e-6, instantaneous power identity
as above. This known-template case tests implementation, not discovery.

Accept only as prototype QA if every gate passes; failures are implementation
or template diagnostics, not rejection of all energy models. Next: separately
preregister matched C-free/G-fixed/G-auto training on the original TRAIN.
Boundary probe0077 is independent and cannot gate this prototype's progress.
