# Loop0036: conservative cohesive-energy bound and numerical boundary audit

Preregistered before implementation, checkpoint834b3ed; frozen M3 only. Claim:
a computable sufficient reduced-model energy region exists, while its coverage
and conservatism must be measured rather than equated with the full-order basin.

Fix gamma=7/5rad without tuning. For26 positive edges define cohesive polytope
|eta_star+B^T delta|<=gamma, fixing generator10 angle. On this convex set,
HessianU >= c*L_K, where the exact rational Taylor bound
c=1-gamma^2/2+gamma^4/24-gamma^6/720 is positive and <=cos(gamma).
With exact binary-float-as-rational K and delta_star, compute L_K inverse by
Fraction Gaussian elimination, and R_e=b_e^T L_K^-1 b_e. Then
Hsafe=(c/2)*min_e[(gamma-|eta_star_e|)^2/R_e] is an analytic lower bound
on the potential energy of every polytope boundary facet. Round its reported
float downward and verify this direction using exact fractions.

The rigorous statement is conditional: initial state inside this polytope and
exact H3<Hsafe cannot exit it under the exact reduced ODE. Compact sublevels,
positive M/Bm/Am and LaSalle (p=0 then omega=0 then E=0) give convergence modulo
angle gauge. This is neither the true critical energy nor a FOM certificate.
Measured event energies computed in floating point remain numerical membership
checks, not interval-certified classifications. No optimizer result is a bound.

Gates: exact inverse residual zero; c>0 and eta_star strictly inside; Hsafe>0;
downward rounding verified; actual float inverse comparison<1e-10; no observed
cohesive exit in numerical points satisfying H<=.99Hsafe and interior by1e-8;
energy identity and48 existing tests continue to pass. Report all development
event margins Hsafe-H and normalized margins, interior status and label/group
strata; numerical ROC/AUC and within-group duration monotonicity are descriptive.

Numerical basin probe: one maximal-initial-H TRAIN direction per15busgroups
(labels not used), plus17 deterministic random29-state directions(seed36).
Normalize directions by local quadratic energy to Hsafe; amplitudes
.25,.5,1,2,4,8, fixed60s RK4maxstep.005. Retain all192 initial states, energies,
cohesive exit, physical LOS and final energy. Finite60s survival is not a proof
of membership in the true ROA. Check first3 near-Hsafe points with DOP853 to
maxrelative state error<1e-4. No ray/threshold selection from validation or test.
Do not expand scans to rescue a conservative bound; quantify limited coverage.
