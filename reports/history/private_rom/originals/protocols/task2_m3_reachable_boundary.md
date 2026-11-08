# Loop0037: reachable-direction numerical loss-of-synchronism boundaries

Preregistered before implementation; checkpoint37921ba. Loop0036's small
energy-normalized grid contains no LOS, so it cannot locate the actual physical
escape proxy. Hypothesis: along observed TRAIN directions a finite60s LOS
transition is bracketable and lies far outside the conservative Hsafe sublevel.
No model, energy formula or LOS threshold change.

Use same maximal-initial-H TRAIN event per15busgroups, but preserve the physical
state amplitude: x(rho)=equilibrium+rho*(x_event-equilibrium), remove only uniform
angle offset. Fixed rho grid0..2 step.125; all255 points integrated60s,
RK4maxstep.005 with the original LOS definition. Report nonmonotone label
sequences. For each direction with a stable-to-LOS adjacent bracket, refine the
first transition with12 bisections. Do not assume other parts of the ray are
monotone or call finite-time survival a global ROA proof.

Numerical acceptance: finite trajectories; H decreases (max positive step
relative<=1e-6); no point inside the certified sublevel/cohesive set exits it;
at leastone LOS bracket exists (otherwise hypothesis fails, report no boundary).
Recheck first3 bracket pairs with DOP853 (rtol1e-10/atol1e-11), requiring same
LOS labels. Grid and midpoint initial states, H values and decisions are all
retained. Report interval widths, Hsafe-to-transition-H ratios and differences
between bound, cohesive exit, numerical LOS and asymptotic attraction concepts.
No validation/test direction selection, parameter fitting, or deployment gate
based on fault bus. Synthetic radial points are not claimed to be full-order
reachable; only their directions originate from observed TRAIN clearing states.
