# Loop0067 preregistration — joint long-horizon closure campaign

Starting a6a37b7; authority `docs/TASK2_after_loop_66.md`. Resume authorized
2026-09-14; the prior pause is superseded. No historical result is rejudged.

## Question / allowed change

Does reopening the existing representation under a common-motion-aware
physical objective improve long-horizon prediction or instability detection
over a matched Xi-only fit? Source S is `private_data/task2_long_0060_5s`
(SHA d34d9f5c36fb296e1d49e59f5f7b18a5da33b56bf371a16169f82bc25664995c).
0058 and M3 remain references. C changes only active Xi; J changes GRU/head,
c/amplitude, active alpha/Xi and decoder. This complete joint-training
algorithm is the single comparison; no per-loss ablation chain.
All buffers, physical units/M3,100ms window, architecture,r3,normalization,
q_center/q_transform, ports, masks/library remain bitwise frozen. No whitening,
constant drift, state clipping, new terms or post-fit initialization shift.

## Fixed training configuration

Both arms start independently from S:800 steps, seed67, fresh Adam lr1e-4,
batch32, one CPU thread. Steps1–100 use2s;101–800 cycle1/2/5s (first1s).
Only final800 is evaluated. No extension, early stopping or adaptive weight.
Each batch alternates stable/unstable labels (16 each), then uniformly samples
an available fault group within that label and an event within that group,
with replacement. Labels/groups never enter model inputs. C/J use identical
random draws. Local endpoint uniformly10..len(x)-2 over FULL TRAIN; local
rollout min(.2s,true remaining prefix). Long rollout starts at x[10], uses
11 genuine initial states once, and stops each event's loss at its true end.
Preserve sampled IDs/endpoints/horizons; no padded/frozen truth or exclusions.

For each observed point let
e=.5*(mean(COI(delta_error)^2)/s_delta^2+
mean(COI(omega_error)^2)/s_omega^2)+(mass_mean(omega_error)/s_omega)^2,
with existing s_delta=.47675314588683515,s_omega=1.0033667249602924.
Average within each event, then risk=.8*batch_mean+.2*worst-ceil(10%) mean.
Local loss uses the same point error, event averaged. Short latent consistency
is existing one-step integrated discrepancy divided by.1, both encoder sides
differentiable. Electrical terms use actual local force, Pe,V,PL truth:
F=mean(((memory(q+a)-force)/force_scale)^2),
Z=(normalized Pe MSE+V MSE+PL MSE)/3.
Total =10*risk+local+.1*latent_consistency+.1*(F+Z)
+1e-4*mean(abs(masked_Xi))+mean(E(zeros(11,30))^2).
All weights fixed here from physical scales/existing TRAIN conventions, not
validation tuning. Electrical gradients MUST be active for J; they are
constant diagnostics for C. No covariance/orthogonality penalty. No gradient
clipping. Nonfinite loss/gradient/parameters stops that fit, saves failure
checkpoint and traces; optimization failure is not family falsification.

## L1 baseline and frozen evaluation rule

Before fits, augment S/0058 existing validation forecasts with the original
LOS rule: absolute rotor-angle spread>pi AND max abs COI speed>.02pu,
5 consecutive .01s samples. Restore delta_star and omega/omega_s+1 exactly.
Carry the observed preforecast hold history; retain all raw TRAIN/validation
coverage denominators, including loss/no future by100ms as latency-uncovered.
Use no TEST payload. Genuine future scores remain separate from extrapolation.
Report detected LOS/time, no-LOS-in-window, numerical-unresolved and a separate
window-settled flag (last1s angle<pi and COI speed<.005pu). Settling is a
finite-window prediction, never an asymptotic or energy certificate.
Report true-unstable misses, false detections and false window-settled counts,
their denominators, timing errors, common-frequency RMSE and per-group counts.

Use existing1/2/5/8 and true-prefix benchmark. For LOS require finite33-state
integration and .005/.01 normalized step discrepancy<=1e-3 up to the later
of the two trigger endpoints (8s if either has none), same detected/nonhit
status and hit times within.02s. If needed exactly one .0025 comparison to
.005; unresolved cases stay explicit, not physical instability. Full-error
metrics use genuine truth only; no inference from missing high-order tails.
Compute time/speed baseline on same hardware and matched intervals; log GRU,
integration,total and throughput separately from100ms observation latency.
Historical whole-generation wall time is not a speed denominator. Any ANDES
replay/timing setup is verified on explicit TRAIN events before ratio claims.

## Prospective screen / interpretation

Prerequisites: causality/split/frozen checks, finite forward+gradient, f(0)=0
within1e-10, valid integration on all scored/claimed intervals. No new
numerically-unresolved events versus C or S may count as a safety improvement.
J must not increase eligible true-unstable misses, false LOS detections or
false window-settled counts versus either C or S. Additionally, J's1/2s pooled
and CVaR, unstable-prefix mean/pooled/CVaR, and5s common-frequency pooled
errors must be <=1.05 times both references (floor1e-8).
Given these, shortlist J if either (a)5s complete mean AND pooled improve
by>=5% versus C, CVaR<=1.05*C and all three<=1.05*S; or (b)at least2 fewer
true-unstable misses versus C, with5s mean/pooled/CVaR<=1.05 of C and S.
These are screening, not final promotion:8s/Pareto/speed and up to3 total
seeds of the shortlisted algorithm remain. J-S/B2/capacity probes require
their own bounded configuration; none is auto-triggered midfit. No freshTEST.

First preserve/reproduce the baseline, then C/J preflight, then fits; B1 may
proceed on frozen S while training runs under a separate short preregistration.
New-gradient/LOS/time-alignment tests, smoke, root review before scientific
runs; independent final metrics once afterward. Final closure <=one page,
details in existing per-event artifacts; no registry/framework or audit chain.
