# Loop0071 preregistration — physical-output modal baseline

Authority after66 L3; starting checkpoint2ba0a83. Parallel read-only local
diagnostic while0070 fits; no new training candidate or change to0070.
S remains the current provisional long-horizon reference. This is a cheap
alignment screen, not the final model's deeper physical interpretation.

Question: does S already reproduce a majority of the verified FOM oscillatory
electromechanical modes, beyond merely having stable own-system poles?
Hypothesis/screen: at least7 of the13 modes elected in0035b admit distinct,
physically similar S partners under the fixed conditions below. M3 provides
a descriptive baseline. Failure rejects this modal-fidelity claim only,
not local stability, prediction validity or the latent dimension.

Sources: `private_data/task2_fullorder_descriptor_0035/descriptor.npz` and
its provenance; `task2_verified_modes_0035b/evidence.json` and M3 matrix;
frozen S, M3 and `task2_coupled_equilibrium_0065/coupled_equilibrium.npz`.
Hash all these concrete files before/after. Do NOT use the rejected160-mode
algebraically substituted matrix as the FOM operator. No H5/TEST payload,
new ANDES simulation, fit, threshold tuning or model modification.

Recompute generalized left/right eigenvectors from the original699-state
pencil A=[[fx,fy],[gx,gy]], E=diag(Tf,zeros479), retaining finite pairs by
the existing0035b relative homogeneous cutoff1e-10. Require150 finite,
549 infinite, no0/0 pairs, eigen residual<=1e-8; match recomputed finite
poles one-to-one to all150 verified poles within relative1e-6. Keep the SAME
13 positive-imaginary/identifiable/EM-participation>=.1 targets from0035b,
not a new hand-picked frequency range. Unreliable individual eigenvectors
(existing left-E-right overlap<1e-12) cannot support a match claim.

Verify name mapping exactly delta GENROU1..10 and omega GENROU1..10, raw
delta rad and omega absolute pu. Check delta-dot/omega Jacobian conversion
equals omega_s=2*pi*60 and inertia Tf[omega]=M3.mass*omega_s (relative1e-10).
The prior case SHA/equilibrium identity is preserved, not re-exported.

Use common physical output y=[Pcoi*delta/s_delta, FULL omega_rad_s/s_omega],
Pcoi=I-1*m'/sum(m), s_delta=.47675314588683515,
s_omega=1.0033667249602924. Keep common frequency; remove only common angle.
For FOM multiply omega perturbations by omega_s; for S/M3 use the gauge
embedding into their original physical states. Evaluate each mode's actual
complex output vector, not latent covariance or component labels.

MAC=abs(v_f^H*v_r)^2/(norm(v_f)^2*norm(v_r)^2). A possible pairing requires
relative frequency difference<=.15 (FOM denominator), absolute damping-ratio
difference<=.05, MAC>=.8. Define d_f=relative frequency difference/.15,
d_z=absolute damping difference/.05, d_m=(1-MAC)/.2. On eligible edges use
cost=(d_f^2+d_z^2+d_m^2)/1000, invalid edges1e6,13 private dummy columns cost1.
Hungarian assignment thus maximizes eligible one-to-one cardinality before
quality; no many-to-one rescue or mere nearest-frequency claims. Report ALL
pair metrics and explicit unmatched FOM/ROM modes, matched counts for S/M3.
Exclude only nonoscillatory ROM poles (imag<=1e-6), with their full spectrum
still retained; null physical-output modes cannot count as matched.

Before execution: synthetic complex phase/scale MAC invariance, one-to-one
assignment and unit/gauge mapping tests; root code review. If the integrity
checks fail, classify implementation/source inconsistency, not science.
If S has<7 qualifying matches, close the claim rejected and preserve the
specific frequency/damping/shape limitations. If it passes, retain only a
screen result: small-perturbation response confirmation and deeper physical
interpretation remain reserved for the final shortlisted model. No stability
margin transfer, energy certificate or final L1/L2/L3 completion claim.
