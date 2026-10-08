# Loop0090 — original-descriptor harmonic reaction, without memory attribution

Start9d22c2f; authority after75. Diagnostic; promotion none.0088/0089 closed.
Question: does frozen original G3000 approximate the FOM's physically normalized
harmonic generator reaction on the13 fixed electromechanical frequencies?
Screen: aggregate relative Frobenius error<=.20 AND every frequency<=.50,
conditional on all source/numerical QA. This is a new diagnostic convention,
not an old gate, confirmatory modal test or permission to tune a model.

## Scientific correction before registration

Independent review rejected the initially proposed direct/memory attribution.
Metadata-only inspection found170 positive/50 zero time constants among220
nominal states, plus479 algebraics; minimum positive time constant .02s.
This agrees with0035's known PSS algebraic-state constraints and does not
reopen its verified150 finite/549 infinite descriptor spectrum. Naively
eliminating only zero-T variables is not a validated170/150-state reduction.
Use the ORIGINAL699 pencil; no inverse mass on all states, pseudoinverse,
regularization, controller disablement or substitute reduced FOM operator.

The root adopts the review's narrower question: total sampled harmonic
reaction only. No properness/steady-state/causal-memory/trajectory-error
attribution claim. Regularity of the full pencil does NOT prove well-posedness
after clamping generator motions; that is checked separately at each frequency.

## Exact physical equations

FOM A=[[fx,fy],[gx,gy]], E=diag(Tf,zeros479). Retained coordinates are FIRST
20: delta GENROU1..10 in rad and omega GENROU1..10 in absolute pu. h=remaining
679 coordinates, including algebraic and dynamic states. Verify these names,
delta-row A_delta,omega=Omega I, E_delta=I and all other delta-row terms=0;
Omega=2*pi*60, E_omega=M*Omega, with exact accepted M3 mass.

For each s=i*2*pi*f, where f is each of the SAME13 verified target frequencies
in ID order62,64,66,68,71,73,77,79,84,91,98,100,104, prescribe delta=d and
omega_pu=s*d/Omega. For all10 basis columns simultaneously solve

  H_F=s*diag(E_h)-A_hh,
  H_F X_F=A_h,delta+(s/Omega)*A_h,omega,
  T_F=A_omega,delta+(s/Omega)*A_omega,omega+A_omega,h X_F.

T is the endogenous normalized generator torque RHS per unit angle. The
externally imposed reaction needed to enforce this motion is

  R_F=diag(M)*s^2-T_F.

Use G3000 external33 AD matrix from0083 baseline ONLY; h=(p10,q3)13 states:

  H_G=s I13-A_hh,
  H_G X_G=A_h,delta+s A_h,omega,
  T_G=diag(M)*(A_omega,delta+s A_omega,omega+A_omega,h X_G),
  R_G=diag(M)*s^2-T_G.

Normalize reaction K=diag(1/(M*S_omega))*R*S_delta, S_delta=.47675314588683515,
S_omega=1.0033667249602924. These scales normalize operators, not a claim that
finite .477rad perturbations are linear. Preserve ALL10 angle input directions
and common frequency; do not project away uniform harmonic motion.
Aggregate error=sqrt(sum_f||K_G-K_F||F^2/sum_f||K_F||F^2), equal frequency
weight; per-frequency ratio analogous. Require denominator>1e-12 (each and
aggregate Frobenius norm), otherwise comparison inconclusive, no changed metric.
Also report endogenous torque operators and their errors descriptively: the
shared inertial reaction can mask relative feedback error. No separate gate or
replacement choice for these secondary quantities.

Report Pcoi output and common-frequency weighted-average output discrepancies
as descriptive absolute norms (Pcoi=I-1*m'/sum(m)); do not claim an additive
Euclidean attribution because Pcoi is generally oblique. No direct/memory split.

## Source and numerical validity

Exact descriptor SHA6648fca0340fbd0823b5f0d50e99f19ffa00ce6ee32bfb36002219938371c4da;
descriptor evidence37d82fb8d1c3af8bb5d4cd7266c14d04913b10aa7caec77ac8beeb29336435fa;
verified0035b evidence8a1d3e53a1b4aeba5ceb2f1c7017378dc3ceb51be64a372a3562da8b9a7970ef.
0083 matrices/evidence and G3000/model/source links inherit0089's fixed anchors
and passing checks. No source change accepted. Hash inputs/code before/after;
same full-order physical signature/equilibrium, exact150 finite target provenance.
No model loading, eigenvalue recomputation, H5/VAL/TEST or new simulation.

Each main solve uses complex128 with deterministic max-absolute row then column
equilibration: Dr=1/rowmax(H), Dc=1/colmax(Dr H), Heq=Dr H Dc, beq=Dr B;
solve Heq Y=beq once by scipy.linalg.solve, X=Dc Y. Zero scaling rows/columns,
LinAlgWarning/error, nonfinite result or failed check stop the affected work;
no fallback/refinement/retry/frequency shift/channel exclusion.

For original and equilibrated systems, maximum columnwise2-norm backward error
||H x-b||/(||H||F||x||+||b||)<=1e-10. Compute singular values of Heq once;
kappa2=smax/smin, gamma_n=n*eps/(1-n*eps),
rho=kappa2*max(equilibrated backward error,gamma_n), require rho<=1e-6.
Report all condition/error quantities. A measured zero residual is not proof of
zero forward error. These are numerical engineering gates, not a rigorous error
certificate or a coordinate-invariant physical property.

Check original hidden descriptor residual after reconstructing all699 rows and
angle kinematics, normalized<=1e-10. Speed-row residual MUST equal R_F, not0;
ROM full33 residual analogously equals imposed acceleration R_G/M in speed rows.
Units/kinematics/inertia and cached G common-angle checks<=1e-10 maxrelative.
Save raw matrices, solutions and reaction/torque operators before scientific gates.

Independent checker uses separately written raw (unequilibrated) NumPy solves
on the SAME26 systems, no fallback, and reconstructs original equations and
physical operators independently. Compare each normalized reaction and torque
operator to maxabs/max(1,maxabs(both))<=1e-8 and saved metric arithmetic<=1e-10
absolute. No new frequencies or model. A failed independent numerical check
leaves the physical comparison inconclusive, not a model-family rejection.

## Finite stage budget and implementation

ONE main13-frequency invocation:13 FOM679 +13 ROM13 multi-RHS solves and26
conditioning SVDs maximum,10min. ONE independent26-solve check,10min. Count52
solves total if complete, not two experiments of unspecified size. Aggregate
1200s including two prior metadata shell calls3.6526071s (no solves); main and
independent costs recorded separately. Synthetic implementation QA is separate;
one coherent preexecution repair, no real retry. Close/review at completion/cap.

Synthetic tests: complex multi-RHS solve/reconstruction; invariance to an
invertible hidden coordinate transform; simple known harmonic oscillator with
physical Omega/M conversion; descriptor algebraic feedthrough without falsely
calling it memory; singular clamped block and failure/cap preservation. Root
reviews physical assembly and final code before real execution. Use one short
script/helper as needed, existing sources, no framework or general DAE reduction.
Fresh artifacts private_data/task2_harmonic_reaction_0090, refuse overwrite.

Failure of sampled fidelity rejects only this frozen-system/grid claim; failure
of well-posedness/numerics is inconclusive. Neither result supplies a unique
missing mechanism, latent order, causal explanation of rollout error, global
stability or a physically identified memory kernel. A future partition requires
a separately validated treatment of polynomial/algebraic feedthrough.
