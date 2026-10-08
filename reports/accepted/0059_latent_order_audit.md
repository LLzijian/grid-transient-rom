# Loop0059 diagnostic: three relevant short-time latent modes, not a covariance claim

## Question, hypothesis and scope

Does unwhitened0056 contain three driven/observable dynamic modes with
autonomous predictive contributions, or a removable third coordinate?
Starting checkpointc159dd8; preregistration2dd4286; analytic implementation
f777806 and frozen rollout implementation3f978f4. No model fitted or changed.
0058 had already passed all22 unchanged preregistered1s gates before this loop.
0056 remained the mandated primary object even though0058 scored better.

Classification: **diagnostic**. The evidence supports three locally effective
dynamical directions in frozen0056 and rejects the tested near-preserving
rank2 reduction. It does not retroactively accept0056, prove globally minimal
physical order, identify three named full-order controller states, or establish
long-horizon prediction/stability.

## Decisive linear-system evidence

The zero physical/latent reference has exactly zero complete vector-field
residual: it is an equilibrium of this frozen ODE. Aq is constant because its
retained q dependence is linear. Its eigenvalues are
**0, -.174568612, -.368163052 s^-1**: an integrator and two nonoscillatory
relaxation modes, with decay times5.728407 and2.716188s.
Eigenvector condition3.25907; three separated real poles, not three oscillators.

The integrator is a latent electrical-output direction, not the rotational
angle gauge: its Cq action is nonzero. Aq is only the latent subsystem with
physical x treated as driving input. Its zero pole neither proves nor rules
out instability of the full feedback-coupled ROM. Infinite-horizon stable
Gramians are not assumed.

| Reference matrix | Rank at1e-12/1e-10/1e-8 | Singular values | Condition |
|---|---|---|---:|
| O=[Cq;Cq Aq;Cq Aq^2] |3/3/3|3.978766,2.671115,2.307114|1.724564|
| R, physical input units |3/3/3|10.623266,4.523968,2.628534|4.041518|
| R, TRAIN-RMS input units |3/3/3|5.907800,2.523261,1.452137|4.068350|

Across1320 preregistered operating points from346 TRAIN events, both physical
and normalized reachability ranks remain3. Normalized smallest singular value
min/median/max=.791913/1.424420/2.083479; condition2.375730/4.137962/6.769068.
Every mode's PBH controllability rank remains3. Physical PBH minimum smallest
singular values across the region are .891696/.843831/.830843 in pole order
0/-.174568612/-.368163052. No nearly invisible/unreachable direction is exposed
by these diagnostics.

Raw O/R and Cq singular spectra/conditions are coordinate- and unit-dependent;
they are reported as requested, not used as invariant independence gates.
Force singular values3.741095/2.434258/2.285646 reproduce0056 and show available
spatial rank, not by themselves three effective dynamic states.

The TRAIN-pooled finite-time Hankel singular values, invariant under the
checked similarity transform with fixed physical input/output metrics, are:

| Horizon | HSV1 | HSV2 | HSV3 |
|---|---:|---:|---:|
|1s|15.398124|5.161619|2.893463|
|2s|31.052379|8.878118|4.299391|
|8s|127.863110|17.128717|6.399021|

These use event-balanced local Bscaled Bscaled^T and Cq^T Cq, not latent
sample covariance or whitening. They characterize a pooled local-linear
input/output system, not global nonlinear controllability or a truncation
error theorem. Integrator growth explains why horizon sensitivity matters.

Actual TRAIN physical-motion forcing and initialized action are nonzero in
each invariant mode (RMS over outputs/time, then equal weighting per event):

| Pole (s^-1) | RMS(Cq P_i g(x)) | RMS(Cq P_i q_initial) | Reference scaled residue norm |
|---|---:|---:|---:|
|0|1.375544|.697723|16.492475|
|-.174568612|.820447|.611092|7.346640|
|-.368163052|1.445338|.518472|6.350204|

g(0)=0 here, so the forcing-minus-reference RMS is identical. These physical
output-weighted quantities and modal residues are unchanged by latent
similarity/eigenvector rescaling. They distinguish actual observed excitation
from merely available B input directions. Physical inputs are constrained by
the surrounding dynamics; local input rank alone is not a full-system control
claim.

## Autonomous ablation and no-refit reduction

All107 eligible validation events are initialized from the same saved100ms
encoder output. Full autonomous1s reproduction differs from0056 by at most
6.57253e-14. No subsequent encoder call or observed future enters integration.
The96 complete1s events comprise95 stable and one unstable; shorter unstable
prefixes are retained separately.

|1s arm|Mean|Pooled|CVaR10|
|---|---:|---:|---:|
|Full0056|.305931762|.400976544|.824500471|
|All memory force off; algebraic retained|.600948939|1.017677677|2.431292875|
|Original coordinate1 force off|.908988420|1.569478913|4.286674218|
|Original coordinate2 force off|.338554496|.504834185|.992490044|
|Original coordinate3 force off|.554626429|.712011687|1.586264443|
|Remove dynamic pole0|.452735775|.579261567|1.283664184|
|Remove dynamic pole-.174568612|.599239366|1.046939546|2.455468026|
|Remove dynamic pole-.368163052|.404065175|.675097870|1.430954537|
|TRAIN1s balanced rank2|.382029054|.622595328|1.316607563|

Every true modal removal worsens both mean and pooled by substantially more
than the preregistered1%. Original-coordinate ablations also worsen both, but
are coordinate-dependent diagnostics, not the primary order evidence.

Balanced rank2 uses the fixed two leading TRAIN1s Hankel directions, WV=I,
z0=Wq0, zdot=W fq(x,Vz), output Cq(Vz+a(x)). No refit or validation basis
selection. Its mean/pooled/CVaR ratios are approximately1.249/1.553/1.597,
failing the preregistered1.02/1.02/1.05 near-preservation limits. Do not promote
this rejected reduction; failure of this projection is not proof that every
possible nonlinear two-state model must fail. Historical retrained r2/0055 is
also not a nested reduction of0056 (1s .383885/.641701/1.183394).

## Long-horizon boundary of the conclusion

|Horizon|Full mean/pooled/CVaR|Balanced rank2 mean/pooled/CVaR|
|---|---|---|
|2s|.379837/.438959/.826939|.481364/.720900/1.475796|
|5s|.661234/1.260692/3.162840|.669937/1.258551/3.043369|
|8s|1.535884/3.147601/7.583820|1.720941/3.651521/9.367467|

All arrays remain finite, which is NOT a boundedness or stability certificate.
The1s contribution result does not extend uniformly to8s: e.g. removing the
-.1746 mode improves8s mean but worsens pooled/tail; removing the-.3682 mode
improves8s pooled/tail but worsens mean. The complete8s cohort is95 stable
trajectories; these are not unstable-classification results.

RK4.01/.005 maximum eventwise normalized discrepancies over all107 fixed
forecasts, full versus rank2, are:1s2.296e-6/2.338e-6;
2s2.315e-5/5.313e-5;5s.003751/.131085;8s.029688/2.78955.
Both pass the frozen1s numerical gate. Long-horizon discrepancies and
model-error growth must not be hidden or interpreted as validated dynamics.
No dt/weight/order was selected from these results.

Independent saved-array arithmetic reproduced all nine1s arm aggregates and
all eventwise integration discrepancies to1e-10. The largest8s discrepancies
above occur beyond observed unstable prefixes, but numerical sensitivity is
NOT confined to unobserved tails. Among the95 complete8s stable events, the
maximum is .003486175 for full0056 (pilot_0150) and .019253351 for rank2
(pilot_0237). Complete-cohort maxima at1/2/5s are respectively
1.187e-6/2.819e-6/9.663e-6 for full and1.182e-6/3.970e-6/3.904e-5 for rank2.
Thus8s remains stress evidence, not a precision-controlled long-time claim.

## Secondary0057 and implications

0057 also has local observable/reachable rank3 throughout the same1320
operating points. Its poles are-.599636656,-.766280175,-.930544041,
with decay times1.667677,1.305006,1.074640s; O singular values
4.545765/2.776488/2.185990, condition2.079499. Physical R singular values
14.375217/10.362472/5.007920, condition2.870496. Its1s/2s/8s HSV are
[7.678265,4.799875,1.360465]/[10.292201,5.676252,1.662377]/
[11.718442,5.999887,1.785476]. This is a different learned system, not a
whitened similarity realization of0056. Its historical tail/contribution
failure remains valid; covariance conditioning did not guarantee useful prediction.

Root decision: retain r=3 as a justified prospective candidate order; no r>3.
**Cancel covariance approximate-orthogonality as a FUTURE acceptance gate.**
Normalization/conditioning may remain an optimization convention, never an
independence claim. Historical0056 failure and0058 original acceptance are not
rewritten. No encoder, library, topology, model weight or loss changed here.

Next: follow the user's revised `docs/TASK2_after_loop_59.md`. Begin Loop0060
as a champion/challenger campaign: warm-start0058, combine full-trajectory
TRAIN local windows with fixed-initialization2/5s autonomous curriculum, and
use8s first as development audit. Keep r3, encoder architecture, topology and
candidate library initially unchanged. Compare performance on a Pareto basis,
without a covariance gate. Formal full-system stability/energy work waits for
a basically frozen long-horizon ODE. Fresh TEST remains sealed until all final
model, training, selection and stability-evaluation definitions are frozen.

## Reproducibility

Scientific venv: E:/03-nn_Transfer_Func/kron-reduce/.venv/Scripts/python.exe.
Commands (no retraining):

- experiments/audit_latent_order.py --output private_data/task2_latent_order_0059
- Same script with --model private_data/task2_joint_0057/model.pt --normalization
  private_data/task2_joint_0057/normalization.npz --output private_data/task2_latent_order_0059_secondary57
- experiments/audit_latent_order_rollout.py --audit private_data/task2_latent_order_0059
  --output private_data/task2_latent_order_rollout_0059

57 tests and both synthetic check suites passed. Analytic B maximum FD error
1.292e-10; Aq7.376e-12. Topology-force singular values reproduce source evidence.
Similarity transfer errors<6.1e-16, HSV relative errors<8.2e-14; nonlinear
physical rollout similarity over all107 events differs by<7.106e-14.
Primary model SHA2568c6e666bb4683311c89d6c564273d9d3ab5f0f081cfdc268dd3c2663f65c061d.
Full-precision matrices, per-region derivatives/PBH diagnostics and excitation
are in the primary/secondary evidence.json and matrices.npz. All-arm trajectories,
per-event/group metrics and numerical discrepancies are in rollout
predictions.npz/per_event.json. Frozen TEST was never read.

## Reproducible derivative values at the reference

Rows below follow the three original0056 coordinates; physical input/output
indices follow the frozen ten-generator ordering. Values rounded here;
full precision and1320 operating-point B matrices are in the NPZ/JSON.

Aq (s^-1), constant throughout the retained library:

| row | q1 | q2 | q3 |
|---|---:|---:|---:|
| 1 | -0.174569 | 0.207787 | 0 |
| 2 | 0 | -0.368163 | 0 |
| 3 | 0 | 0.373086 | 0 |

Bq = [B_delta | B_omega | B_p]. Delta in radians, synchronous omega in rad/s,
p=Pm-Pm_star in system pu. These are physical-unit, unnormalized derivatives.

| row | delta1 | delta2 | delta3 | delta4 | delta5 | delta6 | delta7 | delta8 | delta9 | delta10 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 1.236718 | 1.243004 | -2.479722 | 0 | 0 | 0 | 0 | 0.916747 | 0 | -0.916747 |
| 2 | -2.928663 | 0 | -0.098043 | 1.155574 | 0.105102 | 0 | 0 | 0 | 0.281616 | 1.484414 |
| 3 | -3.029347 | 1.775813 | 0.166875 | 1.253534 | 0.489707 | -7.336404 | 6.283197 | -0.476109 | 0.097766 | 0.774968 |

| row | omega1 | omega2 | omega3 | omega4 | omega5 | omega6 | omega7 | omega8 | omega9 | omega10 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0 | 0 | 0.230771 | 0.119985 | 0.123605 | 0 | 0.171383 | -0.154093 | 0 | -0.422521 |
| 2 | 0.233135 | -0.050617 | 0 | 0 | 0 | 0 | 0 | -0.182268 | -0.174029 | 0.219455 |
| 3 | 0 | 0 | -0.063136 | 0 | 0 | 0 | 0 | 0 | 0 | -0.312240 |

| row | p1 | p2 | p3 | p4 | p5 | p6 | p7 | p8 | p9 | p10 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0.886127 |
| 2 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | -0.561129 |
| 3 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1.283317 |

Actual Cq=S_G B_net C_normalized diag(amplitude), not just the unit-direction
matrix; positive output is the electrical correction SUBTRACTED from the swing
force. C_unit is separately preserved in matrices.npz/evidence.json.

| row | q1 | q2 | q3 |
|---|---:|---:|---:|
| 1 | -0.468269 | 2.689196 | -0.792527 |
| 2 | -0.185098 | -0.263675 | -0.488057 |
| 3 | 1.859993 | 1.101559 | -1.215647 |
| 4 | 0.407833 | 0.019068 | -0.730204 |
| 5 | 0.586039 | 0.211067 | -0.816890 |
| 6 | 1.119696 | 0.291507 | -1.218639 |
| 7 | 0.200971 | -0.005330 | -0.695169 |
| 8 | -0.321504 | -0.087134 | -0.450281 |
| 9 | -0.198955 | -0.259703 | -0.491642 |
| 10 | -1.531076 | -1.175279 | -1.040710 |
