# Loop0062 preregistration — conditional sparse coefficient identifiability

Starting checkpoint0f057c3. Authority docs/TASK2_after_loop_59.md Phase4/5.
0061 supports retaining r3; current uncertainty is group-wise equation
identifiability, not another latent-order test. The36-entry mask was imposed
and frozen: mask equality across0060 is not evidence of support discovery.

## Question / hypothesis

Can one equation on the current support describe the frozen encoder's latent
increments across all24 TRAIN fault-bus groups without dependence on a single
group? Hypothesis: the supported coefficients are numerically identifiable,
and leave-one-TRAIN-group-out diagnostic fits preserve the pooled fit's
held-out-group output-projected dynamics within the fixed10% screens below.

This is conditional encoder-reference consistency, NOT physical latent truth,
unbiased cross-validation, accepted model refitting, or proof of unique support.
The encoder was trained on these TRAIN groups. A failure cannot by itself
reject the successful autonomous model: it may expose encoder/ODE mismatch,
collinear library terms, group dependence, or fixed-support inadequacy.

## Frozen components and data

Model `private_data/task2_long_0060_5s/model.pt`, SHA256
d34d9f5c36fb296e1d49e59f5f7b18a5da33b56bf371a16169f82bc25664995c.
Freeze every tensor, normalization, M3 backbone, r3, topology, encoder,
library, and support. No fitting of deployment weights; diagnostic coefficient
arrays are saved separately, never as a replacement model.pt. No validation
values enter this analysis; existing shared reader may load validation but
only full TRAIN is used. No old/fresh TEST payload is read.

Use all346 eligible full TRAIN events, same initial100ms availability rule.
At each real sample i>=10 encode only x[i-10:i+1] to obtain qenc_i. This
sliding encoder is diagnostic training-reference construction, not an
autonomous prediction protocol. Never pad, extend, or delete short events.
Before fits report encode(zero physical history), its actual initial force
and full field; do not assume it equals the model equilibrium q=0. This
reference check is descriptive, not an extra acceptance veto.

## Small integrated regression, no neural optimization

For each event partition indices10..last into successive windows of at most
10 steps (.1s), retaining a final shorter nonempty window. At every genuine
sample build the existing76-column feature row Theta(x_i,qenc_i), exactly
as the frozen JointROM does, without adding interactions or an intercept.
The existing bias column remains constrained by the saved mask.

For a window [a,b] form

```
X_ab = trapz(Theta, actual .01s grid)/(t_b-t_a)
y_ab = (qenc_b-qenc_a)/(t_b-t_a)
```

Thus y is an integrated latent derivative, not a finite-difference ground
truth. Give every event equal weight and each of its windows equal weight:
row weight1/sqrt(N_events*N_windows_in_event). Physical long/unstable event
length does not determine its total weight.

For each latent output separately fit ONLY its12 saved active columns with
SVD least squares, rcond1e-10, no ridge/threshold search. Use pooled TRAIN
weighted RMS of each design column for numerical scaling, floor1e-8;
unscale coefficients before reporting. This is column unit conditioning,
not a latent covariance rotation or a new state-independence criterion.
Save full singular spectra, conditions, and ranks at1e-12/1e-10/1e-8 for
pooled, each group alone, and leave-one-group-out designs. If rank deficient,
retain the SVD minimum-norm diagnostic but label coefficients non-unique.

Compute one pooled fit and24 leave-one-group-out fits on all other TRAIN
events, retaining original event-balanced weighting within each subset.
No optimization curriculum, prospective validation, or model selection is
performed. Report masks, active indices, signed coefficients and changes
from the deployed Xi. Identical fixed masks never count as stable discovery.

## Predetermined diagnostic screens and interpretation

For every group g evaluate genuine windows of that group with (i) the pooled
fit, (ii) the fit excluding g, and (iii) the frozen deployed Xi. Define an
event-balanced output-projected reference-defect RMS

E_g(Xi)=RMS_event/window/generator[ Cq (y_ab-X_ab Xi) ],

and field-disagreement ratio

D_g=RMS[ Cq X_ab (Xi_without_g-Xi_pooled) ]
    / max(RMS[ Cq X_ab Xi_pooled ],1e-8).

Report numerator/denominator, physical units (force per second), all groups,
stable/unstable event contributions, and ratios. No selective averaging.
Common-group consistency screen: EVERY g must satisfy
E_g(Xi_without_g)/max(E_g(Xi_pooled),1e-8)<=1.10 and D_g<=.10.
These are prospective sensitivity-screen tolerances, not physical constants.
Conditional numerical identifiability requires all12 columns in each pooled
and leave-one-group-out output design to have rank12 at rtol1e-10. Group-alone
ranks and other tolerances are descriptive rather than extra vetoes.
Nonfinite quantities invalidate the computation, not the hypothesis.

Compare signed coefficient distributions and Aq spectra descriptively;
no stability constraint or new order conclusion is imposed. Report the
deployed model's reference defect even if the diagnostic fit is much better:
its physical rollout objective and encoder-consistency target are different.

## Checks / next implication

Synthetic exact linear ODE integrals must recover coefficients; tests must
verify event balancing, partial windows, causal history indices, no inactive
terms, and rank-deficiency labeling. Root reviews code before TRAIN analysis.
Independently recompute decisive matrices/metrics from saved window arrays.

A pass supports conditional group consistency only; support alternatives and
deployment-loss sensitivity remain unproved. A failure is diagnosed from
spectra, reference defects and group contributions before changing support.
No covariance gate, encoder redesign, r>3, extra loss, or library expansion
is authorized. Continue to the next smallest justified experiment; this
diagnostic is not the project Goal's completion.

## Closure

Loop0062 completed with97 tests and independent saved-array QA passing.
The rank screen passes but10/24 groups fail common-field sensitivity; root
rejects the conditional encoder-reference consistency hypothesis without
rejecting the deployment model. No trained weight changed. See
`reports/failed/0062_group_identifiability.md`. No thresholds were revised.
