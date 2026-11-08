# Loop0040: expanded TRAIN electrical residual atlas, no M4 training

Preregistered before implementation, checkpointc1e9f8a. Follow the new
TASK2_after_loop_39 plan: all369 accepted TRAIN events in Pilot+expansion,
no validation or either test payload. Use true0..8s available prefixes without
padding, central interior samples; report pre50ms-LOS events separately.

Keep the plan's sign r_e=M*omega_dot-p+E0+Domega. Since the model candidate
uses electrical force -BgCq, matching that candidate requires BgCq=-r_e (after
instantaneous correction), not +r_e. This is a coordinate/sign convention,
not a change to the plan's storage identity. State it explicitly in later fits.

Frozen full-order GENROU D=0 for all10 machines (case source verified), so
M*omega_dot=tm-te. Use exported te to compute r_e=Pmstar-te+E0+Domega;
independently compare with central finite-difference acceleration. This avoids
mistaking differentiation noise for hidden memory. Require normalized FD
discrepancy median<.01 and95thpercentile<.1, denominator max(1,RMS(tm-te)).
All samples finite; no event exclusion to achieve a favorable rank.

Bg uses the26positive frozen-M3 edges. Its rank must be9. Separate Euclidean
zero-sum force from the orthogonal common/net-force component; report both
energies. BgC cannot represent nonzero net force. Uncentered event-balanced
second moments define spatial force-energy spectra for full/balanced residual,
by shard, outcome and disturbance group. These spectra are not latent dimensions.
Check exact energy decomposition and min-norm edge reconstruction<1e-10.
Bg pseudoinverse coordinates are nonunique and not sparse physical discovery.

For first3 diagnostic balanced modes only: event-wise autocorrelation1/e lag
within1s, detrended/Hann dominant frequency only for>=2s available traces,
leave-one-group spatial-mode alignment, and correlations with full-order hidden
states (exclude rotor angle/speed). Report signed cross-group consistency of
the top5 associations per mode; correlation does not establish causality or
genuine memory after instantaneous-state conditioning. Do not select r=2/3,
fit r_inst, train an observer or call a PCA projection a deployable closure.

Acceptance is correctness/completeness of this diagnostic atlas and numerical
gates, not a forced low-rank finding. Save per-event residuals/time privately,
moments,force/edge basis,group/stratum records and dataset/model/code identity.
Next loop must test the smallest constrained instantaneous correction in
autonomous rollout before any dynamic residual-mode adoption.
