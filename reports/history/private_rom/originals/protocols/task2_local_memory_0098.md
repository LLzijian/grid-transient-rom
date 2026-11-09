# Loop0098 / E1 — shared local versus global structured memory

Prospective card, 2026-09-22. User explicitly resumed research using
`HANDOFF_after_loop_0097.md` and `TASK2_after_loop_97.md`; after97 now supplies
the roadmap. Starting HEAD `6057946`. No earlier result is reclassified.
Work type: idea study. Promotion: none. TEST remains closed.

## Question, comparison, scope

Compare newly jointly trained G3-S (33 states), L10-S (40), HL11-S (41),
using one small shared scalar-energy / constant J,R / explicit-port grammar.
Does machine-local memory improve autonomous capabilities at this budget?
Memory dimension, spatial organization, and encoder aggregation are a package;
this does not isolate locality from capacity. Historical M3/0058/S/C3000/G3000
are frozen references, not matched fresh fits. A poor finite search does not
falsify the architecture. No L2 region or final L1/L3 claim is authorized.

## Data and fixed physics

Original explicit TRAIN only for fits, scales and selection: raw369,
eligible346 (311 stable,35 unstable),24 fault groups. Preserve every original
event and duration; complete1/2/5/8s counts315/311/311/311. TRAIN reader must
not load validation during pilot/selection. Final development evaluation uses
the existing107 eligible validation events (raw117; unstable12 of raw22).
No TEST payload read, boundary0075/0077 merge, fabricated tail or difficult
event deletion. Existing TRAIN normalization is reused without refitting.
Starting assets and code hashes plus dirty-tree status are saved in artifacts;
unrelated AGENTS/README/engineering_loop/report/history edits are preserved.

Physical interface is [delta,omega_rad_s,p,q], p=tm-pm_star. Internal
pi=M*omega, M=raw_inertia/(2*pi*60); retain common frequency, remove only
one common angle for modal analysis. Encoder runs once on 11 angle/speed
observations over100ms, actual p0 remains known. Future RHS reads its state
only. No old latent labels, whitening or trained closure is transferred.

## Explicit finite model grammar

Reuse StructuredROM's M3 embedding, active26 angle-difference edges,
nonlinear sin/cos, positive empirical k,d,J_pi,p,Rpp,p-storage; zero damping
starts1e-4 in all arms. Inertia, equilibrium, support and scales are fixed.
Base physical degrees of freedom are identical across arms. A single learned
positive q-storage scalar is shared across all memory coordinates.

Let v=Wq: W=I for L10; for G3 its columns are normalized known-parameter
vectors [1,M/mean(M),rate/mean(rate)]; HL11 uses [I,1/sqrt(10)]. These are
fixed maps, not free machine IDs or independently fitted node equations.
Shared local GRU7->8->1 reads normalized relative angle, full speed,
two weighted-neighbor summaries and known M/rate/gain ratios. G3 GRU12->8->3
reads W-weighted aggregates of the four dynamic features; HL adds GRU4->8->1
on their mean. Anchored output=0.1*(raw(history)-raw(zero history)), with
identical constant machine features in the two encoder evaluations.

Eight optional shared template groups (node/edge instances share scalars):
H_pq=sum(p_i/p_scale_i)*v_i;
H_edge_sin=sum normalized_k*(v_i-v_j)*sin_diff(phi+eta);
H_edge_cos=sum normalized_k*(v_i+v_j)*cos_diff(phi+eta);
J_pi,q with W; J_p,q with diag(1/p_scale)W;
R_qq=rho I; R_pi,q=rho sum a_i a_i^T with normalized
a_i=(unit_pi_i,W_i); and one local port template acting on pi_i.
The port shares six coefficients multiplying normalized omega,p,v,
neighbor sin_diff, neighbor cos_diff and neighbor memory difference.
All extra matrices exclude angle rows/columns. Storage cross terms are
not individually claimed positive. J is skew, R PSD; actual port supply
is retained in Hdot. End-point reversal must leave edge energy unchanged.

Start with J_pi,q=1e-3 and R_q=1e-2 active; all other templates inactive.
At updates100 and400, rank remaining templates using actual differentiated
0.2s autonomous multistep loss on the fixed TRAIN panel; activate at most3
per opportunity (six total). Signed starts oppose the derivative at magnitude
1e-3; dissipative starts1e-2 and require favorable one-sided derivative.
Port ranking varies its six coefficients as one atomic template, choosing
their negative gradient direction at norm1e-3. No one-step proxy ranking.
One whole-template ablation prune at800: remove an optional noninitial group
only if its removal increases the same panel autonomous objective by <0.1%.
No hidden library/rank/port-family additions. Preserve pre/post states.

## Joint objective and optimization

All active encoder, physical empirical parameters, storage, J,R and port
coefficients learn together, float64/RK4 dt.01. No global Hdot<0 loss.
Per-point error reuses0078: .5*(mean COI-delta squared/Sdelta^2 + mean
COI-speed squared/Somega^2)+common-speed squared/Somega^2+mean p squared/
p_scale^2. Fixed Sdelta=.47675314588683515,Somega=1.0033667249602924.
Each event contributes its genuine available prefix mean. Long loss uses
the first100ms only; local integral .2s uses a random causal TRAIN window.
Total=10*(.8 mean+.2 worst10% event error)+local+.1 latent-one-step
consistency/.1^2+.1 physical-torque output loss+1e-4 active-template penalty.
Torque truth follows the audited0078 algebra p-M*omega_dot; no Pe/torque
substitution. Auxiliary voltage/load decoder is omitted equally in all arms;
its absence is a scoped implementation choice, not a recovered L2 output.

Batch12: existing stable/unstable/group-balanced sampling for10 entries,
plus two entries from a seeded shuffled complete-TRAIN cycle. Record coverage
and IDs; full eligible pool is never filtered to the panel. Seed98 for all.
Adam block rates: encoder3e-4, signed/templates1e-3, positive raw1e-2;
gradient norm cap100. One setting only: conclusions explicitly limited to it.
Pilot uses twelve deterministic events: three distinct unstable groups then
nine stable groups, group/event lexicographic round-robin; all real prefixes.
Up to60 pilot updates per arm, horizons .2/1s alternating, no support growth.
Pilot states are diagnostic only; formal fits restart the same initial model.

Formal milestones1000/3000/6000. First100 updates horizon1s;101..300 alternate
1/2s;301 onward cycle1/2/5s. Fixed panel monitor at0,500,1000 and every500;
use average1/2/5s event risk, retain severe rows separately. Extend1000->3000
only if both0->500 and500->1000 improve>=1%; extend3000->6000 only if both
2000->2500 and2500->3000 improve>=1%. No extension on validation results.
Stop an arm on nonfinite loss/gradient/parameters, structural failure, or
panel severe risk >10 times initial at two consecutive monitors; retain
the failing checkpoint. Continuous improvement at cap=optimization limited.

Select among completed formal milestones using the lowest fixed-panel risk,
tie earliest; separately report last valid checkpoint. If resource cap occurs
before1000, select last valid checkpoint as an explicitly incomplete-budget
result, not a milestone winner. No pilot or validation-based selection.

## Finite resource budget, evidence and decision

Total scientific execution cap12000 process-wall seconds (sum across arms,
parallelism does not reset it): QA/smoke/pilot1200, formal3x2400=7200,
evaluation/exports2400, implementation-retry contingency1200. Count failed
runs and diagnostics. At most one implementation retry for an identified
bug; preserve all affected results and hashes, do not treat them as valid
science. Memory cap8GB/process; one torch/BLAS thread per arm. Smoke estimates
are recorded before formal fitting; caps cannot silently rise. Pilot cap
180s/arm within the shared1200 allowance; formal cap2400s/arm includes its
monitors/growth/checkpoints. Steps or time cap, whichever comes first.

Report fixed-compute and selected/last results, per-task curves, initial
mismatch, counts of templates/instances/parameters/expanded RHS, 1/2/5/8s
complete and severe prefixes, common frequency, paired group changes,
classification denominators/errors/unresolved rows, original one-to-one
modal matches (.15 frequency,.05 damping,.80 MAC), local alpha, power QA
and full actual ports. Use existing validation evaluator dt.005/.01 plus
at most one .0025 refinement trigger; unresolved numerics quarantine the
affected scores. Use cached aligned FOM eigenvectors, not new FOM runs.
Energy identity is QA; V/region/coverage are unestablished in E1. Inference
timing is only qualified as useful speed if numerical/accuracy scope permits.

Before route adoption obtain one independent bounded read-only challenge.
At closure: retain at most2 complementary research candidates if useful;
otherwise redirect to the after97 E2 same-organization F/S comparison or
record optimization limitation, based on the complete capability profile.
No requirement to beat10% for exploratory value, no gate lowered, no final
promotion or frozen-test access. Root owns all scientific decisions.

## Independent preflight challenge and root response (before any fit)

Agent `/root/e1_challenge`, independent of proposal and implementation,
read-only review: proceed with implementation/smoke; no theoretical veto.
Required checks adopted: activate every template/port coefficient for the
permutation/edge reversal tests; use directed D for sine summaries but
absolute D for cosine summaries. Record initial q, Wq and physical-memory
RHS RMS, since a common0.1 encoder scale does not ensure equal physical
perturbation. No scale tuning from these diagnostics is authorized here.
Root accepts the limitations: different dimensions/aggregations remain a
bundle; M3 rate/gain are fixed TRAIN-derived empirical backbone descriptors,
not newly measured machine parameters; panel metrics are training diagnostics;
wall-cap results cannot establish model-family inadequacy. Continuous
complexity regularization must depend on active coefficient magnitudes,
not merely a discrete count. No V/region claim follows from power QA.

Implementation clarification before smoke/fitting: edge energy weights are
fixed k0/mean(k0); port sine/cosine summaries are fixed k0-weighted sums
normalized by each node's incident weight. Dissipator proposals use the
physical rho derivative at zero via a continuous mask, removing softplus
chain scaling from ranking. Accepted sequential deletions update their
ablation baseline. Probe state is restored even on a failed proposal;
source hash drift quarantines a run. The formal process reserves60s within
its cap (pilot30s) for triggered checks and endpoint serialization; actual
fit/selection, endpoint and total process costs are all retained. There is
no permission to spend unused wall time as unregistered optimizer updates.
All24 targeted synthetic tests passed before data smoke, including all-arm
heterogeneous machine permutation, full-template energy gradient, original
MechanicalROM embedding, zero-mask dissipative gradient and prune-regression
checks. No real training result informed these implementation corrections.

## Closure pointer (2026-09-22; protocol above unchanged)

Completed capability, numerical quarantine, independent final challenge and
root stage decision: [0098_local_memory_capability.md](0098_local_memory_capability.md).
All three fits stopped below1000 steps at their preregistered resource caps;
no promotion. Full TRAIN and development evidence is preserved under
`private_data/task2_local_memory_0098/`; this stage is closed.
