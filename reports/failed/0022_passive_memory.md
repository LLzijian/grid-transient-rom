# Loop0022 rejected: passive force memory still carries static bias

Start758b24a; experiment519af5a; preregistration in that commit's
`reports/task2_m4_passive_q1.md`. Command:
`python experiments/run_private_m4.py --output private_data/task2_m4_0022`.
Ruff and48tests pass; scientific screen fails. No threshold adjustment.

Reciprocal drive=.00176917*c, decay again hits.01/s. This prevents0021's huge
active-feedback explosion but maintains a near-constant extra torque. At.2s
pooled .253497 improves M3 .334328; at2s1.069741,5s2.761683,8s5.185263 are worse
than .740507/.565149/.464055. Three new false-unstable events (0056,0061,0150).
History observability is unchanged and not the limiting issue.

Post-failure TRAIN-only diagnostic: equilibrium-anchored sin(angle differences)
explains99.1285% of the scalar force residual squared norm; adding speed gives
99.3196%, adding Pm gives99.8071%. Fits are event-balanced fixedridge1e-6, no
validation selection. Therefore the leading SVD direction is predominantly an
instantaneous electrical-response defect, not demonstrated memory. A standalone
decaying q should not be tasked with absorbing this static constitutive response.

Retain444 per-event/horizon records and classification in `0022_m4_evidence.json`;
private complete run remains available. Revert failed memory implementation.
Next hypothesis: separate instantaneous angle response from residual scalar
memory, and require an explicit memory-off ablation to identify q's contribution.
M2 remains frozen; this is M4 electrical-closure discovery, not M2 optimization.
