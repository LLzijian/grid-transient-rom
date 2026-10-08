# Loop0071 — physical-output modal baseline

Classification: **rejected** for the preregistered majority-modal-fidelity
claim on S, not for its local stability or trajectory predictions.
Preregistration `reports/task2_physical_modes_0071.md`; run65173c6.

Five synthetic tests pass, including phase/scale invariance, physical-unit
mapping and cardinality-first one-to-one assignment. The original699-state
descriptor reproduces all150 verified finite poles exactly, with549 infinite
and0 indeterminate modes; maximum eigen residual1.03403e-15. Inertia/unit
mapping error<=1.223e-16; source identities/hashes are unchanged. No rejected
160-mode operator, simulation, H5 payload, fitting or TEST access was used.

S has **1/13** qualifying distinct matches versus the fixed>=7 screen;
M3 has0/13. S has11 eligible-output oscillatory modes, M3 has10. The sole
S match is FOM84 (.641414Hz, damping.185242) to S14 (.702749Hz,
damping.152627): relative frequency error.095624, absolute damping
error.0326154, physical-output MAC.841443. All pair metrics and unmatched
modes are retained; no nearest-frequency or many-to-one rescue.

Limitations are not solely pole frequency: eight of the nine conventional
electromechanical targets (all except84) have maximum S output MAC<.8
across all oscillatory candidates. FOM91 has strong shape similarity
(MAC.955) but its strongest-shape partner has29.5% frequency error; FOM104
has MAC.849 but damping difference.182. These descriptive examples do not
change the fixed matching rule. A stable learned Jacobian does not establish
high-order modal fidelity, nor does this local mismatch reject r3 globally.

Evidence: `private_data/task2_physical_modes_0071/evidence.json` and
`modal_arrays.npz`; exact full spectra, physical-output vectors and pair
metrics are saved. Independent cached-array recomputation in
`independent_check.json` agrees on all143/130 S/M3 pairs within3.33e-15
and reproduces the matching cardinalities. S/M3 checkpoints are unchanged;
the read-only reproducer
is retained. No small-perturbation confirmation or energy/stability-margin
transfer is claimed. Continue the fixed0070 fit/evaluation, then choose the
next bounded scientific experiment under after66; no modal-threshold tuning.
