# Loop0075 — TRAIN boundary midpoint batch: rejected

**Question / hypothesis.** Can one real midpoint per20 TRAIN transition
groups yield a refit-ready supplement:>=16 qualified pairs, including>=4
stable and>=4 unstable? Preregistered in `reports/task2_train_boundary_0075.md`
at75c7428; starting checkpoint207b36d, generator/config e749e3d, QA44b97c4.
Selection used fixed bus/reactance curves and did not bridge UNKNOWN grid
slots. No global duration monotonicity or critical clearing time was assumed.

**Change / budget.** Exactly20 fixed conditions, each at .005/.0025s;
40 FOM runs, no adaptive replacement or third step. Same operating point,
restored system, controllers,15s stable horizon and physical LOS rules.
Coarse/fine builds finished in517.693/846.652s, run concurrently (not a speed
benchmark). Both exit0;20/20 accepted by each generator,0 discarded.

**Evidence.** Both roots contain11 stable/9 unstable, all20 beyond100ms;
all20 labels agree. Existing mechanical QA and reference case, equilibrium,
inertia, mapping and postfault-signature checks pass. No duplicate old TRAIN
conditions, no VAL/TEST events. Eighteen targeted tests pass. Independent
raw-array metric recalculation matches within5.56e-17 and all20 gate flags.

| Registered gate | Result |
|---|---|
| Qualified pairs>=16/20 | **7/20: fail** |
| Qualified stable>=4 | 4: pass |
| Qualified unstable>=4 | **3: fail** |
| Per-pair normalized COI error<=.001 | 13 fail; maximum .304471327 (bus28) |
| Per-pair normalized common-frequency error<=.001 | 7 fail; maximum .051337484 (bus33) |
| Unstable LOS-time difference<=.02s | 1 fails: bus28,1.3301 vs1.2301s after clearing |

Passing pair IDs: `boundary_0000,0001,0003,0006,0008,0010,0011`
(buses2,3,6,12,15,18,20). Fail counts overlap. All raw outcomes/prefixes are
retained; numerical-unqualified does not replace the stored physical label.

**Classification / interpretation.** Rejected: this fixed acquisition
protocol did not produce the preregistered numerically qualified supplement.
The bottleneck here is FOM step consistency, not missing label diversity,
latency, failed Newton solves, or a new test of the latent model family.
Two step sizes do not establish that the finer trajectory is exact, or
identify the mechanism of the observed sensitivity. No gate relaxation,
third-step rescue, data-pool promotion, model refit or model-family rejection.

**Preservation / next implication.** All40 results and7 passing pair records
remain available. Correct scoped generation/QA reproducers are retained;
there was no model change to roll back. S/0058/M3 hashes and old index/split
hashes are unchanged. Before a later matched-data refit, numerical boundary
fidelity needs a separately authorized/preregistered investigation. **PAUSED
after0075 at the user's request; no0076 or background experiment started.**

Evidence: `private_data/ieee39_boundary_0075`, `ieee39_boundary_fine_0075`,
`task2_boundary_integrity_0075/evidence.json` (includes selection/endpoints
and source hashes); logs `.loop-work/boundary0075_{coarse,fine,qa}*.log`.
Reproduce builds with `generate_private_dataset.py --stage boundary`, the
two `configs/private_dataset_boundary{,_fine}_0075.json` files, fresh output
and separate workspace-local `--solver-workdir` paths, `--workers 1`.
QA: `check_train_boundary.py --selection .loop-work/boundary_selection0075.json`.
The reused QA report's `pilot_ready=false` is its irrelevant VAL-count
heuristic for these ALL-TRAIN roots; its generic holdout-access footer is
not an access record. No TEST trajectory was read; old H5 access was limited
to global physical provenance/equilibrium, never old event payloads.
