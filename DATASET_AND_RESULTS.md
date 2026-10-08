# Curated private data and research results

Snapshot built at 2026-10-08T04:35:58+00:00 from the local working tree. The Git repository contains 666 current source, configuration, protocol, report, paper, and lightweight result files (13,638,297 bytes). It is a current-state snapshot, without the local Git commit history. Historical TEST-bearing numerical files under \`results/e06/*/test\`, test-margin files, and \`results/e08\` are excluded.

The private release \`curated-data-2026-10-08\` contains 619 selected \`private_data/\` files (4,127,249,540 source bytes) in 3 numbered volumes. It contains:
- All TRAIN and validation events from \`ieee39_postfault\` (227 TRAIN, 74 validation) and \`ieee39_expansion\` (142 TRAIN, 43 validation), retaining original event trajectories, labels, times, equilibrium, inertia, and filtered indices. The HDF5 files are explicitly derived subsets, not byte-identical source files. All 105 TEST event payloads were omitted without reading their trajectory values.
- Two TRAIN-only boundary-analysis datasets (20 events each).
- 601 selected scientific result files (1,155,069,121 bytes): terminal evidence, per-event rows, stage plans and cost ledgers, formal model weights/checkpoint descriptions, and six decisive matched validation trajectory arrays.

No TEST-only holdout supplement, full training rollout duplicates, interim monitor arrays, optimizer/RNG states, timing retries, trace logs, caches, or temporary runs are included. Unselected local originals remain in the source workspace. Scientific conclusions and adverse outcomes remain documented in \`reports/\`.

Download all \`selected_private_data.tar.partNNN\` release assets into one directory. From the cloned repository, run:
\`\`\`powershell
python restore_private_data.py path\to\downloaded\parts
\`\`\`
The script verifies volume and file SHA-256 values using \`selected_private_data_volumes.json\` and \`selected_private_data_manifest.jsonl\`, then restores \`private_data/\`. It refuses to overwrite an existing \`private_data/\` directory. The exported \`splits.json\` files deliberately have an empty \`test\` list; \`export_scope.json\` records source hashes and omitted TEST counts. Do not treat these subset files as the full historical data release.
