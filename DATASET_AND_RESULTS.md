# Original datasets and curated research results

This private snapshot contains byte-for-byte copies of the data files used by this project, including TRAIN, validation, TEST, and holdout events. No HDF5 event was filtered, regenerated, relabeled, or resampled for publication. Merely backing up the TEST bytes does not constitute a model evaluation or change the scientific TEST gate.

The Git repository is a current working-tree snapshot of source code, configuration, protocols, reports, paper, and lightweight results. It does not transfer the local Git history. The full data and selected scientific result artifacts are attached to the private Release curated-data-2026-10-08, not stored in Git history.

The Release contains 654 files (5,315,134,456 source bytes) in four numbered parts:

- The original private_data/ieee39_postfault and private_data/ieee39_expansion directories, with their complete dataset.h5 files, indices, original split files, and QA reports. These include all TRAIN, validation, and TEST events.
- The original TRAIN-only boundary datasets and TEST-only holdout supplement, with their original metadata.
- The historical canonical dataset referenced by configs/dataset.json, under external_data/ieee39_postclear_canonical_v1. Its original source and checksum metadata are preserved.
- 601 selected files from the closed TASK2 stages 0098–0106: terminal scientific evidence, per-event results, stage cost ledgers, formal model weights/checkpoint descriptions, and six decisive matched validation trajectory arrays.

Repeated monitor arrays, duplicate full-TRAIN rollout arrays, timing retries, trace logs, optimizer/RNG states, caches, and temporary runs are omitted. Earlier findings and adverse classifications remain in reports/. The original project workspace is unchanged.

## Download and restore

A private-repository member needs GitHub CLI (gh) authenticated for the following
Windows PowerShell commands. Run them from the directory where the clone should go:

    gh repo clone LLzijian/grid-transient-rom
    Set-Location grid-transient-rom
    gh release download curated-data-2026-10-08 -R LLzijian/grid-transient-rom -D .\release_parts
    python restore_private_data.py .\release_parts

Alternatively, download the four original_research_data.tar.partNNN assets in a
browser, place them together in one directory, and run the final Python command
from the repository root with that directory as its argument. Do not try to
extract one part independently.

The script verifies SHA-256 for every part and every archived file against
original_research_data_volumes.json and original_research_data_manifest.jsonl.
It prints "Restored 654 files successfully" only after all checks pass. It
creates private_data/ (including private_data/ieee39_postfault/dataset.h5)
and external_data/ in the clone, and refuses to overwrite existing copies.
For a checksum check without extraction, use:

    python restore_private_data.py .\release_parts --verify-only

The historical canonical dataset is restored into
external_data/ieee39_postclear_canonical_v1. configs/dataset.json still names
its original absolute sibling-project path; before rerunning historical e-series
code on a different computer, point dataset_dir at the restored directory or
place that dataset at the configured path. The current TASK2 private_data paths
restore in their original relative locations.

The Release uses byte-split tar volumes to package all original datasets and selected results together with one recovery procedure. The postfault HDF5 file itself is below GitHub's current 2 GiB limit for an individual Release asset, so this split is a packaging choice, not a requirement for that one file. Reassembled and extracted files are byte-identical to the source files, as recorded by the file-level SHA-256 manifest.
