# Original datasets and curated research results

This private snapshot contains byte-for-byte copies of the data files used by this project, including TRAIN, validation, TEST, and holdout events. No HDF5 event was filtered, regenerated, relabeled, or resampled for publication. Merely backing up the TEST bytes does not constitute a model evaluation or change the scientific TEST gate.

The Git repository is a current working-tree snapshot of source code, configuration, protocols, reports, paper, and lightweight results. It does not transfer the local Git history. The full data and selected scientific result artifacts are attached to the private Release curated-data-2026-10-08, not stored in Git history.

The Release contains 654 files (5,315,134,456 source bytes) in four numbered parts:

- The original private_data/ieee39_postfault and private_data/ieee39_expansion directories, with their complete dataset.h5 files, indices, original split files, and QA reports. These include all TRAIN, validation, and TEST events.
- The original TRAIN-only boundary datasets and TEST-only holdout supplement, with their original metadata.
- The historical canonical dataset referenced by configs/dataset.json, under external_data/ieee39_postclear_canonical_v1. Its original source and checksum metadata are preserved.
- 601 selected files from the closed TASK2 stages 0098–0106: terminal scientific evidence, per-event results, stage cost ledgers, formal model weights/checkpoint descriptions, and six decisive matched validation trajectory arrays.

Repeated monitor arrays, duplicate full-TRAIN rollout arrays, timing retries, trace logs, optimizer/RNG states, caches, and temporary runs are omitted. Earlier findings and adverse classifications remain in reports/. The original project workspace is unchanged.

Download all original_research_data.tar.partNNN assets from the private Release into one directory. From a fresh clone of the repository, run:

    python restore_private_data.py path\to\downloaded\parts

The restore script checks SHA-256 for every part and every extracted file against original_research_data_volumes.json and original_research_data_manifest.jsonl. It restores private_data/ and external_data/ and refuses to overwrite existing copies. The historical canonical dataset is restored into external_data/ within the clone; configs/dataset.json still names its original sibling-project location, so place or configure that data at the expected location before rerunning historical e-series code.

The Release uses byte-split tar volumes to package all original datasets and selected results together with one recovery procedure. The postfault HDF5 file itself is below GitHub's current 2 GiB limit for an individual Release asset, so this split is a packaging choice, not a requirement for that one file. Reassembled and extracted files are byte-identical to the source files, as recorded by the file-level SHA-256 manifest.
