# Kiki

Scientific research on physics-structured sparse reduced-order models for IEEE39
post-fault dynamics, with transient prediction, stability and modal-analysis objectives.

## Start here

- [Agent authority and delegation](AGENTS.md): the single source of agent rules.
- [Scientific engineering loop](docs/engineering_loop.md): preregistration, execution,
  evidence, stage review and promotion.
- [Current research state](docs/current_state.md): current baselines, latest outcome,
  active/next experiment, evidence and TEST status.
- Follow the active TASK roadmap and loop preregistration linked from current_state.

Read historical reports only as needed; reports and experiment artifacts hold the
scientific evidence. This README does not duplicate the agent rules or current status.

## Repository

- `src/kiki/`: model, data and evaluation code.
- `experiments/`: scientific experiment and diagnostic entry points.
- `tests/`: targeted implementation checks.
- `configs/`: experiment/data/release configuration.
- `reports/`: preregistrations, conclusions and historical evidence summaries.
- `docs/`: protocol, current state and roadmaps.

Scientific validity takes priority over benchmark improvement. Development results,
implementation QA and model promotion are distinct. Frozen-test access follows the
registered final evaluation procedure; opening the repository does not authorize it.

## Private data and result backup

The Code tab contains the project source and documentation. The original datasets
and selected scientific results are in the private
[Release](https://github.com/LLzijian/grid-transient-rom/releases/tag/curated-data-2026-10-08)
because private_data/ is excluded from Git.

These root files were added only for the backup:

- DATASET_AND_RESULTS.md explains the contents and recovery procedure.
- original_research_data_manifest.jsonl lists each archived file and its SHA-256.
- original_research_data_volumes.json lists the four parts and their SHA-256.
- restore_private_data.py verifies and restores the files.
- selection_policy.json records what was included and omitted.
- .gitattributes keeps text line endings consistent in the repository.

From a fresh clone on Windows, after signing in with GitHub CLI:

    gh release download curated-data-2026-10-08 -R LLzijian/grid-transient-rom -D .\release_parts
    python restore_private_data.py .\release_parts

This creates private_data/ and external_data/ in the clone. The script refuses to
overwrite either existing directory. For browser downloads or historical canonical
dataset path setup, see [DATASET_AND_RESULTS.md](DATASET_AND_RESULTS.md).


## TASK4 latest results

The independent TASK4 project's closed T4_0002 results, explicit models, scientific
limitations and reproducibility records are available in
[TASK4 results (2026-10-09 publication)](TASK4/RESULTS_2026-10-09.md).
TASK4 research remains paused; retained candidates are not deployment models.
