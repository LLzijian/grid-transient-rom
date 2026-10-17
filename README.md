# Kiki

Scientific research on physics-structured sparse reduced-order models for IEEE39
post-fault dynamics, with transient prediction, stability and modal-analysis objectives.

GitHub repository: [LLzijian/grid-transient-rom](https://github.com/LLzijian/grid-transient-rom)
(`https://github.com/LLzijian/grid-transient-rom.git`).

## Start here

- [Agent authority and delegation](AGENTS.md): the single source of agent rules.
- [Scientific engineering loop](docs/engineering_loop.md): preregistration, execution,
  evidence, stage review and promotion.
- [Current research state](docs/current_state.md): current baselines, latest outcome,
  active/next experiment, evidence and TEST status.
- Follow the active TASK roadmap and loop preregistration linked from current_state.

After each execution that produces or changes a design, implementation or experimental
result, commit and push the key design decisions, decisive results and reproduction
information to GitHub, whether the improvement is accepted, rejected or unresolved.
Keep disposable intermediate data local. Follow the selective archival and push procedure
in [Scientific engineering loop](docs/engineering_loop.md#github-archival-after-each-execution).

Read historical reports only as needed; reports and experiment artifacts hold the
scientific evidence. This README does not duplicate the agent rules or current status.

## Closure equation discovery study

The finite A–E implementation and study are documented in the [final report](reports/closure_discovery_20261017_final.md), with [explicit equations and an independent NumPy package](results/closure_discovery_20261017/E/packages/main_2701/README.md). Prediction, structure recovery and stability findings retain their reported scope.

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
