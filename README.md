# Kiki

Scientific research on physics-structured sparse reduced-order models for IEEE39
post-fault dynamics, with transient prediction, stability and modal-analysis objectives.

GitHub repository: [LLzijian/grid-transient-rom](https://github.com/LLzijian/grid-transient-rom)
(https://github.com/LLzijian/grid-transient-rom.git).

## Start here

- [AGENTS.md](AGENTS.md): top-level research principles, aesthetics, authority and
  navigation. Rules are not experiment logs.
- [kiki.md](kiki.md): scientific task design — deployment contract, data/information
  boundary, metric definitions and capability layers. Current model instances live separately.
- [docs/architecture_design.md](docs/architecture_design.md): first-level learning problem, physical assumptions, identifiability and competing explanations.
- [docs/current_design.md](docs/current_design.md): versioned current architecture, separate from the deployment contract.
- [agents/runtime.md](agents/runtime.md): task ownership, model identity, dependencies and resource scheduling.
- [docs/engineering_loop.md](docs/engineering_loop.md): Campaign → Run → Job tree,
  optimization adequacy vs mechanism evidence, three independent decisions,
  stop/extend and review.
- [docs/research_policy.md](docs/research_policy.md): evidence standards, dual archives,
  budgets, Git archival and the minimal release package.
- [campaigns/](campaigns/): one card per scientific hypothesis, its Run tree and beliefs.
- [docs/current_state.md](docs/current_state.md): current snapshot, pointers to the
  active campaign/run, evidence and next executable action.

After each execution that changes a design, implementation or experimental result,
selectively commit and push the key decisions, decisive results and reproduction
information to GitHub — accepted, rejected, failed or unresolved alike. Keep disposable
intermediate data, private data and raw artifacts local. Follow
[docs/research_policy.md](docs/research_policy.md) §6.

Read historical reports only as needed; reports, results and experiment artifacts hold
the scientific evidence. This README does not duplicate agent rules or current status.

## Closure equation discovery study

The finite A–E implementation and study are documented in the [final report](reports/closure_discovery_20261017_final.md), with [explicit equations and an independent NumPy package](results/closure_discovery_20261017/E/packages/main_2701/README.md). Prediction, structure recovery and stability findings retain their reported scope.

## GPU training

Future closure fits use the CUDA-default entry [experiments/train_closure.py](experiments/train_closure.py) in a separate GPU environment. See the [GPU training guide](docs/gpu_training.md) for setup, resume and measured performance. Historical experiments retain their original runners and frozen evidence.

## Repository

- src/kiki/: model, data and evaluation code.
- experiments/: scientific experiment and diagnostic entry points.
- tests/: targeted implementation checks.
- configs/: experiment/data/release configuration.
- reports/: preregistrations, conclusions and historical evidence summaries.
- docs/: protocol, current state and long-term rules.
- campaigns/: campaign hypothesis cards and Run trees.
- templates/: campaign, run protocol, run closure and mechanism card templates.

Scientific validity takes priority over benchmark improvement. Development results,
implementation QA and model promotion are distinct. Frozen-test access follows the
registered final evaluation procedure; opening the repository does not authorize it.

## Offline harness checks

The v2 harness keeps the existing Campaign/Run/Job and three decisions. It adds
architecture design, separate optimization-health/adequacy records, confirmation
independent of promotion, and advisory task scheduling. It does not resume paused
science or dispatch models. See [runtime](agents/runtime.md).

```bash
python3 tools/research_harness.py check --root .
python3 -B -m unittest discover -s tests -p "test_harness_*.py"
```

Examples are synthetic or explicitly dated repository snapshots, never live authorization.

Integration and local verification: [2026-10-30 harness v2 record](reports/harness_v2_integration_20261030.md).
