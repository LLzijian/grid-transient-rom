# Repository baseline acceptance

- Date: 2026-09-05
- Scope: import the audited project state into Git and register the closed-loop
  engineering policy.
- Expected result: a readable `main` repository with ignored generated caches,
  a machine-readable gate registry, and explicit accept-or-revert rules.

## Verification

- Git repository initialized on branch `main`.
- `configs/acceptance_gates.json` parses as JSON.
- Required loop documentation and report directories exist.
- Existing project files are preserved without scientific-result changes.

## Boundary

This checkpoint records the inherited project. It does not accept the inherited
scientific claims. Those claims remain governed by the audit findings and must
pass the later claim-specific gates before promotion.

