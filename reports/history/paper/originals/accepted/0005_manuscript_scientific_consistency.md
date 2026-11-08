# Accepted loop 0005: manuscript scientific consistency

Date: 2026-09-06

Starting commit: `607db0a`

Decision: **accepted; no new numerical result is claimed**

## Registered claims

1. The method description must include both internal whole-block deletion and
   the accepted total modular-candidate versus freshly refitted backbone gate.
2. The normalized residual impulse ratio 1.018 means that the residual and
   mechanical-input impulses are comparable in scale; it does not mean that the
   backbone explains 98% of the dynamics.
3. H100 denotes 100 integration intervals and therefore 101 endpoint-inclusive
   samples.
4. The inherited mechanical-power outcome is a legacy development diagnostic,
   not an accepted post-audit module decision.
5. Paper figures must use the same terminology as the audited text and should be
   reproducible without timestamp-only PDF changes.

The loop fails if either manuscript retains the 98%-explained inference, omits
the total gate, promotes the legacy power-module result, or if source, PDF, and
generated figures disagree.

## Changes

- Added the total candidate-versus-fresh-M0 adoption gate to the English and
  Chinese method descriptions and to Figure 1.
- Reclassified inherited module instances and the power-module table as legacy
  diagnostics pending refit.
- Corrected the residual-ratio interpretation and made H100 endpoint semantics
  explicit in Chinese.
- Added the verified DOI and issue metadata for the original Clopper--Pearson
  paper: <https://doi.org/10.1093/biomet/26.4.404>.
- Fixed Table I and III notes that visually flowed beside their tables.
- Regenerated the paper figures and PDF from checked-in sources.

## Gate record

| Gate | Result |
|---|---|
| claim-specific text search | pass: no positive-certificate, one-time-test, legacy-modal-headline, or 98%-explained assertion |
| figure generation | pass: 34 stable + 9 unstable F1 validation events; 315 legacy aggregate rows; no test payload reopen |
| deterministic figure hashes | pass: two complete builds produced identical SHA-256 for all 3 PDF + 3 PNG files |
| bibliography | pass: original Clopper--Pearson DOI, no duplicate keys |
| LaTeX | pass: 8 pages, 0 undefined references, 0 overfull boxes |
| visual PDF | pass: all 8 pages inspected; no overlap, clipping, stale figure terminology, or illegible table notes |
| Python syntax/unit/lint | pass: compileall, 26 tests, Ruff |
| G6 JSON | pass: 75 files, 0 errors, 92 inherited non-finite warnings |
| diff integrity | pass: `git diff --check` |

## Scientific boundary

This loop improves internal consistency only. It does not rehabilitate legacy
numerical artifacts. The next evidence-producing step is the five-fold refit
under the accepted gate, followed by corrected validation experiments.
