# Accepted loop 0004: manuscript re-scoped to candidate-energy screen / exploratory evidence

Date: 2026-09-05

Starting commit: `4ce8565`

Decision: **accepted manuscript honesty rewrite** (completes the partial rewrite found in flight;
aligns `paper/main.tex` + `paper/main_CN.md` with accepted loops 0002/0003 and `paper/STATUS.md`)

## Registered claim

The manuscript must stop presenting the local candidate-energy search as a stability
certificate and must not present any number read from the consumed 315-event canonical build as
confirmatory out-of-sample evidence. It must: (a) label local storage checks as structure audits and
the clearing-state rule as a candidate-energy screen with no formal certificate; (b) report 0/51 as a
descriptive retrospective screen over 9 independent source/fault-line groups with the exact one-sided
95% Clopper--Pearson upper bound 0.283; (c) label all 315-event and development numbers as
retrospective/exploratory, removing the false "single read of the frozen test" statement; (d) not
retain the legacy best-pair modal frequency/damping values as claims; (e) document the blocked,
preregistered single-pass release that is required for confirmatory inference. The change fails if any
of those over-claims survives verbatim.

## What changed (paper files only)

- `paper/main.tex`: title/abstract/contributions re-scoped ("Integration-Free Candidate Screening");
  removed "single read"; all 315 results labeled retrospective/exploratory; energy section now
  "Energy-Structure Audits and the Integration-Free Candidate Screen" with local audits + candidate
  critical energy; safety section reports 0/51 over 9 groups + CP 0.283 (cites
  `clopper1934`); modal section withdraws legacy numbers, describes corrected all-pair endpoint;
  added Section II subsection "Confirmatory release protocol" (`\label{sec:release}`) describing the
  blocked preregistered release; Discussion/Conclusion rewritten to the same honest framing
  (removed "lowest escape energy implies bounded motion" theorem; screen is empirical).
- `paper/bib/references.bib`: added `clopper1934`.
- `paper/main_CN.md`: mirrored the same rescope (version bumped to v2; §3 audits+screen; claim ②
  descriptive 0/51/9 groups; claim ③ modal legacy numbers withdrawn; §5 boundaries expanded;
  §6/§7 layout + evidence-tier notes updated).
- `paper/main.pdf`: regenerated (8 pages).

## Gate record

| Gate | Result |
|---|---|
| LaTeX | pass: `latexmk -pdf` → **8 pages, 0 overfull, 0 undefined refs/citations** |
| G1 syntax | pass: `compileall -q src experiments tests` (exit 0) |
| G2 unit | pass: `pytest` → 26 passed |
| G6 JSON | pass: 75 files, errors 0, warnings 92 (legacy Infinity) |
| diff integrity | pass: `git diff --check` |

## Scientific boundary

This is a prose/evidence-tier correction only: no number, result file, or code path changed. The paper
still reports no confirmatory out-of-sample result; that remains blocked behind
`configs/test_release_v1.json` until a genuinely new dataset build and gate-compliant refits exist.
Observation/reference documents (not committed here): `reports/manuscript_screen_rewrite_plan_v1.md`,
`reports/evidence_governance_0003_manuscript_v2.md`,
`reports/history/manuscript_rewrite_inflight_verify_v1.md` (archived process snapshot).
