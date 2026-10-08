# Manuscript status

The current `main.tex`, `main_CN.md`, figures, and PDF incorporate the critical
energy and evidence-lifecycle audits through loop 0005. They compile and use the
current screening terminology, but remain **not submission-ready** because the
body and PDF have not yet incorporated the completed five-fold refit and
corrected modal results, and no fresh confirmatory holdout exists.

Current evidence is in `reports/accepted/0010_project_conclusion_audit.md`.
Modules pass the total gate in F1/F3/F5 only. Strict H100 evaluation includes
177 stable events and excludes all 35 unstable validation events; final pooled
R2 is -0.0843, with F5 pooled R2 -1.6983. Modal evaluation succeeds for 167/177
eligible events, with median event frequency error 19.49% and MAC 0.376.
The body and figures must not be used as the current numerical report until
these results and their limitations have been incorporated.

The local SLSQP/root search produces feasible boundary candidates and
unverified saddle candidates. It does not produce a verified global lower bound
on escape energy. Therefore historical fields named `certified`, the positive
“stability margin” verdict, and the claimed zero missed-instability guarantee
must be interpreted as a retrospective heuristic screen, not a theorem or a
safety guarantee.

The observed screen result remains a descriptive datum: 0 of 51 unstable events
were screened stable. At the declared independent source/fault-line unit there
are only 9 unstable groups, so the one-sided 95% exact Clopper--Pearson upper
bound is 0.283. The revised manuscript reports this uncertainty and the
test-reuse limitation explicitly.

The 315-event test build was also reopened by separate E6b, E6c, and E8 scripts.
It is therefore consumed exploratory evidence, not a one-time confirmatory
release. The legacy modal headline is likewise non-confirmatory because it used
selection validation events, pre-final full models, and the lowest-error match
within each event. The corrected modal protocol uses final pruned models and
every complete Hungarian-assigned pair, but must be run only as a validation
diagnostic until a fresh holdout is available.

Legacy trajectory, module-selection, baseline, and mechanical-power numbers are
now labelled development or retrospective exploratory evidence. The former
modal 7.2%/0.04 headline is withdrawn rather than reinterpreted. The next
scientific work is failure diagnosis and the remaining validation diagnostics,
followed by manuscript revision; the confirmatory release remains blocked on
a new dataset build.

Authoritative implementation semantics are in
`PrunableEnergyModel.critical_energy_level` and
`PrunableEnergyModel.stability_margin`; current repository status is in
`results/README.md` and the accepted loop reports.
