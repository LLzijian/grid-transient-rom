# Synthetic record fixtures — not research results

These files test record logic. There is no IEEE39 experiment, measured confidence
interval, actual independent reviewer, model promotion, or new statistical or
mathematical certification behind these records. `synthetic=true` is intentional.
The assessor checks declarations and reference structure; it does not read or
validate the scientific contents of this page. Do not copy these values into a
real run as evidence.

## Examples and expected interpretation

| File | Expected interpretation |
|---|---|
| `run_single_negative.json` | Keep a single valid negative observation; healthy diagnostics do not establish adequate optimization or a null mechanism. |
| `run_single_matched_gain.json` | A matched, referenced gain can support a budget-specific observation despite insufficient optimization; it cannot promote a model. |
| `run_two_seeds_not_null.json` | The proposed strong conclusion is intentionally denied: two seeds, booleans and an interval overlapping zero do not establish practical equivalence. |
| `run_performance_equivalence.json` | All declared requirements for the package's stronger, two-sided practical-equivalence criterion are present. This does not establish mechanism absence or mechanism irrelevance. |
| `run_functional_prediction_equivalence.json` | Adds a registered functional target, matched probe and explicitly scoped independent-review record. The legacy-named `mechanism_null_within_scope` is limited to that registered measurable prediction; it never means that the physical mechanism does not exist. |
| `run_reviewed_counterexample.json` | Illustrates a precisely scoped mathematical counterexample with a separate declared review; empirical replication and optimization are irrelevant to this particular mathematical route. |

## Synthetic empirical protocol, results and inference

The declared metric is normalized trajectory error reduction, with positive delta
meaning benefit of the candidate over its matched comparator. The fixture margin
is 0.05, declared as registered before the fixture outcomes. The synthetic interval
[-0.03, 0.04] is wholly inside [-0.05, 0.05]. It is an arbitrary logic fixture,
not an interval estimated from data. The broad interval [-0.2, 0.2] fails this
criterion, even though it contains zero.

For the complete-equivalence fixture only, this page stands in for the registered
evaluation contract, fixed budget, matched comparison, optimizer-adequacy report,
probe justification, uncertainty methodology and adaptive-selection accounting.
The scope is exactly the scope string in the JSON. Three named seed units stand
in for individual result references. Their presence does not prove independence
or sufficient power: a real analysis must justify the inferential unit, exposure,
coverage and uncertainty at the stated scope. The dataset in those seed examples
is explicitly development-seen. Seed replication is not new-condition testing.

The functional fixture additionally stands in for a preregistered prediction:
“Within the named fixture cohort, this specified intervention changes this
registered functional metric by at least the declared practical margin.” Its
matched probe is declared to operate correctly. Practical equivalence contradicts
the proposed material difference at this registered metric and scope, not the
existence of a mechanism and not arbitrary alternative mechanisms.

`synthetic_author_A` and `synthetic_reviewer_B` are fictional identities. The
review record declares that the latter supports the stated functional conclusion
after considering the protocol, probe, interval and selection accounting. This is
only a fixture for provenance fields. A boolean or these invented names cannot
substitute for an actual independent review.

## Synthetic exact-counterexample record

The statement is: every smooth scalar autonomous ODE with equilibrium at zero has
a Lyapunov-stable zero equilibrium. The counterexample is x_dot = x. It is smooth,
zero is an equilibrium, and x(t) = exp(t) x(0). Fix epsilon = 1. For any positive
delta, choosing 0 < x(0) < min(delta, 1) gives a finite t with x(t) > 1. Thus this
universal statement is false. This example is not a counterexample to passivity,
to a particular power-system model, or to any claim outside its stated domain.

`synthetic_proof_author_A` and `synthetic_math_reviewer_B` are fictional provenance
entries. The separate reviewer declaration covers the exact statement,
assumptions and argument. The assessor validates the presence and consistency of
those fields; it does not prove the argument or authenticate those identities.

## Conditional record extensions used by strong examples

- `inference.claim_target`: `algorithm_performance`,
  `registered_functional_prediction`, or `mathematical_statement`.
- `effect_interval` always uses candidate gain as positive; the only supported
  explicit `effect_direction` is `candidate_gain_positive`.
- Important judgments have a corresponding indexed `*_reference`. These may
  point to one report; there is no requirement to create a separate file per field.
- `optimization_scope` must match the record scope for an adequacy assertion.
- `confirmation_record` names its kind, scope, independent units and their
  references, frozen protocol, exposure, independence and adequacy assessment.
  `data_exposure` is `unseen_for_selection`, `development_seen`, `mixed` or `unknown`.
  New conditions with unknown/seen exposure do not qualify as independent
  confirmation. A unit count by itself establishes no sufficient sample size.
- `review_record` names the reviewer, evidence authors, scope, reviewed references,
  verdict and exactly the claims/decisions supported. A reviewer cannot also be a
  declared evidence author. These are auditable declarations, not authenticated facts.
- `probe_record` binds a matched probe to the registered functional prediction,
  protocol, results and scope. It is required for a functional null conclusion.
- `proof_record` identifies `theorem` or `exact_counterexample`, the precise target
  statement, assumptions, proof and scope. Reviewed mathematical falsification is
  separate from empirical practical equivalence.

`valid` and `schema_valid` are identical and mean only schema validity.
`promotion_eligible` and the `allowed_*` lists indicate record requirements, never
automatic approval. The module does not check file existence; the optional CLI
check does that separately. Neither check establishes scientific truth.
For every fixture the assessor returns `assessment_mode=synthetic_fixture` and
`actual_use_eligible=false`, even if a formal record gate is satisfied. For a
real record, actual-use eligibility remains `null` (not assessed or authorized).
`science_decision_basis` specifies whether a supported decision rests only on a
budget-specific performance observation or on a stronger scoped claim. Known
unhealthy optimization blocks strong empirical conclusions and promotion until
the inconsistency with an adequacy declaration has been resolved.
