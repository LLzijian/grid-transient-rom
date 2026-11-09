# Loop0078 — bounded structure-discovery screen rejected

Authority: after75 stage B; preregistration
`reports/task2_structure_campaign_0078.md`. All authorized fits/evaluations
are terminal. Implementation6382d3b; extension selection/launch b0172b8.
Attempt1 duplicate-R implementation failure is quarantined separately;
the scientific decision uses only corrected attempt2 and its exact resumes.

## Question, change and decision

Can automatically selected shared-gradient structure (G-auto, r3/100ms)
retain S's behavior and improve complete5s pooled error by >=5% over matched
ordinary sparse C-free? The three arms release the preregistered empirical
bones under matched objectives, samples and bounded parameter budgets.

**Rejected at the registered fixed1200 screen.** G-auto/C pooled ratio
5.112943 fails <=.95 and retention fails. TRAIN-only ranking independently
selected C and G-fixed, not G-auto, for their one1201..3000 extension.
G-fixed3000 also fails retention; it cannot rescue the G-auto hypothesis.
C3000 passes trajectory retention but fails its separate majority-modal
screen (1/13 versus>=7). No seed confirmation or final promotion follows.

## Completed extension evidence

| Quantity | S reference | C-free3000 | G-fixed3000 |
|---|---:|---:|---:|
| Complete1s mean | .298393264 | .253897258 | .363935628 |
| Complete1s pooled | .373210293 | .341181022 | .461582964 |
| Complete1s CVaR10 | .767440577 | .740749303 | .955088301 |
| Complete2s mean | .328373414 | .249002255 | .348087144 |
| Complete2s pooled | .381051143 | .301891462 | .432555198 |
| Complete2s CVaR10 | .734655202 | .616628150 | .914171407 |
| Complete5s mean | .288326166 | .198073952 | .247496741 |
| Complete5s pooled | .333570888 | .240044721 | .308373972 |
| Complete5s CVaR10 | .628207978 | .490982770 | .650510385 |
| Unstable-prefix pooled | 1.230200075 | 1.238937861 | 1.219458725 |
| Unstable-prefix CVaR10 | 1.667156312 | 1.741488599 | 1.836243907 |
| Common5 RMS, rad/s | .452087182 | .216180851 | .394182587 |
| Eligible unstable detected | 10/12 | 10/12 | 9/12 |
| False LOS / false window-settled | 0 / 2 | 0 / 2 | 0 / 2 |

G-fixed fails all six1/2s S-relative mean/pooled/CVaR retention gates,
unstable-prefix CVaR (ratio1.101422760, allowed1.05), and detection>=10/12.
It passes5s/common/prefix-pooled/false-count/numerical gates, but these do
not compensate the failed gates. Misses: pilot_0366, expansion_0186/0189.
C passes every trajectory-retention gate, including prefix CVaR ratio
1.044586273. Neither endpoint is selected by best-validation checkpoint.

G-fixed extension takes17271.40s. TRAIN average risk1200->3000 decreases
2.927119378->.893819713; horizon1/2/5 risks at3000 are
.602125361/1.018217748/1.061116031. C decreases .434825174->.320789836.
Both continue exact1200 model/Adam/RNG; all independent continuation,
1800-step trace, frozen-buffer/support/moments, finite-gradient/field and
source checks pass. G-fixed retains H8/J4/R2/ports2/count143; C Xi34/alpha18/
count141. No extra growth/pruning/loss/seed/LR/budget changes occurred.

Each validation retains117 raw/107 eligible/10 latency and22 raw/12 eligible
unstable. Complete1/2/5 counts96/95/95. All488 available scored intervals
(535 total rows) plus required LOS intervals pass. No finer diagnostic is
triggered; reporting stays .005. Independent direct NumPy cached arithmetic
agrees within2.23e-16 for G and5.56e-17 for C. Separate shared-helper
physical/numerical replay reproduces107 classifications in each arm with
no discrepancy. That replay reads explicit development H5 histories and
metadata; it is not independently implemented software. Direct metric
arithmetic uses caches only. No TEST payload is read. Historical per-step
event-ID logs are absent; RNG/trace audits do not manufacture that evidence.

## Interpretation, rollback and next implication

This rejects the finite-budget discovery/retention claim, not the entire
structured family or r3. G-fixed improves greatly under the permitted
extension, so optimization difficulty remains real. Both G arms fail while
C works as a trajectory control: diagnose the learned bounded template
before adding capacity or losses.1200 TRAIN decomposition found relative
angle dominant, not mechanical-power overweight; local origin instability
was also not the simple explanation.

No accepted/default model was overwritten. Keep experimental modules and
quarantined checkpoints as reproducible evidence, do not deploy rejected
G models. Accepted M3/0058/S remain the clean operational checkpoint;
C3000 remains a control candidate, not a structured discovery or L3 success.

Next smallest question: does the frozen G-fixed3000 electrical-force map
leave most of its observed-state force error outside the span reachable by
changing q, compared descriptively with matched C3000? An affine-in-q
projection can separate a learned instantaneous representation floor from
within-span encoder/dynamics mismatch without fitting, oracle rollout,
new dimensions or losses. Preregister this diagnosis separately; it cannot
retroactively rescue0078 or claim the whole template is impossible.

## Reproducible artifacts

- `reports/0078_fixed1200_evidence.md`: fixed1200 endpoints/selection.
- `reports/0078_C3000_evidence.md`: retained ordinary control.
- `reports/failed/0078_control_physical_modes.md`: separate modal rejection.
- `private_data/task2_structure_0078_attempt2_{C-free,G-fixed,G-auto}` and
  corresponding `_eval`: fixed1200 fits and audits.
- `private_data/task2_structure_0078_extend_{C-free,G-fixed}` and corresponding
  `_eval`: exact extensions, independent integrity, fixed forecasts and audits.
