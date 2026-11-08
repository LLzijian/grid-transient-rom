# 数值结果与执行回执

`results/` 按研究或维护事项保存小体积结果、逐项摘要、公开执行回执和来源清单；原始数据、模型、预测与缓存分别在 `private_data/` 或 `artifacts/`，位置以实际清单为准。

先从[研究报告首页](../reports/README.md)按科学问题阅读，再查[正式 Run 导航](../harness/README.md)中的结果与执行来源。目录、回执状态和模型采用是不同信息，不按结果目录名选择当前模型。

历史结果保留原路径与异常值口径，不为整齐重命名或重新计算。大文件的哈希身份、保存位置和备份能力分别记录；旧本地迁移从[已有清单](private_data_cleanup_20261107/migration_map.csv)接续，不补造历史 Job。

96个公开结果目录与本地产物的关系见[关联入口](repository_organization/20261109_history05/README.md)及[可筛选总表](repository_organization/20261109_history05/storage_associations.csv)。已核实的登记引用、历史组和元数据观察分别记录，缺少来源不自动表示可清理。

## 早期 E 系列的原有解释与纠正

以下为原结果说明的历史范围；整理前的[字节原件](../reports/history/early_e_series/originals/results_status_before_organization_20261109.md)另存，旧相对链接以主题材料表及迁移映射查找。

## Corrected historical development evidence (e10–e15)

The accepted `e10_refit_v2`, `e11_modal_validation_v2`, and `e12_refit_audit`
directories contain the corrected five-fold refit, final-model modal diagnostic,
and event-derived audit summary. They are post-selection validation evidence.
Strict H100 covers only 177 stable events; 35 unstable events are excluded.
Modal evaluation succeeds for 167/177 eligible stable events. See the audit in
`reports/accepted/0010_project_conclusion_audit.md` for negative findings and limits.

`e13_failure_diagnosis` and `e14_force_balance` are accepted frozen-model
diagnostics (loops 0011/0012), not improved checkpoints. F1 has static cut-capacity
violations; F5's large dynamic errors remain after smaller integration steps,
observed future Pm, or zero hidden initialization. Zero initialization reduces
pooled error but worsens event-median error in both audited folds. Neither
diagnostic establishes a deployable repair or a stability certificate.

`e15_electrical_balance` (loop 0013) substitutes offline terminal Pe in a swing
impulse audit. This reduces the residual for many F5 cases but is not a forecast:
the source equation uses electromagnetic torque te, and the Pe substitution
leaves damping, armature losses and sampling effects unresolved.

## Pre-fix and exploratory snapshots (e00–e09)

The legacy e00–e09 numerical files predate Git baseline `f764fe0`. They are
retained as immutable audit inputs, not as results reproduced by the current
code.

In particular, the historical pruning outputs were generated before the
physical-horizon and total module-adoption fixes accepted in loop 0001. They
must not be promoted as current evidence. A future accepted experiment must
write a new run directory, record the starting commit and data identity, and
pass the gates in `docs/engineering_loop.md`.

Loop 0002 also established that historical fields named `certified`,
`critical_energy`, and `margin` came from local SLSQP/root candidates rather
than a verified global escape-energy lower bound. They are retained only as a
retrospective heuristic screen. They do not support a formal stable verdict or
a safety guarantee.

Ten inherited JSON files contain Python-style `NaN` or `Infinity` tokens. The
current JSON gate reports these Python extensions as warnings rather than
rejecting inherited snapshots. They are not rewritten, because doing so would
alter the snapshots' recorded provenance. New accepted artifacts must use
standard finite JSON values or explicit `null` plus a documented status field.

The historical E5/E5b/E5c modal headline is also not confirmatory: it evaluates
validation events that participated in model selection, uses pre-final full
models in several folds, and reports a per-event best-match statistic. The
replacement protocol uses final pruned models and all pairs from a complete
Hungarian assignment, macro-aggregated by event; it remains a post-selection
validation diagnostic until confirmed on a fresh holdout.

The historical E6b/E6c/E8 outputs came from multiple reads of the same test
build. That build is listed as contaminated in `configs/test_release_v1.json`
and cannot be re-armed as a confirmatory release.
