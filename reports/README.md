# 研究记录导航

从 [当前研究状态](../docs/current_state.md) 进入有效路线、当前实验和基线。
具体实验以该入口指向的预注册、结论及证据为准；不要按文件名排序或仅按
`accepted/` 目录选择当前结论。工作类型、结果和模型晋级分别记录，见
[工程循环](../docs/engineering_loop.md)。

## 记录用途

| 位置 | 用途 |
|---|---|
| 当前状态指向的 `task*.md` | 有效实验预注册；其他 task 文档可能已关闭，保留作追溯 |
| `accepted/` | 当时接受的实现、诊断或模型记录；不自动表示模型晋级，局部解释可能被后续纠正 |
| `failed/` | 被拒绝、实现失败或其他未通过记录；属于必要证据，不能因失败而删除 |
| 顶层证据与审查报告 | 某个阶段的证据汇总或独立维护审查，按报告标注的范围使用 |
| `history/` | 已结束的过程快照，不作为待办或当前科学结论 |

## 历史 E 系列与旧论文记录

13 份 `E*.md` 已标为历史记录，保留原位置和正文以维持引用。
E0–E3/E7 主要是旧开发与实现记录；E2 剪枝、E4–E6/E8 的部分强主张
已有明确纠正。旧 paper_* / synthesis / 改写方案也不能作为当前投稿状态。
检索命中历史文件时，先读顶部说明，再沿纠错链接查看有效解释。

- [0001：物理时域与模块采用门修正](accepted/0001_pruning_horizons_and_module_adoption.md)
- [0002：候选能量不是稳定性证书](accepted/0002_candidate_energy_not_certificate.md)
- [0003：测试集访问与模态统计修正](accepted/0003_holdout_lifecycle_and_modal_reporting.md)
- [0009：重拟合与 0008 局部解释纠正](accepted/0009_five_fold_refit_acceptance.md)
- [0010：旧 e 系列综合结论边界](accepted/0010_project_conclusion_audit.md)
- [论文状态](../paper/STATUS.md) · [数值结果用途](../results/README.md)
- [保留价值审查与整理记录](records_retention_audit_20260922.md)

已归档过程快照：[早期门控建设](history/gate_status_audit_v1.md)、
[论文改写中途检查](history/manuscript_rewrite_inflight_verify_v1.md)、
[旧版排版检查](history/paper_tex_quality_v1.md)。历史错误和纠错链均保留。
