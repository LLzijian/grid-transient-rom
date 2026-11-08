# 研究记录导航

## 先按问题阅读

- **[当前主线：静态拓扑、隐幅值调制与力记忆](../campaigns/gftd_kuramoto_discovery_20261106.md)**：先读当前研究候选、证据范围与未决问题；[静态恢复](gftd_sparse_topology_20261107/README.md)、[调制辨别](gftd_modulation_20261107/README.md)、[真实暂态迁移](gftd_fit297_20261107/README.md)提供正式登记与原执行／续跑导航。
- **[初代暂态模型：模块选择、能量筛查与误差诊断](history/early_e_series/README.md)**：13 份 E 报告和旧保留审计已收拢。先读历史认识，再查采用／拒绝决定和解释更正。
- **[早期私有模型：机械功率、记忆闭合与结构诊断](history/private_rom/README.md)**：按问题串起 Loop0017–0106，接受与未采用记录同时保留；[有路径依赖的材料](history/private_rom/protected.md)列明引用者和解除条件。
- **[闭合方程发现：结构搜索、因果初始化与工程核验](history/closure_discovery/README.md)**：串起 10-17 的 A—E 研究与 10-19 的后续工作，分别查看科学结论、工程接受和未得到有效比较的记录。
- **[早期论文：成稿、图表、数字核对与证据修订](history/paper/README.md)**：区分当时的成稿／排版完成与后续科学验收，连接旧稿、修订方案和最终接受记录。
- **[早期工程门控：基线审查与后续实现](maintenance/early_gates/README.md)**：维护记录单独归类，历史待办不进入当前研究任务。
- [当前研究状态](../docs/current_state.md)：今天的研究路线、候选与下一动作。历史综述不代替当前状态。
- [已登记 Run 中文总表](run_index.md)与[材料导航命令](../harness/README.md)：21 份正式登记按中文问题查询；缺失字段显示 unknown，科学状态沿原记录解释。

本目录保存实验结果、分析、审查及其冻结来源。具体算法方案见 [designs/](../designs/README.md)，
旧算法方案见 [设计档案](../designs/history/README.md)；新 Run 协议随 Campaign 放 `campaigns/`。
已有冻结协议与原始模型咨询记录属于实验来源证据，保留原身份；不据此在本目录继续新增活动设计。

从 [当前研究状态](../docs/current_state.md) 进入有效路线、当前实验和基线。
具体实验以该入口指向的预注册、结论及证据为准；不要按文件名排序或仅按
`accepted/` 目录选择当前结论。工作类型、结果和模型晋级分别记录，见
[工程循环](../docs/engineering_loop.md)。

## 记录用途

| 位置 | 用途 |
|---|---|
| 已有 `task*.md` 与冻结协议 | 历史或当前 Run 的预登记原件，采用范围以当前状态和该 Run 为准；新协议归 Campaign 材料 |
| `accepted/` | 当时接受的实现、诊断或模型记录；不自动表示模型晋级，局部解释可能被后续纠正 |
| `failed/` | 被拒绝、实现失败或其他未通过记录；属于必要证据，不能因失败而删除 |
| 顶层证据与审查报告 | 某个阶段的证据汇总或独立维护审查，按报告标注的范围使用 |
| `history/` | 历史研究综述与过程快照；按各页范围和更正链阅读，不作为当前候选或任务状态 |
| `maintenance/` | harness、环境与恢复等维护事项；不作为科学 Run 或模型晋升 |

## 历史 E 系列与旧论文记录

13 份 E 系列报告与引用它们的保留审计已一同归入[主题档案](history/early_e_series/README.md)，原件字节保持不变。旧链接语境通过已核验的固定提交版本保留。
E0–E3/E7 主要是旧开发与实现记录；E2 剪枝、E4–E6/E8 的部分强主张
已有明确纠正。旧 paper_* / synthesis / 改写方案也不能作为当前投稿状态。
检索命中历史文件时，先读顶部说明，再沿纠错链接查看有效解释。

- [0001：物理时域与模块采用门修正](history/early_e_series/originals/accepted/0001_pruning_horizons_and_module_adoption.md)
- [0002：候选能量不是稳定性证书](history/early_e_series/originals/accepted/0002_candidate_energy_not_certificate.md)
- [0003：测试集访问与模态统计修正](history/early_e_series/originals/accepted/0003_holdout_lifecycle_and_modal_reporting.md)
- [0009：重拟合与 0008 局部解释纠正](history/early_e_series/originals/accepted/0009_five_fold_refit_acceptance.md)
- [0010：旧 e 系列综合结论边界](accepted/0010_project_conclusion_audit.md)
- [论文状态](../paper/STATUS.md) · [数值结果用途](../results/README.md)
- [保留价值审查与整理记录](history/early_e_series/originals/records_retention_audit_20260922.md)（原文阅读入口见主题档案）

已归档过程快照：[早期门控建设](maintenance/early_gates/README.md)、
[论文改写中途检查](history/paper/originals/notes/manuscript_rewrite_inflight_verify_v1.md)、
[旧版排版检查](history/paper/README.md)。历史错误和纠错链均保留。
