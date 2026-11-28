# 算法设计

当前主线（Scientist `gpt-6-astra` 已审定 v3.1，待 AGY 独立复核及用户放行）为 **[PDD v3.1（OSGKD-Decoupled）架构简报](pdd_v3_architecture_brief.md)**，面向 NeurIPS 2027，聚焦“训练期特权物理观测（$\mathcal{O}_1$）与部署期短时部分观测（$\mathcal{O}_0$）下的开放网络化动力学可辨识性与两阶段解耦发现”。配套规划见 [PDD v3 Campaign](../campaigns/pdd_v3/campaign.md)。

前代已封卷端到端架构为 **[OSGKD 2.0](osgkd_v2_architecture.md)**（保留作为端到端消融与 IEEE-39 旗舰应用参照）。实施次序和比较条件见 [v2 Campaign](../campaigns/osgkd_v2/campaign.md)，已取得的结果见[当前状态](../docs/current_state.md)。

[历史索引](history/README.md#osgkd-v1)收录 v1 的四份设计，包括因冻结运行器逐路径校验而保留原位的三份来源。前序 GFTD 可从[同一索引](history/README.md#gftd-20261113)查阅。

通用设计方法在 [docs/](../docs/architecture_design.md)，研究协议随 [Campaign](../campaigns/README.md) 保存，实验结果、分析和审查归 [reports/](../reports/README.md)。

当前规范与修订证据：[v3.1 Scientist 规范](../reports/pdd_v3_architecture_20261128/v3_1_scientist_spec.md) · [修订台账](../reports/pdd_v3_architecture_20261128/revision_ledger_v3_1.md)。旧 v3.0 审阅结论只对应其冻结输入；当前执行仍暂停。
