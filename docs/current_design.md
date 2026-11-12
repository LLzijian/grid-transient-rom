# 当前架构与待辨别问题

当前采用 GFTD-Kuramoto 的 5N 修订作为研究候选：共享自治方程显式保留相角、转速、
机械功率、隐幅值和力记忆。IEEE39 十机实例为 50 维；采用研究候选不等于模型晋升。
部署与数据合同见 [kiki](../kiki.md)，通用设计方法见 [architecture_design](architecture_design.md)。

## 设计入口

| 内容 | 来源 |
|---|---|
| 当前方程、父模型映射与资格要求 | [5N 修订设计](../designs/gftd_repair_design_20261107.md) |
| 数学与同初态数值资格通过的有界调制研究候选（原生产方程不变） | [储能可行性与有限时域存在](../designs/gftd_storage_feasibility_20261109.md) |
| 原297个样本已通过六项数值检查的同核初始化 | [同核初始化设计](../designs/gftd_bounded_initialization_20261109.md) |
| 已闭合的匹配预测Run（完整L1/L2/L3与独立审查） | [匹配预测设计](../designs/gftd_matched_prediction_20261109.md) |
| 已完成三点历史比较，静态保留父候选，有界调制选中 1/4 | [三点更新设计](../designs/gftd_objective_alignment_20261112.md) · [结果审查](../reports/gftd_objective_alignment_20261112/result_review.md) |
| 固定调制方向的补偿探针已执行并复核，两档未决 | [补偿见证设计](../designs/gftd_joint_distinguishability_20261112.md) |
| 研究假说、竞争解释与 Run 树 | [GFTD Campaign](../campaigns/gftd_kuramoto_discovery_20261106.md) |
| 未知拓扑 → 动态调制与接口 → train fit297 同期比较 | [三步主线](../designs/gftd_mainline_20261107_architecture.md) |
| 原Run 3的输入、工作量及判据 | [fit297 协议](https://github.com/LLzijian/grid-transient-rom/blob/145847532bacd3454dca8869e819778efc8258f6/reports/gftd_fit297_20261107_protocol.md) |
| v2.0 草案及其更正 | [原架构简报](../designs/gftd_kuramoto_architecture_brief.md) |

节点局域核、候选边族和耗散形式属于建模先验；边支撑、调制作用和参数恢复由证据判断。
完整方程、初始化细节和实验成绩保留在上述设计及报告中。

## 下一待辨别问题

固定调制方向的两档补偿结果均超过观测残余门。求解到限后返回的可行解提供上界，优化充分性仍为 unknown。科学执行与独立结果复核均已完成，最终主张审查通过，root 已闭合本轮。下一步只拟定已保存线性盒约束问题的可行上界和最优值下界。尚未释放新计算，不开放模型 RHS 或新评分。[独立结果审查及更正](../reports/gftd_joint_distinguishability_20261112/result_review_correction.md) · [复核记录 v2](../results/gftd_joint_distinguishability_20261112/result_audit_v2.json)

已有证据界定了问题范围。Run 1 支持有限 K4 静态恢复，Run 2 建立拟合、存取、因果初始化和自主预测接口。匹配预测完成原 train 297 的同期评价，两模型均有局部吸引集证书、实际初态各 0/297 获证、参考模态各配对 10/13。有界调制在三点比较中的 1/4 更新改善了原 24 包的历史选择目标，联合恢复、预测改善和实际初态证书覆盖仍待对应证据。[当前状态](current_state.md)

既有 40 维基线和已接受锚点保留原适用范围。default none、incumbent 与 protected anchors 不变。
历史架构见[设计档案](../designs/history/README.md)。
