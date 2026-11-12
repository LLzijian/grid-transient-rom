# 当前研究状态

**2026-11-12：匹配预测与三点历史比较均已完成执行和独立结果审查。三点比较中，静态保留父候选，有界调制的四分之一步在原 24 包的历史选择目标上改善 0.135032%，选中该步并结束这条方向的搜索。下一问是调制系数变化能否由共享静态系数和合法隐初态补偿，当前在准备有界设计。**

## 三点历史比较

六个候选全部完成，144/144 套原六门通过。静态三个尺度均无合格改善；有界调制的 1/4、1/8 均通过原严格改进门，1/4 的目标由 0.036866139234346 降至 0.036816358081140。该目标是 24 包等权归一化历史 MSE 加开放项惩罚。1/4 的未惩罚 MSE 在 10 包下降、14 包上升。旧端点和有序 24 包身份均已核对。科学 RHS 56,004,480 次，包含 288 个优化任务和 144 次固定父坐标残差诊断的成本。三点比较未产生新的真实预测、L2 覆盖或联合恢复证据。[协议](../campaigns/gftd_objective_alignment_20261112/protocol.md) · [终态回执](../results/gftd_objective_alignment_20261112/science/execution.json) · [科学判读](../reports/gftd_objective_alignment_20261112/final_assessment.md) · [独立审查](../reports/gftd_objective_alignment_20261112/result_review.md)

决定为代码 `retain_branch`、模型 `retain_evidence`、科学 `continue`。root 已正式闭合本轮，工作量已关闭，45 份执行源码的活动冻结已解除。后续研究须另行冻结和审查。[最终决定](../results/gftd_objective_alignment_20261112/final_adoption.json) · [源码档案](../results/gftd_objective_alignment_20261112/source_archive.json)

## 匹配预测已闭合

在同一 train fit297 上完成两路 fit、594 次部署、L2/L3 与一次真实未来评分。两模型在 fit 后均保留各自父参数，联合辨识和优化充分性为 unknown。完整可比范围的 risk 未支持有界调制总体收益，真实可用区间多检出 4 个失稳样本。原 Run 3 的总体 L1 仍按其原数值资格保留 null。[最终决定](../results/gftd_matched_prediction_20261109/final_adoption.json) · [科学判读](../reports/gftd_matched_prediction_20261109/final_assessment.md) · [部署摘要](../results/gftd_matched_prediction_20261109/deploy_digest.json) · [L1 摘要](../results/gftd_matched_prediction_20261109/l1_digest.json)

| train 范围／指标 | 静态 | 有界调制 |
|---|---:|---:|
| 初始化原六门 | 297/297 | 297/297 |
| 完整 1 s 严格参考确认 | 297/297 | 289/297，8 个到限未决 |
| 稳定 267，5 s risk | 12.312022 | null，264/267 合格 |
| 失稳 30，真实区间 risk | 9.612748 | 11.805068 |
| 失稳 30，真实区间漏检 | 21/30 | 17/30 |
| 失稳 30，8 s 漏检 | 7/30 | 7/30，事件身份不同 |
| L2 正式初态获证 | 0/297 | 0/297 |
| L3 参考模态配对 | 10/13 | 10/13 |

risk 是逐事件时间平均归一化平方误差的加权风险。失稳样本实际评分时长为部署后 0.03–1.44 s。静态 3 个误报在调制侧均为数值未决；调制稳定 5/8 s 均保留原分母。两模型均有局部 ROM 吸引集证书，全部正式初态均未获证书覆盖，确定性初始化误差界为 null。L3 用于描述物理对应。[独立主张审查](../reports/gftd_matched_prediction_20261109/final_claim_review.md)

## 下一问与边界

以有界调制四分之一步为诊断工作候选，在该点附近检验沿父候选到四分之一步的固定 β 方向能否在合法 100 ms 完整自治历史中由共享静态系数 c 和盒内幅值、力记忆初态 a/r 补偿。后续 Run 为 `gftd_joint_distinguishability_20261112`，当前准备有界设计，尚未释放科学执行。一般拓扑与隐调制联合恢复、预测能力目标和实际初态 L2 覆盖仍待解决。[目标与证据更新](../reports/gftd_objective_alignment_20261112/goal_evidence_update.md) · [研究假说](../campaigns/gftd_kuramoto_discovery_20261106.md) · [架构入口](current_design.md)

部署沿用 100 ms 因果历史、末点真实 p₀ 与冻结常数。原 val/test 关闭，旧 test 退出选择，新 FOM 为 0，TASK4 暂停。default none、incumbent 与 protected anchors 不变。
