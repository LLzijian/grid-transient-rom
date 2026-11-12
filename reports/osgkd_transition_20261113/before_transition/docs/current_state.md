# 当前研究状态

**2026-11-13：已按用户明确要求暂停自主研究，主目标状态为 paused。独立实现复查已中止，下一轮资格检查和科学计算均未启动；当前修改保留在本地。未发现本项目使用的 Python 3.12 或项目虚拟环境计算进程，等待用户明确恢复。**

**2026-11-12：固定 β 方向的局部补偿探针已完成执行和独立结果复核。两档均未达到见证门，结论未决。最终主张审查通过，root 已闭合本轮。**

## 固定调制方向的局部补偿

以有界调制四分之一步为诊断候选，用原 24 包，将同一补偿向量按两档尺度施加到共享系数与合法幅值、力记忆初态。扰动幅度 1/64、1/128 下，补偿后的全体归一化观测 RMS 残余分别为 2.304e-5 和 1.153e-5，均超过 1e-6。相对于各自对照，残余减少约 96.8%，两档均未通过见证门。TRF 达到 100 次外层迭代上限，101 次 LSMR 均达到 1,000 次迭代上限，优化充分性仍为 unknown。共 12,912 次历史映射、9,296,640 次模型 RHS，成本已知。[独立结果审查及更正](../reports/gftd_joint_distinguishability_20261112/result_review_correction.md) · [复核记录 v2](../results/gftd_joint_distinguishability_20261112/result_audit_v2.json)

本轮代码 `retain_branch`、模型 `retain_evidence`、科学 `continue`。工作量关闭，47 份活动源码的本轮冻结已解除，执行档案保持不变。[最终决定](../results/gftd_joint_distinguishability_20261112/final_adoption.json) · [科学判读及计时更正](../reports/gftd_joint_distinguishability_20261112/final_assessment_correction.md) · [最终主张审查](../reports/gftd_joint_distinguishability_20261112/final_claim_review.md)

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

下一轮针对已保存的固定线性问题计算精确可行上界和对偶下界。设计已通过独立审查并由 root 采用，实现已完成，独立实现复查因用户暂停中止；恢复后仍须先完成审查和资格检查。只有上下界仍跨越目标时才允许一次密集有界求解，最多 1,051 次内部线性分解。科学执行尚未释放，本轮模型 RHS 和未来评分均为 0。[有限协议](../campaigns/gftd_linear_compensation_bounds_20261112/protocol.md) · [设计采用](../results/gftd_linear_compensation_bounds_20261112/design_adoption.json)一般拓扑与隐调制联合恢复、预测能力目标和实际初态 L2 覆盖仍待解决。[研究假说](../campaigns/gftd_kuramoto_discovery_20261106.md) · [架构入口](current_design.md)

部署沿用 100 ms 因果历史、末点真实 p₀ 与冻结常数。原 val/test 关闭，旧 test 退出选择，新 FOM 为 0，TASK4 暂停。default none、incumbent 与 protected anchors 不变。
