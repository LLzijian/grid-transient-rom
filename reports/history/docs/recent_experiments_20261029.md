# 近期实验材料导航（2026-10-29）

**当前科研状态：PAUSED_BY_USER。** 用户要求本轮结束后暂停。记忆转运轮次已实际结束，五个原生阶段均正常闭合，活动锁已移除，临时保活任务已删除且配置无残留；当前无运行中的科研作业，也无保活定时任务。本页用于查阅已完成材料，不启动后续工作。状态入口见[当前状态](current_state.md)。

## 五轮问题、结论与证据

下表调用数均为各轮实际完成的 optimizer 调用；NNLS单列，不与训练调用相加。

| 阶段与问题 | 决定及保留范围 | 实际调用 | 材料入口 |
|---|---|---|---|
| **force_graph_20261028**：空间作用应进入静态耦合还是记忆传播？ | 两个来源中，传播记忆M相对静态S具有稳定风险与8s漏判收益；共同频率和p误差存在权衡。M作为研究起点保留，未采用科学默认模型。 | **1020**＝QA108＋校准48＋正式864 | [方案](../reports/force_graph_20261028_protocol.md) · [结论](../reports/force_graph_20261028_conclusion.md) · [公开证据](../results/force_graph_20261028/completed_stage_public.json) |
| **force_exchange_20261028**：单参数p-r互易交换是否有价值？ | 同源共同96时，C1相对C0改善失稳风险与8s漏判；未满足冻结父模型的稳定精度保护，未保留为默认方案，无延长或第二种子。C1探索资产与局部收益保全。 | **264**＝QA72＋正式192 | [方案](../reports/force_exchange_20261028_protocol.md) · [结论](../reports/force_exchange_20261028_conclusion.md) · [公开证据](../results/force_exchange_20261028/completed_stage_public.json) |
| **state_gated_memory_20261028**：角差调制是否改善有效传播？ | A/B/C各96，真实区间漏判均21/30；逐边C的局部门已起作用，但原决定为 **NOT_RETAINED**，无延长或第二种子。A96保留为失稳风险方面的科学候选，R_N精度与关键事件参照不变。 | **396**＝QA108＋正式288 | [方案](../reports/state_gated_memory_20261028_protocol.md) · [结论](../reports/state_gated_memory_20261028_conclusion.md) · [公开证据](../results/state_gated_memory_20261028/completed_stage_public.json) |
| **weak_force_identification_20261029**：消去隐力后，一次直接辨识能否提供自治训练起点？ | 人工已知ROM恢复通过；唯一真实弱解λ＝0，按严格正参数门记 **CANDIDATE_NOT_ADMITTED**。没有D/W训练比较，不救回边界解，也不将未执行组记作模型失败。 | **optimizer 0**（QA0、正式0）；人工NNLS **4**，真实NNLS **1** | [方案](../reports/weak_force_identification_20261029_protocol.md) · [结论](../reports/weak_force_identification_20261029_conclusion.md) · [公开证据](../results/weak_force_identification_20261029/completed_stage_public.json) |
| **memory_transport_20261029**：固定二维平面的保守转运S能否优于原A和耗散路径P？ | A/S/P各128，原共同128决定为 **NOT_RETAINED**；无延长或第二种子。三组端点及不利证据保全，不晋升。P是预定对称路径，不能称独立容量匹配。 | **492**＝QA108＋正式384；NNLS **0** | [方案](../reports/memory_transport_20261029_protocol.md) · [结论](../reports/memory_transport_20261029_conclusion.md) · [公开证据](../results/memory_transport_20261029/completed_stage_public.json) |

## 模型身份与保全位置

- **可用模型档案**保持各自原接受范围；这些研究轮次没有替换可用模型或新增科学默认模型，默认仍为`none`。
- **科学参照与来源**分开：R_N是N_M192参照；记忆转运的唯一来源是N_M192→SGM A_28311 step96（A96），冻结执行源为`181d7f8bba2eb530a0ad5afe0130035755e974a2`。来源选择不等于最终采用。
- **候选资产**包括M-A512、M-N、交换C1、SGM各端点与本轮A/S/P端点。其检查点、学习曲线、主细预测及恢复证据按原位置保留；资产归档不自动赋予活跃前沿或部署身份。原失败分类、试算后丢弃的更新诊断及未采用性能原型均不改判。

公开包提供各轮指标、分母、来源与原始证据SHA。原始事件、预测数组和Adam状态仍按既有本地保全规则管理；本页只增加链接，不移动或重命名绑定证据。未用额度按原轮关闭，不转入后续轮次。成本使用各轮原生Job的既定范围，不相加声称整个项目成本；旧r_memory为`INTERRUPTED_RUNTIME`且存在未知成本，外层controller与前置准备也须分列。train fit297、报告用train385与noise88不混作独立确认；风险、物理单位RMSE、归一化RMSE及训练J各沿原定义。

main已合入[PR34规则更新](https://github.com/LLzijian/grid-transient-rom/pull/34)与[PR35条件证明](https://github.com/LLzijian/grid-transient-rom/pull/35)，未采用新的科学模型。这些结果不等于L1/L2完成、真实物理图识别或同精度成本胜出。原val/test关闭，TASK4暂停，新FOM条件0。最新Astra的RC条件准备尚未采用、未分配预算；历史报告中的后续动作保留为当时记录，当前以用户明确暂停为准。
