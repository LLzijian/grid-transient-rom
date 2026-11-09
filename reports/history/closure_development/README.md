# 闭合模型的后续辨别：函数形式、训练时域与执行条件

**这组研究把三个问题逐步分开：一个函数能否改善预测、数据与训练时域能否区分机制、实现是否已经具备有效比较的执行条件。三类结果分别保留，工程通过与模型采用独立判断。**

本页于 2026-11-09 追溯整理，主要覆盖 2026-10-20—25 的材料。原协议、当时的预测与采用决定不改写；人工算例、真实 train 和已用于开发的合成 val 保留各自范围。今天的候选与授权从[当前状态](../../../docs/current_state.md)进入。

## 按科学问题阅读

| 问题 | 原记录支持的认识 | 来源 |
|---|---|---|
| 边记忆能改善某个误差，是否就应采用？ | 二次边记忆降低部分失稳误差，但稳定风险和漏检保护未通过；转速幅值耗散的稳定风险收益伴随共同频率与关键事件代价，均按原门槛未采用。 | [二次边记忆](https://github.com/LLzijian/grid-transient-rom/blob/145847532bacd3454dca8869e819778efc8258f6/reports/quadratic_memory_edge_20261020_conclusion.md)、[转速耗散](https://github.com/LLzijian/grid-transient-rom/blob/145847532bacd3454dca8869e819778efc8258f6/reports/frequency_dissipation_20261021_conclusion.md) |
| 短窗预测通过是否唯一指定机制？ | 两种预给定边记忆均可通过原短窗标准；删除新增储能后也可通过。较长自主预测给三个候选提供了额外的选择信息，不等于已恢复未知公式。 | [两机制比较](https://github.com/LLzijian/grid-transient-rom/blob/145847532bacd3454dca8869e819778efc8258f6/reports/edge_memory_discrimination_20261021_conclusion.md)、[整项删除](https://github.com/LLzijian/grid-transient-rom/blob/145847532bacd3454dca8869e819778efc8258f6/reports/edge_closure_deletion_20261021_conclusion.md)、[长时域辨别](https://github.com/LLzijian/grid-transient-rom/blob/145847532bacd3454dca8869e819778efc8258f6/reports/long_horizon_discrimination_20261021_conclusion.md) |
| 函数形式是否影响因果预测？ | 固定的 12 模型比较在 0% 与 1% 噪声下，有理端口相对 tanh 的合成 val 结果在六个配对实例同向；这是给定形式间的比较，不是算法发现未知形式。 | [完整比较](https://github.com/LLzijian/grid-transient-rom/blob/145847532bacd3454dca8869e819778efc8258f6/reports/checkpoint_continuation_20261022_conclusion.md) |
| 给出有限语法池，是否就能选中生成机制？ | 一轮因真值轨迹数值资格而未完成三实例确认；后续增长受控的六机制比较完成，但三个 train 实例均未选中生成机制。数值可区分的排序与正确机制恢复分别判断。 | [原资格受限结果](https://github.com/LLzijian/grid-transient-rom/blob/145847532bacd3454dca8869e819778efc8258f6/reports/finite_grammar_execution_20261022_conclusion.md)、[完整六机制比较](https://github.com/LLzijian/grid-transient-rom/blob/145847532bacd3454dca8869e819778efc8258f6/reports/bounded_edge_grammar_20261022_conclusion.md) |
| 训练时域如何影响机制选择？ | 固定两机制、三实例的 2048 更新比较中，0.5 s 训练窗口选择竞争机制，2 s 窗口选择植入机制。窗口与更新次数的作用分开记录；结果是既有 train 条件内的辨别，未作为真实系统独立确认。 | [训练窗口研究](https://github.com/LLzijian/grid-transient-rom/blob/145847532bacd3454dca8869e819778efc8258f6/reports/horizon_pair_20261023_conclusion.md) |

## 工程结果如何使用

- [公共乘积合并](https://github.com/LLzijian/grid-transient-rom/blob/145847532bacd3454dca8869e819778efc8258f6/reports/fused_closure_runtime_20261021_conclusion.md)支持固定 N/D 结构与登记合成完整负载上的等价计算及耗时降低；不据此重开旧成本拒绝或采用科学模型。
- [三次记忆端口](https://github.com/LLzijian/grid-transient-rom/blob/145847532bacd3454dca8869e819778efc8258f6/reports/cubic_memory_port_20261021_conclusion.md)、[语法外因果试验](https://github.com/LLzijian/grid-transient-rom/blob/145847532bacd3454dca8869e819778efc8258f6/reports/outside_grammar_causal_20261021_conclusion.md)、[有理与 tanh 原成本检查](https://github.com/LLzijian/grid-transient-rom/blob/145847532bacd3454dca8869e819778efc8258f6/reports/rational_control_20261021_conclusion.md)分别保留实现通过与正式研究未准入的原因。没有正式训练的阶段不计作预测能力的阴性比较。
- [多项式计算路径](https://github.com/LLzijian/grid-transient-rom/blob/145847532bacd3454dca8869e819778efc8258f6/reports/edge_polynomial_runtime_20261021_conclusion.md)保留未通过的配置一致性检查；通过子集不替代完整工程合同。
- [训练效率作业中断](https://github.com/LLzijian/grid-transient-rom/blob/145847532bacd3454dca8869e819778efc8258f6/reports/training_efficiency_20261024_interruption.md)保留缺少完成回执的事实，没有将未执行的 opt1 写成训练加速成果。[热路径审查](https://github.com/LLzijian/grid-transient-rom/blob/145847532bacd3454dca8869e819778efc8258f6/reports/code_efficiency_inspection_20261025.md)与该作业的冻结来源分别保存。
- [进程监督与恢复主题](../../maintenance/execution_runtime/README.md)解释执行器、身份检查和检查点故障，避免将它们归因于物理机制。

## 原件与继承关系

工程与执行记录见[边幅值 QA](../../../results/edge_memory_amplitude_20261021/qa01_summary.json)、[多项式 QA](../../../results/edge_polynomial_runtime_20261021/qa02_summary.json)、[有限语法执行回执](../../../results/finite_grammar_execution_20261022/completion_summary.json)及[饱和端口核验结论](https://github.com/LLzijian/grid-transient-rom/blob/145847532bacd3454dca8869e819778efc8258f6/reports/tanh_port_runtime_20261019_conclusion.md)。回执完成状态与上文的科学结论分别解释。

[材料表](catalog.md)列出本主题原件、原标题、原位置及固定提交阅读入口；[保护表](protected.md)列出仍受路径约束的相关材料。跟随冻结原文内部链接时使用固定提交版本，本地原件不改写链接。

这一阶段继承 [A—E 闭合发现](../closure_discovery/README.md)，后续进入 [R-memory、空间作用与 p 通道](../force_memory/README.md)。这些联系描述研究问题的推进，不补造历史 Run 或独立确认。

5份规划原始返回已与原整理稿同目录保存，位置见[第七批原件表](../../../results/repository_organization/20261109_history07/originals.md)。QA05 原始返回已完成[旧路径恢复核验](../../../results/repository_organization/20261109_rational_runtime_recovery23/README.md)并归档。

另存有[2026-10-25外部项目综合咨询](../../consensus_project_review_20261025/README.md)，与正式Run结论及独立审查分别阅读。

[历史报告恢复核验](../../../results/repository_organization/20261109_closure_20261020_runtime_recovery31/README.md)列出旧路径、冻结 SHA256 和原校验入口。
