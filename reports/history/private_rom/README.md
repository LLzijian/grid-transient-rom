# 早期私有模型：机械功率、记忆闭合与结构诊断

**这条研究线从统一数据合同上的 M2／M3 基线出发，逐步检查机械功率动态、记忆状态、初始化、能量边界和模态对应。接受记录与未采用方案共同构成证据；它们分别回答具体条件下的问题，不能合成一次模型族的整体成功或失败。**

本页于 2026-11-08 追溯整理，覆盖原 Loop0017–0106 及相关 TASK1／TASK2 材料。前批收拢 Loop0017–0097，本批补上后续联合训练与初始化研究的阅读入口。
早期协议包含 50 ms 等不同部署版本；原协议是历史来源，当前合同与授权以 [kiki.md](../../../kiki.md) 和[当前研究状态](../../../docs/current_state.md)为准。

## 从科学问题进入

| 问题 | 这组材料留下的认识 | 先读哪些记录 |
|---|---|---|
| 数据可用于研究，是否意味着模型已通过？ | smoke／Pilot 接受的是数据链与 QA；M2 接受为比较基准，M3 的机械功率贡献通过当时的 val 筛选。三种接受对象各不相同。 | [数据链](https://github.com/LLzijian/grid-transient-rom/blob/145847532bacd3454dca8869e819778efc8258f6/reports/accepted/0017_private_smoke.md)、[Pilot](https://github.com/LLzijian/grid-transient-rom/blob/145847532bacd3454dca8869e819778efc8258f6/reports/accepted/0018_private_pilot.md)、[M2](https://github.com/LLzijian/grid-transient-rom/blob/145847532bacd3454dca8869e819778efc8258f6/reports/accepted/0019_private_m2.md)、[M3](https://github.com/LLzijian/grid-transient-rom/blob/145847532bacd3454dca8869e819778efc8258f6/reports/accepted/0020_private_m3_pm.md) |
| 能从残差找到低维量，是否就能得到自主闭合？ | 初态的非因果对照显示初始化有改进空间，时变力反演又暴露单一衰减状态的表达缺口；这两种解释需要分别检验。 | [初态与结构对照](https://github.com/LLzijian/grid-transient-rom/blob/145847532bacd3454dca8869e819778efc8258f6/reports/accepted/0027_memory_oracles.md)、[非中心化对照](https://github.com/LLzijian/grid-transient-rom/blob/145847532bacd3454dca8869e819778efc8258f6/reports/accepted/0030_uncentered_oracle.md)、[方案全表](catalog.md) |
| 局部模态、能量界与真实系统安全是什么关系？ | 冻结 M3 的局部谱与保守能量界各有明确数学对象；全阶谱还经历了约化伪模态的纠正。这些证据保留各自 ROM 范围和覆盖条件。 | [模态审计](https://github.com/LLzijian/grid-transient-rom/blob/145847532bacd3454dca8869e819778efc8258f6/reports/accepted/0035_m3_modes.md)、[能量界](https://github.com/LLzijian/grid-transient-rom/blob/145847532bacd3454dca8869e819778efc8258f6/reports/accepted/0036_m3_energy_boundary.md)、[方向相关边界](https://github.com/LLzijian/grid-transient-rom/blob/145847532bacd3454dca8869e819778efc8258f6/reports/accepted/0037_m3_reachable_boundary.md) |
| 轨迹改善是否足以通过完整评价？ | Pilot 的正式 test 检查中，轨迹相对贡献通过，而新增失稳漏检使联合主张未通过；该 test 随后已消费，不能用于新模型选择。 | [原确认记录](https://github.com/LLzijian/grid-transient-rom/blob/145847532bacd3454dca8869e819778efc8258f6/reports/failed/0038_private_confirmatory.md) |
| 闭合误差应归于表示、参数还是初始化？ | 后续空间力投影、隐状态阶数和联合训练记录逐项检查这些解释。例如 0064 支持其冻结 train 条件下以已有作用空间内的失配为主，不能外推为所有模型都应维持同一状态维数。 | [物理力投影](https://github.com/LLzijian/grid-transient-rom/blob/145847532bacd3454dca8869e819778efc8258f6/reports/accepted/0064_physical_force_projection.md)、[隐状态阶数](https://github.com/LLzijian/grid-transient-rom/blob/145847532bacd3454dca8869e819778efc8258f6/reports/accepted/0059_latent_order_audit.md)、[长时程记录](../../accepted/0060_long_horizon_campaign.md) |
| 同一梯度结构通过实现检查，是否已经完成结构发现？ | 0076 接受的是实现与模板表达能力检查。后续系数、局部稳定性和模态平衡诊断仍需按各自冻结对象解释，不能由一次诊断直接宣布模型晋升。 | [结构原型](https://github.com/LLzijian/grid-transient-rom/blob/145847532bacd3454dca8869e819778efc8258f6/reports/accepted/0076_structure_prototype.md)、[复方向模态平衡](https://github.com/LLzijian/grid-transient-rom/blob/145847532bacd3454dca8869e819778efc8258f6/reports/accepted/0097_complex_balance.md) |

上表直接引用原报告。本次没有重算指标、加载模型或重新读取原始数据；也没有把后来已暴露的数据重新标为独立确认。

## 后续联合训练与初始化：Loop0098–0106

这部分进一步区分优化充分性、数值资格与初态推断。以下是原决定的阅读摘要，没有重新评价或选择模型。

| 问题 | 这组记录的结论范围 | 来源 |
|---|---|---|
| 继续联合训练是否得到完整目标上的有效比较？ | 0099 两组各追加 600 次更新，L10 完整目标仍数值未决；0100 的共同 100 步细步长比较未恢复完整目标资格。两轮均不构成模型晋升或模型族否证。 | [0099](https://github.com/LLzijian/grid-transient-rom/blob/145847532bacd3454dca8869e819778efc8258f6/reports/0099_joint_adequacy_capability.md)、[0100](https://github.com/LLzijian/grid-transient-rom/blob/145847532bacd3454dca8869e819778efc8258f6/reports/0100_joint_numerical_pair_capability.md) |
| 结构研究候选与普通预测基线是什么关系？ | 0102 分别保留 S-O 作为结构研究候选、F-O 作为普通稀疏 L1 基线；0103–0104 的端点没有晋升，不能混成一个“当前最好模型”。 | [0102](https://github.com/LLzijian/grid-transient-rom/blob/145847532bacd3454dca8869e819778efc8258f6/reports/0102_prediction_optimization_capability.md)、[0103](https://github.com/LLzijian/grid-transient-rom/blob/145847532bacd3454dca8869e819778efc8258f6/reports/0103_cross_dissipation_capability.md)、[0104](https://github.com/LLzijian/grid-transient-rom/blob/145847532bacd3454dca8869e819778efc8258f6/reports/0104_lr_adaptation_capability.md) |
| 离线初态改善能否转成因果预测收益？ | 0105 固定模型、12 条 train 事件上的未来真值初态优化显示乐观余地；0106 的短程单组探索未得到可用因果端点。两个结果的对象、样本范围与指标不同，不能串成一项改进率。 | [0105](https://github.com/LLzijian/grid-transient-rom/blob/145847532bacd3454dca8869e819778efc8258f6/reports/0105_q0_scope_conclusion.md)、[0106](https://github.com/LLzijian/grid-transient-rom/blob/145847532bacd3454dca8869e819778efc8258f6/reports/0106_oracle_distillation_pilot_capability.md) |

[阶段状态原快照](https://github.com/LLzijian/grid-transient-rom/blob/145847532bacd3454dca8869e819778efc8258f6/reports/history/current_state_before_closure_discovery_20261017.md)保留当时的起点、关闭原因和恢复线索；其中 RUNNING、BLOCKED 及预算只描述历史时点。[后续闭合方程研究](../closure_discovery/README.md)另按主题阅读。

## 接受、未采用与不可判定分别保留

- **数据、代码或基线接受**：说明对应合同或检查通过，不能自动扩大成预测和泛化结论。
- **诊断接受**：可以排除或加强一个限定的竞争解释，未必改变模型参数或采用决定。
- **候选未采用**：保留具体门槛、比较对象和失败原因，不据文件夹名称否定整个方法族。
- **实现偏离、优化状态或资源检查未通过**：原记录有单独身份，不统计成一次有效的科学比较。

[完整材料表](catalog.md)同时列出原协议、接受／拒绝记录、原始标题和机读证据；原分类作为历史属性保留。
机读证据从旧 `reports/` 收入 `results/private_rom_history/`，不混入当前 Run 的结果目录。原始私有数据、模型、预测数组和代码位置保持不变。

## 有路径依赖的材料为什么仍在原处

[原位保护入口](protected.md)连接最新全表、具体引用者和解除条件。前批数量属于当时范围；跨项目综述在最新总表按主主题展示，不把重新分类当作解除依赖。
其中包括当前论文状态仍引用的早期总审计、旧项目总览绑定的来源、脚本直接读取的证据，以及受保护记录继续引用的协议。
“被引用”和“被接受”是不同属性，保护不改变任何科学判断。

## 原文与后续入口

归档原件保持本地字节；部分原件的 Windows 换行与 Git 版本不同，固定提交已核对为仅换行不同、文本一致。跟随原文内部链接时，使用材料表中的固定提交版本；本地相对链接没有通过改写原件来修补。

- [初代公开算例与纠正链](../early_e_series/README.md)
- [旧论文与改写来源](../paper/README.md)
- [历史设计入口](../../../designs/history/README.md)：设计原件已经在此前批次整理，本轮不移动或改写。
- [本批迁移与当前累计路径](../../../results/repository_organization/20261108_early_history04/migration.json)
- [整理批次总表](../../../results/repository_organization/README.md)

已闭合的15个历史代码入口另见[代码档案](../../../experiments/history/private_rom/README.md)；报告与科学库原始字节不变，旧执行按完整旧版本恢复。

第八批在前序代码／概览归档后复核保护边界，再解除6份报告／证据的旧路径依赖；[最新累计位置](../../../results/repository_organization/20261109_history08/migration.json)与本主题保护表接续原记录。
