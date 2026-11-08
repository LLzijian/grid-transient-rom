# 初代暂态模型：模块选择、能量筛查与误差诊断

**这组研究留下了三个可继续使用的认识：模块需要与重新拟合的主干按同一尺度比较；局部能量候选不能替代稳定性证明；预测误差需要通过物理对照定位，不能由拟合损失或平均分数解释。**

这里整理 2026-09-04 至 09-08 的 E 系列及其后续纠正，沿用 09-22 保留审计确认的解释。
本页是 2026-11-08 补建的历史综述，不是新实验或当前模型采用决定。当前路线见[当前研究状态](../../../docs/current_state.md)。

## 先看这四个问题

数值模态结果可从[纠正后的 e11 摘要](../../../results/e11_modal_validation_v2/summary.json)回查，适用范围仍以综合审计为准，不作为新确认。

| 问题 | 后续记录支持的认识 | 优先阅读 |
|---|---|---|
| 加入记忆模块，是否在所有交叉验证组都值得采用？ | 修正总采用门后，F1/F3/F5 采用模块，F2/F4 回退主干；五组交叉验证的结果不能写成所有组都采用。 | [五组重拟合验收](originals/accepted/0009_five_fold_refit_acceptance.md) |
| 有能量函数、找到鞍点，就能判断稳定吗？ | 旧算法取得的是局部可行能量候选，未建立逃逸边界的有效下界；保留启发式筛查，撤回正式证书解释。 | [候选能量与稳定性证书的区别](originals/accepted/0002_candidate_energy_not_certificate.md) |
| 旧频率精度和 test 表能否直接用作成果？ | 模态须使用最终模型与全部指派对；旧 test 已被消费，继续保留为探索性记录。 | [评价协议修正](originals/accepted/0003_holdout_lifecycle_and_modal_reporting.md)、[综合审计](../../accepted/0010_project_conclusion_audit.md) |
| 大误差主要是积分步长、Pm 输入，还是隐状态初值？ | 在当时冻结模型和对照范围内，缩小步长、给定真实未来 Pm、改用零隐初值均不足以消除主要误差；电气信息有解释价值，尚未唯一定位缺失机制。 | [积分与输入对照](originals/accepted/0011_failure_mechanism_diagnosis.md)、[隐初值对照](originals/accepted/0012_force_balance_diagnosis.md)、[电气功率对照](originals/accepted/0013_electrical_substitution_audit.md) |

## 模块选择：把比较口径与模型精度分开

早期 E2 形成了“主干拟合 → 候选模块提议 → 联合拟合 → 整块删除”的实现与开发记录。
后续增加了总采用门：最终候选必须相对重新拟合的主干，取得至少 2% 的标准化 val 损失改善，且滚动失败率不增加。
删除某一模块不利于局部结果，并不自动说明整个模块模型优于主干。[采用门修正](originals/accepted/0001_pruning_horizons_and_module_adoption.md)

随后发现，模型保存后没有保留训练归一化尺度，导致一次 F1 比较使用了不同分母。
该次结果被拒绝；修复保存和载入后重新比较，接受 F1 重跑。**这里接受的是同口径比较下的结果，不能把两次不同尺度的分数写成模型性能改进。**
[被拒绝的尺度比较](originals/failed/0007_f1_prune_validation_scale_roundtrip.md) → [修复后的 F1 重跑](originals/accepted/0008_f1_prune_rerun_acceptance.md) → [完整重拟合验收](originals/accepted/0009_five_fold_refit_acceptance.md)

完整重拟合按 F1–F5 分成五组，每次用对应的 val 评价。其历史采用结果为：

| 比较组 | 当时的模块采用决定 | H100 合格／声明 val 样本 | 最终 pooled R² |
|---|---|---:|---:|
| F1 | 采用 6 个可选实例 | 34／43 | 0.765788 |
| F2 | 回退重新拟合的主干 | 32／37 | 0.713422 |
| F3 | 采用 5 个可选实例 | 34／42 | 0.857565 |
| F4 | 回退重新拟合的主干 | 38／43 | 0.676278 |
| F5 | 采用 6 个可选实例 | 39／47 | -1.698347 |

来源：[0009 验收原表](originals/accepted/0009_five_fold_refit_acceptance.md)。H100 指 100 个积分区间、101 个采样点，实际终点约 1 s。
此处 pooled R² 沿用原报告合并角度与转速的定义，不是带统一物理单位的 RMSE。
合格的 177 个样本全部稳定，排除的 35 个样本全部失稳；这张表不能表示失稳样本的固定时长预测能力。

这组结果说明了“采用门通过”和“绝对预测精度足够”是两个判断；F5 的结果也完整进入了接受的数值记录。
[综合审计](../../accepted/0010_project_conclusion_audit.md)进一步限定：这些是用于选择后的 val 诊断，不能替代独立 test 确认。

## 能量与模态：保留数值观察，采用修正后的解释

**能量筛查。** 旧 E4/E6/E8 使用局部约束优化和不完整鞍点搜索构造能量裕度。
后续审查指出，局部可行值给出的界方向不足以证明稳定区域；旧“证书 PASS／获证判稳”退出有效解释。
历史 51 个失稳样本中没有被筛为稳定的观察仍被保留，但独立来源／故障线路组只有 9 个，不能转述为零风险保证。
[原问题与修正推导](originals/accepted/0002_candidate_energy_not_certificate.md)

**模态对应。** E5/E5b/E5c 的旧标题采用过每事件最佳匹配及旧模型。
修正诊断使用最终模型，保留完整指派中的全部匹配对，再按事件汇总。
177 个稳定合格事件中 167 个计算成功、10 个失败；成功事件的频率相对误差汇总值为 19.49%、阻尼比绝对误差为 0.07537、MAC 为 0.37647。
这些指标各有定义，不能与旧“7.2%／0.04”标题作同口径升降比较，也不能将失败事件算作已覆盖。
[模态协议](originals/accepted/0003_holdout_lifecycle_and_modal_reporting.md)、[结果与分母](../../accepted/0010_project_conclusion_audit.md)

**test 使用。** 旧 315 事件 build 经 E6b/E6c/E8 多次使用，已退出确认性用途。
整理可以保留全部旧记录，不能恢复这个 build 的独立性。[访问与用途修正](originals/accepted/0003_holdout_lifecycle_and_modal_reporting.md)

## 接受与未接受：各自在研究中留下了什么

下面列的是当时有出处的决定。`accepted/`、`failed/` 只是原存放位置，不代替这些决定的对象与范围。

| 当时的处理 | 材料及用途 | 此后如何使用 |
|---|---|---|
| 接受实现／解释修正 | [物理时域与总采用门](originals/accepted/0001_pruning_horizons_and_module_adoption.md)、[证书语义](originals/accepted/0002_candidate_energy_not_certificate.md)、[test／模态口径](originals/accepted/0003_holdout_lifecycle_and_modal_reporting.md) | 理解原实验口径；不等于采用当前模型 |
| 接受数值记录 | [F1 重跑](originals/accepted/0008_f1_prune_rerun_acceptance.md)、[五组重拟合](originals/accepted/0009_five_fold_refit_acceptance.md) | 保留正、负和回退结果；0008 的局部过强解释以 0009 为准 |
| 接受诊断证据，未晋升预测器 | [容量与积分检查](originals/accepted/0011_failure_mechanism_diagnosis.md)、[力平衡](originals/accepted/0012_force_balance_diagnosis.md)、[电气功率替换](originals/accepted/0013_electrical_substitution_audit.md) | 区分竞争解释；未来真值只用于离线诊断 |
| 拒绝不合法比较 | [归一化尺度遗漏](originals/failed/0007_f1_prune_validation_scale_roundtrip.md) | 保存错误原因、纠正方法和被撤回的比较 |
| 未采用候选，保留结果 | [真实 Pm 电气重拟合](originals/failed/0014_causal_pm_integral_refit.md) | 该次登记精度门未通过；不外推为 Pm 动态无用 |
| 未采用候选，保留结果 | [因果隐初态学习](originals/failed/0015_learned_initial_storage.md) | 平均指标改善但尾部恶化，未通过预设门；不外推为隐初态无法学习 |
| 未采用候选，保留结果 | [动力学与隐初态联合优化](originals/failed/0016_joint_rollout_refinement.md) | 该次 H100 尾部门未通过；原记录转向后续私有模型研究，不是当前待执行指令 |

其中 0011 对纯正弦耦合、固定机械输入的模型检查了割集容量必要条件，并将 F1 的部分平衡点失败定位为该冻结模型的静态不可行。
同一诊断中，F5 通过所检验的静态容量条件，却仍有大动态误差。因此这两类问题被分开处理。
后续电气功率替换的观察提供了下一步检查方向，没有把真实未来电气量变成部署输入。

## 哪些旧解释已经修订

| 旧解释 | 阅读时采用的更正 | 更正依据 |
|---|---|---|
| 所有组都保留三类模块 | F1/F3/F5 采用，F2/F4 回退 | [0009](originals/accepted/0009_five_fold_refit_acceptance.md) |
| 残差比 1.018 意味着解释了 98% 动力学 | 它表示残差冲量与机械输入冲量尺度相近，不能换算解释比例 | [0005](../paper/originals/accepted/0005_manuscript_scientific_consistency.md) |
| E3 的长时程结果已是审计后的模块采用 | 保留为原 F1、原时间范围和来源限制下的开发诊断 | [0005](../paper/originals/accepted/0005_manuscript_scientific_consistency.md) |
| 最终系数字节相同，证明中间删除路径未受尺度问题影响 | 只证明该次最终输出相同，不能证明中间决策等价；F1 重跑接受决定仍保留 | [0009 对 0008 的更正](originals/accepted/0009_five_fold_refit_acceptance.md) |
| 有局部能量候选就有稳定性证书 | 保留启发式筛查，正式证书需要另外满足证明条件 | [0002](originals/accepted/0002_candidate_energy_not_certificate.md) |
| E7 证明了某类基线普遍不可用、只有结构模型能外推 | 本页只使用原 F1、原预算与原评价条件下的对照记录；不沿用方法族优劣或正式证书的扩大解释 | [E7 原记录](originals/E7_baselines.md)、[后续审计](originals/records_retention_audit_20260922.md) |

## 按问题查原报告

13 份 E 报告均已接入本页；原件中“当前”“下一步”和“PASS”属于当时时点。

| 问题／用途 | 本地冻结原件 | 应配合阅读 |
|---|---|---|
| 数据通路和因果输入是否接通 | [数据检查 E0](originals/E0_data_check.md) | 历史实现 QA；不继承旧部署合同 |
| 纯摆方程主干的早期基线 | [基座复现 E1](originals/E1_m0_backbone.md) | 单组开发结果 |
| 从残差提议候选模块 | [候选提议 E2](originals/E2_modular_backbone.md) | [残差比例解释更正](../paper/originals/accepted/0005_manuscript_scientific_consistency.md) |
| 全模块联合拟合 | [联合拟合 E2](originals/E2_modular_full.md) | [后续重拟合](originals/accepted/0009_five_fold_refit_acceptance.md) |
| 删除哪些模块、最后是否采用 | [删除审计 E2](originals/E2_modular_prune.md) | [总采用门](originals/accepted/0001_pruning_horizons_and_module_adoption.md)、[重拟合结果](originals/accepted/0009_five_fold_refit_acceptance.md) |
| 机械功率动态与时间范围 | [Pm 方程 E3](originals/E3_pm_ode.md) | [历史开发诊断的采用边界](../paper/originals/accepted/0005_manuscript_scientific_consistency.md) |
| 能量候选与筛查裕度 | [能量筛查 E4](originals/E4_certificates_margin.md) | [证书语义更正](originals/accepted/0002_candidate_energy_not_certificate.md) |
| 单组模态对应 | [频率与振型 E5](originals/E5_oscillation_frequency.md) | [全匹配口径](originals/accepted/0003_holdout_lifecycle_and_modal_reporting.md) |
| 多组频率对应 | [频率汇总 E5b](originals/E5b_frequency_crossfold.md) | [修正模态结果](../../accepted/0010_project_conclusion_audit.md) |
| 多组阻尼对应 | [阻尼汇总 E5c](originals/E5c_damping_crossfold.md) | [修正模态结果](../../accepted/0010_project_conclusion_audit.md) |
| 原多组比较和旧 test | [原汇总 E6](originals/E6_fold_confirmation.md) | [test 用途](originals/accepted/0003_holdout_lifecycle_and_modal_reporting.md)、[综合审计](../../accepted/0010_project_conclusion_audit.md) |
| 有限预算基线对照 | [基线比较 E7](originals/E7_baselines.md) | 单组开发对照；证书和模态解释采用后续更正 |
| 旧能量筛查的阈值敏感性 | [筛查变化 E8](originals/E8_margin_frontier.md) | [候选能量解释](originals/accepted/0002_candidate_energy_not_certificate.md) |

关键数值来源保持原位：[重拟合产物](../../../results/e10_refit_v2/)、[修正模态](../../../results/e11_modal_validation_v2/)、[综合审计结果](../../../results/e12_refit_audit/summary.json)、[失败诊断](../../../results/e13_failure_diagnosis/summary.json)、[力平衡](../../../results/e14_force_balance/summary.json)、[电气替换](../../../results/e15_electrical_balance/summary.json)。

## 本次归档与原文阅读

13 份 E 报告已全部移入 `originals/`。第二批将剩余 7 份及引用它们的保留审计作为同一引用组归档，文件名和字节保留，持续维护的阅读入口已更新。
本批进一步收拢可移动的接受／拒绝记录及机读证据，原存放分类和来源关系保留；[补充材料表](catalog.md)提供全部入口。研究代码、私有模型与原始数据位置保持不变，不新增采用决定。
归档原件目录已设置 Git 不转换换行，后续 Windows 检出仍保持原字节。

移动原件的正文相对链接仍以旧位置为基准。**需要跟随原文内部链接时，使用下面经过 Git 字节核对的固定提交版本**；本地原件用于原文检查与恢复。

| 原文 | 保留旧路径语境的阅读版本 |
|---|---|
| 数据检查 E0 | [阅读原文](https://github.com/LLzijian/grid-transient-rom/blob/145847532bacd3454dca8869e819778efc8258f6/reports/E0_data_check.md) |
| 主干复现 E1 | [阅读原文](https://github.com/LLzijian/grid-transient-rom/blob/145847532bacd3454dca8869e819778efc8258f6/reports/E1_m0_backbone.md) |
| 候选模块提议 E2 | [阅读原文](https://github.com/LLzijian/grid-transient-rom/blob/145847532bacd3454dca8869e819778efc8258f6/reports/E2_modular_backbone.md) |
| 联合拟合 E2 | [阅读原文](https://github.com/LLzijian/grid-transient-rom/blob/145847532bacd3454dca8869e819778efc8258f6/reports/E2_modular_full.md) |
| 机械功率动态 E3 | [阅读原文](https://github.com/LLzijian/grid-transient-rom/blob/145847532bacd3454dca8869e819778efc8258f6/reports/E3_pm_ode.md) |
| 基线对照 E7 | [阅读原文](https://github.com/LLzijian/grid-transient-rom/blob/145847532bacd3454dca8869e819778efc8258f6/reports/E7_baselines.md) |
| 删除审计 E2 | [阅读原文](https://github.com/LLzijian/grid-transient-rom/blob/145847532bacd3454dca8869e819778efc8258f6/reports/E2_modular_prune.md) |
| 能量筛查 E4 | [阅读原文](https://github.com/LLzijian/grid-transient-rom/blob/145847532bacd3454dca8869e819778efc8258f6/reports/E4_certificates_margin.md) |
| 频率与振型 E5 | [阅读原文](https://github.com/LLzijian/grid-transient-rom/blob/145847532bacd3454dca8869e819778efc8258f6/reports/E5_oscillation_frequency.md) |
| 多组频率 E5b | [阅读原文](https://github.com/LLzijian/grid-transient-rom/blob/145847532bacd3454dca8869e819778efc8258f6/reports/E5b_frequency_crossfold.md) |
| 多组阻尼 E5c | [阅读原文](https://github.com/LLzijian/grid-transient-rom/blob/145847532bacd3454dca8869e819778efc8258f6/reports/E5c_damping_crossfold.md) |
| 原多组比较 E6 | [阅读原文](https://github.com/LLzijian/grid-transient-rom/blob/145847532bacd3454dca8869e819778efc8258f6/reports/E6_fold_confirmation.md) |
| 阈值敏感性 E8 | [阅读原文](https://github.com/LLzijian/grid-transient-rom/blob/145847532bacd3454dca8869e819778efc8258f6/reports/E8_margin_frontier.md) |
| 保留审计与旧纠正链 | [阅读原文](https://github.com/LLzijian/grid-transient-rom/blob/145847532bacd3454dca8869e819778efc8258f6/reports/records_retention_audit_20260922.md) |

[第三批迁移与当前累计位置](../../../results/repository_organization/20261108_early_history03/migration.json)、[第二批回执](../../../results/repository_organization/20261108_early_history02/migration.json)和[第一批回执](../../../results/repository_organization/20261108_early_e_series/migration.json)分别保留各自时点的操作、哈希与恢复方法；当前路径以最新映射为准。
本页直接核对原报告及其纠正材料，没有重新读取 train/val/test 数组、重算历史模型或改变历史判断。
