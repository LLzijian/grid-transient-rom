# 当前研究状态

**2026-11-09：长时数值辨别已闭合。五个调制状态在共同参考区间内同时出现幅值增长和可减少的积分误差；static_027的15 s精度仍未决。零项求值修正的20对真实检查未通过冻结总门，28条条件轨迹未释放。下一步优先设计初始化可行性选择与幅值反馈辨别。**

## 当前范围与指针

- Campaign：[GFTD 结构发现](../campaigns/gftd_kuramoto_discovery_20261106.md)；架构入口：[current_design](current_design.md)。
- 当前闭合 Run：[长时数值辨别结论](../reports/gftd_long_horizon_diagnostic_20261109/conclusion.md)、[登记](../harness/runs/gftd_long_horizon_diagnostic_20261109.json)、[协议](../campaigns/gftd_long_horizon_diagnostic_20261109/protocol.md)、[最终独立复核](../reports/gftd_long_horizon_diagnostic_20261109/final_review.md)、[root决定](../results/gftd_long_horizon_diagnostic_20261109/root_adoption.json)、[归档核验](../results/gftd_long_horizon_diagnostic_20261109/archive_verified.json)；执行COMPLETED，工作量closed，有限区间诊断supported_within_scope。
- 父初始化精度Run：[结论](../reports/gftd_initialization_precision_20261109/conclusion.md)、[独立复核](../reports/gftd_initialization_precision_20261109/result_review.md)、[root决定](../results/gftd_initialization_precision_20261109/root_adoption.json)、[归档核验](../results/gftd_initialization_precision_20261109/archive_verified.json)；执行COMPLETED、工作量closed，数值覆盖改善supported_within_scope。
- 父Run 3已COMPLETED、预算closed；[结论](../reports/gftd_fit297_20261107/originals/gftd_fit297_20261107_conclusion.md)、[最终执行回执](../results/gftd_fit297_20261107/resume01/execution.json)、[root决定](../results/gftd_fit297_20261107/root_adoption.json)。静态46/297、调制1/297合格，失稳覆盖均0/30，原总体L1保持null。
- 用户持续授权未知拓扑、隐幅值调制和 train fit297 同期比较；原 val/test 关闭，旧 test 退出选择，新 FOM=0，TASK4 暂停。
- 部署维持 11 点、100 ms 相角与转速历史及末点真实 p，随后共享自治 ODE。
- 本Run代码retain_branch、模型retain_evidence；科学supported_within_scope仅限指定状态的数值误差与区间增长证据。调制恢复与总体预测收益仍未决，生产方程不变。
- default none、incumbent与protected anchors不变。父初始化精度Run的1874份原始产物、695747365字节均留本地，[原始清单](../results/gftd_initialization_precision_20261109/raw_artifacts_manifest.json)的1852条哈希引用全部吻合。父Run 3科学提交`83645d1b9d3593eae4474bc4228dd00335b9eb76`及[归档核验](../results/gftd_fit297_20261107/archive_verified.json)保留。

## 已闭合证据

| 工作 | 结论与范围 | 证据 |
|---|---|---|
| Run 1：K4 静态恢复 | 四个无噪图支撑正确；含噪支撑正确但权重精度下降；平衡设置返回 null | [结论](../reports/history/gftd_campaign/originals/gftd_sparse_topology_20261107/gftd_sparse_topology_20261107_conclusion.md) |
| Run 2：调制与完整接口 | 48/48 数值合格，30/48 达到 1 s 预测门；动态调制恢复未获支持，联合优化充分性 unknown | [结论](../reports/gftd_modulation_20261107/originals/gftd_modulation_20261107_conclusion.md)、[科学判读](../reports/gftd_modulation_20261107/originals/gftd_modulation_20261107_scientific_assessment.md)、[独立复核](../reports/gftd_modulation_20261107/originals/gftd_modulation_20261107_result_review.md) |
| 初始化精度：冻结Run 3参数 | 23个旧失败点有RK4离散误差证据，原14个溢出点有共同参考区间内增长；完整候选新增532个合格输出，15个初始化失败分别保留 | [本轮结论](../reports/gftd_initialization_precision_20261109/conclusion.md)、[完整性核验](../results/gftd_initialization_precision_20261109/full_integrity.json) |
| 长时数值辨别：10个保存状态 | 46条积分均有回执；五状态增长与误差并存；static_027三参考只在3.84 s内一致。268两隐状态延拓不同；零项真实检查10对等价、9对保留活跃异常、1对因相同平方溢出未过分类门 | [结论](../reports/gftd_long_horizon_diagnostic_20261109/conclusion.md)、[逐状态摘要](../results/gftd_long_horizon_diagnostic_20261109/phase1_digest.json) |

Run 3的646个初始化优化任务均收敛，但数值资格有547项未通过，其中14项指数溢出；共享适配351/384任务到限，充分性unknown。原Windows保存失败后41组结果完整保留，一次技术续跑只完成余下553组，没有重拟合或重放。总1030任务、215231残差、9145365尝试RK4步，原/续宿主合计3996.005 s；原始数据和轨迹留本地。

Run 3两模型各35条条件耦合边，调制25条条件调制边；总体L1为null。L2两模型局部证书成立但47实点均未获覆盖；L3各10/13模态配对，K与绝对Kron导纳相关0.440/0.483。详见[科学判读](../reports/gftd_fit297_20261107/originals/gftd_fit297_20261107_scientific_assessment.md)、[独立复核](../reports/gftd_fit297_20261107/originals/gftd_fit297_20261107_result_review.md)和[最终摘要](../results/gftd_fit297_20261107/evidence_digest.json)。

## 下一授权动作

先形成有界设计，比较合法初始化中的动力学可行性选择，并分析负边势能经指数调制产生增长反馈的条件。原历史拟合更好的268_stage2比stage1更早出现增长，不能仅按历史损失选点；本轮没有实施回退。不得在已闭合Run内补网格、容差或第二种修正。

本Run实际2046228次积分RHS及219次采样／单点RHS，合计2046447次；28条条件轨迹未运行，余额不转移。原参数、部署步长与六门不变。78份原始产物共98326340字节[留在本地](../results/gftd_long_horizon_diagnostic_20261109/raw_artifacts_manifest.json)。未选的8个长时异常保持未检验，真实未来与val/test未读取。
标准IEEE39工作簿已找回并与原生成清单哈希一致；全阶descriptor与13参考模式身份已核。L2为ROM局部证明，不能替代全阶安全结论；历史R_N的21/30漏检不作本轮同期成绩。
部署后全体条件信息分析因原进程缓存丢失保持unknown，未重复端口重建/SVD；原24包拟合信息仍保留。
