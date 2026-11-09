# 当前研究状态

**2026-11-09：储能可行性Run已完成。原指数模型族存在储能下降却有限时间逃逸的精确反例；有界调制exp(tanh h)保留零点二阶展开，其完整5N方程满足有限时域存在性。固定268两个旧状态的六条新参考均完成1 s并通过三参考一致性，独立缓存复算通过。候选保留为研究证据，尚未进入合法初始化与生产接口。**

## 当前范围与指针

- Campaign：[GFTD 结构发现](../campaigns/gftd_kuramoto_discovery_20261106.md)；架构入口：[current_design](current_design.md)。
- 最新闭合 Run：[结论](../reports/gftd_storage_feasibility_20261109/conclusion.md)、[设计](../designs/gftd_storage_feasibility_20261109.md)、[协议](../campaigns/gftd_storage_feasibility_20261109/protocol.md)、[独立结果审查](../reports/gftd_storage_feasibility_20261109/result_review.md)、[root决定](../results/gftd_storage_feasibility_20261109/root_adoption.json)、[登记](../harness/runs/gftd_storage_feasibility_20261109.json)、[归档核验](../results/gftd_storage_feasibility_20261109/archive_verified.json)；科学提交`069f54ef`已推送核验，执行COMPLETED，工作量closed，数学命题与指定1 s结构干预supported_within_scope。
- 上轮闭合 Run：[长时数值辨别结论](../reports/gftd_long_horizon_diagnostic_20261109/conclusion.md)、[登记](../harness/runs/gftd_long_horizon_diagnostic_20261109.json)、[协议](../campaigns/gftd_long_horizon_diagnostic_20261109/protocol.md)、[最终独立复核](../reports/gftd_long_horizon_diagnostic_20261109/final_review.md)、[root决定](../results/gftd_long_horizon_diagnostic_20261109/root_adoption.json)、[归档核验](../results/gftd_long_horizon_diagnostic_20261109/archive_verified.json)；执行COMPLETED，工作量closed，有限区间诊断supported_within_scope。
- 父初始化精度Run：[结论](../reports/gftd_initialization_precision_20261109/conclusion.md)、[独立复核](../reports/gftd_initialization_precision_20261109/result_review.md)、[root决定](../results/gftd_initialization_precision_20261109/root_adoption.json)、[归档核验](../results/gftd_initialization_precision_20261109/archive_verified.json)；执行COMPLETED、工作量closed，数值覆盖改善supported_within_scope。
- 父Run 3已COMPLETED、预算closed；[结论](../reports/gftd_fit297_20261107/originals/gftd_fit297_20261107_conclusion.md)、[最终执行回执](../results/gftd_fit297_20261107/resume01/execution.json)、[root决定](../results/gftd_fit297_20261107/root_adoption.json)。静态46/297、调制1/297合格，失稳覆盖均0/30，原总体L1保持null。
- 用户持续授权未知拓扑、隐幅值调制和 train fit297 同期比较；原 val/test 关闭，旧 test 退出选择，新 FOM=0，TASK4 暂停。
- 部署维持 11 点、100 ms 相角与转速历史及末点真实 p，随后共享自治 ODE。
- 本Run代码retain_branch、模型retain_evidence；科学supported_within_scope限于已证明的数学命题与同初态结构干预。调制恢复与总体预测收益仍未决，生产方程不变。
- default none、incumbent与protected anchors不变。父初始化精度Run的1874份原始产物、695747365字节均留本地，[原始清单](../results/gftd_initialization_precision_20261109/raw_artifacts_manifest.json)的1852条哈希引用全部吻合。父Run 3科学提交`83645d1b9d3593eae4474bc4228dd00335b9eb76`及[归档核验](../results/gftd_fit297_20261107/archive_verified.json)保留。

## 已闭合证据

| 工作 | 结论与范围 | 证据 |
|---|---|---|
| Run 1：K4 静态恢复 | 四个无噪图支撑正确；含噪支撑正确但权重精度下降；平衡设置返回 null | [结论](../reports/history/gftd_campaign/originals/gftd_sparse_topology_20261107/gftd_sparse_topology_20261107_conclusion.md) |
| Run 2：调制与完整接口 | 48/48 数值合格，30/48 达到 1 s 预测门；动态调制恢复未获支持，联合优化充分性 unknown | [结论](../reports/gftd_modulation_20261107/originals/gftd_modulation_20261107_conclusion.md)、[科学判读](../reports/gftd_modulation_20261107/originals/gftd_modulation_20261107_scientific_assessment.md)、[独立复核](../reports/gftd_modulation_20261107/originals/gftd_modulation_20261107_result_review.md) |
| 初始化精度：冻结Run 3参数 | 23个旧失败点有RK4离散误差证据，原14个溢出点有共同参考区间内增长；完整候选新增532个合格输出，15个初始化失败分别保留 | [本轮结论](../reports/gftd_initialization_precision_20261109/conclusion.md)、[完整性核验](../results/gftd_initialization_precision_20261109/full_integrity.json) |
| 长时数值辨别：10个保存状态 | 46条积分均有回执；五状态增长与误差并存；static_027三参考只在3.84 s内一致。268两隐状态延拓不同；零项真实检查10对等价、9对保留活跃异常、1对因相同平方溢出未过分类门 | [结论](../reports/gftd_long_horizon_diagnostic_20261109/conclusion.md)、[逐状态摘要](../results/gftd_long_horizon_diagnostic_20261109/phase1_digest.json) |
| 储能可行性：一个事件、两个旧状态 | 有界核六参考全部1 s合格；stage2在共同0–0.92 s内幅值最大值由4.163105变为1.566103，调制因子最大值由7.099988变为1.243586；真实预测效应未评价 | [结论](../reports/gftd_storage_feasibility_20261109/conclusion.md)、[独立审计](../results/gftd_storage_feasibility_20261109/independent_audit.json) |

Run 3的646个初始化优化任务均收敛，但数值资格有547项未通过，其中14项指数溢出；共享适配351/384任务到限，充分性unknown。原Windows保存失败后41组结果完整保留，一次技术续跑只完成余下553组，没有重拟合或重放。总1030任务、215231残差、9145365尝试RK4步，原/续宿主合计3996.005 s；原始数据和轨迹留本地。

Run 3两模型各35条条件耦合边，调制25条条件调制边；总体L1为null。L2两模型局部证书成立但47实点均未获覆盖；L3各10/13模态配对，K与绝对Kron导纳相关0.440/0.483。详见[科学判读](../reports/gftd_fit297_20261107/originals/gftd_fit297_20261107_scientific_assessment.md)、[独立复核](../reports/gftd_fit297_20261107/originals/gftd_fit297_20261107_result_review.md)和[最终摘要](../results/gftd_fit297_20261107/evidence_digest.json)。

## 下一授权动作

先完成**新核与合法初始化使用同一方程**的有限设计：统一幅值字典、完整梯度、100 ms历史推进和核版本身份，在固定共享参数下检查合法初态的数值资格，再决定匹配适配与真实L1比较。具体新Run需先由Scientist设计并经独立挑战；当前Run不追加工作。

本轮12项模拟检查及唯一人工资格通过，科学积分61,899次RHS、人工10次RHS与370次H；程序宿主墙钟15.065 s、CPU13.172 s。六份新轨迹共3,410,113字节[保留本地](../results/gftd_storage_feasibility_20261109/raw_artifacts_manifest.json)，未读取真实未来，未运行新核全体594项初始化，余额不转移。

上轮实际2046228次积分RHS及219次采样／单点RHS，合计2046447次；28条条件轨迹未运行，余额不转移。原参数、部署步长与六门不变。78份原始产物共98326340字节[留在本地](../results/gftd_long_horizon_diagnostic_20261109/raw_artifacts_manifest.json)。未选的8个长时异常保持未检验，真实未来与val/test未读取。
标准IEEE39工作簿已找回并与原生成清单哈希一致；全阶descriptor与13参考模式身份已核。L2为ROM局部证明，不能替代全阶安全结论；历史R_N的21/30漏检不作本轮同期成绩。
部署后全体条件信息分析因原进程缓存丢失保持unknown，未重复端口重建/SVD；原24包拟合信息仍保留。
