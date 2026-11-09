# 当前研究状态

**2026-11-10：有界核在原train fit297上完成297/297因果初始化，全部通过原六项数值检查，289个获得完整1 s严格参考确认；8个参考到限保持未决。本轮已闭合、归档并推送。下一轮匹配拟合与预测的公开接口已接通，44项固定资格全部通过，累计人工28次RHS／748次H；独立复核接受，两个模型已实际启动匹配拟合，部署和真实评分等待各自冻结结果。**

## 当前范围与指针

- Campaign：[GFTD 结构发现](../campaigns/gftd_kuramoto_discovery_20261106.md)；架构入口：[current_design](current_design.md)。
- 当前活动Run：[匹配预测设计](../designs/gftd_matched_prediction_20261109.md)、[协议](../campaigns/gftd_matched_prediction_20261109/protocol.md)、[资格审查](../reports/gftd_matched_prediction_20261109/qualification_review.md)、[资格采用](../results/gftd_matched_prediction_20261109/qualification_adoption.json)、[拟合释放](../results/gftd_matched_prediction_20261109/fit_release.json)、[实际启动](../results/gftd_matched_prediction_20261109/fit_host_start.json)、[登记](../harness/runs/gftd_matched_prediction_20261109.json)；工作量open，两个独立worker执行静态／有界调制的匹配共享适配，源码提交为`a1936bc5`；完成状态以阶段终态回执为准。
- 最新闭合Run：[有界核初始化结论](../reports/gftd_bounded_initialization_20261109/conclusion.md)、[协议](../campaigns/gftd_bounded_initialization_20261109/protocol.md)、[科学判读](../reports/gftd_bounded_initialization_20261109/scientific_assessment.md)、[独立审查](../reports/gftd_bounded_initialization_20261109/result_review.md)、[root决定](../results/gftd_bounded_initialization_20261109/root_adoption.json)、[源码归档](../results/gftd_bounded_initialization_20261109/source_archive.json)、[推送核验](../results/gftd_bounded_initialization_20261109/archive_verified.json)、[登记](../harness/runs/gftd_bounded_initialization_20261109.json)；科学提交`95afe8cd`已推送核验，COMPLETED、closed，代码retain_branch、模型retain_evidence、科学supported_within_scope。
- 储能可行性Run：[结论](../reports/gftd_storage_feasibility_20261109/conclusion.md)、[独立审查](../reports/gftd_storage_feasibility_20261109/result_review.md)、[归档核验](../results/gftd_storage_feasibility_20261109/archive_verified.json)；科学提交`069f54ef`已推送核验，数学命题与指定1 s结构干预supported_within_scope。
- 上轮闭合 Run：[长时数值辨别结论](../reports/gftd_long_horizon_diagnostic_20261109/conclusion.md)、[登记](../harness/runs/gftd_long_horizon_diagnostic_20261109.json)、[协议](../campaigns/gftd_long_horizon_diagnostic_20261109/protocol.md)、[最终独立复核](../reports/gftd_long_horizon_diagnostic_20261109/final_review.md)、[root决定](../results/gftd_long_horizon_diagnostic_20261109/root_adoption.json)、[归档核验](../results/gftd_long_horizon_diagnostic_20261109/archive_verified.json)；执行COMPLETED，工作量closed，有限区间诊断supported_within_scope。
- 父初始化精度Run：[结论](../reports/gftd_initialization_precision_20261109/conclusion.md)、[独立复核](../reports/gftd_initialization_precision_20261109/result_review.md)、[root决定](../results/gftd_initialization_precision_20261109/root_adoption.json)、[归档核验](../results/gftd_initialization_precision_20261109/archive_verified.json)；执行COMPLETED、工作量closed，数值覆盖改善supported_within_scope。
- 父Run 3已COMPLETED、预算closed；[结论](../reports/gftd_fit297_20261107/originals/gftd_fit297_20261107_conclusion.md)、[最终执行回执](../results/gftd_fit297_20261107/resume01/execution.json)、[root决定](../results/gftd_fit297_20261107/root_adoption.json)。静态46/297、调制1/297合格，失稳覆盖均0/30，原总体L1保持null。
- 用户持续授权未知拓扑、隐幅值调制和 train fit297 同期比较；原 val/test 关闭，旧 test 退出选择，新 FOM=0，TASK4 暂停。
- 部署维持 11 点、100 ms 相角与转速历史及末点真实 p，随后共享自治 ODE。
- 储能可行性Run代码retain_branch、模型retain_evidence；科学supported_within_scope限于已证明的数学命题与同初态结构干预。调制恢复与总体预测收益仍未决，生产方程不变。
- default none、incumbent与protected anchors不变。父初始化精度Run的1874份原始产物、695747365字节均留本地，[原始清单](../results/gftd_initialization_precision_20261109/raw_artifacts_manifest.json)的1852条哈希引用全部吻合。父Run 3科学提交`83645d1b9d3593eae4474bc4228dd00335b9eb76`及[归档核验](../results/gftd_fit297_20261107/archive_verified.json)保留。

## 已闭合证据

| 工作 | 结论与范围 | 证据 |
|---|---|---|
| Run 1：K4 静态恢复 | 四个无噪图支撑正确；含噪支撑正确但权重精度下降；平衡设置返回 null | [结论](../reports/history/gftd_campaign/originals/gftd_sparse_topology_20261107/gftd_sparse_topology_20261107_conclusion.md) |
| Run 2：调制与完整接口 | 48/48 数值合格，30/48 达到 1 s 预测门；动态调制恢复未获支持，联合优化充分性 unknown | [结论](../reports/gftd_modulation_20261107/originals/gftd_modulation_20261107_conclusion.md)、[科学判读](../reports/gftd_modulation_20261107/originals/gftd_modulation_20261107_scientific_assessment.md)、[独立复核](../reports/gftd_modulation_20261107/originals/gftd_modulation_20261107_result_review.md) |
| 初始化精度：冻结Run 3参数 | 23个旧失败点有RK4离散误差证据，原14个溢出点有共同参考区间内增长；完整候选新增532个合格输出，15个初始化失败分别保留 | [本轮结论](../reports/gftd_initialization_precision_20261109/conclusion.md)、[完整性核验](../results/gftd_initialization_precision_20261109/full_integrity.json) |
| 长时数值辨别：10个保存状态 | 46条积分均有回执；五状态增长与误差并存；static_027三参考只在3.84 s内一致。268两隐状态延拓不同；零项真实检查10对等价、9对保留活跃异常、1对因相同平方溢出未过分类门 | [结论](../reports/gftd_long_horizon_diagnostic_20261109/conclusion.md)、[逐状态摘要](../results/gftd_long_horizon_diagnostic_20261109/phase1_digest.json) |
| 储能可行性：一个事件、两个旧状态 | 有界核六参考全部1 s合格；stage2在共同0–0.92 s内幅值最大值由4.163105变为1.566103，调制因子最大值由7.099988变为1.243586；真实预测效应未评价 | [结论](../reports/gftd_storage_feasibility_20261109/conclusion.md)、[独立审计](../results/gftd_storage_feasibility_20261109/independent_audit.json) |
| 有界核同核初始化：原297个train样本 | 297/297六门通过，较旧指数新增15个、无丢失；289完整严格参考确认。历史MSE18个降低、279个升高，优化充分性unknown | [结论](../reports/gftd_bounded_initialization_20261109/conclusion.md)、[独立审计](../results/gftd_bounded_initialization_20261109/full_independent_audit.json) |

Run 3的646个初始化优化任务均收敛，但数值资格有547项未通过，其中14项指数溢出；共享适配351/384任务到限，充分性unknown。原Windows保存失败后41组结果完整保留，一次技术续跑只完成余下553组，没有重拟合或重放。总1030任务、215231残差、9145365尝试RK4步，原/续宿主合计3996.005 s；原始数据和轨迹留本地。

Run 3两模型各35条条件耦合边，调制25条条件调制边；总体L1为null。L2两模型局部证书成立但47实点均未获覆盖；L3各10/13模态配对，K与绝对Kron导纳相关0.440/0.483。详见[科学判读](../reports/gftd_fit297_20261107/originals/gftd_fit297_20261107_scientific_assessment.md)、[独立复核](../reports/gftd_fit297_20261107/originals/gftd_fit297_20261107_result_review.md)和[最终摘要](../results/gftd_fit297_20261107/evidence_digest.json)。

## 下一授权动作

按[已冻结匹配预测协议](../campaigns/gftd_matched_prediction_20261109/protocol.md)完成静态与有界调制的两路共享适配：各24个固定样本、两起点1200残差、至多3个完整库提案。两模型冻结后，各执行原297个样本的初始化与自主预测；全部选择冻结后，再读取真实train未来作L1，并报告同一最终ODE的L2/L3。43份执行源码冻结，未知Job先对账，不重放。

[下一主线源设计](../reports/gftd_bounded_initialization_20261109/next_mainline_design_draft.md)及[独立数学审查](../reports/gftd_bounded_initialization_20261109/next_l2_design_review.md)已形成正式预登记。全量原进程PID27908及四个worker已退出；完整回执与独立审计确认全部Job完成，原启动会话退出码仍为null。有界核本轮科学RHS为105,009,270，人工资格另14次RHS／374次H；893份原始产物共72,857,671字节留本地，未读真实未来。

上一轮储能可行性的12项模拟检查及唯一人工资格通过，科学积分61,899次RHS、人工10次RHS与370次H；程序宿主墙钟15.065 s、CPU13.172 s。六份新轨迹共3,410,113字节[保留本地](../results/gftd_storage_feasibility_20261109/raw_artifacts_manifest.json)，未读取真实未来，未运行新核全体594项初始化，余额不转移。

长时数值诊断Run实际2046228次积分RHS及219次采样／单点RHS，合计2046447次；28条条件轨迹未运行，余额不转移。原参数、部署步长与六门不变。78份原始产物共98326340字节[留在本地](../results/gftd_long_horizon_diagnostic_20261109/raw_artifacts_manifest.json)。未选的8个长时异常保持未检验，真实未来与val/test未读取。
标准IEEE39工作簿已找回并与原生成清单哈希一致；全阶descriptor与13参考模式身份已核。L2为ROM局部证明，不能替代全阶安全结论；历史R_N的21/30漏检不作本轮同期成绩。
部署后全体条件信息分析因原进程缓存丢失保持unknown，未重复端口重建/SVD；原24包拟合信息仍保留。
