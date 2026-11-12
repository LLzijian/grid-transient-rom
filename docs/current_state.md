# 当前研究状态

**2026-11-12：匹配预测 Run 的两路 fit 已完成并通过科学判读、独立审查。两模型各自保留父参数，原192套候选数值检查全部通过。原协议的全量部署已启动，4个worker正在执行两模型各297包；真实train未来尚未打开。**

## 当前范围与指针

- Campaign：[GFTD结构发现](../campaigns/gftd_kuramoto_discovery_20261106.md)；[架构入口](current_design.md)。
- [原三步目标与完成要求](../reports/gftd_matched_prediction_20261109/goal_requirements_20261112.md)已按独立审查补齐；整项目标仍未闭合，数值覆盖不替代机制识别或真实预测。
- 活动Run：[设计](../designs/gftd_matched_prediction_20261109.md)、[协议](../campaigns/gftd_matched_prediction_20261109/protocol.md)、[登记](../harness/runs/gftd_matched_prediction_20261109.json)。执行状态RUNNING，当前阶段deploy_running，工作量open；[实际启动回执](../results/gftd_matched_prediction_20261109/deploy_host_start.json)记录宿主及worker身份。
- fit终态：[执行回执](../results/gftd_matched_prediction_20261109/fit/execution.json)、[两模型冻结](../results/gftd_matched_prediction_20261109/fit/model_freeze.json)、[事实摘要](../results/gftd_matched_prediction_20261109/fit_digest.json)、[科学判读](../reports/gftd_matched_prediction_20261109/fit_assessment.md)、[独立审查](../reports/gftd_matched_prediction_20261109/fit_review.md)、[root采用](../results/gftd_matched_prediction_20261109/fit_adoption.json)。原拟合宿主正常退出，未重跑。
- 下一阶段：[部署释放](../results/gftd_matched_prediction_20261109/deploy_release.json)、[资源检查](../results/gftd_matched_prediction_20261109/deploy_preflight.json)。因宿主可用内存减少，将原效率规划8个worker调整为4个，科学工作量不变。
- 部署进展：截至2026-11-12T11:27:21.823937+00:00，完成302/594个组合，302个原六门通过，296个完整严格参考确认；[首批103个审查](../reports/gftd_matched_prediction_20261109/deploy_batch01_review.md)与[第二批137个审查](../reports/gftd_matched_prediction_20261109/deploy_batch02_review.md)已通过，[现场快照](../results/gftd_matched_prediction_20261109/deploy_progress_20261112_batch02.json)保留未决项。
- 43份执行源码仍与prepare一致，科学源码提交`a1936bc5`。当前阶段不追加拟合、不修改算法或数值门。
- 用户持续授权未知拓扑、隐幅值调制和train fit297同期比较；原val/test关闭，旧test退出选择，新FOM=0，TASK4暂停。
- 部署信息维持11点、100 ms相角和转速历史及末点真实p₀，随后共享自治ODE。default none、incumbent与protected anchors不变。

## 已完成的关键证据

| 工作 | 结论与范围 | 证据 |
|---|---|---|
| Run 1：K4静态恢复 | 四个无噪图支撑正确；含噪权重精度下降；平衡设置返回null | [结论](../reports/history/gftd_campaign/originals/gftd_sparse_topology_20261107/gftd_sparse_topology_20261107_conclusion.md) |
| Run 2：调制与接口 | 48/48数值合格，30/48达到1 s预测门；动态调制恢复和联合优化充分性未确认 | [结论](../reports/gftd_modulation_20261107/originals/gftd_modulation_20261107_conclusion.md) |
| 原Run 3 | 静态46/297、调制1/297合格，失稳覆盖均0/30，总体L1为null；两模型L3各10/13模态配对 | [摘要](../results/gftd_fit297_20261107/evidence_digest.json) |
| 初始化精度与长时辨别 | 已区分RK4离散误差、隐状态差异和动力学增长；原异常点不统一归为积分器失败 | [精度结论](../reports/gftd_initialization_precision_20261109/conclusion.md)、[长时结论](../reports/gftd_long_horizon_diagnostic_20261109/conclusion.md) |
| 有界核同核初始化 | 原train297全部通过六门，289个完整1 s严格参考确认，8个参考到限未决；比旧指数新增15个通过、无丢失 | [结论](../reports/gftd_bounded_initialization_20261109/conclusion.md)、[归档核验](../results/gftd_bounded_initialization_20261109/archive_verified.json) |
| 当前匹配fit | 静态与调制均保留父参数；各三次提案对应一个唯一结果。静态45列、调制90列条件信息合格，联合识别unknown | [阶段摘要](../results/gftd_matched_prediction_20261109/fit_digest.json)、[独立审查](../reports/gftd_matched_prediction_20261109/fit_review.md) |

本次fit覆盖固定24包、每模型192次代理优化：静态192次收敛，调制177次收敛、15次到限，所有候选的已选点均来自收敛任务。最终48点相对投影梯度小，步长一致门静态0/24、调制1/24；96次固定方向试算无改善，优化充分性仍unknown。提案未被接受来自历史选择目标，当前候选数值检查未失败。弱式条件提案与盒约束自治历史目标不同，其影响尚未分解。

fit科学RHS为97,812,960；人工资格另28次RHS、748次H，合计RHS97,812,988。host wall为14,750.92 s，worker wall之和20,254.71 s，二者不混用。原始模型、诊断与轨迹留本地。

## 下一授权动作

执行已冻结的两模型各297包simple600→smooth1200初始化、原六门、每个正式点的1 s严格参考与15 s主细自治预测，并分析同一最终ODE的L2/L3。全部594个初始化选择冻结后，单独释放真实train未来评分，报告原分母L1；不按预测结果回改参数或初态。

未决和失败保留原事件、分母与成本。完整5 s稳定样本risk、共同转速RMSE、归一化p RMSE和30个失稳样本的实际可用区间分别报告。历史21/30漏检不作本轮同期成绩；L2只针对最终ROM，L3采用已冻结的13个ANDES物理参考模式及Kron导纳。
