# 当前研究状态

**2026-11-08：三步主线已实现并运行。Run 3完成两次fit、594组初始化和47组双步长预测；静态46/297、调制1/297数值合格，两模型的30个失稳样本均未覆盖，总体L1效应与调制识别未决。Scientist和独立结果审查已完成，本轮工作量关闭，已完成选择性归档与推送核验。**

## 当前范围与指针

- Campaign：[GFTD 结构发现](../campaigns/gftd_kuramoto_discovery_20261106.md)；架构入口：[current_design](current_design.md)。
- 当前 Run：[fit297 登记](../harness/runs/gftd_fit297_20261107.json)，执行状态COMPLETED、科学主张unresolved、预算closed；[结论](../reports/gftd_fit297_20261107/originals/gftd_fit297_20261107_conclusion.md)、[最终执行回执](../results/gftd_fit297_20261107/resume01/execution.json)、[root决定](../results/gftd_fit297_20261107/root_adoption.json)。
- 用户持续授权未知拓扑、隐幅值调制和 train fit297 同期比较；原 val/test 关闭，旧 test 退出选择，新 FOM=0，TASK4 暂停。
- 部署维持 11 点、100 ms 相角与转速历史及末点真实 p，随后共享自治 ODE。
- 代码 retain_branch；模型 retain_evidence；科学 continue。调制恢复主张 unresolved。
- default none、incumbent 与 protected anchors 不变。本轮证据提交为 `83645d1b9d3593eae4474bc4228dd00335b9eb76`，689个文件原字节及远端一致性已核验，见[归档回执](../results/gftd_fit297_20261107/archive_verified.json)。父证据提交为 `66f1308c25ee38c3d7217bd86ee43b61a042a5b6`，已推送并核验原字节。

## 已闭合证据

| 工作 | 结论与范围 | 证据 |
|---|---|---|
| Run 1：K4 静态恢复 | 四个无噪图支撑正确；含噪支撑正确但权重精度下降；平衡设置返回 null | [结论](../reports/history/gftd_campaign/originals/gftd_sparse_topology_20261107/gftd_sparse_topology_20261107_conclusion.md) |
| Run 2：调制与完整接口 | 48/48 数值合格，30/48 达到 1 s 预测门；动态调制恢复未获支持，联合优化充分性 unknown | [结论](../reports/gftd_modulation_20261107/originals/gftd_modulation_20261107_conclusion.md)、[科学判读](../reports/gftd_modulation_20261107/originals/gftd_modulation_20261107_scientific_assessment.md)、[独立复核](../reports/gftd_modulation_20261107/originals/gftd_modulation_20261107_result_review.md) |

Run 3的646个初始化优化任务均收敛，但数值资格有547项未通过，其中14项指数溢出；共享适配351/384任务到限，充分性unknown。原Windows保存失败后41组结果完整保留，一次技术续跑只完成余下553组，没有重拟合或重放。总1030任务、215231残差、9145365尝试RK4步，原/续宿主合计3996.005 s；原始数据和轨迹留本地。

Run 3两模型各35条条件耦合边，调制25条条件调制边；总体L1为null。L2两模型局部证书成立但47实点均未获覆盖；L3各10/13模态配对，K与绝对Kron导纳相关0.440/0.483。详见[科学判读](../reports/gftd_fit297_20261107/originals/gftd_fit297_20261107_scientific_assessment.md)、[独立复核](../reports/gftd_fit297_20261107/originals/gftd_fit297_20261107_result_review.md)和[最终摘要](../results/gftd_fit297_20261107/evidence_digest.json)。

## 下一授权动作

下一主线在既有授权内前瞻区分冻结ROM的积分精度不足与参数/初始化所产生的动力学问题；下一科学动作尚未释放，不回填本轮通过数，不转移关闭余额。
标准IEEE39工作簿已找回并与原生成清单哈希一致；全阶descriptor与13参考模式身份已核。L2为ROM局部证明，不能替代全阶安全结论；历史R_N的21/30漏检不作本轮同期成绩。
部署后全体条件信息分析因原进程缓存丢失保持unknown，未重复端口重建/SVD；原24包拟合信息仍保留。
