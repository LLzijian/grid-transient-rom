# 当前研究状态

**2026-10-31：解析p_backward的两个固定候选均完成数值资格，最终独立审查通过。**
用户已要求按harness开展实验并自动迭代，既定方向有界推进有效；本聊天每小时接续kiki已启用。
最新Run analytic_p_backward_qualification_20261031实际Astra设计、sol实现/执行、独立审查分阶段留痕。
原val/test关闭、旧消费test永久退出选择、新FOM=0、TASK4暂停；100 ms部署合同不变。

**最新证据。** A96_analytic来自SGM A_28311 step96；A16_analytic来自MemoryTransport A_30311 step16。
原严格loader先核父模型，再用显式analytic_p_backward_overlay_v1只替换副本p_backward。
两个候选各297/297事件、2790/2790必要区间全通过，原数学/因果/缓存/序列化与主细资格均通过。
稳定1/2/5/8 s、失稳真实可用区间（最长15 s）及8 s分类确认路径均按冻结规则核查。
五项主指标差全部落在各自新旧主细差之和+1e−12内；不称统计等效或新数据泛化。
最大跨实现COI归一化RMSE分别5.19124e−9/1.91852e−9；分类和窗口状态变化全部0。

**原能力边界。** SGM A96解析候选8 s/真实区间漏判3/30、21/30，A step16解析候选4/30、22/30，
与各自父版完整ID集合一致；两时域稳定误报均0/267，数值未决/未覆盖均0。
T_base分别9/8完整保留，E新旧分别0.0200/0.0075 s、u均0；原科学T_anchor仍9，A step16仍不覆盖。
两者implementation_qualified=true；science_entry_eligible均false。数值入口资格不晋升原不合格检查点。
R_N保护档案、SGM A96原范围、A_30311九点无全门与step96真实20/30、8 s 4/30的权衡均保留。

**三决定。** 代码retain_branch，保留显式解析构造和专用加载；后续协议可引用两个合格候选。
模型retain_evidence，不隐式迁移旧模型，不改main code/working_candidate/incumbent/protected anchors或default none。
科学continue：长期数值行为保持这一前置问题已在登记范围内检验，下一问题回到p形式归因。
本Run无优化，当前优化健康/充分性not_applicable；来源训练的优化充分性仍unknown。

**执行与成本。** 19项最终人工测试；首版四项启动前问题已独立复审关闭，旧新哈希保留。
唯一prepare、正常两份paired（各候选一次）与summary均exit0，完整产物已核hash，记录进程已退出。
真实修复0、重试0、optimizer0、共享参数拟合0；额外修复预测未释放，未用额度关闭、不转移父余额。
原生wall：prepare25.07897 s；两候选228.93754/192.73812 s，并行重叠228.94363 s、单项和421.67566 s；
summary4.46403 s。各工具阶段及包含关系另表，CPU进程秒/LLM/全项目总成本unknown，不宣称加速。

**父证据。** p_reverse_precision_20261031已远端归档be39cfb：Torch64逐位复现旧buffer，
对舍入参考差1.5673e−10；实际初始化/QR/初始场传播通过。历史唯一生成原因unknown。
p_port_diagnostic_20261031来源/平衡297/297、基准差0，M1闭合通过、M2为转速传导与固定p残差混合。
旧反推1e−12门失败及M3六十格归因unknown不回写；上游tm含义限制保留，p按冻结数值坐标解释。
首轮10次离线共享估计、第二轮0拟合，本轮0拟合，各Run成本和关闭余额独立。

**材料。** [本轮结论](../reports/analytic_p_backward_qualification_20261031_conclusion.md)、
[协议](../reports/analytic_p_backward_qualification_20261031_protocol.md)、[审查](../reports/analytic_p_backward_qualification_20261031_review.md)、
[公开结果](../results/analytic_p_backward_qualification_20261031/public_qualification.json)、
[有限聚合](../results/analytic_p_backward_qualification_20261031/evidence_digest.json)。
私有overlay/预测/逐事件身份/原生进程身份留本地；远端凭证见本Run私有archival_verified.json，不能由COMPLETED推断推送成功。

**下一动作。** 先核对本Run私有远端归档凭证；未完成则补齐，核验后才开启下一Run。
下一有限窗口由实际Astra给前瞻设计，优先区分共享p常数不匹配与方程形式误差，旧β只作已暴露估计。
不追加本Run训练，不改状态/部署/数据权限；无活动或未知Job才继续，用户后续暂停优先。
旧memory_transport关闭1356余额、r_memory INTERRUPTED_RUNTIME及UNKNOWN全项目账目不改。
