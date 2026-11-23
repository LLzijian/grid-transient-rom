# 当前研究状态

**2026-11-23。** 当前方案为 [OSGKD 2.0](../designs/osgkd_v2_architecture.md)。[P0 零训练诊断](../reports/osgkd_v2_p0_20261123/result.md)、[P1a 固定末点评价](../reports/osgkd_v2_p1a_20261123/result.md)及[P2a 初态试验](../reports/osgkd_v2_p2a_20261123/result.md)已完成并独立复核。P1a 两边各 200 次成功更新，主细步长评价覆盖全 297 条。

- P1a 的 2 s 与 8 s 训练组，在稳定 267 条的 8 s 运动风险为 39.751 和 2.038，稳定误报为 15/267 和 0/267。8 s 组 1/2 s 风险较高，保住原 C1 三个真实区间检出，仍有 27/30 漏检。[P1a 登记](../harness/runs/osgkd_v2_p1a_20261123.json)
- P2a 固定 P1a 长组方程，一步精修使历史平方残差降低 98.3%。稳定 8 s 风险由 2.038 升至 89.481，稳定误报由 0/267 升至 105/267，失稳真实区间由 3/30 增至 4/30。精修未通过采用守卫，保留原初始化器和长组方程。[P2a 登记](../harness/runs/osgkd_v2_p2a_20261123.json)

- v1 C0/C1 固定参考均有局部增长。去共同角、保留共同转速后的闭环谱最大实部分别为 1.44007 和 1.47467 s⁻¹。局部增长与实际长期预测误差需分开检验。
- C1 在失稳样本真实记录内检出 3/30，26 次触发发生于真实记录结束后，另有 1 次截至 8 s 无触发。27 条真实区间漏检中有 13 条的幅值初态达到边界的 99%。[P0 回执](../harness/runs/osgkd_v2_p0_20261123.json)

**前序证据（2026-11-20）。OSGKD 首轮完整 C0/C1 训练、一次条件适配及全 train fit297 正式评价已完成，最终报告已通过独立复核，证据与工作量已闭合。**

部署为 [motion_only_v1](../kiki.md)：11点/100 ms相角与完整转速历史，真实p0不输入。该轮采用 [OSGKD v1](../designs/osgkd_architecture.md)，协议为 [fit297首轮](../campaigns/osgkd_ieee39/first_fit297_protocol.md)。

- 两组各1500次更新、0非有限拒绝，完整事件顺序和时域配对。均按原规则追加500次，并在最终评分前选定1500次检查点。C1未静态回退；参数6809/6854，状态均50。
- 两模型全部297的8 s主细数值检查通过，最大归一化RMS差分别2.70e-8、9.39e-8。当前大预测误差不能用本步长范围内的积分精度不足解释。
- train稳定267的8 s risk：C0 78.28062、C1 42.60901（低45.57%）；共同转速RMSE：1.47912、1.05578 rad/s。稳定误报60/267→6/267，C1的6例是原60例的子集。
- 稳定1/2 s的C1 risk高18.11%/14.33%；失稳真实区间两模型均只检出3/30、漏检27/30，内部检查12例均漏。8 s的27例未知尾段保留未覆盖，不算成功检出。
- 原来源审计保留blocked/unresolved；5个历史子窗连续闭合未决。端口按经审查的离散观测合同近似监督，样本全部保留。

三个决定：代码 retain_branch，模型 retain_evidence，科学 continue。工作候选为C1第1500次，C0同合同对照同步保留；default none、incumbent和protected anchors不变。原始val/test关闭，新FOM=0，TASK4暂停，GFTD继续暂存。

[早期闭合诊断](../reports/osgkd_v2_early_closure_20261123/result.md)在原 27 条漏检的两窗均覆盖十节点，选出总电气端。[固定末点配对适配](../reports/osgkd_v2_early_electrical_20261123/result.md)已完成各 32 次更新并通过独立复核。候选组在真实上限 0.25 s 的电气与运动均方误差分别比对照组低 1.71% 和 3.43%，均未达到预登记幅度。稳定 8 s 风险为父模型 2.038、对照组 1.115、候选组 2.075，三组仍均检出原 3/30，稳定误报为 0/267。保留 P1a 长组，下一辨别聚焦稳定 8 s 风险上升事件，先区分短窗与长期目标的冲突及优化路径的影响。[Run 登记](../harness/runs/osgkd_v2_early_electrical_20261123.json) · [v2 Campaign](../campaigns/osgkd_v2/campaign.md)

[本轮报告](../reports/osgkd_fit297_20261113/result.md) · [机器结果](../results/osgkd_fit297_20261113/analysis.json) · [科学综合](../reports/osgkd_fit297_20261113/scientific_synthesis.md) · [Run记录](../harness/runs/osgkd_fit297_20261113.json)
