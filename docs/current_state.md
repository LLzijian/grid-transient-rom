# 当前研究状态

**2026-11-23。** 当前方案为 [OSGKD 2.0](../designs/osgkd_v2_architecture.md)。本次完成目标、架构与执行规划更新。v2 执行状态为待登记，未启动训练，现有运行代码和最近实测模型属于 v1。

**前序证据（2026-11-20）。OSGKD 首轮完整 C0/C1 训练、一次条件适配及全 train fit297 正式评价已完成，最终报告已通过独立复核，证据与工作量已闭合。**

部署为 [motion_only_v1](../kiki.md)：11点/100 ms相角与完整转速历史，真实p0不输入。该轮采用 [OSGKD v1](../designs/osgkd_architecture.md)，协议为 [fit297首轮](../campaigns/osgkd_ieee39/first_fit297_protocol.md)。

- 两组各1500次更新、0非有限拒绝，完整事件顺序和时域配对。均按原规则追加500次，并在最终评分前选定1500次检查点。C1未静态回退；参数6809/6854，状态均50。
- 两模型全部297的8 s主细数值检查通过，最大归一化RMS差分别2.70e-8、9.39e-8。当前大预测误差不能用本步长范围内的积分精度不足解释。
- train稳定267的8 s risk：C0 78.28062、C1 42.60901（低45.57%）；共同转速RMSE：1.47912、1.05578 rad/s。稳定误报60/267→6/267，C1的6例是原60例的子集。
- 稳定1/2 s的C1 risk高18.11%/14.33%；失稳真实区间两模型均只检出3/30、漏检27/30，内部检查12例均漏。8 s的27例未知尾段保留未覆盖，不算成功检出。
- 原来源审计保留blocked/unresolved；5个历史子窗连续闭合未决。端口按经审查的离散观测合同近似监督，样本全部保留。

三个决定：代码 retain_branch，模型 retain_evidence，科学 continue。工作候选为C1第1500次，C0同合同对照同步保留；default none、incumbent和protected anchors不变。原始val/test关闭，新FOM=0，TASK4暂停，GFTD继续暂存。

下一步见 [v2 Campaign](../campaigns/osgkd_v2/campaign.md)。本机 C0/C1 第 1500 次检查点和两份逐事件记录均与原清单的大小和 SHA256 一致。P0 可由这些资产检验参考处的谱、初态饱和与失步时间，本次只核验文件身份。首个训练比较拟从同一 C1 检查点配对适配 2 s 与 8 s 目标，30 个失稳样本的真实未来均不足 2 s。

[本轮报告](../reports/osgkd_fit297_20261113/result.md) · [机器结果](../results/osgkd_fit297_20261113/analysis.json) · [科学综合](../reports/osgkd_fit297_20261113/scientific_synthesis.md) · [Run记录](../harness/runs/osgkd_fit297_20261113.json)
