# 当前研究状态

**训练未恢复；checkpoint_selection_20261030 已完成（无训练选择审计，optimizer 0）。**
用户2026-10-30指令“根据项目工程设计，在最新结果的基础上，向前推进”后，本轮实际取得Astra有界设计，
经独立Reviewer审查，完成train fit297的9个A检查点与R_N/A96参照核验。
原val/test关闭、新FOM=0、TASK4暂停、default none；模型与工作指针未改。

**结论。** 已存A_30311九点没有同时通过原精度/关键事件目标及本轮身份/时机守卫。
step16稳定5/8 s风险最低（0.752083/0.576173），但失稳真实区间风险4.096469、漏判22/30，
8 s漏判4/30；R_N对应3.512358、21/30、4/30，A96对应2.369721、21/30、3/30。
step96真实区间漏判20/30、风险3.152346，保住原9个真实区间检出并修复1个漏判；
但8 s相对A96新增1个漏判，仍为4/30，因此不准入。原门本身已无全通过点，不是新增守卫单独导致。

**证据有效性与来源。** 11个评价对象数值资格均通过；两时域误报均0/267，数值未决0。
候选读出前已冻结检出保护集（8 s 27、真实区间9）与T_anchor（9）；缺T覆盖的点E/u为null，
没有通过缩小比较集合获得时机收益。来源模型/Adam/trace/journal对账通过；
step64–128通过原最近64步健康门，step0/16/32/48保留insufficient_history；优化充分性unknown。
A0/A96原双向297事件兼容与共享状态对账通过。同一已消费train、单来源单日程，不是独立确认。

**三决定。** 代码retain_branch；模型retain_evidence，建议来源null，不晋升；科学unresolved。
H3说明稳定精度收益受端点选择影响；本轮进一步定位了这条已存路径的稳定/失稳权衡，未否定A族。
step96的真实区间收益与不利的8 s事件身份均保留，不以总分抵消。

**成本与执行。** 8个新主细评价、3个冻结完整结果复用；CPU float64、最多4个单线程worker。
Stage1审计34.541507 s，Stage2 controller wall562.870089 s，来源核查2.886849 s；
单点并发耗时不相加冒充经过时间；决策进程独立wall null，CPU总进程秒unknown。
所有候选一次完成，无retry；三个原生阶段退出码0；本Run17个已登记PID/创建身份均已退出，活动0。
旧memory_transport的492调用、7472.72 s是旧范围，1356未用余额仍关闭，不转入本轮。

**材料。** [结论](../reports/checkpoint_selection_20261030_conclusion.md)、
[协议](../reports/checkpoint_selection_20261030_protocol.md)、
[Scientist设计](../reports/checkpoint_selection_20261030_scientist.md)、
[独立审查](../reports/checkpoint_selection_20261030_review.md)、
[公开决策](../results/checkpoint_selection_20261030/decision_public.json)、
[执行回执](../results/checkpoint_selection_20261030/execution_receipt.json)。
私有预测、完整逐事件身份、manifest与完成回执留在private_data/checkpoint_selection_20261030/。

**下一问题。** 先诊断机械p端口失配的归属：离散滤波近似、常数/量纲、角速度误差传导还是方程形式。
Astra已写[草案](../reports/p_port_diagnostic_20261030_scientist_draft.md)，DRAFT_UNADOPTED，
未登记新Run、未计算或训练；须冻结协议与独立审查后方可执行无训练诊断。E/s暂不作为首选。

**前序证据。** closure_probe_20261030已完成：H2不作主要解释，H1降级未否证，H3贡献因素，H4开放；
见[报告](../reports/closure_probe_20261030_conclusion.md)。旧r_memory保持INTERRUPTED_RUNTIME，
1165实际/1152持久，正式总成本UNKNOWN；R_N保护档案、A96来源、S/P端点与失败证据均原位保全。
