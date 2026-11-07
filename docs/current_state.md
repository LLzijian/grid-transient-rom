# 当前研究状态

**2026-11-07：主线 Run 1 已完成并经独立复核。候选全图 K4 上，零图、路径图、星图和环图的无噪支撑全部恢复正确，最大边权绝对误差 2.0612e-4。下一步接通动态调制与完整 fit/initialize；Run 2 正式设计已形成，尚未释放执行。**

## 范围与工作指针

- 用户已指定未知稀疏拓扑→隐幅值调制与接口→train fit297 同期真实验证的连续主线，范围内自主推进，无需逐轮用户确认。
- 原始 val/test 关闭；旧消费 test 退出选择；新 FOM=0；TASK4 暂停。部署仍为 11 点、100 ms theta/omega 历史和末点真实 p，随后同一共享自治 ODE。default none，incumbent 与 protected anchors 不变。
- 已知参数、人工三机的工作初始化仍为 simple600→smooth1200。Run 2 将在此基础上明确扩展未知幅值并重新资格；不把既有 a_end=0 的有限成功当作通用初始化已完成。

## Run 1 的有效证据

[结果](../reports/gftd_sparse_topology_20261107_conclusion.md)：完整 p/r 端口、四弱测试和记忆初态 SVD 消元，把父单尺度 c 扩为六条未知候选边。四图×三噪声档的12个有激励批次，在主细两格中均选对支撑，假边/漏边均为0；四个平衡条件的三档、两格共24设置返回 null，区别于有信息的零图空集。

无噪最大边权误差2.0612e-4；高噪 sigma=1e-4 最大误差两格合计0.2637447。支撑正确不等于含噪强度精确。每条件一个固定方向、两档噪声配对，两积分格不是独立重复；结论限登记K4条件，未证明任意图或K10恢复。

32生成格、48观测包、96重建和48设置全部完成；1560代数项执行、1560因低信息跳过。1530次NNLS全部合格。独立矩阵复算最大相对KKT为2.75e-16，全部资格/评分一致；固定有限字典求解充分，连续含噪逆问题充分性unknown。端口、输入与数值检查通过。唯一运行4.702 s，宿主调用6.602 s，16冻结源未变；最终27项工程检查通过，无执行后修复或重放。

[科学判读](../reports/gftd_sparse_topology_20261107_scientific_assessment.md)、[独立结果审查](../reports/gftd_sparse_topology_20261107_result_review.md)、[root决定](../results/gftd_sparse_topology_20261107/root_adoption.json)分别保存。代码retain_branch；模型retain_evidence；科学supported_within_scope，Campaign继续。工作量关闭、余额不转移；started.json是历史启动凭据。

## 下一动作与归档

[三步主线架构](../reports/gftd_mainline_20261107_architecture.md)保持。Run 2 的[正式设计待审](../reports/gftd_modulation_20261107_protocol.md)采用候选自洽的幅值、精确指数列、层级筛选和完整两阶段初始化；代码资格、机制恢复与预测价值分别判断。Run 1 归档核验后进入设计复核、实现与有限执行，不另开噪声支线。

[Run 3接口盘点](../reports/gftd_fit297_20261107_interface_inventory.md)已定位train-only读取和L1复用位置。物理Kron矩阵及其实际原始算例文件尚未找到，L3参考保持unknown；已向用户询问可选本地路径，前两步照常推进。尚未读取fit297事件、执行IEEE39训练或计算L2证书。

父单尺度Run提交0cf535b2d7d3dd01a05925dc1ea1da4a774228e7已推送并核验39文件原字节；本轮正在选择性归档，归档未确认前不释放下一Run。
