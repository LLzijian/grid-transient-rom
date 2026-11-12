# 当前架构与待辨别问题

当前采用 GFTD-Kuramoto 的 5N 修订作为研究候选：共享自治方程显式保留相角、转速、
机械功率、隐幅值和力记忆。IEEE39 十机实例为 50 维；采用研究候选不等于模型晋升。
部署与数据合同见 [kiki](../kiki.md)，通用设计方法见 [architecture_design](architecture_design.md)。

## 设计入口

| 内容 | 来源 |
|---|---|
| 当前方程、父模型映射与资格要求 | [5N 修订设计](../designs/gftd_repair_design_20261107.md) |
| 数学与同初态数值资格通过的有界调制研究候选（原生产方程不变） | [储能可行性与有限时域存在](../designs/gftd_storage_feasibility_20261109.md) |
| 原297个样本已通过六项数值检查的同核初始化 | [同核初始化设计](../designs/gftd_bounded_initialization_20261109.md) |
| 已闭合的匹配预测Run（完整L1/L2/L3与独立审查） | [匹配预测设计](../designs/gftd_matched_prediction_20261109.md) |
| 已采用设计、正在隔离实现的三点历史目标比较 | [三点更新设计](../designs/gftd_objective_alignment_20261112.md) |
| 研究假说、竞争解释与 Run 树 | [GFTD Campaign](../campaigns/gftd_kuramoto_discovery_20261106.md) |
| 未知拓扑 → 动态调制与接口 → train fit297 同期比较 | [三步主线](../designs/gftd_mainline_20261107_architecture.md) |
| 原Run 3的输入、工作量及判据 | [fit297 协议](https://github.com/LLzijian/grid-transient-rom/blob/145847532bacd3454dca8869e819778efc8258f6/reports/gftd_fit297_20261107_protocol.md) |
| v2.0 草案及其更正 | [原架构简报](../designs/gftd_kuramoto_architecture_brief.md) |

节点局域核、候选边族和耗散形式属于建模先验；边支撑、调制作用和参数恢复由证据判断。
完整方程、初始化细节和实验成绩保留在上述设计及报告中。

## 待辨别问题

1. 当前弱式提案的固定缩步能否改善同一自治历史目标？实现、纯mock资格和旧端点聚合身份核验均已通过，科学执行已单独释放，按固定三个步长取证。
2. 在哪些观测与激励条件下，静态边、动态调制和加性记忆能够被区分？
3. 如何在原分母和长时数值资格下改善 train fit297 自主预测，并区分完整方法差异与调制本身的作用？
4. 如何让同一部署 ODE 的严格局部证书覆盖实际初态及其误差集合？已有 L3 图及模态对应保持描述性。

Run 1 支持有限 K4 静态恢复；Run 2 已接通拟合、存取、完整初始化与自主预测，
动态调制恢复仍未获支持。结果范围及下一动作由 [current_state](current_state.md) 连接对应证据。
既有 40 维基线和已接受锚点保留原适用范围；default none，incumbent 与 protected anchors 不变。
历史架构与整理前完整说明见 [设计档案](../designs/history/README.md)。
