# 当前架构与待辨别问题

当前采用 GFTD-Kuramoto 的 5N 修订作为研究候选：共享自治方程显式保留相角、转速、
机械功率、隐幅值和力记忆。IEEE39 十机实例为 50 维；采用研究候选不等于模型晋升。
部署与数据合同见 [kiki](../kiki.md)，通用设计方法见 [architecture_design](architecture_design.md)。

## 设计入口

| 内容 | 来源 |
|---|---|
| 当前方程、父模型映射与资格要求 | [5N 修订设计](../designs/gftd_repair_design_20261107.md) |
| 研究假说、竞争解释与 Run 树 | [GFTD Campaign](../campaigns/gftd_kuramoto_discovery_20261106.md) |
| 未知拓扑 → 动态调制与接口 → train fit297 同期比较 | [三步主线](../designs/gftd_mainline_20261107_architecture.md) |
| 当前比较的输入、工作量及判据 | [fit297 协议](https://github.com/LLzijian/grid-transient-rom/blob/145847532bacd3454dca8869e819778efc8258f6/reports/gftd_fit297_20261107_protocol.md) |
| v2.0 草案及其更正 | [原架构简报](../designs/gftd_kuramoto_architecture_brief.md) |

节点局域核、候选边族和耗散形式属于建模先验；边支撑、调制作用和参数恢复由证据判断。
完整方程、初始化细节和实验成绩保留在上述设计及报告中。

## 待辨别问题

1. 在哪些观测与激励条件下，静态边、动态调制和加性记忆能够被区分？
2. 固定合法初始化、共同模态设置与适配机会，调制开关如何改变 train fit297 的 L1 指标？
3. 同一部署 ODE 能否满足 L2 的完整证明条件？L3 的图及模态对应按原口径报告。

Run 1 支持有限 K4 静态恢复；Run 2 已接通拟合、存取、完整初始化与自主预测，
动态调制恢复仍未获支持。结果范围及下一动作由 [current_state](current_state.md) 连接对应证据。
既有 40 维基线和已接受锚点保留原适用范围；default none，incumbent 与 protected anchors 不变。
历史架构与整理前完整说明见 [设计档案](../designs/history/README.md)。
