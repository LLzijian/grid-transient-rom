# 当前架构：OSGKD

当前活动设计为 [OSGKD 三端联合辨识](../designs/osgkd_architecture.md)，主合同为 [motion_only_v1](../kiki.md)。本次范围是阶段一的算法与控制面切换，尚无新训练或预测结果；架构采用以集中审查和 root 回执为准。

| 部分 | 学习对象与边界 |
|---|---|
| 机械驱动 | 用实际存在的 train 端口标签锚定有效机械作用，重新辨识动态，旧机械模型只作起点 |
| 内部耦合 | 有效图、基础强度及有界调制；互易交换部分零和，图先验与真实线路恢复分开 |
| 环境交换 | 有来源的静态电气作用及必要动态记忆；允许非零和与正负增量，不能把旧 r 直接重命名 |

首版最多沿用 `(theta,omega,p,a,e)` 的 5N 容器，先形成完整 C0/C1，再删组重拟合。部署仅 11 点/100 ms 相角与完整转速历史及声明的冻结常数，由估计器给出 p/a/e 初态；真实 p₀ 只属于 `known_p0_reference`。固定参考、潜态规范、完整功率链式账和同一未来目标见唯一详细设计。

下一阶段首项工作是核对端口字段、单位、符号、参考与样本覆盖，并实现同信息 C0/C1。字段映射、有限训练协议和实现资格尚未完成；缺环境分项时采用声明的有效分解，不造逐机真值。[OSGKD Campaign](../campaigns/osgkd_ieee39/campaign.md) · [当前状态](current_state.md)

GFTD 作为前序证据与可复用组件保留，旧真 p₀ API 不承担新合同：

- [5N 原修订](../designs/gftd_repair_design_20261107.md)、[有界核与储能分析](../designs/gftd_storage_feasibility_20261109.md)、[同核初始化](../designs/gftd_bounded_initialization_20261109.md)。
- [匹配预测](../designs/gftd_matched_prediction_20261109.md)、[三点更新](../designs/gftd_objective_alignment_20261112.md)、[局部补偿](../designs/gftd_joint_distinguishability_20261112.md)。
- [GFTD 收束](../reports/osgkd_transition_20261113/gftd_closure.md)、[原三步主线](../designs/gftd_mainline_20261107_architecture.md)、[设计档案](../designs/history/README.md)。

已接受的 40 维基线及锚点保持原适用范围。default none、incumbent 和 protected anchors 不变。
