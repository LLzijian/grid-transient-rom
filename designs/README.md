# 算法设计

当前主线为 **[OSGKD 三端联合辨识](osgkd_architecture.md)**，分别学习机械驱动、内部耦合和环境交换。
采用状态见 [current_design](../docs/current_design.md)，研究安排见 [OSGKD Campaign](../campaigns/osgkd_ieee39/campaign.md)，运行事实见 [current_state](../docs/current_state.md)。

| 活动设计 | 范围与状态 |
|---|---|
| [OSGKD 架构](osgkd_architecture.md) | 阶段一已采用；主合同为 motion_only_v1，先构造同信息 C0/C1。尚无 OSGKD 训练结果，阶段二未释放。 |

旧 GFTD 设计已归入 [历史档案](history/README.md#gftd-20261113)，按原合同保留为前序证据与可复用组件；本目录不再把 5N GFTD 标为当前候选。
本次入口、归档与角色回执核查见 [补充报告](../reports/osgkd_harness_followup_20261113/README.md)。

通用设计方法与权限见 [architecture_design](../docs/architecture_design.md) 和 [kiki](../kiki.md)。
具体算法设计放 `designs/`，旧版本放 `designs/history/`；Run 协议随 Campaign 放 `campaigns/`，结果、分析及审查放 `reports/`。
版本、父引用和采用状态继续由现有 Campaign 管理，本目录不另建实验树。
