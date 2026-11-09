# 算法设计

本目录说明算法的状态、方程、物理假设、推断原则和设计修订。
通用方法与权限见 [docs/architecture_design](../docs/architecture_design.md) 和 [kiki](../kiki.md)；
当前采用状态见 [current_design](../docs/current_design.md)，运行安排见 [Campaign](../campaigns/gftd_kuramoto_discovery_20261106.md)。

| 设计 | 用途与状态 |
|---|---|
| [GFTD 原始架构简报](gftd_kuramoto_architecture_brief.md) | v2.0 原始方案及更正；当前方程以 5N 修订为准 |
| [5N 修订设计](gftd_repair_design_20261107.md) | 当前研究候选的状态、方程、父模型映射与资格要求 |
| [三步主线](gftd_mainline_20261107_architecture.md) | 未知拓扑、动态调制与接口、train fit297 同期比较 |
| [fit297 L2 设计](gftd_fit297_20261107_l2_design.md) | 同一部署 ODE 的证书推导及验证方法；不表示证书已经获得 |

旧算法方案、模型专用实现说明与先前版本见 [designs/history](history/README.md)。
新 Run 协议随 Campaign 放 `campaigns/`；实验结果、分析和审查报告放 `reports/`。
版本、父引用和采用状态继续由现有 Campaign 管理，本目录不另建实验树。

## fit297 的冻结来源

fit297 Run 已关闭。可从固定提交回查当时绑定的[三步主线原件](https://github.com/LLzijian/grid-transient-rom/blob/145847532bacd3454dca8869e819778efc8258f6/reports/gftd_mainline_20261107_architecture.md)和[L2 设计原件](https://github.com/LLzijian/grid-transient-rom/blob/145847532bacd3454dca8869e819778efc8258f6/reports/gftd_fit297_20261107_l2_design.md)。本目录两份设计副本与冻结原件字节一致，后续修订另记版本。

迁移来源、位置与 SHA256 统一记录在 [迁移清单](../harness/documentation_migration_20261108.json)。
