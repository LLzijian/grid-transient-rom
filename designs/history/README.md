# 历史设计

保存旧算法方案、专项数据与搜索流程、模型专用实现说明，以及旧版本的完整原文。
混合文档中的资格检查、失败和成本记录随方案保留；归档不改变原结论或采用范围。
当前方案见 [设计入口](../README.md)，实验结果与分析见 [reports](../../reports/README.md)。

归档文件保持原字节，相对路径仍具有原文件位置的历史语义；查找迁移位置和哈希使用
[原迁移清单](../../harness/documentation_migration_20261108.json)和[当前归档入口](../../results/repository_organization/current.json)。历史实验按原提交和冻结材料复现。

## GFTD 20261113

11 份前序 GFTD 设计按原字节归档；原相对链接按归档前位置理解。下表“原提交阅读”可连续读取当时上下文。
迁移路径、字节数与 SHA256 见 [本批清单](../../results/repository_organization/20261113_gftd_design_archive/migration.json)；旧协议和回执保持原文，harness 通过已校验映射查找原路径。

| 归档原文 | 原提交阅读 |
|---|---|
| [gftd_bounded_initialization_20261109.md](gftd_20261113/gftd_bounded_initialization_20261109.md) | [原位置](https://github.com/LLzijian/grid-transient-rom/blob/0ff2c9643d9eae11ca00620ed816e58beeba5dc9/designs/gftd_bounded_initialization_20261109.md) |
| [gftd_fit297_20261107_l2_design.md](gftd_20261113/gftd_fit297_20261107_l2_design.md) | [原位置](https://github.com/LLzijian/grid-transient-rom/blob/0ff2c9643d9eae11ca00620ed816e58beeba5dc9/designs/gftd_fit297_20261107_l2_design.md) |
| [gftd_joint_distinguishability_20261112.md](gftd_20261113/gftd_joint_distinguishability_20261112.md) | [原位置](https://github.com/LLzijian/grid-transient-rom/blob/0ff2c9643d9eae11ca00620ed816e58beeba5dc9/designs/gftd_joint_distinguishability_20261112.md) |
| [gftd_kuramoto_architecture_brief.md](gftd_20261113/gftd_kuramoto_architecture_brief.md) | [原位置](https://github.com/LLzijian/grid-transient-rom/blob/0ff2c9643d9eae11ca00620ed816e58beeba5dc9/designs/gftd_kuramoto_architecture_brief.md) |
| [gftd_linear_compensation_bounds_20261112.md](gftd_20261113/gftd_linear_compensation_bounds_20261112.md) | [原位置](https://github.com/LLzijian/grid-transient-rom/blob/0ff2c9643d9eae11ca00620ed816e58beeba5dc9/designs/gftd_linear_compensation_bounds_20261112.md) |
| [gftd_mainline_20261107_architecture.md](gftd_20261113/gftd_mainline_20261107_architecture.md) | [原位置](https://github.com/LLzijian/grid-transient-rom/blob/0ff2c9643d9eae11ca00620ed816e58beeba5dc9/designs/gftd_mainline_20261107_architecture.md) |
| [gftd_matched_prediction_20261109.md](gftd_20261113/gftd_matched_prediction_20261109.md) | [原位置](https://github.com/LLzijian/grid-transient-rom/blob/0ff2c9643d9eae11ca00620ed816e58beeba5dc9/designs/gftd_matched_prediction_20261109.md) |
| [gftd_objective_alignment_20261112.md](gftd_20261113/gftd_objective_alignment_20261112.md) | [原位置](https://github.com/LLzijian/grid-transient-rom/blob/0ff2c9643d9eae11ca00620ed816e58beeba5dc9/designs/gftd_objective_alignment_20261112.md) |
| [gftd_objective_alignment_20261112_draft.md](gftd_20261113/gftd_objective_alignment_20261112_draft.md) | [原位置](https://github.com/LLzijian/grid-transient-rom/blob/0ff2c9643d9eae11ca00620ed816e58beeba5dc9/designs/gftd_objective_alignment_20261112_draft.md) |
| [gftd_repair_design_20261107.md](gftd_20261113/gftd_repair_design_20261107.md) | [原位置](https://github.com/LLzijian/grid-transient-rom/blob/0ff2c9643d9eae11ca00620ed816e58beeba5dc9/designs/gftd_repair_design_20261107.md) |
| [gftd_storage_feasibility_20261109.md](gftd_20261113/gftd_storage_feasibility_20261109.md) | [原位置](https://github.com/LLzijian/grid-transient-rom/blob/0ff2c9643d9eae11ca00620ed816e58beeba5dc9/designs/gftd_storage_feasibility_20261109.md) |

## 较早档案

- [architecture_brief_20261030.md](architecture_brief_20261030.md)
- [closure_discovery_runbook.md](closure_discovery_runbook.md)
- [core_before_cleanup_20261108/architecture_design.md](core_before_cleanup_20261108/architecture_design.md)
- [core_before_cleanup_20261108/current_design.md](core_before_cleanup_20261108/current_design.md)
- [cubic_memory_port_runtime.md](cubic_memory_port_runtime.md)
- [cubic_port_only_runtime.md](cubic_port_only_runtime.md)
- [current_design_before_gftd_kuramoto_20261106.md](current_design_before_gftd_kuramoto_20261106.md)
- [edge_memory_amplitude_runtime.md](edge_memory_amplitude_runtime.md)
- [finite_grammar_discovery.md](finite_grammar_discovery.md)
- [frequency_dissipation_runtime.md](frequency_dissipation_runtime.md)
- [gftd_kuramoto_architecture_brief.md](gftd_kuramoto_architecture_brief.md)
- [navigation_20261108/gftd_kuramoto_architecture_brief.md](navigation_20261108/gftd_kuramoto_architecture_brief.md)
- [outside_grammar_causal.md](outside_grammar_causal.md)
- [p_dissipative_closure_20261031_architecture_brief.md](p_dissipative_closure_20261031_architecture_brief.md)
- [p_minimal_closure_20261031_architecture_brief.md](p_minimal_closure_20261031_architecture_brief.md)
- [quartic_memory_runtime.md](quartic_memory_runtime.md)
- [rational_control_causal.md](rational_control_causal.md)
- [spatial_extrema_runtime.md](spatial_extrema_runtime.md)
- [TASK1_private_dataset_design.md](TASK1_private_dataset_design.md)
- [TASK2_after_loop_101.md](TASK2_after_loop_101.md)
- [TASK2_after_loop_26.md](TASK2_after_loop_26.md)
- [TASK2_after_loop_39.md](TASK2_after_loop_39.md)
- [TASK2_after_loop_52.md](TASK2_after_loop_52.md)
- [TASK2_after_loop_59.md](TASK2_after_loop_59.md)
- [TASK2_after_loop_66.md](TASK2_after_loop_66.md)
- [TASK2_after_loop_75.md](TASK2_after_loop_75.md)
- [TASK2_after_loop_97.md](TASK2_after_loop_97.md)
- [TASK2_memory_sparse_energy_rom_design.md](TASK2_memory_sparse_energy_rom_design.md)
- [temporal_objective_runtime.md](temporal_objective_runtime.md)
