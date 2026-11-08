# 早期私有模型与TASK2代码

本组入口对应已闭合的历史任务。原始字节、失败记录和接受范围保留；源码里的ROOT、同目录导入与相对输入仍按原版本解释。

**阅读先看[所属报告主题](../../../reports/history/private_rom/README.md)。需要恢复旧执行环境时，在单独目录检出完整提交 `145847532bacd3454dca8869e819778efc8258f6`，按原目录和合法输入核查；本目录只保存原件，不能直接运行。**

归档不恢复训练、val/test或新FOM权限；不重新执行旧检查以验证目录整理。

| 原入口 | 用途及原报告 |
|---|---|
| [audit_electrical_port_balance.py](audit_electrical_port_balance.py) | Raw electrical load/loss balance explains the generator-only net residual.；[原报告](../../../reports/accepted/0048_electrical_port_balance.md) |
| [audit_holdout_supplement.py](audit_holdout_supplement.py) | Mechanical TEST coverage QA only; never compute a predictor score.；[原报告](../../../reports/history/private_rom/originals/accepted/0047_holdout_duration_supplement.md) |
| [audit_private_energy.py](audit_private_energy.py) | Exact frozen-M3 storage identity and separate full-order trajectory audit.；[原报告](../../../reports/history/private_rom/originals/accepted/0028_m3_energy.md) |
| [audit_private_energy_boundary.py](audit_private_energy_boundary.py) | Exact-rational conservative M3 barrier; finite-horizon probes are separate.；[原报告](../../../reports/history/private_rom/originals/accepted/0036_m3_energy_boundary.md) |
| [audit_private_expansion.py](audit_private_expansion.py) | Cross-shard identity/coverage QA; never a model score or test feature fit.；[原报告](../../../reports/history/private_rom/originals/accepted/0039_formal_expansion.md) |
| [audit_private_reachable_boundary.py](audit_private_reachable_boundary.py) | Fixed physical-amplitude ray probes; numerical LOS boundary, not full ROA.；[原报告](../../../reports/history/private_rom/originals/accepted/0037_m3_reachable_boundary.md) |
| [check_boundary_probe_0077.py](check_boundary_probe_0077.py) | Independent read-only QA for completed Loop0077 artifacts.；[原报告](../../../reports/history/private_rom/originals/failed/0077_boundary_probe.md) |
| [check_long_step.py](check_long_step.py) | Registered .0025-step convergence diagnostic for saved Loop0060 forecasts.；[原报告](../../../reports/accepted/0060_long_horizon_campaign.md) |
| [check_structure_prototype.py](check_structure_prototype.py) | Loop 0076 prototype QA runner (synthetic and M3 assets only).；[原报告](../../../reports/history/private_rom/originals/accepted/0076_structure_prototype.md) |
| [diagnose_electrical_residual_atlas.py](diagnose_electrical_residual_atlas.py) | TRAIN-only residual decomposition and hidden-state association diagnostics.；[原报告](../../../reports/history/private_rom/originals/accepted/0040_electrical_residual_atlas.md) |
| [diagnose_private_memory.py](diagnose_private_memory.py) | Post0026 numerical oracles: explicitly noncausal diagnostics, never a model.；[原报告](../../../reports/history/private_rom/originals/accepted/0027_memory_oracles.md) |
| [diagnose_topology_force_capacity.py](diagnose_topology_force_capacity.py) | Noncausal forcing controls isolate geometry, coordinate and ODE failures.；[原报告](../../../reports/history/private_rom/originals/accepted/0045_topology_force_capacity.md) |
| [evaluate_joint_adequacy_0099.py](evaluate_joint_adequacy_0099.py) | Bounded reuse of the frozen trajectory/mode and full-TRAIN evaluators.；[原报告](../../../reports/history/private_rom/originals/capabilities/0099_joint_adequacy_capability.md) |
| [run_private_m3.py](run_private_m3.py) | One fit, true-speed diagnostic and autonomous M3-Pm contribution screen.；[原报告](../../../reports/history/private_rom/originals/accepted/0020_private_m3_pm.md) |
| [summarize_shared_closure_0101.py](summarize_shared_closure_0101.py) | Pure-JSON, read-only 0101 evidence extraction; no model or data loading.；[原报告](../../../reports/0101_shared_closure_FS_capability.md) |

[迁移与回退回执](../../../results/repository_organization/20261109_history07/migration.json)含本地字节哈希及完整旧版本标识。
