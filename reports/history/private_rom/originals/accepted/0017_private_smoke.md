# Loop 0017：私有同一后故障系统 smoke 数据链通过

2026-09-08；冻结实现`eb83bc2`。接受生成、导出、QA与单一COI adapter，
不宣称Pilot已完成。细节见`reports/private_dataset_build_report.md`。

- 5例尝试：3稳定、1物理LOS正常接受；1数值TDS失败整例丢弃。
- 稳定路径15s/1501点，LOS路径40ms/5点，无padding、无外推。
- 正常求解后的LOS停止与导出轨迹上的统一持续判据一致。
- 220维full state、479维代数量及必需Pm/Pe和映射齐全。
- post-fault参数/开关哈希一致，初始基准状态一致，故障确实施加并清除。
- 4例保存数据QA通过；44项测试、compileall、新代码Ruff通过。
- 私有轨迹均由`.gitignore`排除，不进入Git；只提交脚本、配置及统计报告。

首轮导出网格错误在实现阶段修正，历史试跑未覆盖。未调整物理阈值，
未恢复失败前缀。下一阶段沿此定义生成384尝试的Pilot，立即为M2/M3-Pm服务。
