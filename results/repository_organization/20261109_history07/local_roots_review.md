# 6个本地根的来源补核

本轮保留全部6个目录及原路径。公开记录、顶层文件名和两份小型已保存评价说明足以缩小用途范围，尚不能确认具体来源Job，也不构成原始数据或预测内容的完整性核验。

| 本地根 | 本次直接观察与来源 | 处理与解除条件 |
|---|---|---|
| `private_data/p_constant_direction_EQ_engineering_20261031a` | 顶层有`test_all_ten_main_files_must_s0`、`test_no_unknown_completed_stag0`、`test_Q_not_released_without_al0`及对应`current`联接；名称与[已有测试函数](../../../tests/test_p_constant_direction_20261031.py)对应。[旧清理记录](../../../reports/maintenance/repository_runtime/originals/private_data_cleanup_20261108.md)记录过受限访问 | 疑似工程测试残留，不能仅凭名称证明实际生成者。未进入子目录或联接；先核对归属、访问与恢复能力，再单独决定迁移或清理 |
| `private_data/p_constant_direction_EQ_engineering_20261031b` | 同上，顶层名称和联接类型相同；本轮能列根目录，不证明旧受限子树已可读 | 同上；不修改ACL，不把根目录可列举记为全部内容已核验 |
| `private_data/p_constant_direction_EQ_engineering_20261031c` | 同上 | 同上 |
| `private_data/task2_long_0060_2s_eval` | 顶层仅列出`evidence.json`、`per_event_rows.json`、`trajectories.npz`及大小；[0060原报告](../../../reports/task2_long_horizon_campaign.md)记载2s阶段及其后续。确切`_eval`生成命令未找到 | 0060评价用途是名称与报告形成的归属推测，具体Job未知。保留预测和证据，待找到原命令／来源清单并验证其报告职责后再迁移 |
| `private_data/task2_structure_0078_extend_C-free_eval` | [原保护清单](../../../reports/private_data_cleanup_plan_20261107.md)明确包括C3000及对应`_eval`。顶层有预测、逐事件行和独立审查文件；读取3,218字节的`independent_evaluation.json`，其方法说明为既有缓存的辅助复核 | 旧评价及辅助复核材料，源Job未知；没有重新运行该方法。先核清被保护C3000的决定性证据和消费者，迁移时逐项存证；本次不打开预测或H5 |
| `private_data/task2_structure_0078_extend_G-fixed_eval` | 同一[原保护清单](../../../reports/private_data_cleanup_plan_20261107.md)明确包括G3000及对应`_eval`。读取1,216字节的`independent_evaluation.json`，其方法说明为既有物理／数值辅助复核 | 同上，保持原保护。小型说明中记录的历史数据访问属于原执行，不是本次重新访问或新的授权 |

本次没有遍历6根的子树、跟随联接、读取`evidence.json`／逐事件行／轨迹／权重或运行其复核脚本。旧文件数、内容哈希和逻辑大小仍以原迁移／清理记录的核验范围为准。旧关联表中的“未发现直接公开引用”保留其编制时点，本页补充证据与推测的区别，不回填历史Run或模型采用状态。
