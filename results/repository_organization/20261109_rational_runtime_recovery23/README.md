# 历史执行与 QA 报告归档

本批归档14份原件，逐件核对原始字节和固定版本。主题路径保护清单现有0项，全项目报告路径保护剩余186项。

| 原标题 | 连续阅读 | 归档原件 |
|---|---|---|
| Astra6：获授权后的执行器修复方案 | [固定版本](https://github.com/LLzijian/grid-transient-rom/blob/145847532bacd3454dca8869e819778efc8258f6/reports/astra6_authorized_executor_repair_20261022.md) | [原件](../../../reports/maintenance/execution_runtime/originals/astra6_authorized_executor_repair_20261022.md) |
| Astra6：获授权的执行机制重构 | [固定版本](https://github.com/LLzijian/grid-transient-rom/blob/145847532bacd3454dca8869e819778efc8258f6/reports/astra6_executor_reconstruction_20261022.md) | [原件](../../../reports/maintenance/execution_runtime/originals/astra6_executor_reconstruction_20261022.md) |
| 实际Astra6：QA05只读方案与root采纳 | [固定版本](https://github.com/LLzijian/grid-transient-rom/blob/145847532bacd3454dca8869e819778efc8258f6/reports/astra6_qa05_20261022.md) | [原件](../../../reports/maintenance/execution_runtime/originals/astra6_qa05_20261022.md) |
| QA05 Astra6 原始返回 | [固定版本](https://github.com/LLzijian/grid-transient-rom/blob/145847532bacd3454dca8869e819778efc8258f6/reports/astra6_qa05_20261022_raw.txt) | [原件](../../../reports/maintenance/execution_runtime/originals/astra6_qa05_20261022_raw.txt) |
| Astra6新授权执行方案：原始返回 | [固定版本](https://github.com/LLzijian/grid-transient-rom/blob/145847532bacd3454dca8869e819778efc8258f6/reports/astra6_rational_execution_20261021.md) | [原件](../../../reports/maintenance/execution_runtime/originals/astra6_rational_execution_20261021.md) |
| 实际Astra6新轮次原始方案与root采纳 | [固定版本](https://github.com/LLzijian/grid-transient-rom/blob/145847532bacd3454dca8869e819778efc8258f6/reports/astra6_spawn_chain_20261022.md) | [原件](../../../reports/maintenance/execution_runtime/originals/astra6_spawn_chain_20261022.md) |
| 启动重构独立挑战与root处理 | [固定版本](https://github.com/LLzijian/grid-transient-rom/blob/145847532bacd3454dca8869e819778efc8258f6/reports/executor_reconstruction_20261022_independent_review.md) | [原件](../../../reports/maintenance/execution_runtime/originals/executor_reconstruction_20261022_independent_review.md) |
| Windows执行机制重构：两方案、一次最终验收 | [固定版本](https://github.com/LLzijian/grid-transient-rom/blob/145847532bacd3454dca8869e819778efc8258f6/reports/executor_reconstruction_20261022_protocol.md) | [原件](../../../reports/maintenance/execution_runtime/originals/executor_reconstruction_20261022_protocol.md) |
| QA05短路径复验与原12模型执行 | [固定版本](https://github.com/LLzijian/grid-transient-rom/blob/145847532bacd3454dca8869e819778efc8258f6/reports/qa05_recheck_20261022_protocol.md) | [原件](../../../reports/maintenance/execution_runtime/originals/qa05_recheck_20261022_protocol.md) |
| 无进程额度限制下的完整因果对照 | [固定版本](https://github.com/LLzijian/grid-transient-rom/blob/145847532bacd3454dca8869e819778efc8258f6/reports/rational_execution_20261021_protocol.md) | [原件](../../../reports/maintenance/execution_runtime/originals/rational_execution_20261021_protocol.md) |
| 唯一实现修复：首次QA的Windows执行问题 | [固定版本](https://github.com/LLzijian/grid-transient-rom/blob/145847532bacd3454dca8869e819778efc8258f6/reports/rational_execution_20261021_repair.md) | [原件](../../../reports/maintenance/execution_runtime/originals/rational_execution_20261021_repair.md) |
| 执行器额外一次修复与恢复训练 | [固定版本](https://github.com/LLzijian/grid-transient-rom/blob/145847532bacd3454dca8869e819778efc8258f6/reports/rational_execution_additional_repair_20261022_protocol.md) | [原件](../../../reports/maintenance/execution_runtime/originals/rational_execution_additional_repair_20261022_protocol.md) |
| 辅助身份记录器修订 | [固定版本](https://github.com/LLzijian/grid-transient-rom/blob/145847532bacd3454dca8869e819778efc8258f6/reports/rational_execution_recorder_amendment_20261022.md) | [原件](../../../reports/maintenance/execution_runtime/originals/rational_execution_recorder_amendment_20261022.md) |
| 子进程链修复：固定五例与一次QA06 | [固定版本](https://github.com/LLzijian/grid-transient-rom/blob/145847532bacd3454dca8869e819778efc8258f6/reports/spawn_chain_repair_20261022_protocol.md) | [原件](../../../reports/maintenance/execution_runtime/originals/spawn_chain_repair_20261022_protocol.md) |

待复核的程序及机器记录引用已逐条收入保护表。实际调用与历史来源记录分别核查。 [保护明细](protected.csv)

迁移回执列出原址、现址、大小、SHA256、入链与出链。恢复时先核对归档字节及原址是否空闲，再移回原件并恢复索引。 [迁移回执](migration.json) · [验收](validation.json)

QA05 和 QA06 的报告已从各自冻结提交恢复到隔离目录。两版历史运行器的报告路径与 SHA256 检查通过，篡改字节均被拒绝。QA05 使用原绑定，QA06 用冻结源中的路径和哈希构造测试绑定。

回查历史输入时，从迁移回执所列提交读取原路径和字节，并核对原绑定。验收范围为公开报告的文件门槛。
