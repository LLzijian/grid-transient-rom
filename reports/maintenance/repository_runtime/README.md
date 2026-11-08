# 仓库与运行规范：历史变更和存储整理

本页集中保存 Git、harness、执行规则与本地存储的维护记录。它们不作为科学 Run，也不改变模型的接受或拒绝状态。

| 事项 | 如何使用 |
|---|---|
| harness 重写、审查修正与物理研究执行规则整合 | 回溯当时的工程决定；今天的规范从 [AGENTS](../../../AGENTS.md)、[工程循环](../../../docs/engineering_loop.md)及[代理运行边界](../../../agents/runtime.md)进入 |
| 原进程额度取消 | 保留用户决定的历史记录；旧报告的费用和关闭余额继续保留，不由目录整理重设额度 |
| Git 清理与归档 | 核查已发生的维护事实；不据旧命令清除今天的工作区 |
| 本地数据与产物整理 | [原清理报告](originals/private_data_cleanup_20261108.md)与[旧迁移清单](../../../results/private_data_cleanup_20261107/migration_map.csv)一起使用；[本次盘点](../../../results/repository_organization/20261108_inventory01/README.md)说明重新核查的范围 |

[材料表](catalog.md)保留原标题和固定提交阅读版本；本组归档原件保持字节不变。迁入 `artifacts/` 不等于释放磁盘空间，目录元数据吻合不等于重新核验了全部大文件内容，Git 哈希也不能代替本地模型与数据的备份。

旧分支合并与可恢复归档另有[原维护回执](../../../results/branch_maintenance_20261108/maintenance.json)。其中的分支数量和核验状态属于该历史时点，不是今天的 Git 状态。

后续规范化进度及可撤回回执统一保存在[整理批次](../../../results/repository_organization/README.md)，不再为每个维护文件新增一篇报告。
