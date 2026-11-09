# 力记忆恢复与p通道检查代码

本组入口对应已闭合的历史任务。原始字节、失败记录和接受范围保留；源码里的ROOT、同目录导入与相对输入仍按原版本解释。

**阅读先看[所属报告主题](../../../reports/history/force_memory/README.md)。需要恢复旧执行环境时，在单独目录检出完整提交 `145847532bacd3454dca8869e819778efc8258f6`，按原目录和合法输入核查；本目录只保存原件，不能直接运行。**

归档不恢复训练、val/test或新FOM权限；不重新执行旧检查以验证目录整理。

| 原入口 | 用途及原报告 |
|---|---|
| [p_channel_diagnostic_20261030.py](p_channel_diagnostic_20261030.py) | EXPLORATORY / POST-HOC: p-channel error decomposition (not pre-registered).；[原报告](https://github.com/LLzijian/grid-transient-rom/blob/145847532bacd3454dca8869e819778efc8258f6/reports/closure_probe_20261030_conclusion.md) |
| [probe64_20261030.py](probe64_20261030.py) | P4 (H3 cheap check): frozen A step64 versus step128 forward evaluation.；[原报告](https://github.com/LLzijian/grid-transient-rom/blob/145847532bacd3454dca8869e819778efc8258f6/reports/closure_probe_20261030_conclusion.md) |
| [r_memory_recovery_evaluate_20261028.py](r_memory_recovery_evaluate_20261028.py) | R0: one new CPU evaluation of the fixed, latest durable A/25101 model.；[原报告](https://github.com/LLzijian/grid-transient-rom/blob/145847532bacd3454dca8869e819778efc8258f6/reports/r_memory_20261028_conclusion.md) |

[迁移与回退回执](../../../results/repository_organization/20261109_history07/migration.json)含本地字节哈希及完整旧版本标识。
