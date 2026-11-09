# 执行与恢复：进程监督、来源身份和检查点接口

**这一组材料解释科学比较能否可靠启动、保存和恢复。执行器通过不表示模型有效；执行器或检查点失败也不评价闭合机制的预测能力。**

本页追溯 2026-10-21—22 的执行维护。原失败、修复次数、来源和费用原样保留；旧命令不作为今天的运行指令。

| 需要回溯什么 | 原记录中的结果 | 来源 |
|---|---|---|
| 初次无进程额度执行入口 | 监督检查未通过，正式12模型训练尚未启动，数学核心和旧结果保留原分类。 | [原关闭结论](https://github.com/LLzijian/grid-transient-rom/blob/145847532bacd3454dca8869e819778efc8258f6/reports/rational_execution_20261021_conclusion.md)、[额外一次修复](https://github.com/LLzijian/grid-transient-rom/blob/145847532bacd3454dca8869e819778efc8258f6/reports/rational_execution_additional_repair_20261022_conclusion.md) |
| 启动链与真实进程身份 | 后续固定五例验证正常结束、孤儿清理和监督退出；完整科学运行又暴露了检查点接口问题。两类故障分别记录。 | [启动链结果](https://github.com/LLzijian/grid-transient-rom/blob/145847532bacd3454dca8869e819778efc8258f6/reports/spawn_chain_repair_20261022_conclusion.md)、[工程集成范围](https://github.com/LLzijian/grid-transient-rom/blob/145847532bacd3454dca8869e819778efc8258f6/reports/windows_job_runtime_20261022_engineering.md) |
| 重构或重新检查为何未直接验收 | 保留测试设置错误、临时路径与未完成检查；不把已走到的代码路径补成完整通过。 | [重构结果](https://github.com/LLzijian/grid-transient-rom/blob/145847532bacd3454dca8869e819778efc8258f6/reports/executor_reconstruction_20261022_conclusion.md)、[QA05复核](https://github.com/LLzijian/grid-transient-rom/blob/145847532bacd3454dca8869e819778efc8258f6/reports/qa05_recheck_20261022_conclusion.md) |
| 接口修复后的科学结果在哪里 | 固定12模型续行的预测结果属于科学比较，另在历史研究主题阅读；本页不把它重复计为工程成绩。 | [闭合函数形式比较](https://github.com/LLzijian/grid-transient-rom/blob/145847532bacd3454dca8869e819778efc8258f6/reports/checkpoint_continuation_20261022_conclusion.md) |

[完整材料表](catalog.md)与[路径保护表](protected.md)保留咨询、协议、审查、失败原因与恢复条件。当前通用执行边界见[代理运行规则](../../../agents/runtime.md)，不从旧执行回执推断现有进程活性。

[QA05后续准备补丁](originals/qa05_followup_prepared.diff)作为历史来源原样保存，补丁中的源码和私有目录仍按原版本解释，本次没有重新应用。
