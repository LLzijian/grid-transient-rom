# 10-19已知支撑对照入口

本组入口对应已闭合的历史任务。原始字节、失败记录和接受范围保留；源码里的ROOT、同目录导入与相对输入仍按原版本解释。

**阅读先看[所属报告主题](../../../reports/history/closure_discovery/README.md)。需要恢复旧执行环境时，在单独目录检出完整提交 `145847532bacd3454dca8869e819778efc8258f6`，按原目录和合法输入核查；本目录只保存原件，不能直接运行。**

归档不恢复训练、val/test或新FOM权限；不重新执行旧检查以验证目录整理。

| 原入口 | 用途及原报告 |
|---|---|
| [closure_known_controls_20261019.py](closure_known_controls_20261019.py) | Run the registered P1 known-support controls sequentially within one cap.；[原报告](../../../reports/history/closure_discovery/originals/mechanism_20261019/closure_mechanism_20261019_progress.md) |

[迁移与回退回执](../../../results/repository_organization/20261109_history07/migration.json)含本地字节哈希及完整旧版本标识。
