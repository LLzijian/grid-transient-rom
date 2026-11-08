# Loop 0018：Pilot 数据可用，进入 TASK2

2026-09-08。沿用验收代码`62cc1ea`，不新增生成基础设施。
384尝试，361接受（305稳定、56物理LOS），23拒绝（22数值失败、1结局未确认）。
所有接受事件共享同一post-fault系统；每个事件经独立QA重读并核对。
train/validation/test为227/74/60例，对应15/5/4母线组，母线/线路分组无泄漏。
稳定Pm峰峰变化最大4.6683446 system pu；220动态状态及479代数量完整。

执行：`python experiments/audit_private_dataset.py private_data/ieee39_postfault`，
结果PASS，`Pilot sufficient for M2/M3-Pm: True`。
私有输出目录内保存dataset.h5、dataset_index.csv、splits.json、qa_report.md。
更多统计和限制见`reports/private_dataset_build_report.md`。

接受的是Pilot可用于研究，不是模型预测或稳定性结论。下一循环M2 clean baseline，
之后M3-Pm → M4；不为增加数据量推迟模型实验。
