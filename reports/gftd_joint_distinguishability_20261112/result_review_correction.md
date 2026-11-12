# 终态审查的 RHS 分项更正

原审查的有限差分小计误含 34,560 次参考 RHS。有限差分的正确小计为 **9,123,840** 次。下表直接按冻结执行回执的 `costs` 主细网格计数相加，三个分项之和与原总账一致。 [执行回执](../../results/gftd_joint_distinguishability_20261112/science/execution.json)

| 工作 | 主格 RHS | 细格 RHS | 小计 |
|---|---:|---:|---:|
| 参考 | 11,520 | 23,040 | 34,560 |
| 有限差分 | 3,041,280 | 6,082,560 | 9,123,840 |
| 两档对照及补偿核验 | 46,080 | 92,160 | 138,240 |
| 合计 | 3,098,880 | 6,197,760 | 9,296,640 |

更正仅涉及原报告与核查记录中后加的有限差分 RHS 小计。原冻结终态、独立数组复算、12,912 次历史映射及两档未决结论继续有效。原报告及 JSON 保留，分项采用更正表与 v2。v2 SHA-256 为 `ebc7380be6a62bebaec884d3562d026ec8d8b1901ba2b80e20b20a244d3eedd7`。 [原审查](result_review.md) · [原核查记录](../../results/gftd_joint_distinguishability_20261112/result_audit.json) · [更正核查记录](../../results/gftd_joint_distinguishability_20261112/result_audit_v2.json)

本次只读冻结 JSON 并重新求和，未重复数组审查、运行模型或调用求解器。
