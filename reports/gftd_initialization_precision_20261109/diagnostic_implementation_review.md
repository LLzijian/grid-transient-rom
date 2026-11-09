# GFTD 固定初态诊断实现独立审查

**接受首次诊断释放。** 冻结实现能按协议执行固定初态的 RK4 步长比较、双方向连续参考和失败子步取证。释放绑定已完成的 prepare 清单、运行源码和父证据。修正分支另行审查。

| 审查对象 | SHA-256 |
|---|---|
| [诊断入口](../../experiments/gftd_initialization_precision_20261109/run.py) | `0385011e15d696a9ed51ea552605c3b5f937bcdb863c299a442bce9271700342` |
| [人工测试](../../tests/test_gftd_initialization_precision_diagnostic.py) | `b4f91587db9341aa989e7bfe431fa2b088fab2184a475607b5a62da15c5a47e4` |
| [前瞻协议](../../campaigns/gftd_initialization_precision_20261109/protocol.md) | `a1aab626927d428e2c6f3c7821a4bd7525c0378248fe64183faf38bff190c649` |

- **状态和因果输入。** prepare 从父档案逐字节认证模型、原 41 项和续跑 553 项诊断、原成功状态及因果观测包，按已保存的变量校验状态构造。诊断阶段重建并匹配冻结选择清单，每项积分记录起点的状态哈希。原成功和失败状态均保持原阶段身份。p 硬锚在加载时检查，原六门表里的 p=0 引用这项前置检查。数值入口只接收冻结模型和状态。事件 ID、稳定标签和旧诊断用于对齐、选例及报告。 [输入认证与选择](../../experiments/gftd_initialization_precision_20261109/run.py#L192)
- **积分和参考。** 固定网格复用原 `core.rk4` 的四级运算，保存 0.01 s 公共网格、反向历史、从各自历史起点开始的正向往返及 1 s 未来。历史数组按积分方向排列，同一比较内方向一致。两档 Radau 和 DOP853 使用登记容差、方向和最大步长，显式计数含有限差分及密集输出的 RHS 调用。参考须正常到达端点并覆盖有限网格，提前停止的参考不进入两两一致性。原门与更细网格的诊断证据分列。 [积分及参考判定](../../experiments/gftd_initialization_precision_20261109/run.py#L375)
- **归因边界。** `reference.qualified` 判断观测投影，程序另存五组状态分量差异和参考轨迹，`continuous_growth` 留空供结果审查。增长判断需核对同一分量、时间方向及共同有效区间。反向历史的快衰减证据读 `decaying_in_integration_direction_rk4_outside`，物理负实部在反向时间会增长。异常记录保存前一接受态、失败子步和边的指数、中心化势能、活跃标记。异常报告可用同一冻结模型的有效调制系数乘保存的势能，单列 `beta*psi`。求解到限和异常保持各自状态。 [谱与异常证据](../../experiments/gftd_initialization_precision_20261109/run.py#L513)
- **成本及中断。** prepare 和诊断各自独占启动凭据，每点保存轨迹、证据、尝试和完成成本。相同目录第二次启动会拒绝。中断时先对账已有证据和最后启动项，如需继续则由 root 单独释放可确认从未启动的工作单元。强制终止后未写出的成本记 unknown，保留中断状态。 [写入与启动凭据](../../experiments/gftd_initialization_precision_20261109/run.py#L104) · [逐点回执](../../experiments/gftd_initialization_precision_20261109/run.py#L659)

本次独立审查核对源码与测试内容，沿用 root 提供的 7 项人工测试和完整人工诊断回执。未读取私有载荷，未重复执行测试或真实积分，自动续跑不属于本次审查的功能。
