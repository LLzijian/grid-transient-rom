# GFTD 固定初态 DP8 小试独立实现审查

**接受固定初态小试释放。** `static_004_selected` 满足分支 A 的触发条件。该点原 1 s 未来主细步 RMS 差为 1.2714047e-5，超过 1e-5 门槛，未来三参考一致性通过，RK4 的两个相邻收敛阶为 4.0008 和 4.0004。冻结诊断已完成 49/49 个点。 [触发点证据](../../results/gftd_initialization_precision_20261109/states/static_004_selected.json) · [诊断完整性回执](../../results/gftd_initialization_precision_20261109/diagnostic_integrity.json)

- **同点、同方程和原六门。** 小试从已认证的冻结状态出发，重建并逐字节核对选择清单，使用原模型 RHS。历史实际步长为 0.0025/0.00125 s，未来为 0.01/0.005 s。固定 DP8 只用八阶主公式，每个完成步调用 RHS 12 次。两次全状态往返、历史观测 RMS、未来观测 RMS 和峰值沿用原门槛。p 门引用已验证的冻结状态硬锚。 [固定步与资格](../../experiments/gftd_initialization_precision_20261109/fixed_state.py#L73) · [八阶公式](../../src/kiki/numerics.py#L46)
- **分开统计原成功点和阶段对照。** 49 个点包括 44 个原选定点及另外 5 个 stage1 对照。44 个原选定点中，5 个为原合格输出。只有旧失败的原选定点新通过六门，且 5 个原合格输出均通过，才有资格进入完整初始化小试。stage1 对照的退化单列，stage1 对照通过不计为旧失败恢复。缺失、异常和门槛失败均留在结果中。 [分支判据](../../experiments/gftd_initialization_precision_20261109/fixed_state.py#L130)
- **来源、成本与启动。** 入口核对父释放、诊断执行和选择清单的哈希、49 个点的身份、触发记录、当前源码与 DP8 系数。每点保存实际 RHS 调用、成功和中断步、最后接受态及失败信息。启动凭据独占写入，第二次启动拒绝。候选资格由小试结果决定，完整初始化和全体验证各有独立释放。 [释放校验](../../experiments/gftd_initialization_precision_20261109/fixed_state.py#L32) · [执行与记录](../../experiments/gftd_initialization_precision_20261109/fixed_state.py#L140)

核对的入口 SHA-256 为 `bcc69e55aec2f76fd4c9ed1cbe988f1436df5e45c4564e3660d7594af2e13a28`，测试为 `7a24611209bde3d01862f9042e9aebb20c7f78785232003ef5c1ee3e39a07c16`，DP8 内核所在文件为 `09363eeee4679b12891340f198dabc91971f76dd8f0380e2ef30a3356dab1f87`。本次审查沿用 5 项人工测试回执，未执行真实积分或读取私有载荷。 [人工检查回执](../../results/gftd_initialization_precision_20261109/fixed_state_qualification.json)
