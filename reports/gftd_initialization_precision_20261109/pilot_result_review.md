# GFTD 固定 DP8 完整初始化小试独立结果复核

**满足全体验证的释放条件。** 44 个冻结观测包各完成一次完整初始化。原 5 个合格包全部保留资格，新增 25 个通过原六门，另有 14 个包因指数溢出在 stage1 资格检查中阻断。全体阶段沿用同源配置，复用 44 份小试结果原字节，并对剩余 550 包各运行一次。 [小试执行回执](../../results/gftd_initialization_precision_20261109/candidate/pilot/execution.json) · [完整性回执](../../results/gftd_initialization_precision_20261109/pilot_integrity.json)

| train fit297 小试子集 | 已执行 | 原合格 | 本次合格 | 新增合格 |
|---|---:|---:|---:|---:|
| static 稳定样本 | 12 | 4 | 12 | 8 |
| modulated 稳定样本 | 20 | 1 | 18 | 17 |
| modulated 失稳样本 | 12 | 0 | 0 | 0 |

小试未抽中 static 失稳样本。14 个阻断包与原指数溢出分组身份一致。小试范围由旧结果分层确定，全体覆盖和长时延拓要在下一阶段逐项判定。 [冻结清单](../../results/gftd_initialization_precision_20261109/selection.json)

- **身份与参数。** 44 个独占启动凭据、任务回执和诊断归档的事件身份逐项相符。候选模型序列化内容除方法包装和初始化诊断字段外与父模型一致，原 fit297 配置完整保存在包装内。核对 13 个执行源码与测试文件、57 个来源文件、3 个父清单、3 份独立审查及 2 份工程检查回执的释放哈希，均相符。 [小试释放](../../results/gftd_initialization_precision_20261109/pilot_release.json)
- **原门与工作量。** 从每包保存值重判已执行的六道门槛，核对 30 个通过状态的 float64 字节哈希及真实 p₀ 硬锚。每个 simple 从零开始，执行的 smooth 从原可行点开始，各阶段均在 600/1200 次残差上限内。逐项成本合计 18,890 次残差、9,430,245 次 RHS、19,248 次积分尝试及 19,234 次完成。四个工作进程的任务 CPU 合计 1,054.859 s，宿主耗时 305.441 s。 [小试完整性回执](../../results/gftd_initialization_precision_20261109/pilot_integrity.json) · [宿主回执](../../results/gftd_initialization_precision_20261109/pilot_host_receipt.json)

复核使用原因果包的 p₀、冻结模型配置和已保存诊断与状态，未执行额外初始化或积分，未读取真实未来。小试执行回执 SHA-256 为 `4d3df9be60c875e68531c64b747b7c0c37d3ed1579d66be8f9c8f07330cce126`。
